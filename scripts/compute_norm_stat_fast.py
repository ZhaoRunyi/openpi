"""Fast norm stats for Piper configs with exact SLAIPiperInputs alignment.

This script speeds up norm-stat computation by reading episode parquet files directly while
preserving the state/action semantics of the original `compute_norm_stats.py` pipeline for
`LeRobotSLAIPiperDataConfig`.

Compared with the generic fast script, this implementation:
- reconstructs the same action-horizon sequences as LeRobotDataset,
- reuses the same Piper state/action extraction logic as `SLAIPiperInputs`,
- preserves drop-last and shuffled-subsampling behavior from the original script.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import tyro

import compute_norm_stats as reference_stats
from openpi.policies import slai_piper_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as _transforms


STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.parquet$")


def _resolve_base_dir(repo_id: str, base_dir: str | None) -> Path:
    if base_dir is not None:
        path = Path(base_dir)
    else:
        path = Path(os.environ.get("HF_LEROBOT_HOME", "/workspace/data")) / repo_id
    if not path.exists():
        raise ValueError(f"Base directory does not exist: {path}")
    return path


def _resolve_dataset_roots(
    piper_datas: tuple[_config.LeRobotSLAIPiperDataConfig, ...],
    base_dir: str | None,
) -> tuple[Path, ...]:
    if len(piper_datas) == 1:
        return (_resolve_base_dir(piper_datas[0].repo_id, base_dir),)

    root = Path(base_dir) if base_dir is not None else Path(os.environ.get("HF_LEROBOT_HOME", "/workspace/data"))
    dataset_roots = tuple(root / piper_data.repo_id for piper_data in piper_datas)
    missing = [str(path) for path in dataset_roots if not path.exists()]
    if missing:
        raise ValueError(f"Dataset directories do not exist: {missing}")
    return dataset_roots


def _validate_piper_data(data_factory: _config.DataConfigFactory) -> _config.LeRobotSLAIPiperDataConfig:
    if not isinstance(data_factory, _config.LeRobotSLAIPiperDataConfig):
        raise TypeError(
            "compute_norm_stats_fast_piper.py only supports LeRobotSLAIPiperDataConfig, "
            f"got {type(data_factory).__name__}."
        )
    if tuple(data_factory.action_sequence_keys) != ("action",):
        raise ValueError(
            "compute_norm_stats_fast_piper.py currently only supports action_sequence_keys=('action',)."
        )
    return data_factory


def _get_piper_data_groups(
    config: _config.TrainConfig,
) -> tuple[tuple[_config.LeRobotSLAIPiperDataConfig, ...], ...]:
    data_factories = tuple(config.datasets) or (config.data,)
    piper_datas = tuple(_validate_piper_data(data_factory) for data_factory in data_factories)
    if config.norm_mode == "per_dataset":
        return tuple((piper_data,) for piper_data in piper_datas)
    return (piper_datas,)


def _validate_piper_config(config: _config.TrainConfig) -> _config.LeRobotSLAIPiperDataConfig:
    groups = _get_piper_data_groups(config)
    if len(groups) != 1 or len(groups[0]) != 1:
        raise ValueError("This comparison mode only supports a single Piper dataset.")
    return groups[0][0]


def _load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")

    lengths: dict[int, int] = {}
    with episodes_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            lengths[int(payload["episode_index"])] = int(payload["length"])

    if not lengths:
        raise ValueError(f"No episode metadata found in {episodes_path}")
    return lengths


def _collect_episode_files(dataset_root: Path) -> dict[int, Path]:
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data directory: {data_root}")

    files: dict[int, Path] = {}
    for parquet_file in sorted(data_root.rglob("*.parquet")):
        match = EPISODE_FILE_RE.search(parquet_file.name)
        if match is None:
            continue
        files[int(match.group(1))] = parquet_file

    if not files:
        raise ValueError(f"No parquet files found under {data_root}")
    return files


def _selected_frame_count(total_frames: int, batch_size: int, max_frames: int | None) -> int:
    if max_frames is not None and max_frames < total_frames:
        num_batches = max_frames // batch_size
        if num_batches == 0:
            raise ValueError(
                f"max_frames={max_frames} is too small for batch_size={batch_size}; original script would emit 0 batches."
            )
    else:
        num_batches = total_frames // batch_size
        if num_batches == 0:
            raise ValueError(
                f"Dataset with {total_frames} frames is too small for batch_size={batch_size}; original script would fail."
            )
    return num_batches * batch_size


def _build_selected_indices(total_frames: int, batch_size: int, max_frames: int | None, seed: int) -> np.ndarray:
    num_samples = _selected_frame_count(total_frames, batch_size, max_frames)
    if max_frames is not None and max_frames < total_frames:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(total_frames, generator=generator)[:num_samples].cpu().numpy()

    return np.arange(num_samples, dtype=np.int64)


def _build_weighted_sample_indices(
    frame_counts: tuple[int, ...],
    dataset_weights: tuple[float, ...] | None,
    batch_size: int,
    max_frames: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = tuple(1.0 for _ in frame_counts) if dataset_weights is None else dataset_weights
    if len(weights) != len(frame_counts):
        raise ValueError("dataset_weights must have the same length as datasets.")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("dataset_weights must be non-negative and contain a positive total weight.")

    num_samples = _selected_frame_count(sum(frame_counts), batch_size, max_frames)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset_indices = torch.multinomial(
        torch.as_tensor(weights, dtype=torch.float64),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    ).cpu().numpy()
    frame_indices = np.empty(num_samples, dtype=np.int64)
    for dataset_index, frame_count in enumerate(frame_counts):
        mask = dataset_indices == dataset_index
        frame_indices[mask] = torch.randint(frame_count, (int(mask.sum()),), generator=generator).cpu().numpy()
    return dataset_indices.astype(np.int64, copy=False), frame_indices


def _load_needed_episode_arrays(
    episode_files: dict[int, Path],
    needed_episode_ids: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    states: dict[int, np.ndarray] = {}
    actions: dict[int, np.ndarray] = {}

    for episode_id in tqdm.tqdm(sorted(int(ep) for ep in needed_episode_ids), desc="Loading Piper episodes"):
        if episode_id not in episode_files:
            raise KeyError(f"Episode {episode_id} missing parquet file.")

        df = pd.read_parquet(episode_files[episode_id], columns=[STATE_COLUMN, ACTION_COLUMN])
        states[episode_id] = np.stack(df[STATE_COLUMN].to_numpy()).astype(np.float32, copy=False)
        actions[episode_id] = np.stack(df[ACTION_COLUMN].to_numpy()).astype(np.float32, copy=False)

    return states, actions


def _build_batch_arrays(
    batch_indices: np.ndarray,
    episode_offsets: np.ndarray,
    episode_states: dict[int, np.ndarray],
    episode_actions: dict[int, np.ndarray],
    action_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.searchsorted(episode_offsets[1:], batch_indices, side="right")
    local_indices = batch_indices - episode_offsets[episode_ids]
    horizon_offsets = np.arange(action_horizon, dtype=np.int64)

    first_episode = int(episode_ids[0])
    full_state_dim = episode_states[first_episode].shape[-1]
    full_action_dim = episode_actions[first_episode].shape[-1]

    batch_states = np.empty((len(batch_indices), full_state_dim), dtype=np.float32)
    batch_actions = np.empty((len(batch_indices), action_horizon, full_action_dim), dtype=np.float32)

    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        episode_id = int(episode_id)
        local_batch_indices = local_indices[mask]
        episode_state = episode_states[episode_id]
        episode_action = episode_actions[episode_id]

        batch_states[mask] = episode_state[local_batch_indices]
        query_indices = np.clip(
            local_batch_indices[:, None] + horizon_offsets[None, :],
            0,
            len(episode_action) - 1,
        )
        batch_actions[mask] = episode_action[query_indices]

    return batch_states, batch_actions


def _build_dataset_metadata(dataset_root: Path) -> dict:
    episode_lengths = _load_episode_lengths(dataset_root)
    episode_files = _collect_episode_files(dataset_root)
    episode_ids = np.array(sorted(episode_lengths), dtype=np.int64)
    lengths = np.array([episode_lengths[int(episode_id)] for episode_id in episode_ids], dtype=np.int64)
    episode_offsets = np.concatenate([[0], np.cumsum(lengths)])
    return {
        "episode_files": episode_files,
        "episode_offsets": episode_offsets,
        "total_frames": int(episode_offsets[-1]),
    }


def _check_mixed_output_dims(piper_datas: tuple[_config.LeRobotSLAIPiperDataConfig, ...]) -> None:
    dims = [
        (
            slai_piper_policy.get_space_dim(piper_data.state_space),
            slai_piper_policy.get_space_dim(piper_data.action_space),
        )
        for piper_data in piper_datas
    ]
    if len(set(dims)) > 1:
        raise ValueError(f"Cannot compute mixed norm stats with inconsistent Piper Inputs output dimensions: {dims}")


def _process_piper_group(
    config: _config.TrainConfig,
    piper_datas: tuple[_config.LeRobotSLAIPiperDataConfig, ...],
    dataset_roots: tuple[Path, ...],
    max_frames: int | None,
) -> tuple[dict[str, normalize.NormStats], tuple | None]:
    if len(piper_datas) > 1:
        _check_mixed_output_dims(piper_datas)

    metadata = tuple(_build_dataset_metadata(dataset_root) for dataset_root in dataset_roots)
    frame_counts = tuple(meta["total_frames"] for meta in metadata)
    if len(piper_datas) == 1:
        frame_indices = _build_selected_indices(frame_counts[0], config.batch_size, max_frames, config.seed)
        dataset_indices = np.zeros(len(frame_indices), dtype=np.int64)
    elif config.dataset_sampling_strategy == "concat_weighted":
        dataset_indices, frame_indices = _build_weighted_sample_indices(
            frame_counts,
            tuple(config.dataset_weights) if config.dataset_weights is not None else None,
            config.batch_size,
            max_frames,
            config.seed,
        )
    else:
        offsets = np.cumsum((0,) + frame_counts)
        frame_indices = _build_selected_indices(sum(frame_counts), config.batch_size, max_frames, config.seed)
        dataset_indices = np.searchsorted(offsets[1:], frame_indices, side="right").astype(np.int64, copy=False)
        frame_indices = frame_indices - offsets[dataset_indices]

    print(f"Selected frames: {len(frame_indices)}")
    print(f"Action horizon: {config.model.action_horizon}")
    caches = []
    for dataset_index, (piper_data, dataset_root, meta) in enumerate(
        zip(piper_datas, dataset_roots, metadata, strict=True)
    ):
        selected_for_dataset = frame_indices[dataset_indices == dataset_index]
        selected_episode_ids = np.unique(
            np.searchsorted(meta["episode_offsets"][1:], selected_for_dataset, side="right")
        )
        print(f"\nReading data from: {dataset_root}")
        print(f"Total frames: {meta['total_frames']}")
        print(f"State space: {piper_data.state_space}")
        print(f"Action space: {piper_data.action_space}")
        episode_states, episode_actions = _load_needed_episode_arrays(meta["episode_files"], selected_episode_ids)
        caches.append((meta["episode_offsets"], episode_states, episode_actions))

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch_start in tqdm.tqdm(
        range(0, len(frame_indices), config.batch_size),
        desc="Processing Piper batches",
        total=len(frame_indices) // config.batch_size,
    ):
        batch_dataset_indices = dataset_indices[batch_start : batch_start + config.batch_size]
        batch_frame_indices = frame_indices[batch_start : batch_start + config.batch_size]
        batch_state_inputs = None
        batch_action_inputs = None
        for dataset_index in np.unique(batch_dataset_indices):
            mask = batch_dataset_indices == dataset_index
            episode_offsets, episode_states, episode_actions = caches[int(dataset_index)]
            batch_states, batch_actions = _build_batch_arrays(
                batch_frame_indices[mask],
                episode_offsets,
                episode_states,
                episode_actions,
                config.model.action_horizon,
            )
            inputs = slai_piper_policy.extract_state_action_inputs(
                batch_states,
                batch_actions,
                state_space=piper_datas[int(dataset_index)].state_space,
                action_space=piper_datas[int(dataset_index)].action_space,
            )
            if batch_state_inputs is None or batch_action_inputs is None:
                batch_state_inputs = np.empty((len(batch_frame_indices), inputs["state"].shape[-1]), dtype=np.float32)
                batch_action_inputs = np.empty(
                    (len(batch_frame_indices), *inputs["actions"].shape[1:]), dtype=np.float32
                )
            batch_state_inputs[mask] = inputs["state"]
            batch_action_inputs[mask] = inputs["actions"]

        stats["state"].update(np.ascontiguousarray(batch_state_inputs))
        stats["actions"].update(np.ascontiguousarray(batch_action_inputs))

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    context = None
    if len(piper_datas) == 1:
        episode_offsets, episode_states, episode_actions = caches[0]
        context = (episode_offsets, episode_states, episode_actions, frame_indices)
    return norm_stats, context


def _compute_reference_stats(config: _config.TrainConfig, max_frames: int | None) -> dict[str, normalize.NormStats]:
    piper_data = _validate_piper_config(config)
    data_config = dataclasses.replace(
        piper_data.create_base_config(config.assets_dirs, config.model),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state": "observation.state",
                        "observation.images.cam_high": "observation.images.cam_high",
                        "observation.images.cam_left_wrist": "observation.images.cam_left_wrist",
                        "observation.images.cam_right_wrist": "observation.images.cam_right_wrist",
                        "actions": "action",
                        "task_index": "task_index",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        data_transforms=_transforms.Group(
            inputs=[
                slai_piper_policy.SLAIPiperInputs(
                    model_type=config.model.model_type,
                    state_space=piper_data.state_space,
                    action_space=piper_data.action_space,
                    image_space=piper_data.image_space,
                )
            ],
            outputs=[
                slai_piper_policy.SLAIPiperOutputs(
                    action_space=piper_data.action_space,
                )
            ],
        ),
        action_sequence_keys=piper_data.action_sequence_keys,
    )

    data_loader, num_batches = reference_stats.create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Reference original stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
    return {key: value.get_statistics() for key, value in stats.items()}


def _summarize_diff(
    fast_stats: dict[str, normalize.NormStats],
    reference: dict[str, normalize.NormStats],
    *,
    atol: float,
    rtol: float,
) -> None:
    all_close = True
    for key in ("state", "actions"):
        print(f"\n[{key}]")
        for field in ("mean", "std", "q01", "q99"):
            fast_value = np.asarray(getattr(fast_stats[key], field))
            reference_value = np.asarray(getattr(reference[key], field))
            max_abs = float(np.max(np.abs(fast_value - reference_value)))
            is_close = bool(np.allclose(fast_value, reference_value, atol=atol, rtol=rtol))
            all_close = all_close and is_close
            print(f"  {field}: allclose={is_close} max_abs_diff={max_abs:.6e}")

    if not all_close:
        raise AssertionError(
            "fast_piper stats did not match the reference stats within tolerance. "
            f"(atol={atol}, rtol={rtol})"
        )


def _compute_reference_stats_with_slaipiperinputs(
    config: _config.TrainConfig,
    piper_data: _config.LeRobotSLAIPiperDataConfig,
    episode_offsets: np.ndarray,
    episode_states: dict[int, np.ndarray],
    episode_actions: dict[int, np.ndarray],
    selected_indices: np.ndarray,
) -> dict[str, normalize.NormStats]:
    """Slow reference path that calls SLAIPiperInputs sample-by-sample."""
    transform = slai_piper_policy.SLAIPiperInputs(
        model_type=config.model.model_type,
        state_space=piper_data.state_space,
        action_space=piper_data.action_space,
        image_space=piper_data.image_space,
    )
    dummy_image = np.zeros((1, 1, 3), dtype=np.uint8)
    horizon_offsets = np.arange(config.model.action_horizon, dtype=np.int64)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}

    for batch_start in tqdm.tqdm(
        range(0, len(selected_indices), config.batch_size),
        desc="Reference SLAIPiperInputs batches",
        total=len(selected_indices) // config.batch_size,
    ):
        batch_indices = selected_indices[batch_start : batch_start + config.batch_size]
        episode_ids = np.searchsorted(episode_offsets[1:], batch_indices, side="right")
        local_indices = batch_indices - episode_offsets[episode_ids]

        batch_states = []
        batch_actions = []
        for episode_id, local_index in zip(episode_ids, local_indices, strict=True):
            episode_id = int(episode_id)
            local_index = int(local_index)
            episode_action = episode_actions[episode_id]
            query_indices = np.clip(local_index + horizon_offsets, 0, len(episode_action) - 1)
            transformed = transform(
                {
                    "observation.state": episode_states[episode_id][local_index],
                    "actions": episode_action[query_indices],
                    "observation.images.cam_high": dummy_image,
                    "observation.images.cam_left_wrist": dummy_image,
                    "observation.images.cam_right_wrist": dummy_image,
                }
            )
            batch_states.append(transformed["state"])
            batch_actions.append(transformed["actions"])

        stats["state"].update(np.stack(batch_states, axis=0))
        stats["actions"].update(np.stack(batch_actions, axis=0))

    return {key: value.get_statistics() for key, value in stats.items()}


def main(
    config_name: str,
    base_dir: str | None = None,
    max_frames: int | None = None,
    compare_with_reference: bool = False,
    compare_with_original: bool = False,
    compare_atol: float = 1e-6,
    compare_rtol: float = 1e-6,
    save: bool = True,
):
    """Compute fast-but-aligned norm stats for a Piper config."""
    config = _config.get_config(config_name)
    piper_data_groups = _get_piper_data_groups(config)

    for piper_datas in piper_data_groups:
        dataset_roots = _resolve_dataset_roots(piper_datas, base_dir)
        norm_stats, reference_context = _process_piper_group(config, piper_datas, dataset_roots, max_frames)

        if save:
            for piper_data in piper_datas:
                output_path = Path(piper_data.assets.assets_dir or config.assets_dirs) / (
                    piper_data.assets.asset_id or piper_data.repo_id
                )
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"\nWriting stats to: {output_path}")
                normalize.save(output_path, norm_stats)

        if compare_with_reference:
            if reference_context is None:
                raise ValueError("compare_with_reference is only supported for per-dataset/single-dataset groups.")
            episode_offsets, episode_states, episode_actions, selected_indices = reference_context
            reference = _compute_reference_stats_with_slaipiperinputs(
                config,
                piper_datas[0],
                episode_offsets,
                episode_states,
                episode_actions,
                selected_indices,
            )
            _summarize_diff(norm_stats, reference, atol=compare_atol, rtol=compare_rtol)
            print("\nComparison against slow SLAIPiperInputs reference passed.")

    if compare_with_original:
        if base_dir is not None:
            raise ValueError(
                "compare_with_original requires base_dir=None so the reference path matches the process-level "
                "HF_LEROBOT_HOME used by compute_norm_stats.py."
            )
        if len(piper_data_groups) != 1 or len(piper_data_groups[0]) != 1:
            raise ValueError("compare_with_original is only supported for a single Piper dataset.")
        reference = _compute_reference_stats(config, max_frames)
        _summarize_diff(norm_stats, reference, atol=compare_atol, rtol=compare_rtol)
        print("\nComparison against compute_norm_stats.py passed.")


if __name__ == "__main__":
    tyro.cli(main)
