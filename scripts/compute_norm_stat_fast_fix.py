"""Compute normalization statistics for EmbodiChain LeRobot datasets quickly.

This script is a fast path for datasets produced by Embodied_Challenge /
EmbodiChain. It avoids video decoding and the generic HuggingFace parquet scan,
but keeps the normalization *semantics* aligned with ``compute_norm_stats.py``:

- read only episodes declared by ``meta/episodes.jsonl``;
- map ``observation.qpos`` -> ``state`` and ``action`` -> ``actions``;
- build the same action chunks as LeRobot ``delta_timestamps``;
- apply the config's ``DeltaActions`` transform when present;
- update ``RunningStats`` with the same batch size/drop-last behavior.
"""

from __future__ import annotations

from collections import OrderedDict
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except Exception:  # pragma: no cover - only used when LeRobot is unavailable.
    HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_dataset_dir(base_dir: str | None, repo_id: str | None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser()
    if repo_id is None:
        raise ValueError("Either base_dir must be provided or config must have repo_id.")
    return Path(HF_LEROBOT_HOME) / repo_id


def _resolve_dataset_dirs(base_dir: str | None, data_configs: tuple[_config.DataConfig, ...]) -> list[Path]:
    if len(data_configs) == 1:
        return [_resolve_dataset_dir(base_dir, data_configs[0].repo_id)]
    root = Path(base_dir).expanduser() if base_dir is not None else Path(HF_LEROBOT_HOME)
    return [root / str(data_config.repo_id) for data_config in data_configs]


def _output_path(
    config: _config.TrainConfig,
    data_factory: _config.DataConfigFactory,
    data_config: _config.DataConfig,
) -> Path:
    return Path(data_factory.assets.assets_dir or config.assets_dirs) / (
        data_config.asset_id or data_config.repo_id
    )


def _episode_file_path(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel_path = data_path.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)
    return dataset_dir / rel_path


def _detect_columns(info: dict[str, Any], state_col: str | None, action_col: str | None) -> tuple[str, str]:
    features = info.get("features", {})

    if state_col is None:
        for candidate in ("observation.qpos", "observation.state", "state"):
            if candidate in features:
                state_col = candidate
                break
    if action_col is None:
        for candidate in ("action", "actions"):
            if candidate in features:
                action_col = candidate
                break

    if state_col is None or action_col is None:
        raise ValueError(
            "Could not infer state/action parquet columns. "
            f"Available feature keys: {sorted(features.keys())}"
        )
    if state_col not in features:
        raise ValueError(f"State column '{state_col}' is not present in meta/info.json features.")
    if action_col not in features:
        raise ValueError(f"Action column '{action_col}' is not present in meta/info.json features.")

    return state_col, action_col


def _validate_fast_transforms(data_config: _config.DataConfig) -> None:
    transform_names = {
        type(transform).__name__
        for transform in data_config.data_transforms.inputs
        if not isinstance(transform, transforms.DeltaActions)
    }
    if not transform_names <= {"EmbodiChainInputs", "LiberoInputs", "SLAIPiperInputs"}:
        raise NotImplementedError(
            "This fast script only supports an explicit norm-stat fast registry "
            f"(got {sorted(transform_names)}). Use compute_norm_stats.py for this config."
        )


def _apply_fast_transform(transform: transforms.DataTransformFn, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    transform_name = type(transform).__name__
    if transform_name in {"EmbodiChainInputs", "LiberoInputs"}:
        return data
    if transform_name == "SLAIPiperInputs":
        from openpi.policies import slai_piper_policy

        return slai_piper_policy.extract_state_action_inputs(
            data["state"],
            data.get("actions"),
            state_space=transform.state_space,
            action_space=transform.action_space,
        )
    if isinstance(transform, transforms.DeltaActions):
        return transform(data)
    raise NotImplementedError(f"Unsupported fast norm transform: {transform_name}")


def _apply_fast_transforms(
    data_config: _config.DataConfig,
    states: np.ndarray,
    action_chunks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = {"state": states, "actions": action_chunks}
    for transform in data_config.data_transforms.inputs:
        data = _apply_fast_transform(transform, data)
    return np.asarray(data["state"], dtype=np.float32), np.asarray(data["actions"], dtype=np.float32)


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_numpy()
    if len(values) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(np.stack(values), dtype=np.float32)


def _load_episode_arrays(path: Path, state_col: str, action_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path, columns=[state_col, action_col])
    states = _stack_column(df, state_col)
    actions = _stack_column(df, action_col)
    if len(states) != len(actions):
        raise ValueError(f"State/action row count mismatch in {path}: {len(states)} != {len(actions)}")
    return states, actions


def _build_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    frame_indices = np.arange(len(actions))[:, None]
    offsets = np.arange(action_horizon)[None, :]
    query_indices = np.minimum(frame_indices + offsets, len(actions) - 1)
    return actions[query_indices]


def _warn_about_extra_parquet(dataset_dir: Path, expected_files: list[Path]) -> None:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return

    actual = {path.resolve() for path in data_dir.glob("**/*.parquet")}
    expected = {path.resolve() for path in expected_files}
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)

    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} episode parquet files declared by metadata:\n{preview}")

    if extra:
        preview = "\n".join(str(path) for path in extra[:10])
        print(
            "\nWarning: found parquet files under data/ that are not declared by meta/episodes.jsonl. "
            "They will be ignored by this fast script because LeRobot would otherwise read them via data_dir.\n"
            f"Extra parquet count: {len(extra)}\n{preview}\n"
        )


class _EpisodeCache:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: OrderedDict[tuple[Path, str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, path: Path, state_col: str, action_col: str) -> tuple[np.ndarray, np.ndarray]:
        key = (path, state_col, action_col)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        arrays = _load_episode_arrays(path, state_col, action_col)
        self._cache[key] = arrays
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return arrays


@dataclasses.dataclass(frozen=True)
class _FastEpisode:
    episode: dict[str, Any]
    path: Path
    state_col: str
    action_col: str
    data_config: _config.DataConfig


def _process_sequential(
    *,
    stats: dict[str, normalize.RunningStats],
    records: list[_FastEpisode],
    action_horizon: int,
    batch_size: int,
    max_samples: int,
) -> tuple[int, int]:
    buffered_states: list[np.ndarray] = []
    buffered_actions: list[np.ndarray] = []
    buffered_count = 0
    processed_samples = 0
    processed_files = 0

    for record in tqdm(records, desc="Processing episodes"):
        if processed_samples >= max_samples:
            break

        full_states, full_actions = _load_episode_arrays(record.path, record.state_col, record.action_col)
        expected_length = int(record.episode["length"])
        if len(full_states) != expected_length:
            raise ValueError(
                f"Episode {record.episode['episode_index']} length mismatch: "
                f"metadata says {expected_length}, parquet has {len(full_states)} rows"
            )

        remaining = max_samples - processed_samples
        states = full_states[:remaining]

        action_chunks = _build_action_chunks(full_actions, action_horizon)
        action_chunks = action_chunks[: len(states)]
        states, action_chunks = _apply_fast_transforms(record.data_config, states, action_chunks)

        buffered_states.append(states)
        buffered_actions.append(action_chunks)
        buffered_count += len(states)
        processed_samples += len(states)
        processed_files += 1

        while buffered_count >= batch_size:
            states_batch = np.concatenate(buffered_states, axis=0)
            actions_batch = np.concatenate(buffered_actions, axis=0)
            stats["state"].update(states_batch[:batch_size])
            stats["actions"].update(actions_batch[:batch_size])

            states_remainder = states_batch[batch_size:]
            actions_remainder = actions_batch[batch_size:]
            buffered_states = [states_remainder] if len(states_remainder) else []
            buffered_actions = [actions_remainder] if len(actions_remainder) else []
            buffered_count = len(states_remainder)

    if buffered_count:
        raise RuntimeError(
            "Internal batching error: sequential processing ended with a non-empty buffer. "
            "The sample count should already be a multiple of batch_size."
        )

    return processed_files, processed_samples


def _process_shuffled_subset(
    *,
    stats: dict[str, normalize.RunningStats],
    records: list[_FastEpisode],
    action_horizon: int,
    batch_size: int,
    max_samples: int,
    frame_weights: np.ndarray | None = None,
) -> tuple[int, int]:
    import torch

    lengths = np.asarray([int(record.episode["length"]) for record in records], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total_frames = int(lengths.sum())
    generator = torch.Generator()
    generator.manual_seed(0)
    if frame_weights is None:
        selected = torch.randperm(total_frames, generator=generator)[:max_samples].numpy()
    else:
        selected = torch.multinomial(
            torch.as_tensor(frame_weights, dtype=torch.double),
            max_samples,
            replacement=True,
            generator=generator,
        ).numpy()

    cache = _EpisodeCache(max_size=64)
    files_seen: set[Path] = set()

    for start in tqdm(range(0, len(selected), batch_size), desc="Processing shuffled batches"):
        batch_indices = selected[start : start + batch_size]
        state_batch: list[np.ndarray] = []
        action_batch: list[np.ndarray] = []

        for global_idx in batch_indices:
            ep_pos = int(np.searchsorted(starts, global_idx, side="right") - 1)
            local_idx = int(global_idx - starts[ep_pos])
            record = records[ep_pos]
            states, actions = cache.get(record.path, record.state_col, record.action_col)
            files_seen.add(record.path)

            query_indices = np.minimum(local_idx + np.arange(action_horizon), len(actions) - 1)
            transformed_state, transformed_actions = _apply_fast_transforms(
                record.data_config,
                states[local_idx][None, :],
                actions[query_indices][None, :, :],
            )
            state_batch.append(transformed_state[0])
            action_batch.append(transformed_actions[0])

        stats["state"].update(np.stack(state_batch, axis=0))
        stats["actions"].update(np.stack(action_batch, axis=0))

    return len(files_seen), max_samples


def main(
    config_name: str,
    base_dir: str | None = None,
    max_frames: int | None = None,
    state_col: str | None = None,
    action_col: str | None = None,
):
    """Compute norm stats for an EmbodiChain LeRobot dataset.

    Args:
        config_name: OpenPI train config name.
        base_dir: Dataset directory. Defaults to ``$HF_LEROBOT_HOME/<repo_id>``.
        max_frames: Match ``compute_norm_stats.py`` sampling semantics when provided.
        state_col: Optional override for the state parquet column.
        action_col: Optional override for the action parquet column.
    """
    config = _config.get_config(config_name)
    data_factories = tuple(config.datasets) or (config.data,)
    data_factory_groups = (data_factories,)
    if config.norm_mode == "per_dataset":
        data_factory_groups = tuple((factory,) for factory in data_factories)

    for data_factories, data_configs in (
        (factories, tuple(factory.create(config.assets_dirs, config.model) for factory in factories))
        for factories in data_factory_groups
    ):
        for data_config in data_configs:
            if data_config.rlds_data_dir is not None:
                raise NotImplementedError("This fast script only supports local LeRobot parquet datasets, not RLDS.")
            if len(data_config.action_sequence_keys) != 1:
                raise NotImplementedError(
                    "This fast script currently supports exactly one action sequence key. "
                    f"Got: {data_config.action_sequence_keys}"
                )
            _validate_fast_transforms(data_config)

        dataset_dirs = _resolve_dataset_dirs(base_dir, data_configs)
        records: list[_FastEpisode] = []
        repo_frame_counts: list[int] = []
        detected_schemas: list[tuple[str, str, int, int]] = []
        first_info: dict[str, Any] | None = None

        for dataset_dir, data_config in zip(dataset_dirs, data_configs, strict=True):
            info_path = dataset_dir / "meta" / "info.json"
            episodes_path = dataset_dir / "meta" / "episodes.jsonl"
            if not info_path.exists() or not episodes_path.exists():
                raise FileNotFoundError(f"Expected LeRobot metadata under {dataset_dir / 'meta'}")

            info = _read_json(info_path)
            first_info = first_info or info
            dataset_state_col, dataset_action_col = _detect_columns(info, state_col, action_col)
            state_dim = int(info["features"][dataset_state_col]["shape"][-1])
            action_dim = int(info["features"][dataset_action_col]["shape"][-1])
            detected_schemas.append((dataset_state_col, dataset_action_col, state_dim, action_dim))

            dataset_episodes = sorted(_read_jsonl(episodes_path), key=lambda ep: int(ep["episode_index"]))
            dataset_files = [_episode_file_path(dataset_dir, info, int(ep["episode_index"])) for ep in dataset_episodes]
            _warn_about_extra_parquet(dataset_dir, dataset_files)

            dataset_frames = sum(int(ep["length"]) for ep in dataset_episodes)
            meta_total_frames = int(info.get("total_frames", dataset_frames))
            if dataset_frames != meta_total_frames:
                raise ValueError(
                    f"{dataset_dir}: episodes.jsonl lengths sum to {dataset_frames}, "
                    f"but info.json says {meta_total_frames}"
                )
            records.extend(
                _FastEpisode(
                    episode=episode,
                    path=episode_file,
                    state_col=dataset_state_col,
                    action_col=dataset_action_col,
                    data_config=data_config,
                )
                for episode, episode_file in zip(dataset_episodes, dataset_files, strict=True)
            )
            repo_frame_counts.append(dataset_frames)

        total_frames = sum(repo_frame_counts)
        batch_size = int(config.batch_size)
        weighted_sampling = config.norm_mode == "mixed" and len(data_configs) > 1
        if max_frames is not None and max_frames < total_frames:
            num_batches = max_frames // batch_size
            shuffle_subset = True
        else:
            num_batches = total_frames // batch_size
            shuffle_subset = weighted_sampling
        max_samples = num_batches * batch_size

        if max_samples < 2:
            raise ValueError(f"Not enough samples to compute stats after drop_last: {max_samples}")

        frame_weights = None
        if weighted_sampling:
            dataset_weights = (
                tuple(config.dataset_weights) if config.dataset_weights is not None else (1.0,) * len(data_configs)
            )
            if len(dataset_weights) != len(repo_frame_counts):
                raise ValueError("dataset_weights must have the same length as datasets.")
            if any(weight < 0 for weight in dataset_weights) or sum(dataset_weights) <= 0:
                raise ValueError("dataset_weights must be non-negative and contain a positive total weight.")
            frame_weights = np.concatenate(
                [
                    np.full(frame_count, float(dataset_weight) / frame_count)
                    for dataset_weight, frame_count in zip(dataset_weights, repo_frame_counts, strict=True)
                ]
            )

        stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

        print(f"Reading dataset: {dataset_dirs}")
        print(f"Config: {config_name}")
        print(f"Norm mode: {config.norm_mode}")
        print(f"Detected schemas: {detected_schemas}")
        print(f"FPS: {first_info.get('fps') if first_info else None}")
        print(f"Episodes: {len(records)}")
        print(f"Total frames from metadata: {total_frames}")
        print(f"Batch size: {batch_size}")
        print(f"Action horizon: {config.model.action_horizon}")
        print(f"Samples used after drop_last: {max_samples}")

        if shuffle_subset:
            files_processed, samples_processed = _process_shuffled_subset(
                stats=stats,
                records=records,
                action_horizon=config.model.action_horizon,
                batch_size=batch_size,
                max_samples=max_samples,
                frame_weights=frame_weights,
            )
        else:
            files_processed, samples_processed = _process_sequential(
                stats=stats,
                records=records,
                action_horizon=config.model.action_horizon,
                batch_size=batch_size,
                max_samples=max_samples,
            )

        print(f"\nProcessed {files_processed} files with {samples_processed} samples")

        norm_stats = {key: value.get_statistics() for key, value in stats.items()}
        for key, stat_result in norm_stats.items():
            print(f"\n{key} statistics:")
            print(f"  Shape: {stat_result.mean.shape}")
            print(f"  Mean: {stat_result.mean}")
            print(f"  Std: {stat_result.std}")
            print(f"  Q01: {stat_result.q01}")
            print(f"  Q99: {stat_result.q99}")

        for data_factory, data_config in zip(data_factories, data_configs, strict=True):
            output_path = _output_path(config, data_factory, data_config)
            print(f"\nWriting stats to: {output_path}")
            normalize.save(output_path, norm_stats)
            print(f"Normalization stats saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
