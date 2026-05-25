"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import pathlib
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _output_path(
    config: _config.TrainConfig,
    data_factory: _config.DataConfigFactory,
    data_config: _config.DataConfig,
) -> pathlib.Path:
    return pathlib.Path(data_factory.assets.assets_dir or config.assets_dirs) / (
        data_config.asset_id or data_config.repo_id
    )


def create_torch_dataloader(
    data_config: _config.DataConfig | tuple[_config.DataConfig, ...],
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    dataset_weights: tuple[float, ...] | None = None,
) -> tuple[_data_loader.Dataset, int]:
    data_configs = data_config if isinstance(data_config, tuple) else (data_config,)
    if any(data_config.repo_id is None for data_config in data_configs):
        raise ValueError("Data config must have a repo_id")
    datasets = [
        _data_loader.TransformedDataset(
            _data_loader.create_torch_dataset(single_data_config, action_horizon, model_config),
            [
                *single_data_config.repack_transforms.inputs,
                *single_data_config.data_transforms.inputs,
                # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
                RemoveStrings(),
            ],
        )
        for single_data_config in data_configs
    ]
    if len(datasets) > 1:
        reference_sample = datasets[0][0]
        for dataset in datasets[1:]:
            sample = dataset[0]
            for key in ("state", "actions"):
                if np.asarray(sample[key]).shape[-1] != np.asarray(reference_sample[key]).shape[-1]:
                    raise ValueError(f"Cannot compute mixed norm stats with inconsistent {key} dimensions.")
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    sampler = None
    if len(datasets) > 1:
        sampler = _data_loader.create_lerobot_weighted_sampler(dataset, dataset_weights or (1.0,) * len(datasets))
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_factories = tuple(config.datasets) or (config.data,)
    data_factory_groups = (data_factories,)
    if config.norm_mode == "per_dataset":
        data_factory_groups = tuple((factory,) for factory in data_factories)

    for data_factories, data_configs in (
        (factories, tuple(factory.create(config.assets_dirs, config.model) for factory in factories))
        for factories in data_factory_groups
    ):
        data_config = data_configs[0] if len(data_configs) == 1 else data_configs

        if len(data_configs) == 1 and data_configs[0].rlds_data_dir is not None:
            data_loader, num_batches = create_rlds_dataloader(
                data_configs[0], config.model.action_horizon, config.batch_size, max_frames
            )
        else:
            data_loader, num_batches = create_torch_dataloader(
                data_config,
                config.model.action_horizon,
                config.batch_size,
                config.model,
                config.num_workers,
                max_frames,
                tuple(config.dataset_weights) if config.dataset_weights is not None else None,
            )

        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        asset_label = "+".join(str(data_config.asset_id or data_config.repo_id) for data_config in data_configs)
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc=f"Computing stats for {asset_label}"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
        for data_factory, data_config in zip(data_factories, data_configs, strict=True):
            output_path = _output_path(config, data_factory, data_config)
            print(f"Writing stats to: {output_path}")
            normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
