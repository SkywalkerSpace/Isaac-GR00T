#!/usr/bin/env python3
"""Launch N1.7 fine-tuning with the local Isaac Lab modality configuration."""

import argparse
from pathlib import Path

import yaml

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("finetune_config.yaml"))
    parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("profiles.yaml"))
    args = parser.parse_args()

    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.experiment.experiment import run
    from profile_utils import load_profiles
    from modality_config import build_modality_config

    config_path = args.config.expanduser().resolve()
    config_data = yaml.safe_load(config_path.read_text())
    output_dir = Path(config_data["training"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        config_data["training"]["output_dir"] = str((config_path.parent / output_dir).resolve())
    config = get_default_config().load_dict(config_data)
    profiles = load_profiles(args.profiles)
    if not profiles:
        raise ValueError("No robot/task profiles configured")
    config.data.datasets = []
    for profile in profiles:
        dataset_path = Path(profile["lerobot_dataset"])
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Profile {profile['name']} dataset is missing {info_path}; "
                "convert its HDF5 input first"
            )
        config.data.datasets.append(
            SingleDatasetConfig(
                dataset_paths=[str(dataset_path)],
                embodiment_tag=profile["embodiment_tag"],
                mix_ratio=float(profile["mix_ratio"]),
                dataset_type="physical_embodiment",
            )
        )
        config.data.modality_configs[profile["embodiment_tag"]] = build_modality_config(profile)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    for profile in profiles:
        MODALITY_CONFIGS[profile["embodiment_tag"]] = build_modality_config(profile)
    config.validate()
    run(config)


if __name__ == "__main__":
    main()
