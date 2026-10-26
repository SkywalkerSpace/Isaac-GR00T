#!/usr/bin/env python3
"""Launch N1.7 fine-tuning with the local Isaac Lab modality configuration."""

import argparse
import os
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
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    )
    from profile_utils import load_profiles
    from modality_config import build_modality_config

    config_path = args.config.expanduser().resolve()
    config_data = yaml.safe_load(config_path.read_text())
    # Flash Attention 2 requires the Qwen3-VL backbone to be loaded in a
    # half precision dtype. TrainingArguments.bf16 only enables autocast; it
    # does not change the dtype used by from_pretrained().
    model_config = config_data.setdefault("model", {})
    if model_config.get("use_flash_attention", True):
        model_dtype = model_config.get("model_dtype", "bfloat16")
        if model_dtype not in ("bfloat16", "float16"):
            raise ValueError(
                "Flash Attention 2 requires model.model_dtype to be 'bfloat16' "
                f"or 'float16'; got {model_dtype!r}"
            )
        model_config["load_bf16"] = model_dtype == "bfloat16"

    output_dir = Path(config_data["training"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        config_data["training"]["output_dir"] = str((config_path.parent / output_dir).resolve())
    config = get_default_config().load_dict(config_data)
    # Isaac Lab profiles store absolute environment commands. Preserve the
    # configured setting and keep modality action representations absolute.
    profiles = load_profiles(args.profiles)
    if not profiles:
        raise ValueError("No robot/task profiles configured")
    static_custom_id = EMBODIMENT_TAG_TO_PROJECTOR_INDEX["isaaclab_custom"]
    if static_custom_id >= config.model.max_num_embodiments:
        raise ValueError(
            f"Static Isaac Lab projector ID {static_custom_id} exceeds model capacity "
            f"(max_num_embodiments={config.model.max_num_embodiments})"
        )
    for profile in profiles:
        tag = profile["embodiment_tag"]
        if tag.startswith("isaaclab_"):
            print(
                f"[ISAACLAB FINETUNE] embodiment tag={tag!r}, "
                f"processor.embodiment_id_mapping[{tag!r}]={static_custom_id}",
                flush=True,
            )
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
        modality_config = build_modality_config(profile)
        config.data.modality_configs[profile["embodiment_tag"]] = modality_config

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    for profile in profiles:
        MODALITY_CONFIGS[profile["embodiment_tag"]] = build_modality_config(profile)
    config.validate()

    # Accelerate probes/imports DeepSpeed while unwrapping models, even when
    # Trainer was not given a DeepSpeed configuration. On minimal servers that
    # can trigger Triton compilation (and require Python development headers).
    # This single-process training entry point does not use DeepSpeed.
    if config.training.num_gpus == 1 or config.training.use_ddp:
        os.environ["GR00T_DISABLE_ACCELERATE_DEEPSPEED"] = "1"
        try:
            from accelerate.utils import other as accelerate_other

            accelerate_other.is_deepspeed_available = lambda: False
        except ImportError:
            pass

    run(config)


if __name__ == "__main__":
    main()
