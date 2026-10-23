# Isaac Lab datasets with GR00T N1.7

These scripts convert DexMimicGen HDF5 demonstrations to the LeRobot layout,
fine-tune N1.7 on one or more datasets, and evaluate a checkpoint in the
matching Isaac Lab environments.

## Profiles

`profiles.yaml` binds each robot/task's HDF5 inputs, converted dataset,
embodiment tag, and Isaac Lab environment. Add one profile per dataset and
environment pair. Each profile has independent state, action, and camera
mappings. Action slices are currently concatenated in profile order, and the
loader verifies that they cover the full action vector. State groups and fields
are also concatenated in their configured order.

Relative dataset paths are resolved from the directory containing
`profiles.yaml`. The profile tag must be unique and start with `isaaclab_`.
`mix_ratio` sets the dataset's relative sampling weight during multi-dataset
fine-tuning.

## Convert HDF5

Convert the first configured profile or select one by name:

```bash
python examples/isaaclab/convert_hdf5_to_lerobot.py --profile pick_place_gr1t2
```

Convert all configured datasets:

```bash
python examples/isaaclab/convert_hdf5_to_lerobot.py --all-profiles
```

Each profile may list multiple HDF5 input files. Converted LeRobot metadata,
parquet episodes, and per-camera videos are written to its `lerobot_dataset`.

## Fine-tune

Convert every configured dataset, then train on all profiles together:

```bash
python examples/isaaclab/convert_hdf5_to_lerobot.py --all-profiles
python examples/isaaclab/finetune.py
```

The trainer checks that every configured dataset exists. Edit
`finetune_config.yaml` for the base checkpoint, GPU count, batch size, and
training schedule.

## Roll out in Isaac Lab

```
apt install -y vulkan-tools python3.10-dev
```

Run with Isaac Lab and this GR00T checkout importable in the same environment:

```bash
CUDA_VISIBLE_DEVICES=3  python examples/isaaclab/rollout_isaaclab.py \
  --model-path /home/mayuhang/Isaac-GR00T/examples/isaaclab/outputs/isaaclab_multi_robot/isaaclab_multi_robot_n1d7/ --profile pick_place_gr1t2
```

Use `--all-profiles` to evaluate every configured environment. Results are
written to a separate subdirectory per profile under `./outputs/isaaclab_rollout`.

## command

### isaaclab

CUDA_VISIBLE_DEVICES=3  uv run --index-strategy unsafe-best-match --extra isaacsim,mimic python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --num_envs 40 --generation_num_trials 1000 --input_file /home/mayuhang/datasets/isaaclab/dataset_annotated_gr1_50.hdf5 --output_file ./datasets/generated_dataset_gr1_50.hdf5

### gr00t

python examples/isaaclab/convert_hdf5_to_lerobot.py --all-profiles

./examples/isaaclab/finetune.sh

rollout_isaaclab 用 isaaclab python 环境

CUDA_VISIBLE_DEVICES=1  python examples/isaaclab/rollout_isaaclab.py   --model-path /home/mayuhang/Isaac-GR00T/examples/isaaclab/outputs/isaaclab_multi_robot/isaaclab_multi_robot_n1d7/ --profile pick_place_gr1t2  --debug-camera
