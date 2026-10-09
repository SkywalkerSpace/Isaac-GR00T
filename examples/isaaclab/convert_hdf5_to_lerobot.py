#!/usr/bin/env python3
"""Convert Isaac Lab demonstration HDF5 episodes to N1.7 LeRobot data."""

import argparse
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from profile_utils import load_profiles, select_profiles


def _write_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    height, width, channels = frames.shape[1:]
    if channels != 3:
        raise ValueError(f"Expected RGB images, got {frames.shape}")
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _read_state(obs: h5py.Group, length: int, state_fields: dict) -> dict[str, np.ndarray]:
    state = {}
    for key, fields in state_fields.items():
        values = [obs[field][:] for field in fields]
        value = values[0] if len(values) == 1 else np.concatenate(values, axis=-1)
        if value.shape[0] < length:
            raise ValueError(f"State {key} has {value.shape[0]} steps, expected {length}")
        state[key] = np.asarray(value[:length], dtype=np.float32)
    return state


def convert(profile: dict) -> None:
    input_files = [Path(path) for path in profile["hdf5_inputs"]]
    output_dir = Path(profile["lerobot_dataset"])
    task_description = profile["task_description"]
    fps = int(profile["fps"])
    action_dim = int(profile["action_dim"])
    action_fields = profile["action_fields"]
    state_fields = profile["state_fields"]
    camera_fields = profile["camera_fields"]
    action_source = profile["action_source"]
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    tasks = [{"task_index": 0, "task": task_description}]
    total_frames = 0
    episode_index = 0
    sample_state = None
    sample_camera_shapes = {}
    action_slices = {}
    state_slices = {}

    for input_path in input_files:
        with h5py.File(input_path, "r") as h5:
            if "data" not in h5:
                raise ValueError(f"Missing /data group in {input_path}")
            demo_names = sorted(h5["data"].keys(), key=lambda name: int(name.split("_")[-1]))
            for demo_name in tqdm(demo_names, desc=input_path.name):
                demo = h5["data"][demo_name]
                if profile["require_success"] and not bool(demo.attrs.get("success", False)):
                    continue
                if action_source not in demo:
                    raise KeyError(f"Missing /data/{demo_name}/{action_source}")
                actions = np.asarray(demo[action_source], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != action_dim:
                    raise ValueError(f"{demo_name} action shape {actions.shape}; expected (T, {action_dim})")

                obs = demo["obs"]
                length = min(int(demo.attrs.get("num_samples", len(actions))), len(actions))
                state = _read_state(obs, length, state_fields)
                camera_frames = {}
                for key, hdf5_key in camera_fields.items():
                    frames = np.asarray(obs[hdf5_key][:length])
                    if frames.ndim != 4 or frames.shape[-1] != 3:
                        raise ValueError(f"{demo_name}/{hdf5_key} has invalid image shape {frames.shape}")
                    if len(frames) != length:
                        raise ValueError(f"{demo_name}/{hdf5_key} has {len(frames)} frames, expected {length}")
                    camera_frames[key] = frames
                    sample_camera_shapes.setdefault(key, list(frames.shape[1:]))

                actions = actions[:length]
                action = {}
                for key, (start, end) in action_fields.items():
                    action[key] = actions[:, start:end]
                    if action[key].shape[1] != end - start:
                        raise ValueError(f"Bad configured action slice for {key}: {(start, end)}")

                flat_state = np.concatenate(list(state.values()), axis=-1)
                flat_action = np.concatenate(list(action.values()), axis=-1)
                if flat_action.shape[-1] != action_dim:
                    raise ValueError(f"Action slices concatenate to {flat_action.shape[-1]}, expected {action_dim}")
                if sample_state is None:
                    sample_state = state
                    index = 0
                    for key, value in state.items():
                        state_slices[key] = {"start": index, "end": index + value.shape[-1]}
                        index += value.shape[-1]
                    index = 0
                    for key, value in action.items():
                        action_slices[key] = {"start": index, "end": index + value.shape[-1]}
                        index += value.shape[-1]

                record = {
                    "observation.state": list(flat_state),
                    "action": list(flat_action),
                    "timestamp": (np.arange(length, dtype=np.float32) / fps).tolist(),
                    "annotation.human.action.task_description": [0] * length,
                    "task_index": [0] * length,
                    "episode_index": [episode_index] * length,
                    "index": list(range(total_frames, total_frames + length)),
                }
                episode_chunk = episode_index // 1000
                data_dir = output_dir / "data" / f"chunk-{episode_chunk:03d}"
                data_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(record).to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)
                for key, frames in camera_frames.items():
                    video_dir = (
                        output_dir
                        / "videos"
                        / f"chunk-{episode_chunk:03d}"
                        / f"observation.images.{key}"
                    )
                    video_dir.mkdir(parents=True, exist_ok=True)
                    path = video_dir / f"episode_{episode_index:06d}.mp4"
                    _write_video(frames, path, fps)
                episodes.append({"episode_index": episode_index, "tasks": [task_description], "length": length})
                total_frames += length
                episode_index += 1

    if not episodes:
        raise ValueError("No successful episodes were converted")

    state_dim = sum(value.shape[-1] for value in sample_state.values())

    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [action_dim]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
    }
    for key in camera_fields:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": sample_camera_shapes[key],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "mpeg4",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps({
        "codebase_version": "v2.0",
        "robot_type": profile["embodiment_tag"],
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(camera_fields),
        "total_chunks": math.ceil(len(episodes) / 1000),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }, indent=2))

    modality = {
        "state": {key: {**value, "original_key": "observation.state"} for key, value in state_slices.items()},
        "action": {key: {**value, "original_key": "action"} for key, value in action_slices.items()},
        "video": {key: {"original_key": f"observation.images.{key}"} for key in camera_fields},
        "annotation": {
            "human.action.task_description": {
                "original_key": "annotation.human.action.task_description"
            }
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2))
    (meta_dir / "episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in episodes))
    (meta_dir / "tasks.jsonl").write_text(json.dumps(tasks[0]) + "\n")
    print(f"Converted {len(episodes)} episodes / {total_frames} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("profiles.yaml"))
    parser.add_argument("--profile", help="Profile name; defaults to the first profile")
    parser.add_argument("--all-profiles", action="store_true", help="Convert each configured dataset")
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    for profile in select_profiles(profiles, args.profile, args.all_profiles):
        convert(profile)


if __name__ == "__main__":
    main()
