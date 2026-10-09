#!/usr/bin/env python3
"""Evaluate N1.7 in one or more profile-matched Isaac Lab environments."""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("profiles.yaml"))
parser.add_argument("--profile", help="Profile name; defaults to the first profile")
parser.add_argument("--all-profiles", action="store_true", help="Evaluate all configured environments")
parser.add_argument("--model-path", required=True, type=Path)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--action-steps", type=int, default=8)
parser.add_argument("--language", default=None)
parser.add_argument("--video-dir", type=Path, default=Path("./outputs/isaaclab_rollout"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args, enable_cameras=True)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from isaaclab_tasks.utils import parse_env_cfg

from profile_utils import load_profiles, select_profiles


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _env_observation_to_policy(obs: dict, instruction: str, modality: dict, profile: dict) -> dict:
    policy_obs = obs["policy"] if "policy" in obs else obs
    videos = {}
    for key in modality["video"].modality_keys:
        image = _numpy(policy_obs[profile["camera_fields"][key]])
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        videos[key] = image[None, None, ...]

    states = {}
    for key in modality["state"].modality_keys:
        fields = profile["state_fields"][key]
        values = [_numpy(policy_obs[field]) for field in fields]
        value = values[0] if len(values) == 1 else np.concatenate(values, axis=-1)
        if value.ndim > 1:
            value = value[0]
        states[key] = np.asarray(value, dtype=np.float32)[None, None, :]

    language_key = modality["language"].modality_keys[0]
    return {"video": videos, "state": states, "language": {language_key: [[instruction]]}}


def _action_to_env(action_dict: dict[str, np.ndarray], profile: dict, index: int) -> np.ndarray:
    action = np.concatenate([action_dict[key][0, index] for key in profile["action_fields"]], axis=-1)
    if action.shape != (profile["action_dim"],):
        raise ValueError(f"Policy action has shape {action.shape}; expected ({profile['action_dim']},)")
    return action.astype(np.float32)


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def evaluate_profile(profile: dict) -> None:
    output_dir = args.video_dir / profile["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    env_id = profile["environment_id"]
    env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=1)
    env_cfg.sim.render_interval = 1
    env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    policy_env = env.unwrapped
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(profile["embodiment_tag"]),
        model_path=str(args.model_path),
        device=args.device,
    )
    modality = policy.modality_configs
    instruction = args.language or profile["task_description"]
    camera_key = next(iter(profile["camera_fields"]))
    camera_env_key = profile["camera_fields"][camera_key]
    results_path = output_dir / "results.csv"
    successes = 0
    try:
        with results_path.open("w", newline="") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=["episode", "success", "steps"])
            writer.writeheader()
            for episode in range(args.episodes):
                obs, _ = env.reset()
                policy.reset()
                frames = []
                success = False
                steps = 0
                for _ in range(args.max_steps):
                    policy_input = _env_observation_to_policy(obs, instruction, modality, profile)
                    actions, _ = policy.get_action(policy_input)
                    done = False
                    chunk_len = min(args.action_steps, len(modality["action"].delta_indices))
                    for chunk_index in range(chunk_len):
                        env_action = _action_to_env(actions, profile, chunk_index)
                        obs, _, terminated, truncated, info = env.step(
                            torch.as_tensor(env_action, device=policy_env.device).unsqueeze(0)
                        )
                        steps += 1
                        image = _numpy((obs.get("policy", obs))[camera_env_key])
                        if image.ndim == 4:
                            image = image[0]
                        frames.append(np.asarray(image, dtype=np.uint8))
                        try:
                            success_value = policy_env.termination_manager.get_term("success")
                            success = success or bool(_numpy(success_value).any())
                        except (AttributeError, ValueError, KeyError):
                            success = success or bool(_numpy(info.get("success", False)).any())
                        done = bool(_numpy(terminated).any() or _numpy(truncated).any())
                        if done or steps >= args.max_steps:
                            break
                    if done or steps >= args.max_steps:
                        break
                successes += int(success)
                video_path = output_dir / f"episode_{episode:04d}.mp4"
                _save_video(frames, video_path, int(profile["fps"]))
                writer.writerow({"episode": episode, "success": int(success), "steps": steps})
                result_file.flush()
                print(f"[{profile['name']}] episode {episode}: success={success}, steps={steps}")
        print(
            f"[{profile['name']}] success rate: {successes}/{args.episodes} "
            f"= {successes / max(args.episodes, 1):.3f}"
        )
    finally:
        env.close()


def main() -> None:
    profiles = load_profiles(args.profiles)
    for profile in select_profiles(profiles, args.profile, args.all_profiles):
        evaluate_profile(profile)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
