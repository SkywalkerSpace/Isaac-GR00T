#!/usr/bin/env python3
"""Evaluate N1.7 in one or more profile-matched Isaac Lab environments."""

import argparse
import csv
from datetime import datetime, timezone
import sys
import traceback
from pathlib import Path
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("profiles.yaml"))
parser.add_argument("--profile", help="Profile name; defaults to the first profile")
parser.add_argument("--all-profiles", action="store_true", help="Evaluate all configured environments")
parser.add_argument("--model-path", required=True, type=Path)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument(
    "--action-steps", type=int, default=16,
    help="How many predicted 20 Hz actions to execute before observing again (default: 16)",
)
parser.add_argument("--language", default=None)
parser.add_argument("--video-dir", type=Path, default=Path("./outputs/isaaclab_rollout"))
parser.add_argument(
    "--debug-camera", action="store_true",
    help="Log camera observation statistics and save raw/reset PNG snapshots",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.action_steps < 1:
    parser.error("--action-steps must be at least 1")


def log(message: str) -> None:
    print(f"[ISAACLAB ROLLOUT] {message}", flush=True)

app_launcher = AppLauncher(args, enable_cameras=True)
simulation_app = app_launcher.app
log("Simulation application initialized; loading rollout dependencies")

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401 - importing the package registers Mimic Gym IDs
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from isaaclab_tasks.utils import parse_env_cfg

from profile_utils import load_profiles, select_profiles


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _camera_frame(obs: dict, camera_env_key: str) -> np.ndarray:
    policy_obs = obs.get("policy", obs)
    if camera_env_key not in policy_obs:
        raise KeyError(
            f"Camera key {camera_env_key!r} not found in observation; "
            f"available keys: {list(policy_obs.keys())}"
        )
    image = _numpy(policy_obs[camera_env_key])
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected one environment in camera batch, got {image.shape}")
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected camera image with shape HxWxC (C>=3), got {image.shape}")
    return image


def _log_camera_stats(label: str, image: np.ndarray) -> None:
    image = np.asarray(image)
    finite = image[np.isfinite(image)] if np.issubdtype(image.dtype, np.number) else image
    if finite.size:
        stats = (
            f"min={finite.min():.4g}, max={finite.max():.4g}, "
            f"mean={finite.mean():.4g}, nonzero={np.count_nonzero(finite)}/{finite.size}"
        )
    else:
        stats = "no finite values"
    channel_means = np.mean(image[..., :3], axis=(0, 1)) if image.ndim == 3 else []
    log(
        f"Camera {label}: shape={image.shape}, dtype={image.dtype}, {stats}, "
        f"RGB_means={np.array2string(channel_means, precision=3)}"
    )


def _save_camera_snapshot(image: np.ndarray, path: Path) -> None:
    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.size and np.nanmax(frame) <= 1.0:
            frame = frame * 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not save camera snapshot to {path}")
    log(f"Saved raw camera snapshot to {path}")


def _env_observation_to_policy(obs: dict, instruction: str, modality: dict, profile: dict) -> dict:
    policy_obs = obs["policy"] if "policy" in obs else obs
    videos = {}
    for key in modality["video"].modality_keys:
        image = _numpy(policy_obs[profile["camera_fields"][key]])
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and np.nanmax(image) <= 1.0:
                image = image * 255.0
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
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


def _save_actions(actions: list[tuple[datetime, np.ndarray]], path: Path, profile: dict) -> None:
    """Save the actions actually sent to the environment, in profile order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        f"{key}_{dimension}"
        for key, (start, end) in profile["action_fields"].items()
        for dimension in range(end - start)
    ]
    with path.open("w", newline="") as action_file:
        writer = csv.writer(action_file)
        writer.writerow(["step", "timestamp", *columns])
        for step, (timestamp, action) in enumerate(actions):
            writer.writerow([
                step,
                timestamp.isoformat(timespec="milliseconds"),
                *np.asarray(action, dtype=np.float32).tolist(),
            ])
    log(f"Saved {len(actions)} environment actions to {path}")


def _save_predicted_actions(
    chunks: list[tuple[int, int, int, np.ndarray]], path: Path, profile: dict
) -> None:
    """Save every predicted action and whether it was sent to the environment."""
    columns = [
        f"{key}_{dimension}"
        for key, (start, end) in profile["action_fields"].items()
        for dimension in range(end - start)
    ]
    with path.open("w", newline="") as action_file:
        writer = csv.writer(action_file)
        writer.writerow(["chunk", "action_index", "executed", *columns])
        for chunk, action_index, executed_count, action in chunks:
            writer.writerow([
                chunk,
                action_index,
                int(action_index < executed_count),
                *np.asarray(action, dtype=np.float32).tolist(),
            ])
    log(f"Saved {len(chunks)} predicted actions to {path}")


def _save_video(frames: list[np.ndarray], path: Path, fps: int, debug: bool = False) -> None:
    if not frames:
        raise RuntimeError(f"No camera frames were captured; cannot write {path}")
    normalized = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError(f"Expected an HxWx3 camera frame, got shape {frame.shape}")
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.size and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        normalized.append(frame)

    height, width = normalized[0].shape[:2]
    if debug:
        _log_camera_stats("first frame before video encoding", normalized[0])
        _log_camera_stats("last frame before video encoding", normalized[-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"OpenCV could not open an MP4 writer for {path}. Install an OpenCV build with "
            "FFmpeg/GStreamer support or choose a different --video-dir/container."
        )
    try:
        for frame in normalized:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video writer finished without creating a non-empty file: {path}")
    log(f"Saved {len(frames)} frames to {path} ({path.stat().st_size} bytes)")
    if debug:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, decoded = capture.read()
            if not ok:
                log(f"WARNING: OpenCV could not decode the first frame from {path}")
            else:
                _log_camera_stats("first frame decoded from MP4 (BGR)", decoded)
        finally:
            capture.release()


def evaluate_profile(profile: dict) -> None:
    output_dir = args.video_dir / profile["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir.resolve()}")
    env_id = profile["environment_id"]
    log(f"Creating environment {env_id}")
    env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=1)
    env_cfg.sim.render_interval = 1
    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array")
    policy_env = env.unwrapped
    log(f"Loading policy from {args.model_path}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(profile["embodiment_tag"]),
        model_path=str(args.model_path),
        device=args.device,
    )
    modality = policy.modality_configs
    configured_horizon = len(modality["action"].delta_indices)
    log(
        f"Action horizon: runtime={configured_horizon}, "
        f"requested execution chunk={args.action_steps}; max_steps={args.max_steps}"
    )
    instruction = args.language or profile["task_description"]
    camera_fields = profile["camera_fields"]
    results_path = output_dir / "results.csv"
    successes = 0
    try:
        with results_path.open("w", newline="") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=["episode", "success", "steps"])
            writer.writeheader()
            for episode in tqdm(range(args.episodes), desc=f"{profile['name']} episodes", unit="episode"):
                log(f"Starting {profile['name']} episode {episode}")
                obs, _ = env.reset()
                policy.reset()
                reset_images = {
                    key: _camera_frame(obs, env_key)
                    for key, env_key in camera_fields.items()
                }
                if args.debug_camera:
                    for key, image in reset_images.items():
                        _log_camera_stats(f"episode {episode} after reset ({key})", image)
                    if episode == 0:
                        log(
                            "Render diagnostics: "
                            f"render_enabled={getattr(policy_env, 'render_enabled', 'n/a')}, "
                            f"sim.is_rendering={getattr(policy_env.sim, 'is_rendering', 'n/a')}, "
                            f"render_interval={getattr(policy_env.cfg.sim, 'render_interval', 'n/a')}"
                        )
                        for key, env_key in profile["camera_fields"].items():
                            try:
                                _log_camera_stats(
                                    f"episode 0 reset ({env_key})",
                                    _camera_frame(obs, env_key),
                                )
                            except (KeyError, ValueError) as exc:
                                log(f"Camera diagnostic failed for {key} -> {env_key}: {exc}")
                            sensor = getattr(policy_env.scene, "sensors", {}).get(env_key)
                            if sensor is None:
                                log(f"Camera sensor {env_key!r} not found in env.scene.sensors")
                            else:
                                log(
                                    f"Camera sensor {env_key!r}: class={type(sensor).__name__}, "
                                    f"prim_path={getattr(sensor.cfg, 'prim_path', 'n/a')}, "
                                    f"renderer_cfg={getattr(sensor.cfg, 'renderer_cfg', 'n/a')}"
                                )
                    if episode == 0:
                        for key, image in reset_images.items():
                            _save_camera_snapshot(image, output_dir / f"debug_reset_{key}.png")
                frames_by_camera = {key: [] for key in camera_fields}
                executed_actions = []
                predicted_actions = []
                success = False
                steps = 0
                step_progress = tqdm(
                    total=args.max_steps,
                    desc=f"episode {episode} steps",
                    unit="step",
                    leave=False,
                )
                effective_chunk_steps = min(args.action_steps, configured_horizon)
                chunk_limit = (
                    args.max_steps + effective_chunk_steps - 1
                ) // effective_chunk_steps
                for chunk_number in range(chunk_limit):
                    policy_input = _env_observation_to_policy(obs, instruction, modality, profile)
                    actions, _ = policy.get_action(policy_input)
                    prediction_length = len(next(iter(actions.values()))[0])
                    if prediction_length != configured_horizon:
                        raise ValueError(
                            f"Policy returned {prediction_length} actions, but runtime modality "
                            f"horizon is {configured_horizon}"
                        )
                    done = False
                    chunk_len = min(
                        args.action_steps,
                        prediction_length,
                        args.max_steps - steps,
                    )
                    chunk_predictions = [
                        (chunk_number, index, chunk_len, _action_to_env(actions, profile, index).copy())
                        for index in range(prediction_length)
                    ]
                    predicted_actions.extend(chunk_predictions)
                    executed_in_chunk = 0
                    for chunk_index in range(chunk_len):
                        env_action = _action_to_env(actions, profile, chunk_index)
                        executed_actions.append((datetime.now(timezone.utc), env_action.copy()))
                        obs, _, terminated, truncated, info = env.step(
                            torch.as_tensor(env_action, device=policy_env.device).unsqueeze(0)
                        )
                        steps += 1
                        executed_in_chunk += 1
                        images = {
                            key: _camera_frame(obs, env_key)
                            for key, env_key in camera_fields.items()
                        }
                        if args.debug_camera and steps == 1:
                            for key, image in images.items():
                                _log_camera_stats(f"episode {episode} after first step ({key})", image)
                                if episode == 0:
                                    _save_camera_snapshot(
                                        image, output_dir / f"debug_step1_{key}.png"
                                    )
                        for key, image in images.items():
                            frames_by_camera[key].append(np.asarray(image).copy())
                        try:
                            success_value = policy_env.termination_manager.get_term("success")
                            success = success or bool(_numpy(success_value).any())
                        except (AttributeError, ValueError, KeyError):
                            success = success or bool(_numpy(info.get("success", False)).any())
                        done = bool(_numpy(terminated).any() or _numpy(truncated).any())
                        if done:
                            termination_manager = getattr(policy_env, "termination_manager", None)
                            triggered_terms = []
                            if termination_manager is not None:
                                for term_name in termination_manager.active_terms:
                                    try:
                                        term_value = _numpy(termination_manager.get_term(term_name))
                                        if bool(term_value.any()):
                                            triggered_terms.append(term_name)
                                    except (AttributeError, KeyError, ValueError) as exc:
                                        triggered_terms.append(f"{term_name}=unavailable({exc})")
                            log(
                                f"[{profile['name']}] episode {episode} ended at step {steps}: "
                                f"terminated={bool(_numpy(terminated).any())}, "
                                f"truncated={bool(_numpy(truncated).any())}, "
                                f"triggered_terms={triggered_terms or 'none reported'}"
                            )
                        if done or steps >= args.max_steps:
                            break
                    for index in range(len(predicted_actions) - prediction_length, len(predicted_actions)):
                        chunk, action_index, _, predicted_action = predicted_actions[index]
                        predicted_actions[index] = (
                            chunk, action_index, executed_in_chunk, predicted_action
                        )
                    step_progress.update(executed_in_chunk)
                    if done or steps >= args.max_steps:
                        break
                step_progress.close()
                if not done and steps < args.max_steps:
                    log(
                        f"[{profile['name']}] episode {episode} loop exhausted without done: "
                        f"steps={steps}, max_steps={args.max_steps}, "
                        f"chunks={chunk_number + 1}/{chunk_limit}, "
                        f"action_steps={args.action_steps}, runtime_horizon={configured_horizon}, "
                        f"effective_chunk_steps={effective_chunk_steps}"
                    )
                elif not done and steps >= args.max_steps:
                    log(
                        f"[{profile['name']}] episode {episode} reached max_steps="
                        f"{args.max_steps} after {chunk_number + 1}/{chunk_limit} chunks"
                    )
                successes += int(success)
                for key, frames in frames_by_camera.items():
                    video_path = output_dir / f"episode_{episode:04d}_{key}.mp4"
                    _save_video(frames, video_path, int(profile["fps"]), debug=args.debug_camera)
                actions_path = output_dir / f"episode_{episode:04d}_actions.csv"
                _save_actions(executed_actions, actions_path, profile)
                predictions_path = output_dir / f"episode_{episode:04d}_predicted_actions.csv"
                _save_predicted_actions(predicted_actions, predictions_path, profile)
                writer.writerow({"episode": episode, "success": int(success), "steps": steps})
                result_file.flush()
                log(f"[{profile['name']}] episode {episode}: success={success}, steps={steps}")
        log(
            f"[{profile['name']}] success rate: {successes}/{args.episodes} "
            f"= {successes / max(args.episodes, 1):.3f}"
        )
        with results_path.open("a", newline="") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=["episode", "success", "steps"])
            writer.writerow(
                {
                    "episode": "success_rate",
                    "success": f"{successes / max(args.episodes, 1):.6f}",
                    "steps": "",
                }
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
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        log("Closing simulation application")
        simulation_app.close()
