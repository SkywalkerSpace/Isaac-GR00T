"""Load and validate robot/task profiles shared by Isaac Lab scripts."""

from pathlib import Path

import yaml


def load_profiles(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    profiles = raw.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"{path} must contain a non-empty 'profiles' list")

    names, tags = set(), set()
    resolved = []
    for original in profiles:
        profile = dict(original)
        for field in (
            "name", "embodiment_tag", "environment_id", "lerobot_dataset",
            "task_description", "action_fields", "state_fields", "camera_fields",
        ):
            if field not in profile:
                raise ValueError(f"Profile is missing required field {field!r}: {profile}")
        if profile["name"] in names:
            raise ValueError(f"Duplicate profile name: {profile['name']}")
        if profile["embodiment_tag"] in tags:
            raise ValueError(f"Duplicate embodiment tag: {profile['embodiment_tag']}")
        names.add(profile["name"])
        tags.add(profile["embodiment_tag"])
        if not isinstance(profile.get("hdf5_inputs"), list) or not profile["hdf5_inputs"]:
            raise ValueError(f"Profile {profile['name']} must list at least one hdf5_inputs file")
        if not profile["state_fields"] or not profile["camera_fields"] or not profile["action_fields"]:
            raise ValueError(f"Profile {profile['name']} needs state, action, and camera mappings")
        if not profile["embodiment_tag"].startswith("isaaclab_"):
            raise ValueError("Custom Isaac Lab embodiment tags must start with 'isaaclab_'")

        profile.setdefault("fps", 20)
        profile.setdefault("action_horizon", 16)
        profile.setdefault("action_source", "actions")
        profile.setdefault("require_success", True)
        profile.setdefault("mix_ratio", 1.0)
        profile.setdefault("checkpoint", "./outputs/isaaclab_multi_robot")
        if float(profile["mix_ratio"]) <= 0:
            raise ValueError(f"mix_ratio must be positive in profile {profile['name']}")
        if int(profile["fps"]) <= 0 or int(profile["action_horizon"]) <= 0:
            raise ValueError(f"fps and action_horizon must be positive in profile {profile['name']}")
        action_fields = {}
        ranges = []
        for key, bounds in profile["action_fields"].items():
            start, end = map(int, bounds)
            if start < 0 or end <= start:
                raise ValueError(f"Invalid action range for {key}: {bounds}")
            action_fields[key] = (start, end)
            ranges.append((start, end))
        sorted_ranges = sorted(ranges)
        cursor = 0
        for start, end in sorted_ranges:
            if start != cursor:
                raise ValueError(
                    f"Action fields in {profile['name']} must cover dimensions "
                    "without gaps or overlap"
                )
            cursor = end
        profile["action_dim"] = cursor
        profile["action_fields"] = action_fields
        state_cursor = 0
        profile["state_slices"] = {}
        for key, fields in profile["state_fields"].items():
            profile["state_slices"][key] = {"start": state_cursor, "end": None}
            state_cursor += len(fields)

        for field in ("lerobot_dataset", "checkpoint"):
            value = Path(profile[field]).expanduser()
            profile[field] = str((path.parent / value).resolve()) if not value.is_absolute() else str(value)
        profile["hdf5_inputs"] = [
            str((path.parent / Path(value).expanduser()).resolve())
            if not Path(value).expanduser().is_absolute()
            else str(Path(value).expanduser())
            for value in profile["hdf5_inputs"]
        ]
        resolved.append(profile)
    return resolved


def select_profiles(
    profiles: list[dict], name: str | None = None, all_profiles: bool = False
) -> list[dict]:
    if all_profiles:
        return profiles
    if name is None:
        return profiles[:1]
    matches = [profile for profile in profiles if profile["name"] == name]
    if not matches:
        raise ValueError(f"Unknown profile {name!r}; choices: {', '.join(p['name'] for p in profiles)}")
    return matches
