"""Build N1.7 modality configs from Isaac Lab robot/task profiles."""

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def build_modality_config(profile: dict) -> dict[str, ModalityConfig]:
    video_keys = list(profile["camera_fields"])
    state_keys = list(profile["state_fields"])
    action_keys = list(profile["action_fields"])
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(int(profile["action_horizon"]))),
            modality_keys=action_keys,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in action_keys
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.action.task_description"]
        ),
    }
