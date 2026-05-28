import dataclasses

import einops
import numpy as np

from typing import List, Literal, Union

from openpi import transforms
from openpi.models import model as _model

"""THIS MODULE REFERRED https://github.com/huagailuowen/openpi, many thanks!"""


def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        # "": np.random.rand(12),
        # NOTE: TODO: it seems that pi0 ask state and action be the same meaning, but informed that not in need
        # 10d = xyz, R[:3, 0], R[:3, 1], gripper
        "observation.state": np.random.rand(10),
        "observation.images.eye_in_hand": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation.images.agentview": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


##### STATE / ACTION SPACE #####

# Raw vector layout: 0~2 pos, 3~8 6d_rot, 9 gripper (width, max=0.04).
FIELD_SLICES = {
    "pos": slice(0, 3),
    "6d_rot": slice(3, 9),
    "gripper": slice(9, 10),
}

# str -> list of field names (ids). Used when ids is a str key.
IDS_MAP = {
    "10d": ["pos", "6d_rot", "gripper"],
    "9d": ["pos", "6d_rot"],
}

GRIPPER_FULL_WIDTH = 0.04


def _resolve_ids(ids: Union[str, List[str]]) -> List[str]:
    """Resolve ids to list of field names. str -> lookup in IDS_MAP; list -> use as-is."""
    if isinstance(ids, str):
        return list(IDS_MAP.get(ids, IDS_MAP["10d"]))
    return list(ids)


def _validate_fields(fields: List[str]) -> None:
    """Raise if any field is not in FIELD_SLICES."""
    invalid = [f for f in fields if f not in FIELD_SLICES]
    if invalid:
        raise ValueError(
            f"Invalid field names {invalid}; allowed: {list(FIELD_SLICES)}"
        )


@dataclasses.dataclass(frozen=True)
class GripperConfig:
    """Gripper encoding. For '01': value/GRIPPER_FULL_WIDTH >= threshold -> 1 else 0."""

    type: Literal["width", "01"] = "width"
    threshold: float = 0.01  # used when type=="01", relative to max width 0.04


@dataclasses.dataclass(frozen=True)
class StateSpaceConfig:
    """State space: ids is str key into IDS_MAP or list of field names; gripper=None when ids omit gripper."""

    ids: Union[str, List[str]] = "10d"
    gripper: GripperConfig | None = dataclasses.field(default_factory=GripperConfig)

    def __post_init__(self) -> None:
        fields = _resolve_ids(self.ids)
        _validate_fields(fields)
        if "gripper" not in fields and self.gripper is not None:
            object.__setattr__(self, "gripper", None)


@dataclasses.dataclass(frozen=True)
class ActionSpaceConfig:
    """Action space: ids is str key into IDS_MAP or list of field names; gripper=None when ids omit gripper."""

    ids: Union[str, List[str]] = "10d"
    gripper: GripperConfig | None = dataclasses.field(default_factory=GripperConfig)

    def __post_init__(self) -> None:
        fields = _resolve_ids(self.ids)
        _validate_fields(fields)
        if "gripper" not in fields and self.gripper is not None:
            object.__setattr__(self, "gripper", None)


def _fields_from_state_config(c: StateSpaceConfig) -> List[str]:
    """Resolve state config to list of field names; validate against FIELD_SLICES."""
    fields = _resolve_ids(c.ids)
    _validate_fields(fields)
    return fields


def _fields_from_action_config(c: ActionSpaceConfig) -> List[str]:
    """Resolve action config to list of field names; validate against FIELD_SLICES."""
    fields = _resolve_ids(c.ids)
    _validate_fields(fields)
    return fields


def _apply_gripper_01(value: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize gripper: value/GRIPPER_FULL_WIDTH >= threshold -> 1 else 0."""
    norm = value / GRIPPER_FULL_WIDTH
    return (norm >= threshold).astype(value.dtype)


def _extract_vec(full: np.ndarray, fields: List[str], gripper_cfg: GripperConfig | None) -> np.ndarray:
    """Extract and optionally binarize gripper. full is 10d (pos, 6d_rot, gripper)."""
    indices: list[int] = []
    for f in fields:
        s = FIELD_SLICES[f]
        indices.extend(range(s.start, s.stop))
    vec = full[indices]
    if "gripper" in fields and gripper_cfg is not None and gripper_cfg.type == "01":
        gripper_len = FIELD_SLICES["gripper"].stop - FIELD_SLICES["gripper"].start
        start = len(vec) - gripper_len
        vec = np.concatenate([
            vec[:start],
            _apply_gripper_01(vec[start : start + gripper_len], gripper_cfg.threshold),
        ], axis=-1)
    return vec


@dataclasses.dataclass(frozen=True)
class SLAIFrankaInputs(transforms.DataTransformFn):
    """
    This class is used to converted the lerobo-formartted robocasa dataset, collected by human in OpenDrawer environment and do the 'open the right drawer' task
    """

    model_type: _model.ModelType
    state_space: StateSpaceConfig = dataclasses.field(default_factory=StateSpaceConfig)
    action_space: ActionSpaceConfig = dataclasses.field(default_factory=ActionSpaceConfig)

    def __call__(self, data: dict) -> dict:
        agentview_image = _parse_image(data["observation.images.agentview"])
        eye_in_hand_image = _parse_image(data["observation.images.eye_in_hand"])

        inputs: dict = {}

        ##### STATE #####
        state_fields = _fields_from_state_config(self.state_space)
        full_state = np.asarray(data["observation.state"])
        inputs["state"] = _extract_vec(full_state, state_fields, self.state_space.gripper)

        ##### IMAGES #####
        inputs["image"] = {
            "base_0_rgb": agentview_image,
            "left_wrist_0_rgb": eye_in_hand_image,
            "right_wrist_0_rgb": np.zeros_like(eye_in_hand_image),
        }
        inputs["image_mask"] = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
        }
        inputs["task_index"] = data["task_index"]

        ##### ACTIONS #####
        if "actions" in data:
            action_fields = _fields_from_action_config(self.action_space)
            actions_2d = np.asarray(data["actions"])
            indices: list[int] = []
            for f in action_fields:
                s = FIELD_SLICES[f]
                indices.extend(range(s.start, s.stop))
            actions = actions_2d[:, indices]
            if "gripper" in action_fields and self.action_space.gripper is not None and self.action_space.gripper.type == "01":
                gripper_len = FIELD_SLICES["gripper"].stop - FIELD_SLICES["gripper"].start
                start = actions.shape[-1] - gripper_len
                actions = np.concatenate([
                    actions[:, :start],
                    _apply_gripper_01(actions[:, start : start + gripper_len], self.action_space.gripper.threshold),
                ], axis=-1)
            inputs["actions"] = actions

        # NOTE: THERE IS NO PROMPT, see TransforedDataset defination in openpi/training/data_loader.py #L544,
        # The dataset that carries this SLAIFankaInputs Transform will be wrapped by PromptFromLerobotTask
        # SO SLAIFankaInputs will be called first with only task_index no prompt,
        # then PromptFromLerobotTask will be called to transform task_index to prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

@dataclasses.dataclass(frozen=True)
class SLAIFrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    action_space: ActionSpaceConfig = dataclasses.field(default_factory=ActionSpaceConfig)

    def __call__(self, data: dict) -> dict:
        action_fields = _fields_from_action_config(self.action_space)
        dim = sum(FIELD_SLICES[f].stop - FIELD_SLICES[f].start for f in action_fields)
        actions = np.asarray(data["actions"][:, :dim])
        if "gripper" in action_fields and self.action_space.gripper is not None and self.action_space.gripper.type == "01":
            gripper_len = FIELD_SLICES["gripper"].stop - FIELD_SLICES["gripper"].start
            start = actions.shape[-1] - gripper_len
            actions = np.concatenate([
                actions[:, :start],
                _apply_gripper_01(actions[:, start : start + gripper_len], self.action_space.gripper.threshold),
            ], axis=-1)
        return {"actions": actions}
