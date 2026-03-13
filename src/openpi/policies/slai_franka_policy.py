import dataclasses

import einops
import numpy as np

from typing import List, Union

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

SLAI_FRANKA_STATE_SLICES = {
    # 0~2: 3d pos
    "pos": slice(0, 3),
    # 3~8: 6d_rot
    "6d_rot": slice(3, 9),
    # 9: 1d gripper
    "gripper": slice(9, 10),
}

STATE_SPACE_STR_MAPPING = {
    # 10d = pos(3) + 6d_rot(6) + gripper(1)
    "10d": ["pos", "6d_rot", "gripper"],
    # 9d = pos(3) + 6d_rot(6)
    "9d": ["pos", "6d_rot"],
}


def _check_state_space(state_space: List[str]) -> bool:
    return all(name in SLAI_FRANKA_STATE_SLICES for name in state_space)


SLAI_FRANKA_ACTION_SLICES = SLAI_FRANKA_STATE_SLICES

ACTION_SPACE_STR_MAPPING = {
    "10d": ["pos", "6d_rot", "gripper"],
    "9d": ["pos", "6d_rot"],
}


def _check_action_space(action_space: List[str]) -> bool:
    return all(name in SLAI_FRANKA_ACTION_SLICES for name in action_space)


GRIPPER_FULL_WIDTH = 0.04


def _apply_gripper_mode_01_1d(
    value: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Map gripper width to {0,1} with a threshold in normalized width."""
    norm = value / GRIPPER_FULL_WIDTH
    return (norm >= threshold).astype(value.dtype)

@dataclasses.dataclass(frozen=True)
class SLAIFrankaInputs(transforms.PromptFromLeRobotTask):
    """
    This class is used to converted the lerobo-formartted robocasa dataset, collected by human in OpenDrawer environment and do the 'open the right drawer' task
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.

    state_space: Union[str, List[str]] = "10d"
    action_space: Union[str, List[str]] = "10d"

    state_gripper_type: str = "width"  # "width" | "01"
    state_gripper_threshold: float = 0.01
    action_gripper_type: str = "width"  # "width" | "01"
    action_gripper_threshold: float = 0.01

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        agentview_image = _parse_image(data["observation.images.agentview"])
        eye_in_hand_image = _parse_image(data["observation.images.eye_in_hand"])

        # NOTE TODO: for initial "open right drawer" trial, we use 1 image_right camera as base, as it look at the right; 
        # and image_right camera as left_wrist, like imaginary idle left arm wrist camera
        # and wrist_image camera as right_wrist

        # Create inputs dict. Do not change the keys in the dict below.
        inputs: dict = {}

        ##### STATE #####
        if isinstance(self.state_space, str):
            state_fields = STATE_SPACE_STR_MAPPING.get(self.state_space)
            if state_fields is None:
                state_fields = STATE_SPACE_STR_MAPPING["10d"]
        else:
            state_fields = self.state_space

        if not _check_state_space(state_fields):
            raise ValueError(f"Invalid state_space fields for SLAIFrankaInputs: {state_fields}")

        state_indices: list[int] = []
        for field in state_fields:
            s = SLAI_FRANKA_STATE_SLICES[field]
            state_indices.extend(range(s.start, s.stop))

        full_state = np.asarray(data["observation.state"])
        state_vec = full_state[state_indices]

        if "gripper" in state_fields and self.state_gripper_type == "01":
            # gripper is always the last field in our mappings if present
            gripper_len = SLAI_FRANKA_STATE_SLICES["gripper"].stop - SLAI_FRANKA_STATE_SLICES["gripper"].start
            gripper_start = len(state_vec) - gripper_len
            gripper_slice = state_vec[gripper_start: gripper_start + gripper_len]
            gripper_bin = _apply_gripper_mode_01_1d(gripper_slice, self.state_gripper_threshold)
            state_vec = np.concatenate([state_vec[:gripper_start], gripper_bin], axis=-1)

        inputs["state"] = state_vec

        ##### IMAGES #####
        inputs["image"] = {
            "base_0_rgb": agentview_image,
            "left_wrist_0_rgb": eye_in_hand_image,
            # Pad any non-existent images with zero-arrays of the appropriate shape.
            "right_wrist_0_rgb": np.zeros_like(eye_in_hand_image),
        }
        inputs["image_mask"] = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
            "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
        }
        inputs["task_index"] = data["task_index"]

        ##### ACTIONS #####
        if "actions" in data:
            if isinstance(self.action_space, str):
                action_fields = ACTION_SPACE_STR_MAPPING.get(self.action_space)
                if action_fields is None:
                    action_fields = ACTION_SPACE_STR_MAPPING["10d"]
            else:
                action_fields = self.action_space

            if not _check_action_space(action_fields):
                raise ValueError(f"Invalid action_space fields for SLAIFrankaInputs: {action_fields}")

            action_indices: list[int] = []
            for field in action_fields:
                s = SLAI_FRANKA_ACTION_SLICES[field]
                action_indices.extend(range(s.start, s.stop))

            actions = np.asarray(data["actions"])[:, action_indices]

            if "gripper" in action_fields and self.action_gripper_type == "01":
                gripper_len = SLAI_FRANKA_ACTION_SLICES["gripper"].stop - SLAI_FRANKA_ACTION_SLICES["gripper"].start
                gripper_start = actions.shape[-1] - gripper_len
                gripper_slice = actions[:, gripper_start: gripper_start + gripper_len]
                gripper_bin = _apply_gripper_mode_01_1d(gripper_slice, self.action_gripper_threshold)
                actions = np.concatenate([actions[:, :gripper_start], gripper_bin], axis=-1)

            inputs["actions"] = actions

        # NOTE: add prompt from task_index
        data = super().__call__(data)

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class SLAIFrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_space: Union[str, List[str]] = "10d"
    action_gripper_type: str = "width"
    action_gripper_threshold: float = 0.01

    def __call__(self, data: dict) -> dict:
        if isinstance(self.action_space, str):
            action_fields = ACTION_SPACE_STR_MAPPING.get(self.action_space)
            if action_fields is None:
                action_fields = ACTION_SPACE_STR_MAPPING["10d"]
        else:
            action_fields = self.action_space

        if not _check_action_space(action_fields):
            raise ValueError(f"Invalid action_space fields for SLAIFrankaOutputs: {action_fields}")

        # Compute how many dimensions we need based on fields.
        dim = 0
        for field in action_fields:
            s = SLAI_FRANKA_ACTION_SLICES[field]
            dim += (s.stop - s.start)

        actions = np.asarray(data["action"][:, :dim])

        if "gripper" in action_fields and self.action_gripper_type == "01":
            gripper_len = SLAI_FRANKA_ACTION_SLICES["gripper"].stop - SLAI_FRANKA_ACTION_SLICES["gripper"].start
            gripper_start = actions.shape[-1] - gripper_len
            gripper_slice = actions[:, gripper_start: gripper_start + gripper_len]
            gripper_bin = _apply_gripper_mode_01_1d(gripper_slice, self.action_gripper_threshold)
            actions = np.concatenate([actions[:, :gripper_start], gripper_bin], axis=-1)

        return {"action": actions}
