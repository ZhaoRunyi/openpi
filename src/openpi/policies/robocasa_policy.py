import dataclasses

import einops
import logging
import numpy as np

from typing import List, Dict, Union

from openpi import transforms
from openpi.models import model as _model

def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        # "": np.random.rand(12),
        # NOTE: TODO: it seems t5hat pi0 ask state and action be the same meaning, but informed that not in need
        "state": np.random.rand(25),
        "image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

##### STATE #####

ROBOCASA_STATES = { # for daixianjie/robocasa_lerobot dataset
    "robot0_eef_pos": np.arange(0, 3),
    "robot0_eef_quat": np.arange(3, 7),
    "robot0_gripper_qpos": np.arange(7, 9),
    "robot0_gripper_qvel": np.arange(9, 11),
    "robot0_base_to_eef_pos": np.arange(11, 14),
    "robot0_base_to_eef_quat": np.arange(14, 18),
    "robot0_base_pos": np.arange(18, 21),
    "robot0_base_quat": np.arange(21, 25),
}

def _check_state_space(state_space: List):
    check_ret = True
    for state_name in state_space:
        if state_name not in ROBOCASA_STATES.keys():
            check_ret = False
            break
    return check_ret

STATE_SPACE_STR_MAPPING = {
    # NOTE: see https://github.com/robocasa/robocasa/issues/11, align with paper
    "16d": [
        "robot0_base_to_eef_pos", # 11-14， 3
        "robot0_base_to_eef_quat", # 14-18， 4
        "robot0_base_pos", # 18-21，3 
        "robot0_base_quat", # 21-25， 4
        "robot0_gripper_qpos" # 7-9， 2
    ], # add up to 16
    "25d": [robocasa_state_name for robocasa_state_name in ROBOCASA_STATES.keys()], # add up to 16
}

##### IMAGE #####

OPENPI_IMAGES = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
ROBOCASA_IMAGES = ["image_left", "image_right", "wrist_image"]
DEFAULT_ROBOCASA_IMAGE_SIZE=(224, 224, 3)

def _check_image_space(image_space: Dict):
    check_ret = True
    for openpi_img_name, robocasa_img_name in image_space.items():
        if (openpi_img_name not in OPENPI_IMAGES) or (robocasa_img_name not in ROBOCASA_IMAGES):
            check_ret = False
            break
    return check_ret

IMAGE_SPACE_STR_MAPPING = {
    "2views": {
        "base_0_rgb": "image_left",
        "left_wrist_0_rgb": "wrist_image",
    },
    "3views": {
        "base_0_rgb": "image_left",
        "left_wrist_0_rgb": "wrist_image",
        "right_wrist_0_rgb": "image_right",
    },
}

##### ACTIONS #####

ROBOCASA_ACTIONS = { # for daixianjie/robocasa_lerobot dataset
    "rel_pose_6d": np.arange(0, 6), # corresponding to "right" _action_split_indices
    "gripper": np.arange(6, 7),
    "base": np.arange(7, 10),
    "torso": np.arange(10, 11),
    "base_mode": np.arange(11, 12), # NOTE: https://github.com/robocasa/robocasa/issues/141
}

def _check_action_space(action_space: List):
    check_ret = True
    for action_name in action_space:
        if action_name not in ROBOCASA_ACTIONS.keys():
            check_ret = False
            break
    return check_ret

ACTION_SPACE_STR_MAPPING = {
    # NOTE: see https://github.com/robocasa/robocasa/issues/11, align with paper
    "7d": [
        "rel_pose_6d", # 0-6， 6
        "gripper" # 6-7, 1
    ], # add up to 7
    "12d": [robocasa_action_name for robocasa_action_name in ROBOCASA_ACTIONS.keys()], # add up to 12
}


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """
    This class is used to converted the lerobot-formartted robocasa dataset, collected by human in OpenDrawer environment and do the 'open the right drawer' task
    """

    state_space: Union[str, List[str]]
    image_space: Union[str, Dict]
    action_space: Union[str, List[str]]
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
        
        inputs = {}
        ##### STATE #####

        if isinstance(self.state_space, str):
            # self.state_space = STATE_SPACE_STR_MAPPING.get(self.state_space) # NOTE: fail as @dataclasses.dataclass(frozen=True)
            state_space = STATE_SPACE_STR_MAPPING.get(self.state_space)
            if state_space is None:
                logging.warning(f"String-format state_space {self.state_space} property of RoboCasaInputs is not registerec in STATE_SPACE_STR_MAPPING {list(STATE_SPACE_STR_MAPPING.keys())}")
        else:
            state_space = self.state_space

        state_check_ret  = _check_state_space(state_space)
        if not state_check_ret:
            state_kv_pairs = ", ".join([f"{k}={v}" for k, v in state_space.items()])
            logging.warning(f"Dict-format state_space got invalid content: {state_kv_pairs}")

        state_dict = {}
        all_state_ids = []
        for robocasa_state_name in state_space:
            state_ids = ROBOCASA_STATES[robocasa_state_name]
            all_state_ids.extend(state_ids)

        state_dict["state"] = data["state"][all_state_ids]

        inputs.update(state_dict)

        ##### IMAGE #####

        if isinstance(self.image_space, str):
            image_space = IMAGE_SPACE_STR_MAPPING.get(self.image_space)
            if image_space is None:
                logging.warning(f"String-format image_space {self.image_space} property of RoboCasaInputs is not registerec in IMAGE_SPACE_STR_MAPPING {list(IMAGE_SPACE_STR_MAPPING.keys())}")
        else:
            image_space = self.image_space

        img_check_ret  = _check_image_space(image_space)
        if not img_check_ret:
            img_kv_pairs = ", ".join([f"{k}={v}" for k, v in image_space.items()])
            logging.warning(f"Dict-format image_space got invalid content: {img_kv_pairs}")

        image_dict = {
            "image": {openpi_img_name: np.zeros(DEFAULT_ROBOCASA_IMAGE_SIZE, dtype=np.uint8) for openpi_img_name in OPENPI_IMAGES},
            "image_mask": {openpi_img_name: np.False_ for openpi_img_name in OPENPI_IMAGES}
        }
        for openpi_img_name, robocasa_img_name in image_space.items():
            parsed_img = _parse_image(data[robocasa_img_name])
            image_dict["image"].update(
                {openpi_img_name: parsed_img}
            )
            image_dict["image_mask"].update(
                {openpi_img_name: np.True_}
            )

        inputs.update(image_dict)

        ##### ACTIONS #####

        # Pad actions to the model action dimension.
        # Actions are only available during training.
        if "actions" in data:

            if isinstance(self.action_space, str):
                action_space = ACTION_SPACE_STR_MAPPING.get(self.action_space)
                if action_space is None:
                    logging.warning(f"String-format action_space {self.action_space} property of RoboCasaInputs is not registerec in ACTION_SPACE_STR_MAPPING {list(ACTION_SPACE_STR_MAPPING.keys())}")
            else:
                action_space = self.action_space
            
            action_check_ret  = _check_action_space(action_space)
            if not action_check_ret:
                action_kv_pairs = ", ".join([f"{k}={v}" for k, v in action_space.items()])
                logging.warning(f"Dict-format action_space got invalid content: {action_kv_pairs}")
        
            action_dict = {}
            all_action_ids = []
            for robocasa_action_name in action_space:
                action_ids = ROBOCASA_ACTIONS[robocasa_action_name]
                all_action_ids.extend(action_ids)

            action_dict["actions"] = data["actions"][:, all_action_ids]

            inputs.update(action_dict)

        # NOTE: add prompt from task_index
        data = super().__call__(data)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_space: Union[str, List[str]]

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.

        if isinstance(self.action_space, str):
            action_space = ACTION_SPACE_STR_MAPPING.get(self.action_space)
            if action_space is None:
                logging.warning(f"String-format action_space {self.action_space} property of RoboCasaInputs is not registerec in ACTION_SPACE_STR_MAPPING {list(ACTION_SPACE_STR_MAPPING.keys())}")
        else:
            action_space = self.action_space

        action_check_ret  = _check_action_space(action_space)
        if not action_check_ret:
            action_kv_pairs = ", ".join([f"{k}={v}" for k, v in action_space.items()])
            logging.warning(f"Dict-format action_space got invalid content: {action_kv_pairs}")
    
        action_dim = 0
        for robocasa_action_name in action_space:
            robocasa_action_ids = ROBOCASA_ACTIONS[robocasa_action_name]
            action_dim += len(robocasa_action_ids)

        return {"actions": np.asarray(data["actions"][:, :action_dim])}
