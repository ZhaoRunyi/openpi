import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

"""THIS MODULE REFERRED https://github.com/huagailuowen/openpi, many thanks!"""

def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        # "": np.random.rand(12),
        # NOTE: TODO: it seems t5hat pi0 ask state and action be the same meaning, but informed that not in need
        "state": np.random.rand(25),
        "image_left": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "image_right": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

# TODO: a sweeping 'Inputs' Strategy:
# 1. across tasks, fixed image mapping or dynamic image mapping:
    # e.g. for 'open the right drawer', base_0_rgb=image_right, but for 'open the right drawer', base_0_rgb=image_left?
# 2. actions can be variable in \
    # overall index: "Raw"=all 12, "Eef"=only [0:6] eef pose + [6:7] gripper
    # eef pose delta and abs: "Raw"=delta (to t-1 frame) eef pose, "Abs"=transformed to abs with initial state, NOTE: "delta"=delta to first state in action chunk as it's pi0 pretraining
# 3. : TODO

@dataclasses.dataclass(frozen=True)
class RobocasaOpenRightDrawerRawInputs(transforms.PromptFromLeRobotTask):
    """
    This class is used to converted the lerobo-formartted robocasa dataset, collected by human in OpenDrawer environment and do the 'open the right drawer' task
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.

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
        image_left = _parse_image(data["image_left"])
        image_right = _parse_image(data["image_right"])
        wrist_image = _parse_image(data["wrist_image"])

        # NOTE TODO: for initial "open right drawer" trial, we use 1 image_right camera as base, as it look at the right; 
        # and image_right camera as left_wrist, like imaginary idle left arm wrist camera
        # and wrist_image camera as right_wrist

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": image_right,
                "left_wrist_0_rgb": image_left,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "task_index": data["task_index"]
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # NOTE: add prompt from task_index
        data = super().__call__(data)

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class RobocasaOpenRightDrawer2ViewsInputs(transforms.PromptFromLeRobotTask):
    """
    This class is used to converted the lerobo-formartted robocasa dataset, collected by human in OpenDrawer environment and do the 'open the right drawer' task
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.

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
        image_left = _parse_image(data["image_left"])
        image_right = _parse_image(data["image_right"])
        wrist_image = _parse_image(data["wrist_image"])

        # NOTE TODO: for initial "open right drawer" trial, we use 1 image_right camera as base, as it look at the right; 
        # and image_right camera as left_wrist, like imaginary idle left arm wrist camera
        # and wrist_image camera as right_wrist

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": image_left,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(image_left),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "task_index": data["task_index"]
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # NOTE: add prompt from task_index
        data = super().__call__(data)

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            print(data["prompt"])
            exit()
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class RobocasaRawOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :12])}
