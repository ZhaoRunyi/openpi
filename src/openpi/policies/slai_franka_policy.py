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
        # NOTE: TODO: it seems that pi0 ask state and action be the same meaning, but informed that not in need
        "observation.state": np.random.rand(9), # 9d = xyz, R[:3, 0], R[:3, 1]
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

@dataclasses.dataclass(frozen=True)
class SLAIFrankaInputs(transforms.PromptFromLeRobotTask):
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
        agentview_image = _parse_image(data["observation.images.agentview"])
        eye_in_hand_image = _parse_image(data["observation.images.eye_in_hand"])

        # NOTE TODO: for initial "open right drawer" trial, we use 1 image_right camera as base, as it look at the right; 
        # and image_right camera as left_wrist, like imaginary idle left arm wrist camera
        # and wrist_image camera as right_wrist

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": agentview_image,
                "left_wrist_0_rgb": eye_in_hand_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(eye_in_hand_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "task_index": data["task_index"]
        }

        # Pad action to the model action dimension. Keep this for your own dataset.
        # Action are only available during training.
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
class SLAIFrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # TODO: 10D action: xyz + R[:3, 0] + R[:3, 1] + gripper
        return {"action": np.asarray(data["action"][:, :10])}
