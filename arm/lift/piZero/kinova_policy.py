# Based on https://github.com/Physical-Intelligence/openpi/src/openpi/policies/libero_policy.py
# Adapted for KINOVA two-arm robot setup in pot lifting task

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_kinova_example() -> dict:
    """Creates a random input example for the KINOVA two-arm policy."""
    return {
        "observation/state": np.random.rand(20), 
        "observation/images/robot0_eye_in_hand": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/images/robot1_eye_in_hand": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "pick up the pot with both robots and lift it together",
    }


def _parse_image(image) -> np.ndarray:
    """Parse and normalize image data for Pi0 input."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KinovaInputs(transforms.DataTransformFn):
    """
    Transform inputs for KINOVA two-arm robot policy.
    
    This class converts KINOVA robot observations (two arm states + two camera images)
    to the format expected by Pi0 model for training and inference.
    """

    action_dim: int = 14

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if len(state) < self.action_dim:
            state = transforms.pad_to_dim(state, self.action_dim)
        else:
            state = state[:self.action_dim]

        robot0_image = _parse_image(data["observation/images/robot0_eye_in_hand"])
        robot1_image = _parse_image(data["observation/images/robot1_eye_in_hand"])
        
        #TODO, here we need a policy for EACH ROBOT! Since this is a 2 robot task!

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(robot0_image),  # Pad with zeros since no base camera - FOR ANTHONY - test if this works, according to Pi Zero it should.
                "left_wrist_0_rgb": robot0_image, 
                "right_wrist_0_rgb": robot1_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,  # Mask base camera since KINOVA doesn't have one
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "action" in data:
            actions = data["action"]
            if len(actions) != self.action_dim:
                actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaRobot0Inputs(transforms.DataTransformFn):
    """
    IMPORTANT: FOR THE SCENARIO IF WE WANT 2 DIFFERENT POLICIES FOR EACH ROBOT - THIS IS FOR ROBOT 0!
    """
    
    action_dim: int = 7 
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        full_state = data["observation/state"]
        robot0_state = full_state[:self.action_dim] if len(full_state) >= self.action_dim else transforms.pad_to_dim(full_state[:self.action_dim], self.action_dim)
        
        robot0_image = _parse_image(data["observation/images/robot0_eye_in_hand"])
        
        inputs = {
            "state": robot0_state,
            "image": {
                "base_0_rgb": np.zeros_like(robot0_image),  
                "left_wrist_0_rgb": robot0_image,
                "right_wrist_0_rgb": robot0_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        
        if "action" in data:
            full_actions = data["action"]
            robot0_actions = full_actions[:self.action_dim]
            if len(robot0_actions) != self.action_dim:
                robot0_actions = transforms.pad_to_dim(robot0_actions, self.action_dim)
            inputs["actions"] = robot0_actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaRobot1Inputs(transforms.DataTransformFn):
    """
    IMPORTANT: FOR THE SCENARIO IF WE WANT 2 DIFFERENT POLICIES FOR EACH ROBOT - THIS IS FOR ROBOT 1!
    """
    
    action_dim: int = 7 
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        full_state = data["observation/state"]
        robot1_state = full_state[7:14] if len(full_state) >= 14 else transforms.pad_to_dim(full_state[7:], self.action_dim)
        
        robot1_image = _parse_image(data["observation/images/robot1_eye_in_hand"])
        
        inputs = {
            "state": robot1_state,
            "image": {
                "base_0_rgb": np.zeros_like(robot1_image), 
                "left_wrist_0_rgb": robot1_image,  
                "right_wrist_0_rgb": robot1_image, 
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_, 
                "right_wrist_0_rgb": np.True_,
            },
        }
        
        if "action" in data:
            full_actions = data["action"]
            robot1_actions = full_actions[7:14]
            if len(robot1_actions) != self.action_dim:
                robot1_actions = transforms.pad_to_dim(robot1_actions, self.action_dim)
            inputs["actions"] = robot1_actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaOutputs(transforms.DataTransformFn):
    """
    Transform outputs from Pi0 model back to KINOVA two-arm action format.
    
    This class converts Pi0 model outputs to the 14-dimensional action space
    needed for KINOVA two-arm control.
    """

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :self.action_dim])
        
        robot0_actions = actions[:, :7]
        robot1_actions = actions[:, 7:14]
        
        return {
            "actions": actions,
            "robot0_actions": robot0_actions,
            "robot1_actions": robot1_actions
        }


def split_kinova_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split 14-dim KINOVA actions into two 7-dim robot actions.
    
    Args:
        actions: (batch_size, 14) actions for both robots
        
    Returns:
        robot0_actions: (batch_size, 7) actions for robot 0
        robot1_actions: (batch_size, 7) actions for robot 1
    """
    return actions[:, :7], actions[:, 7:14]


def combine_kinova_actions(robot0_actions: np.ndarray, robot1_actions: np.ndarray) -> np.ndarray:
    """
    Combine two 7-dim robot actions into 14-dim KINOVA actions.
    
    Args:
        robot0_actions: (batch_size, 7) actions for robot 0
        robot1_actions: (batch_size, 7) actions for robot 1
        
    Returns:
        actions: (batch_size, 14) combined actions
    """
    return np.concatenate([robot0_actions, robot1_actions], axis=-1)


def parse_kinova_state(state_dict: dict) -> np.ndarray:
    """
    Parse KINOVA observation state from dictionary format.
    
    Args:
        state_dict: Dictionary containing robot states
        
    Returns:
        state: (20,) flattened state vector
    """
    robot0_pos = state_dict.get("robot0_eef_pos", np.zeros(3))
    robot0_quat = state_dict.get("robot0_eef_quat_site", np.zeros(4)) 
    robot0_gripper = state_dict.get("robot0_gripper_pos", 0.0)
    
    robot1_pos = state_dict.get("robot1_eef_pos", np.zeros(3))
    robot1_quat = state_dict.get("robot1_eef_quat_site", np.zeros(4))
    robot1_gripper = state_dict.get("robot1_gripper_pos", 0.0)

    state = np.concatenate([
        robot0_pos, robot0_quat, [robot0_gripper],
        robot1_pos, robot1_quat, [robot1_gripper]
    ])
    
    return state.astype(np.float32)