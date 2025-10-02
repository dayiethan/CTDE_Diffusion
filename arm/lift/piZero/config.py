
# This code is based on and should be added to the https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
# Adapted for KINOVA two-arm robot setup

# Add these imports to the original config.py
import dataclasses
import pathlib
from typing import Sequence
from openpi.training import config as openpi_config
from openpi.training.config import TrainConfig, DataConfig, DataConfigFactory
from openpi import transforms as _transforms
from openpi.training.config import ModelTransformFactory
from openpi.training import weight_loaders
from openpi.models import pi0_config, pi0_fast

# Import kinova_policy - this should be adjusted based on how you use this config
# Option 1: If running from piZero directory
import kinova_policy
# Option 2: If this is added to OpenPi's config.py, use absolute import:
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'path/to/CTDE_Diffusion/arm/lift/piZero'))
# import kinova_policy

@dataclasses.dataclass(frozen=True)
class KinovaDataConfig(DataConfigFactory):
    """
    Data configuration for KINOVA two-arm robot setup.
    
    This config handles transforms for the two-arm pot lifting task with:
    - 14-dimensional actions (7 per robot: 3 pos + 3 rot + 1 gripper)
    - 20-dimensional state (10 per robot: 3 pos + 6 rot6d + 1 gripper)  
    - Two eye-in-hand cameras (robot0_eye_in_hand, robot1_eye_in_hand)
    - Language instructions for collaborative manipulation
    """

    action_sequence_keys: Sequence[str] = ("action",)

    def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
        # Repack transform to map KINOVA observation keys to Pi0 format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/robot0_eye_in_hand": "observation.images.robot0_eye_in_hand",
                        "observation/images/robot1_eye_in_hand": "observation.images.robot1_eye_in_hand",
                        "observation/state": "observation.state",
                        "action": "action", 
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms for KINOVA two-arm setup
        data_transforms = _transforms.Group(
            inputs=[kinova_policy.KinovaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[kinova_policy.KinovaOutputs(action_dim=model_config.action_dim)],
        )

        # KINOVA action masking for delta actions
        # For two-arm setup: 14 actions total
        # Robot 0: [pos(3), rot(3), gripper(1)] - positions/rotations as delta, gripper as absolute
        # Robot 1: [pos(3), rot(3), gripper(1)] - positions/rotations as delta, gripper as absolute
        # Mask: [True]*6 + [False] + [True]*6 + [False] = 12 delta + 2 absolute
        delta_action_mask = _transforms.make_bool_mask(6, -1) + _transforms.make_bool_mask(6, -1)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (tokenization, etc.) - keep as is
        model_transforms = ModelTransformFactory()(model_config)

        # Return complete data config
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class KinovaRobot0DataConfig(DataConfigFactory):
    """
    Data configuration for KINOVA Robot 0 individual policy.
    
    Processes only Robot 0's state and actions (7-dim each).
    """

    action_sequence_keys: Sequence[str] = ("action",)

    def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
        # Repack transform to map KINOVA observation keys to Pi0 format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/robot0_eye_in_hand": "observation.images.robot0_eye_in_hand",
                        "observation/images/robot1_eye_in_hand": "observation.images.robot1_eye_in_hand",
                        "observation/state": "observation.state",
                        "action": "action", 
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms for KINOVA Robot 0
        data_transforms = _transforms.Group(
            inputs=[kinova_policy.KinovaRobot0Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[kinova_policy.KinovaOutputs(action_dim=model_config.action_dim)],
        )

        # Robot 0 action masking for delta actions (7 actions)
        # [pos(3), rot(3), gripper(1)] - positions/rotations as delta, gripper as absolute
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (tokenization, etc.)
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class KinovaRobot1DataConfig(DataConfigFactory):
    """
    Data configuration for KINOVA Robot 1 individual policy.
    
    Processes only Robot 1's state and actions (7-dim each).
    """

    action_sequence_keys: Sequence[str] = ("action",)

    def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
        # Repack transform to map KINOVA observation keys to Pi0 format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/robot0_eye_in_hand": "observation.images.robot0_eye_in_hand",
                        "observation/images/robot1_eye_in_hand": "observation.images.robot1_eye_in_hand",
                        "observation/state": "observation.state",
                        "action": "action", 
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms for KINOVA Robot 1
        data_transforms = _transforms.Group(
            inputs=[kinova_policy.KinovaRobot1Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[kinova_policy.KinovaOutputs(action_dim=model_config.action_dim)],
        )

        # Robot 1 action masking for delta actions (7 actions)
        # [pos(3), rot(3), gripper(1)] - positions/rotations as delta, gripper as absolute
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (tokenization, etc.)
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


_CONFIGS = [
    # KINOVA Two-Arm Pi0 Configurations
    
    TrainConfig(
        # Pi0 with LoRA for KINOVA two-arm pot lifting
        name="pi0_kinova_two_arm_lora",
        model=pi0_config.Pi0Config(
            action_dim=14,  # 14 actions for two KINOVA arms
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora", 
            action_horizon=30
        ),

        data=KinovaDataConfig(
            repo_id="YOUR_REPO_ID",  # Replace with your HuggingFace dataset repo
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        
        # Training steps - may need more for two-arm coordination
        num_train_steps=10_000,

        freeze_filter=pi0_config.Pi0Config(
            action_dim=14,
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora", 
            action_horizon=30
        ).get_freeze_filter(),

        # Turn off EMA for LoRA finetuning
        ema_decay=None,
    ),

    TrainConfig(
        # Full Pi0 training for KINOVA two-arm
        name="pi0_kinova_two_arm",
        model=pi0_config.Pi0Config(
            action_dim=14,
            paligemma_variant="gemma_2b", 
            action_expert_variant="gemma_300m", 
            action_horizon=30
        ),

        data=KinovaDataConfig(
            repo_id="YOUR_REPO_ID",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        
        num_train_steps=10_000,
    ),

    TrainConfig(
        # Pi0-FAST with LoRA for KINOVA two-arm
        name="pi0_kinova_fast_lora",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=14,  # Critical: 14 actions for two KINOVA arms
            paligemma_variant="gemma_2b_lora", 
            action_horizon=20,  # Slightly longer for coordination
            max_token_len=200
        ),

        data=KinovaDataConfig(
            repo_id="YOUR_REPO_ID",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),

        num_train_steps=7_500,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=14, 
            paligemma_variant="gemma_2b_lora", 
            action_horizon=20, 
            max_token_len=200
        ).get_freeze_filter(),

        ema_decay=None,
    ),

    TrainConfig(
        # Pi0-FAST full training for KINOVA two-arm
        name="pi0_kinova_fast",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=14,
            paligemma_variant="gemma_2b", 
            action_horizon=20,
            max_token_len=200
        ),

        data=KinovaDataConfig(
            repo_id="YOUR_REPO_ID",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),

        num_train_steps=7_500,
    ),

    TrainConfig(
        # Pi0.5 for KINOVA two-arm (smaller, faster model)
        name="pi05_kinova_two_arm",
        model=pi0_config.Pi0Config(
            action_dim=14,
            pi05=True, 
            action_horizon=20
        ),

        data=KinovaDataConfig(
            repo_id="YOUR_REPO_ID",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),        
        num_train_steps=8_000,
    ),
    
    # Individual Robot Configurations for Dual Robot Setup
    TrainConfig(
        # Pi0 for KINOVA Robot 0 (7-dim actions, left robot)
        name="pi0_kinova_robot0",
        model=pi0_config.Pi0Config(
            action_dim=7,
            action_horizon=20
        ),

        data=KinovaRobot0DataConfig(
            repo_id="YOUR_REPO_ID",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
    ),
    
    TrainConfig(
        # Pi0 for KINOVA Robot 1 (7-dim actions, right robot)  
        name="pi0_kinova_robot1",
        model=pi0_config.Pi0Config(
            action_dim=7,
            action_horizon=20
        ),

        data=KinovaRobot1DataConfig(
            repo_id="YOUR_REPO_ID", 
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
    ),
]
