import pickle
import time
import numpy as np
from two_arm_lift_env import TwoArmLiftRole
from robosuite.controllers import load_composite_controller_config

def load_and_replay(pkl_path):
    # Load your existing data
    with open(pkl_path, "rb") as f:
        rollout = pickle.load(f)

    # Initialize environment (same config as during recording)
    CAMERA_NAMES = ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', 'robot1_robotview', 'robot1_eye_in_hand']
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera=None,
    camera_names=CAMERA_NAMES,
    camera_heights=256,
    camera_widths=256,
    camera_depths=True,
    camera_segmentations='instance',
    )

    # Initialize simulation
    env.reset()
    
    for obs in rollout:
        # Set full simulation state
        env.sim.set_qpos(obs["robot0_joint_pos"])
        env.sim.set_qvel(obs["robot0_joint_vel"])
        
        # Set pot position
        pot_id = env.sim.model.body_name2id("pot")
        env.sim.model.body_pos[pot_id] = obs["pot_pos"]
        env.sim.model.body_quat[pot_id] = obs["pot_quat"]
        
        # Update visualization
        env.sim.forward()
        env.render()
        time.sleep(1/env.control_freq)

if __name__ == "__main__":
    load_and_replay("rollout.pkl")