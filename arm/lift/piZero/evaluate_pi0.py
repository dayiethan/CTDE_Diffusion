#!/usr/bin/env python3

# Adapted for KINOVA two-arm robot setup with Pi0 policy
# Based on https://github.com/huggingface/lerobot/blob/main/examples/lekiwi/evaluate.py
# Requires OpenPi, LeRobot, and robosuite to be installed
# Run `python evaluate_pi0.py` to run the model

import os
# Configure JAX memory allocation before importing JAX-related modules
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import time
import numpy as np
import argparse
from typing import Dict, Any

# Import Pi0 model from openpi
from openpi.training import config as pi0_config
from openpi.policies import policy_config
from huggingface_hub import snapshot_download

# Import your KINOVA environment and policy transforms
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory to path
from env import TwoArmLiftRole
import kinova_policy
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

def setup_environment():
    """Setup the two-arm KINOVA environment."""
    # Get absolute path to controller config
    controller_path = os.path.join(os.path.dirname(__file__), '..', 'kinova.json')
    controller_config = load_composite_controller_config(robot="Kinova3", controller=controller_path)
    
    env = TwoArmLiftRole(
        robots=["Kinova3", "Kinova3"],
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=True,
        render_camera=None,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_eye_in_hand", "robot1_eye_in_hand"],
        camera_heights=[256, 256],
        camera_widths=[256, 256],
        camera_depths=[False, False] 
    )
    
    return env

def format_observation_for_pi0(obs: Dict[str, Any], state_keys: list = None) -> Dict[str, Any]:
    """
    Convert robosuite observation to Pi0 input format.
    
    Args:
        obs: Robosuite observation dictionary
        state_keys: List of state keys to include (if None, use default)
        
    Returns:
        Formatted observation for Pi0
    """
    # Default state representation (simplified)
    if state_keys is None:
        state_keys = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"
        ]
    
    # Extract and flatten state
    state_components = []
    for key in state_keys:
        if key in obs:
            val = obs[key]
            if np.isscalar(val):
                state_components.append([val])
            else:
                state_components.append(val.flatten())
    
    state = np.concatenate(state_components) if state_components else np.zeros(14)
    
    # Format for Pi0 - note the _image suffix in robosuite
    pi0_obs = {
        "observation/state": state.astype(np.float32),
        "observation/images/robot0_eye_in_hand": obs.get("robot0_eye_in_hand_image", np.zeros((256, 256, 3), dtype=np.uint8)),
        "observation/images/robot1_eye_in_hand": obs.get("robot1_eye_in_hand_image", np.zeros((256, 256, 3), dtype=np.uint8)),
    }
    
    return pi0_obs

def convert_actions_to_robosuite(actions: np.ndarray, current_obs: Dict[str, Any]) -> np.ndarray:
    """
    Convert Pi0 actions to robosuite format.
    
    Args:
        actions: (14,) Pi0 actions [robot0: pos(3), rot(3), grip(1), robot1: pos(3), rot(3), grip(1)]
        current_obs: Current observation for reference
        
    Returns:
        robosuite_actions: (14,) actions in robosuite format
    """
    # Pi0 outputs delta actions, but they should already be in the right format
    # Just ensure it's the right shape
    return actions.flatten()[:14]

def main():
    parser = argparse.ArgumentParser(description='Evaluate Pi0 policy on KINOVA two-arm robot')
    parser.add_argument('--config_name', type=str, default="pi0_kinova_two_arm", 
                       help='Pi0 config name (from config.py)')
    parser.add_argument('--checkpoint_repo', type=str, required=True,
                       help='HuggingFace repo ID for trained model')
    parser.add_argument('--task_description', type=str, 
                       default="Pick up the pot with both robots and lift it together",
                       help='Task description for the policy')
    parser.add_argument('--actions_per_chunk', type=int, default=15,
                       help='Number of actions to execute from each prediction')
    parser.add_argument('--fps', type=int, default=20,
                       help='Control frequency')
    parser.add_argument('--max_episode_steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()

    # Setup environment
    print("Setting up KINOVA two-arm environment...")
    env = setup_environment()
    
    # Load Pi0 model
    print(f"Loading Pi0 model with config: {args.config_name}")
    config = pi0_config.get_config(args.config_name)
    
    print(f"Downloading model from: {args.checkpoint_repo}")
    checkpoint_dir = snapshot_download(repo_id=args.checkpoint_repo)
    pi0_policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Pi0 model loaded successfully")

    # Evaluation loop
    success_count = 0
    total_episodes = 0
    
    for episode in range(args.num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{args.num_episodes}")
        print(f"Task: {args.task_description}")
        print(f"{'='*50}")
        
        # Reset environment
        obs = env.reset()
        step = 0
        episode_reward = 0
        last_actions = None
        action_index = 0
        pred_times = []
        
        while step < args.max_episode_steps:
            t0 = time.perf_counter()
            
            # Predict new actions when needed
            if last_actions is None or action_index >= min(args.actions_per_chunk, len(last_actions)):
                # Format observation for Pi0
                pi0_obs = format_observation_for_pi0(obs)
                pi0_obs["prompt"] = args.task_description
                
                # Run inference
                t_pred_start = time.perf_counter()
                output = pi0_policy.infer(pi0_obs)
                t_pred = time.perf_counter() - t_pred_start
                
                pred_times.append(t_pred)
                pred_times = pred_times[-10:]  # Keep last 10
                
                last_actions = output["actions"]
                action_index = 0
                
                print(f"  Step {step}: Predicted {len(last_actions)} actions, executing {min(args.actions_per_chunk, len(last_actions))}")
                print(f"  Prediction time: {t_pred:.3f}s (avg: {np.mean(pred_times):.3f}s)")
            
            # Execute action
            if action_index < len(last_actions):
                action = last_actions[action_index]
                
                # Convert to robosuite format
                robosuite_action = convert_actions_to_robosuite(action, obs)
                
                # Step environment
                obs, reward, done, info = env.step(robosuite_action)
                episode_reward += reward
                action_index += 1
                step += 1
                
                if args.render:
                    env.render()
                
                # Check for success/failure
                if done:
                    if reward > 0:  # Assuming positive reward indicates success
                        success_count += 1
                        print(f"  ✓ Episode {episode + 1} SUCCEEDED after {step} steps!")
                    else:
                        print(f"  ✗ Episode {episode + 1} failed after {step} steps")
                    break
            else:
                print(f"  Warning: No more actions available at step {step}")
                break
            
            # Maintain FPS
            sleep_time = max(1.0 / args.fps - (time.perf_counter() - t0), 0.0)
            time.sleep(sleep_time)
        
        total_episodes += 1
        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Success rate so far: {success_count}/{total_episodes} ({100*success_count/total_episodes:.1f}%)")
    
    # Final results
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {success_count}")
    print(f"Success rate: {100*success_count/total_episodes:.1f}%")
    print(f"Average prediction time: {np.mean(pred_times):.3f}s")
    
    env.close()

if __name__ == "__main__":
    main()