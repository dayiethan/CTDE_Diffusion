#!/usr/bin/env python3

# Dual Robot Pi0 Evaluation for KINOVA Lift Pot Task
# Runs separate Pi0 policies for each robot in collaborative lifting

import os
# Configure JAX memory allocation before importing JAX-related modules
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import time
import numpy as np
import argparse
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

# Import Pi0 model from openpi
from openpi.training import config as pi0_config
from openpi.policies import policy_config
from huggingface_hub import snapshot_download

# Import your KINOVA environment and policy transforms
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env import TwoArmLiftRole
import kinova_policy
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


class DualRobotPi0Controller:
    """
    Dual Robot Pi0 Controller for KINOVA lift pot task.
    
    Runs separate Pi0 policies for each robot and coordinates their actions.
    """
    
    def __init__(self, robot0_model_path: str, robot1_model_path: str):
        """
        Initialize dual robot controller.
        
        Args:
            robot0_model_path: Path to Robot 0's Pi0 model
            robot1_model_path: Path to Robot 1's Pi0 model
        """
        # Load Pi0 policies for each robot
        self.robot0_policy = self._load_policy(robot0_model_path, robot_id=0)
        self.robot1_policy = self._load_policy(robot1_model_path, robot_id=1)
        
        # Policy transforms
        self.robot0_transform = kinova_policy.KinovaRobot0Inputs(action_dim=7)
        self.robot1_transform = kinova_policy.KinovaRobot1Inputs(action_dim=7)
        
        # Action histories for smoothing
        self.action_history_len = 3
        self.robot0_action_history = []
        self.robot1_action_history = []
        
    def _load_policy(self, model_path: str, robot_id: int):
        """Load Pi0 policy from checkpoint."""
        # Download model if it's a HuggingFace repo
        if "/" in model_path and not os.path.exists(model_path):
            model_path = snapshot_download(model_path)
        
        # Load policy configuration
        config = pi0_config.get_config(f"pi0_kinova_robot{robot_id}")
        policy = policy_config.create_policy(config, model_path)
        
        return policy
    
    def format_observation_for_robot(self, obs: Dict[str, Any], robot_id: int) -> Dict[str, Any]:
        """
        Format robosuite observation for specific robot.
        
        Args:
            obs: Robosuite observation dictionary
            robot_id: 0 or 1 for robot selection
            
        Returns:
            Formatted observation for Pi0
        """
        # Extract state components for both robots
        state_keys = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"
        ]
        
        state_components = []
        for key in state_keys:
            if key in obs:
                val = obs[key]
                if np.isscalar(val):
                    state_components.append([val])
                else:
                    state_components.append(val.flatten())
        
        # Combine into full state (both robots)
        full_state = np.concatenate(state_components) if state_components else np.zeros(14)
        
        # Format for Pi0
        pi0_obs = {
            "observation/state": full_state.astype(np.float32),
            "observation/images/robot0_eye_in_hand": obs.get("robot0_eye_in_hand_image", np.zeros((256, 256, 3), dtype=np.uint8)),
            "observation/images/robot1_eye_in_hand": obs.get("robot1_eye_in_hand_image", np.zeros((256, 256, 3), dtype=np.uint8)),
            "prompt": "Pick up the pot with both robots and lift it together"
        }
        
        return pi0_obs
    
    def get_actions(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get actions from both robot policies.
        
        Args:
            obs: Robosuite observation
            
        Returns:
            (robot0_actions, robot1_actions): 7-dim actions for each robot
        """
        # Format observation
        pi0_obs = self.format_observation_for_robot(obs, robot_id=0)  # Both robots get full obs
        
        # Get inputs for each robot policy
        robot0_inputs = self.robot0_transform(pi0_obs)
        robot1_inputs = self.robot1_transform(pi0_obs)
        
        # Run policies
        robot0_output = self.robot0_policy(robot0_inputs)
        robot1_output = self.robot1_policy(robot1_inputs)
        
        # Extract actions
        robot0_actions = robot0_output["actions"][0]  # Remove batch dimension
        robot1_actions = robot1_output["actions"][0]
        
        # Apply action smoothing
        robot0_actions = self._smooth_actions(robot0_actions, self.robot0_action_history)
        robot1_actions = self._smooth_actions(robot1_actions, self.robot1_action_history)
        
        return robot0_actions, robot1_actions
    
    def _smooth_actions(self, actions: np.ndarray, history: list) -> np.ndarray:
        """Apply temporal smoothing to actions."""
        history.append(actions.copy())
        if len(history) > self.action_history_len:
            history.pop(0)
        
        # Average over history
        if len(history) > 1:
            return np.mean(history, axis=0)
        else:
            return actions


def setup_environment():
    """Setup the two-arm KINOVA lift environment."""
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


def evaluate_dual_robot_lift(
    robot0_model_path: str,
    robot1_model_path: str,
    num_episodes: int = 10,
    render: bool = True,
    save_results: bool = True
):
    """
    Evaluate dual robot Pi0 policies on lift pot task.
    
    Args:
        robot0_model_path: Path to Robot 0's model
        robot1_model_path: Path to Robot 1's model  
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        save_results: Whether to save evaluation results
    """
    print("Setting up dual robot Pi0 evaluation...")
    
    # Setup environment
    env = setup_environment()
    
    # Setup dual robot controller
    controller = DualRobotPi0Controller(robot0_model_path, robot1_model_path)
    
    # Evaluation metrics
    success_count = 0
    episode_lengths = []
    episode_rewards = []
    
    print(f"Starting evaluation for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 500
        
        while not done and step_count < max_steps:
            # Get actions from both robots
            robot0_actions, robot1_actions = controller.get_actions(obs)
            
            # Combine actions for robosuite
            combined_actions = kinova_policy.combine_kinova_actions(
                robot0_actions[np.newaxis, :], 
                robot1_actions[np.newaxis, :]
            )[0]
            
            # Execute action
            obs, reward, done, info = env.step(combined_actions)
            
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
                time.sleep(0.02)  # Slow down for visualization
            
            # Check for success
            if info.get("success", False):
                success_count += 1
                print(f"Episode {episode + 1}: SUCCESS in {step_count} steps!")
                break
        
        if step_count >= max_steps:
            print(f"Episode {episode + 1}: TIMEOUT after {max_steps} steps")
        
        episode_lengths.append(step_count)
        episode_rewards.append(episode_reward)
        
        print(f"Episode reward: {episode_reward:.3f}")
    
    # Calculate results
    success_rate = success_count / num_episodes
    avg_episode_length = np.mean(episode_lengths)
    avg_reward = np.mean(episode_rewards)
    
    print(f"\n{'='*50}")
    print(f"DUAL ROBOT PI0 EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
    print(f"Average Episode Length: {avg_episode_length:.1f} steps")
    print(f"Average Episode Reward: {avg_reward:.3f}")
    print(f"{'='*50}")
    
    # Save results
    if save_results:
        results = {
            "success_rate": success_rate,
            "success_count": success_count,
            "num_episodes": num_episodes,
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "avg_episode_length": avg_episode_length,
            "avg_reward": avg_reward
        }
        
        np.save("dual_robot_pi0_results.npy", results)
        print(f"Results saved to dual_robot_pi0_results.npy")
    
    env.close()
    
    return success_rate, avg_episode_length, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dual Robot Pi0 on KINOVA Lift Task")
    parser.add_argument("--robot0_model", type=str, required=True, 
                       help="Path to Robot 0's Pi0 model (local path or HuggingFace repo)")
    parser.add_argument("--robot1_model", type=str, required=True,
                       help="Path to Robot 1's Pi0 model (local path or HuggingFace repo)")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--no_render", action="store_true",
                       help="Disable rendering")
    parser.add_argument("--no_save", action="store_true", 
                       help="Don't save results")
    
    args = parser.parse_args()
    
    success_rate, avg_length, avg_reward = evaluate_dual_robot_lift(
        robot0_model_path=args.robot0_model,
        robot1_model_path=args.robot1_model,
        num_episodes=args.num_episodes,
        render=not args.no_render,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
