# This script is used to sample the Conditional ODE model for the Two Arm Lift task and execute the demo.
# It uses the 3-dimensional rotation vector of the arm's state and action.

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle as pkl
import copy
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from conditional_Action_DiT import Conditional_ODE
from env import TwoArmLiftRole
from scipy.spatial.transform import Rotation as R
from transform_utils import SE3_log_map, SE3_exp_map, quat_to_rot6d, rotvec_to_rot6d, rot6d_to_quat, rot6d_to_rotvec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon  # we'll create one sub-trajectory starting at each time step

    # Resulting array shape: (n_traj * n_subtraj, planning_horizon, state_dim)
    result = []

    for traj in expert_data:
        for start_idx in range(n_subtraj):
            # If not enough steps, pad with the last step
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                # Need padding
                sub_traj = traj[start_idx:]
                padding = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, padding], axis=0)
            result.append(sub_traj)

    result = np.stack(result, axis=0)
    return result

class PolicyPlayer:
    def __init__ (self, env, render = False):
        self.env = env

        self.control_freq = env.control_freq
        self.dt = 1.0 / self.control_freq
        self.max_time = 10
        self.max_steps = int(self.max_time / self.dt)

        self.render = render

        # Extract the base position and orientation (quaternion) from the simulation data
        robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))

        robot1_base_body_id = self.env.sim.model.body_name2id("robot1_base")
        self.robot1_base_pos = self.env.sim.data.body_xpos[robot1_base_body_id]
        self.robot1_base_ori_rotm = self.env.sim.data.body_xmat[robot1_base_body_id].reshape((3,3))

        # Rotation matrix of robots for the home position, both in their own base frame
        self.R_be_home = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]])

        self.n_action = self.env.action_spec[0].shape[0]

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset

    def reset(self, seed = 0):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []

        return obs
        
    def load_model(self, extra, state_dim=7, action_dim=7):
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 25  # horizon, length of each trajectory
        T = 250 # total time steps

        # Load expert data
        expert_data = np.load("data/expert_actions_rotvec_20.npy")
        expert_data1 = expert_data[:, :, :7]
        expert_data2 = expert_data[:, :, 7:14]
        expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
        expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

        # Compute mean and standard deviation
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        self.mean = np.mean(combined_data, axis=(0,1))
        self.std = np.std(combined_data, axis=(0,1))

        # Normalize data
        expert_data1 = (expert_data1 - self.mean) / self.std
        expert_data2 = (expert_data2 - self.mean) / self.std

        env_desc = TwoArmLift(state_size=state_dim, action_size=action_dim) # Renamed to avoid conflict

        # Prepare conditional vectors for training
        with open("data/pot_states_rotvec_20.npy", "rb") as f:
            obs = np.load(f)
        obs_init1 = expert_data1[:, 0, :3]
        obs_init2 = expert_data2[:, 0, :3]
        obs_init1_cond = expert_data1[:, 4, :3]
        obs = np.repeat(obs, repeats=T, axis=0)
        obs1 = np.hstack([obs_init1, obs])
        obs2 = np.hstack([obs_init2, obs_init1_cond, obs])
        obs1 = torch.FloatTensor(obs1).to(device)
        obs2 = torch.FloatTensor(obs2).to(device)
        attr1 = obs1
        attr2 = obs2
        attr_dim1 = attr1.shape[1]
        attr_dim2 = attr2.shape[1]

        # Preparing expert data for training
        actions1 = expert_data1[:, :H, :]
        actions2 = expert_data2[:, :H, :]
        actions1 = torch.FloatTensor(actions1).to(device)
        actions2 = torch.FloatTensor(actions2).to(device)
        sigma_data1 = actions1.std().item()
        sigma_data2 = actions2.std().item()

        # Load the model
        action_cond_ode = Conditional_ODE(env_desc, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
        action_cond_ode.load(extra=extra)

        return action_cond_ode
    
    def _extract_planning_states_from_env_obs(self, env_obs):
        """
        Extracts 7D (pos + rotvec + gripper) end-effector states for both robots
        from the robosuite environment observation.
        """
        # Robot 0
        r0_pos = env_obs['robot0_eef_pos']
        r0_quat = env_obs['robot0_eef_quat']
        r0_rotvec = R.from_quat(r0_quat).as_rotvec()
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        r0_gripper = self.env.sim.data.qpos[qpos_index_0]
        robot0_planning_state = np.hstack([r0_pos, r0_rotvec, r0_gripper])
        # robot0_planning_state = (robot0_planning_state - self.mean) / self.std

        # Robot 1
        r1_pos = env_obs['robot1_eef_pos']
        r1_quat = env_obs['robot1_eef_quat']
        r1_rotvec = R.from_quat(r1_quat).as_rotvec()
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]
        r1_gripper = self.env.sim.data.qpos[qpos_index_1]
        robot1_planning_state = np.hstack([r1_pos, r1_rotvec, r1_gripper])
        # robot1_planning_state = (robot1_planning_state - self.mean) / self.std

        return robot0_planning_state, robot1_planning_state
    
    
    def execute_mpc_online(self, ode_model, pot_handles_obs_condition, segment_length=25, total_steps=250, n_implement=2):
        """
        Executes MPC online, replanning and interacting with the environment.

        Parameters:
        - ode_model: The Conditional_ODE model.
        - pot_handles_obs_condition: A numpy array (e.g., shape (6,)) representing the static eef positions of the two pot handles for conditioning the model.
        - segment_length: Planning horizon for the ode_model (H).
        - total_steps: Total environment steps to run (T).
        - n_implement: Number of steps from the plan to execute before replanning.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Mean and Std for action denormalization are not set. Call load_model first.")
        
        current_env_obs_dict = self.env._get_observations() # Get initial observation from env
        # breakpoint()

        planned_actions_segment_r0 = None
        planned_actions_segment_r1 = None
        current_plan_step_idx = 0 # To iterate through the currently planned segment

        for env_step_count in range(total_steps):
            if env_step_count % n_implement == 0:  # Time to replan
                r0_current_planning_state, r1_current_planning_state = self._extract_planning_states_from_env_obs(current_env_obs_dict)

                # Plan for robot 0
                cond0_list = [r0_current_planning_state[:3], pot_handles_obs_condition]
                cond0 = np.hstack(cond0_list)
                cond0_tensor = torch.tensor(cond0, dtype=torch.float32, device=device).unsqueeze(0)
                sampled0 = ode_model.sample(attr=cond0_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                planned_actions_segment_r0 = sampled0.cpu().detach().numpy()[0]  # Shape: (segment_length, action_size)

                # Plan for robot 1
                cond1_list = [r1_current_planning_state[:3], planned_actions_segment_r0[n_implement, :3], pot_handles_obs_condition]
                cond1 = np.hstack(cond1_list)
                cond1_tensor = torch.tensor(cond1, dtype=torch.float32, device=device).unsqueeze(0)
                sampled1 = ode_model.sample(attr=cond1_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=1)
                planned_actions_segment_r1 = sampled1.cpu().detach().numpy()[0] # Shape: (segment_length, action_size)

                current_plan_step_idx = 0  # Reset index for the new plan

            action_r0_normalized = planned_actions_segment_r0[current_plan_step_idx]
            action_r1_normalized = planned_actions_segment_r1[current_plan_step_idx]
            current_plan_step_idx += 1

            action_r0_denormalized = action_r0_normalized * self.std + self.mean
            action_r1_denormalized = action_r1_normalized * self.std + self.mean
            
            combined_action_to_env = np.hstack([action_r0_denormalized, action_r1_denormalized])

            next_env_obs_dict, reward, done, info = self.env.step(combined_action_to_env)

            if self.render:
                self.env.render()
            
            current_env_obs_dict = next_env_obs_dict # Update current observation

    
    def get_demo(self, seed, cond_idx, extra, H=25, T=250, n_implement_steps=2): # Added n_implement_steps
        """
        Main file to run the MPC online execution.
        H: Planning horizon for the model.
        T: Total environment steps.
        n_implement_steps: Number of steps to execute from plan before replanning.
        """
        obs_dict_after_reset = self.reset(seed) # Resets env and self.rollout
        model = self.load_model(extra=extra, state_dim=7, action_dim=7)

        with open("data/pot_states_rot6d_20.npy", "rb") as f:
            pot_handles_conditions = np.load(f)
        
        current_pot_handles_condition = pot_handles_conditions[cond_idx]
        
        print("Starting online MPC execution...")
        self.execute_mpc_online(
            ode_model=model,
            pot_handles_obs_condition=current_pot_handles_condition,
            segment_length=H,      # Planning horizon
            total_steps=T,         # Total env steps
            n_implement=n_implement_steps # Steps to execute per plan
        )

    
        
if __name__ == "__main__":
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = False)
    cond_idx = 1
    player.get_demo(seed = cond_idx*10, cond_idx = cond_idx, extra="_lift_mpc_P25E2_50ksteps", H=25, T=250)
