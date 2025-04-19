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

class PolicyPlayer:
    def __init__ (self, env, render = False):
        self.env = env

        self.control_freq = env.control_freq
        self.dt = 1.0 / self.control_freq
        self.max_time = 10
        self.max_steps = int(self.max_time / self.dt)

        self.render = render

        # robot0_base_body_id = env.sim.model.body_name2id("robot0:base")
        # possible: 'robot0_base', 'robot0_fixed_base_link', 'robot0_shoulder_link'

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

        # robot0_init_rotm_world = R.from_quat(obs['robot0_eef_quat_site'], scalar_first = False).as_matrix()
        # robot1_init_rotm_world = R.from_quat(obs['robot1_eef_quat_site'], scalar_first = False).as_matrix()

        self.n_action = self.env.action_spec[0].shape[0]

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset

    def reset(self, seed = 0, mode = 1):
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
        
    def load_model(self, type = "rotvec", state_dim = 7, action_dim = 7):
        n_gradient_steps = 100_000
        batch_size = 64
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 250 # horizon, length of each trajectory

        expert_data = np.load("data/expert_actions_"+type+".npy")
        expert_data1 = expert_data[:, :, :action_dim]
        expert_data2 = expert_data[:, :, action_dim:action_dim*2]

        pot_states = np.load("data/pot_states_"+type+"_100.npy")

        # Compute mean and standard deviation
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        # Normalize data
        expert_data1 = (expert_data1 - mean) / std
        expert_data2 = (expert_data2 - mean) / std

        # Prepare Data for Training
        X_train1 = []
        Y_train1 = []
        for traj in expert_data1:
            for i in range(len(traj) - 1):
                X_train1.append(traj[i])  # Current state + goal
                Y_train1.append(traj[i + 1])  # Next state
        X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32) # Shape: (N, 7)
        Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32) # Shape: (N, 7)

        X_train2 = []
        Y_train2 = []
        for traj in expert_data2:
            for i in range(len(traj) - 1):
                X_train2.append(traj[i])  # Current state + goal
                Y_train2.append(traj[i + 1])  # Next state
        X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 7)
        Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 7)

        env = TwoArmLift(state_size=state_dim, action_size=action_dim)

        obs1 = torch.FloatTensor(pot_states).to(device)
        obs2 = torch.FloatTensor(pot_states).to(device)
        attr1 = obs1
        attr2 = obs2
        attr_dim1 = attr1.shape[1]
        attr_dim2 = attr2.shape[1]

        actions1 = expert_data1[:, :H-1, :]
        actions2 = expert_data2[:, :H-1, :]
        actions1 = torch.FloatTensor(actions1).to(device)
        actions2 = torch.FloatTensor(actions2).to(device)
        sigma_data1 = actions1.std().item()
        sigma_data2 = actions2.std().item()

        action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
        action_cond_ode.load(extra="_T250_"+type+"_pot_100")

        return action_cond_ode

    
    def get_demo(self, seed, mode, file_name = "rollouts_pot/rollout_seed0_mode2.pkl"):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed, mode)

        expert_data = np.load("data/expert_actions_rotvec_100.npy")
        expert_data1 = expert_data[:, :, :7]
        expert_data2 = expert_data[:, :, 7:14]
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        model = self.load_model(type = "rotvec", state_dim = 7, action_dim = 7)

        with open("data/pot_states_rot6d_100.npy", "rb") as f:
            obs = np.load(f)
        obs1 = torch.FloatTensor(obs[0]).to(device).unsqueeze(0)
        obs2 = torch.FloatTensor(obs[0]).to(device).unsqueeze(0)

        traj_len = 250
        n_samples = 1

        sampled1 = model.sample(obs1, traj_len, n_samples, w=1., model_index=0)
        sampled2 = model.sample(obs2, traj_len, n_samples, w=1., model_index=1)
        sampled1 = sampled1.cpu().detach().numpy()[0]
        sampled2 = sampled2.cpu().detach().numpy()[0]
        sampled1 = sampled1 * std + mean
        sampled2 = sampled2 * std + mean


        for i in range(len(sampled1)):
            action = np.hstack([sampled1[i], sampled2[i]])
            obs, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()

    
        
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
    player.get_demo(seed = 0, mode = 3, file_name = "rollouts_pot/rollout_seed0_mode2.pkl")
    # for i in range(100):   
    #     rollout = player.get_demo(seed = i*10, mode = 2)
    #     with open("rollouts/rollout_seed%s_mode2.pkl" % (i*10), "wb") as f:
    #         pkl.dump(rollout, f)
    #     rollout = player.get_demo(seed = i*10, mode = 3)
    #     with open("rollouts/rollout_seed%s_mode3.pkl" % (i*10), "wb") as f:
    #         pkl.dump(rollout, f)
