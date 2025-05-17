# This script is used to sample the Conditional ODE model for the Two Arm Handover task and execute the demo.
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
from env import TwoArmHandoverRole
from scipy.spatial.transform import Rotation as R
from transform_utils import SE3_log_map, SE3_exp_map, quat_to_rot6d, rotvec_to_rot6d, rot6d_to_quat, rot6d_to_rotvec, quat_to_rotm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmHandover():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmHandover"

class PolicyPlayer:
    def __init__ (self, env, render= True):
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

        obs = self.reset()
        self.n_action = self.env.action_spec[0].shape[0]

    def reset(self, seed = 0, mode = 1):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()
        self.handle_length = self.env.hammer.handle_length
        self.hammer_headsize = 2*self.env.hammer.head_halfsize

        self.hammer_pos0 = self.robot0_base_ori_rotm.T @ (self.env._hammer_pos - self.robot0_base_pos)
        self.hammer_pos1 = self.robot1_base_ori_rotm.T @ (self.env._hammer_pos - self.robot1_base_pos)
        self.hammer_rotm = quat_to_rotm(self.env._hammer_quat)
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []

        return obs
        
    def load_model(self, type = "rotvec", state_dim = 7, action_dim = 7):
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 340 # horizon, length of each trajectory

        # Load data
        expert_data = np.load("data_pickup_pos/expert_actions_"+type+"_200.npy")
        expert_data1 = expert_data[:, :, :action_dim]
        expert_data2 = expert_data[:, :, action_dim:action_dim*2]
        hammer_states = np.load("data_pickup_pos/hammer_states_"+type+"_200.npy")

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

        env = TwoArmHandover(state_size=state_dim, action_size=action_dim)

        # Prepare conditional vectors
        obs1 = torch.FloatTensor(hammer_states).to(device)
        obs2 = torch.FloatTensor(hammer_states).to(device)
        attr1 = obs1
        attr2 = obs2
        attr_dim1 = attr1.shape[1]
        attr_dim2 = attr2.shape[1]

        # Prepare expert data
        actions1 = expert_data1[:, :H-1, :]
        actions2 = expert_data2[:, :H-1, :]
        actions1 = torch.FloatTensor(actions1).to(device)
        actions2 = torch.FloatTensor(actions2).to(device)
        sigma_data1 = actions1.std().item()
        sigma_data2 = actions2.std().item()

        # Load the model
        action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
        action_cond_ode.load(extra="_T340_"+type+"_hammer_pickup_pos_200")

        return action_cond_ode
    

    def reactive_mpc_plan(self, ode_model, env, initial_states, obs, segment_length=25, total_steps=100, n_implement=5):
        """
        Plans a full trajectory (total_steps long) by iteratively planning
        segment_length-steps using the diffusion model and replanning at every timestep.
        
        Parameters:
        - ode_model: the Conditional_ODE (diffusion model) instance.
        - env: your environment, which must implement reset_to() and step().
        - initial_state: a numpy array of shape (state_size,) (the current state).
        - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
        - model_i: the index of the agent/model being planned for.
        - segment_length: number of timesteps to plan in each segment.
        - total_steps: total length of the planned trajectory.
        
        Returns:
        - full_traj: a numpy array of shape (total_steps, state_size)
        """
        full_traj = []
        current_states = initial_states.copy()

        for seg in range(total_steps // n_implement):
            segments = []
            for i in range(len(current_states)):
                if i == 0:
                    cond = [current_states[0], current_states[1], obs]
                    cond = np.hstack(cond)
                    cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                    sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                    seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                    if seg == 0:
                        segments.append(seg_i[0:n_implement,:])
                        current_states[i] = seg_i[n_implement-1,:]
                    else:
                        segments.append(seg_i[1:n_implement+1,:])
                        current_states[i] = seg_i[n_implement,:]

                else:
                    cond = [current_states[1], current_states[0], obs]
                    cond = np.hstack(cond)
                    cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                    sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                    seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                    if seg == 0:
                        segments.append(seg_i[0:n_implement,:])
                        current_states[i] = seg_i[n_implement-1,:]
                    else:
                        segments.append(seg_i[1:n_implement+1,:])
                        current_states[i] = seg_i[n_implement,:]
            
            seg_array = np.stack(segments, axis=0)
            full_traj.append(seg_array)

        full_traj = np.concatenate(full_traj, axis=1) 
        return np.array(full_traj)

    
    def get_demo(self, seed, mode, cond_idx, H=34, T=340):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed, mode)

        # Loading
        expert_data = np.load("data_pickup_pos/expert_actions_rotvec_200.npy")
        expert_data1 = expert_data[:, :, :7]
        expert_data2 = expert_data[:, :, 7:14]
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        model = self.load_model(type = "rotvec", state_dim = 7, action_dim = 7)

        with open("data_pickup_pos/hammer_states_rotvec_200.npy", "rb") as f:
            obs = np.load(f)
        obs1 = torch.FloatTensor(obs[cond_idx]).to(device).unsqueeze(0) # The index of the condition you want from pot_states, this should correlate to the seed and mode that are being sampled
        obs2 = torch.FloatTensor(obs[cond_idx]).to(device).unsqueeze(0) # The index of the condition you want from pot_states, this should correlate to the seed and mode that are being sampled

        # Sampling
        traj_len = 340
        n_samples = 1

        planned_trajs = self.reactive_mpc_plan(model, env, [expert_data1[cond_idx, 0, :3], expert_data2[cond_idx, 0, :3]], obs[cond_idx], segment_length=H, total_steps=T, n_implement=5)
        planned_traj1 =  planned_trajs[0] * std + mean
        # np.save("sampled_trajs/mpc_P34E5/mpc_traj1_%s.npy" % i, planned_traj1)
        planned_traj2 = planned_trajs[1] * std + mean
        # np.save("sampled_trajs/mpc_P34E5/mpc_traj2_%s.npy" % i, planned_traj2)

        # Run the sampled trajectory in the environment
        for i in range(len(planned_traj1)):
            action = np.hstack([planned_traj1[i], planned_traj2[i]])
            obs, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()

    
        
if __name__ == "__main__":
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmHandoverRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = True)
    cond_idx = 0
    rollout = player.get_demo(seed = cond_idx*10, mode = 1, cond_idx = cond_idx, H=34, T=340)