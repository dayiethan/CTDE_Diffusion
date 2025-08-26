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
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

class ImitationNet(nn.Module):
    def __init__(self, input_size=7, hidden_size=256, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

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

    def load_model(self, *_, state_dim=7, action_dim=7):
        """
        Load two BC models (one per arm). Assumes self.pot_dim has been set
        in get_demo(); if not, we infer it from the pot file (no stats loaded).
        """
        import os
        import torch.nn as nn

        # Small BC MLP (match training hidden size if you changed it there)
        class ImitationNet(nn.Module):
            def __init__(self, input_size=7, hidden_size=256, output_size=7):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.act = nn.ReLU()
            def forward(self, x):
                x = self.act(self.fc1(x))
                x = self.act(self.fc2(x))
                return self.fc3(x)

        # Ensure pot_dim is known (prefer what get_demo set)
        pot_dim = getattr(self, "pot_dim", None)
        if pot_dim is None:
            pot_init = np.load("data/pot_start_newslower_20.npy")
            self.pot_dim = pot_init.shape[1]
            pot_dim = self.pot_dim

        in_size = 7 + pot_dim
        save_dir = "trained_models/bc"

        bc1 = ImitationNet(input_size=in_size, hidden_size=256, output_size=7).to(device)
        bc2 = ImitationNet(input_size=in_size, hidden_size=256, output_size=7).to(device)
        bc1.load_state_dict(torch.load(os.path.join(save_dir, "model1_retry.pth"), map_location=device))
        bc2.load_state_dict(torch.load(os.path.join(save_dir, "model2_retry.pth"), map_location=device))
        bc1.eval(); bc2.eval()
        return (bc1, bc2)


    
    def obs_to_state(self, obs):
        """
        Read the two arms’ current 7D states (local pos, rotation-vector, gripper)
        using the correct end-effector SITE frame for rotation.
        """

        # --- Robot 0 ---
        # 1) World->local for position
        world_pos0 = obs["robot0_eef_pos"]
        local_pos0 = self.robot0_base_ori_rotm.T @ (world_pos0 - self.robot0_base_pos)
        
        # 2) Get rotation vector from the SITE quaternion
        quat0_site = obs["robot0_eef_quat_site"]
        R_world_to_site0 = R.from_quat(quat0_site).as_matrix()
        R_base_to_site0 = self.robot0_base_ori_rotm.T @ R_world_to_site0
        rotvec0 = R.from_matrix(R_base_to_site0).as_rotvec()

        # 3) Gripper joint position
        grip0 = self.env.sim.data.qpos[self.qpos_index_0]
        state0 = np.hstack([local_pos0, rotvec0, grip0])

        # --- Robot 1 ---
        # 1) World->local for position
        world_pos1 = obs["robot1_eef_pos"]
        local_pos1 = self.robot1_base_ori_rotm.T @ (world_pos1 - self.robot1_base_pos)

        # 2) Get rotation vector from the SITE quaternion
        quat1_site = obs["robot1_eef_quat_site"]
        R_world_to_site1 = R.from_quat(quat1_site).as_matrix()
        R_base_to_site1 = self.robot1_base_ori_rotm.T @ R_world_to_site1
        rotvec1 = R.from_matrix(R_base_to_site1).as_rotvec()

        # 3) Gripper joint position
        grip1 = self.env.sim.data.qpos[self.qpos_index_1]
        state1 = np.hstack([local_pos1, rotvec1, grip1])

        return state0, state1
    
    
    def reactive_mpc_plan(self, bc_models, initial_states, pot_init_raw, segment_length=25, total_steps=325, n_implement=2):
        """
        MPC rollout using BC models (one per arm), conditioned on the pot's initial state.
        - bc_models: (bc_arm1, bc_arm2)
        - initial_states: [norm_state_arm1, norm_state_arm2]  (7-d each, normalized)
        - pot_init_raw: unnormalized pot initial vector (pot_dim,)
        """
        bc1, bc2 = bc_models
        full_traj = []
        current_states = [s.copy() for s in initial_states]  # normalized

        # Normalize pot init once and reuse
        pot_norm = (pot_init_raw - self.pot_mean) / self.pot_std  # (pot_dim,)

        for seg in range(total_steps // n_implement):
            segments = []
            base_states = [s.copy() for s in current_states]

            # 1) plan a normalized segment for each arm via iterative one-step BC
            for i, (base, model) in enumerate([(base_states[0], bc1), (base_states[1], bc2)]):
                seg_list = []
                curr = base.copy()  # normalized current arm state (7,)

                for _ in range(segment_length):
                    # concatenate [arm_state_norm, pot_init_norm] -> (7 + pot_dim,)
                    inp = np.hstack([curr, pot_norm])
                    s_t = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        next_norm = model(s_t).cpu().numpy().squeeze()   # predicts x_{t+1} (normalized)
                    seg_list.append(next_norm)
                    curr = next_norm

                seg_i = np.stack(seg_list, axis=0)  # (segment_length, 7)

                # keep your original pick logic for n_implement
                if seg == 0:
                    step_block = seg_i[0:n_implement, :]
                    next_norm  = seg_i[n_implement-1, :]
                else:
                    step_block = seg_i[1:n_implement+1, :]
                    next_norm  = seg_i[n_implement, :]

                segments.append(step_block)
                current_states[i] = next_norm

            # 2) execute those n_implement actions on the real robot (denorm per arm)
            for t in range(n_implement):
                action1 = segments[0][t] * self.std_arm1 + self.mean_arm1
                action2 = segments[1][t] * self.std_arm2 + self.mean_arm2
                action  = np.hstack([action1, action2])

                obs_env, reward, done, info = self.env.step(action)
                if self.render:
                    self.env.render()

                # 3) re-condition on true observed state (convert back to normalized)
                state1, state2 = self.obs_to_state(obs_env)
                current_states = [
                    (state1 - self.mean_arm1) / self.std_arm1,
                    (state2 - self.mean_arm2) / self.std_arm2,
                ]

            full_traj.append(np.stack([s for s in segments], axis=0))  # (2, n_implement, 7)

        full_traj = np.concatenate(full_traj, axis=1)  # (2, total_steps, 7)
        print("Full trajectory shape: ", np.shape(full_traj))
        return np.array(full_traj)

    
    def get_demo(self, seed, cond_idx, H=25, T=700):
        """
        Main file to get the demonstration data.
        Recomputes normalization (arms + pot) and stores on self.
        """
        obs = self.reset(seed)

        # ---------- load expert & split arms ----------
        expert = np.load("data/expert_actions_newslower_20.npy")  # (N, T, 14)
        arm1 = expert[:, :, :7]
        arm2 = expert[:, :, 7:14]

        n_traj, horizon, _ = arm1.shape

        # ---------- stats from RAW trajectories (match training) ----------
        eps = 1e-8
        self.mean_arm1 = arm1.mean(axis=(0, 1))
        self.std_arm1  = arm1.std(axis=(0, 1)) + eps
        self.mean_arm2 = arm2.mean(axis=(0, 1))
        self.std_arm2  = arm2.std(axis=(0, 1)) + eps

        # ---------- windows ONLY to get per-window current states (NO window stats) ----------
        arm1_w = create_mpc_dataset(arm1, planning_horizon=H)  # just to get [:,0,:]
        arm2_w = create_mpc_dataset(arm2, planning_horizon=H)

        # normalized per-window "x_t" using RAW stats above
        obs_init1 = (arm1_w[:, 0, :] - self.mean_arm1) / self.std_arm1
        obs_init2 = (arm2_w[:, 0, :] - self.mean_arm2) / self.std_arm2

        # ---------- use LIVE env state at reset for the chosen window ----------
        obs = self.reset(seed)
        state1_live, state2_live = self.obs_to_state(obs)
        init1_norm = (state1_live - self.mean_arm1) / self.std_arm1
        init2_norm = (state2_live - self.mean_arm2) / self.std_arm2

        # keep your array-based API: overwrite the index you actually use
        n_traj, horizon, _ = arm1.shape
        total_windows = n_traj * horizon
        cond_idx = cond_idx % total_windows
        obs_init1[cond_idx] = init1_norm
        obs_init2[cond_idx] = init2_norm

        # ---------- pot initial (static) + stats (match training) ----------
        pot_init_all = np.load("data/pot_start_newslower_20.npy")   # shape: (n_traj, pot_dim)
        self.pot_dim  = pot_init_all.shape[1]
        self.pot_mean = np.mean(pot_init_all, axis=0)
        self.pot_std = np.std(pot_init_all, axis=0)

        # map window index -> trajectory index (one pot init per trajectory)
        traj_idx = cond_idx // horizon
        pot0_raw = pot_init_all[traj_idx] 

        # ---------- load BC models now that pot_dim is known ----------
        bc_models = self.load_model()

        # ---------- plan with MPC (BC uses [arm_norm, pot_norm] internally) ----------
        planned_trajs = self.reactive_mpc_plan(
            bc_models,
            [init1_norm, init2_norm],    # normalized live starts
            pot0_raw,                    # raw pot init; normalized inside
            segment_length=H,
            total_steps=T*2,
            n_implement=10
        )


        # ---------- denorm for logging/saving ----------
        planned_traj1 = planned_trajs[0] * self.std_arm1 + self.mean_arm1
        planned_traj2 = planned_trajs[1] * self.std_arm2 + self.mean_arm2
        return planned_traj1, planned_traj2


        
if __name__ == "__main__":
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    T = 700
    H = 25

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    horizon=T*2,
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = False)
    cond_idx = 0
    player.get_demo(seed = cond_idx*10, cond_idx = cond_idx, H=H, T=T)
