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
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

# -------------------------
# Residual MLP block (works on tensors of shape (..., dim))
# -------------------------
class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * expansion)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h  # residual


# =========================
# Bigger, non-Sequential models
# =========================

class GenNet(nn.Module):
    """
    Generator: s -> a
    - Residual MLP stack with LayerNorm, GELU, Dropout.
    - Configurable width, depth, and MLP expansion.
    """
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden_size: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
        expansion: int = 4,
        final_activation: Optional[nn.Module] = None,  # e.g., nn.Tanh() if you want bounded outputs
    ):
        super().__init__()
        self.inp = nn.Linear(s_dim, hidden_size)
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size, dropout=dropout, expansion=expansion)
            for _ in range(num_layers)
        ])
        self.pre_head_norm = nn.LayerNorm(hidden_size)
        self.head_fc1 = nn.Linear(hidden_size, hidden_size)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, a_dim)
        self.final_activation = final_activation

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        h = self.inp(s)                      # (B, hidden)
        for blk in self.blocks:
            h = blk(h)                       # (B, hidden)
        h = self.pre_head_norm(h)
        h = self.head_fc1(h)
        h = self.head_act(h)
        h = self.head_drop(h)
        a = self.out(h)                      # (B, a_dim)
        if self.final_activation is not None:
            a = self.final_activation(a)
        return a

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
        Load two MAGAIL models (one per arm). Assumes self.pot_dim has been set
        in get_demo(); if not, we infer it from the pot file (no stats loaded).
        """

        pot_dim = getattr(self, "pot_dim", None)
        if pot_dim is None:
            pot_init = np.load("data/pot_start_newslower_20.npy")
            self.pot_dim = pot_init.shape[1]
            pot_dim = self.pot_dim

        in_size = 7 + pot_dim
        save_dir = "trained_models/magail_big"  # where training saved G1/G2

        G1 = GenNet(in_size, 7, 64).to(device)
        G2 = GenNet(in_size, 7, 64).to(device)
        G1.load_state_dict(torch.load(os.path.join(save_dir, "G1.pth"), map_location=device))
        G2.load_state_dict(torch.load(os.path.join(save_dir, "G2.pth"), map_location=device))
        G1.eval(); G2.eval()
        return (G1, G2)


    
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
    
    
    def reactive_mpc_plan(self, magail_models, initial_states, pot_init_raw,
                      segment_length=25, total_steps=325, n_implement=2):
        """
        MPC rollout using MAGAIL generators (one per arm), conditioned on the pot's initial state.
        - magail_models: (G1, G2) where each maps [arm_t_norm(7), pot0_norm(p)] -> arm_{t+1}_norm(7)
        - initial_states: [norm_state_arm1, norm_state_arm2]  (each 7-d, already normalized)
        - pot_init_raw: unnormalized pot initial vector (shape: pot_dim,)
        """
        G1, G2 = magail_models
        # normalize static pot-initial once and reuse every step
        pot_norm = (pot_init_raw - self.pot_mean) / self.pot_std

        # ensure we don't request more steps than a planned segment
        n_impl = int(max(1, min(n_implement, segment_length)))

        full_traj = []
        # working copy of current normalized states per arm
        current_states = [np.array(initial_states[0], dtype=np.float32),
                        np.array(initial_states[1], dtype=np.float32)]

        # number of MPC iterations (each executes n_impl steps)
        n_iters = total_steps // n_impl

        for seg in range(n_iters):
            segments = []
            base_states = [s.copy() for s in current_states]  # starting normalized states for this segment

            # --- plan a segment for each arm via iterative one-step prediction
            for i, (base, G) in enumerate([(base_states[0], G1), (base_states[1], G2)]):
                seg_list = []
                curr = base.copy()  # normalized current arm state (7,)

                for _ in range(segment_length):
                    inp = np.hstack([curr, pot_norm]).astype(np.float32)  # (7 + pot_dim,)
                    s_t = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        next_norm = G(s_t).cpu().numpy().squeeze().astype(np.float32)  # arm_{t+1}_norm
                    seg_list.append(next_norm)
                    curr = next_norm

                seg_i = np.stack(seg_list, axis=0)  # (segment_length, 7)

                # choose the n_impl actions to execute and the next state for provisional reconditioning
                if seg == 0:
                    step_block = seg_i[0:n_impl, :]
                    next_prov  = seg_i[n_impl - 1, :]
                else:
                    step_block = seg_i[1:n_impl + 1, :]
                    next_prov  = seg_i[n_impl, :]

                segments.append(step_block)
                # provisional update; will be overwritten below by true observed state after env.step
                current_states[i] = next_prov

            # --- execute planned actions in the real env and recondition on observations
            for t in range(n_impl):
                # denormalize per arm
                action1 = segments[0][t] * self.std_arm1 + self.mean_arm1
                action2 = segments[1][t] * self.std_arm2 + self.mean_arm2
                action  = np.hstack([action1, action2]).astype(np.float32)

                obs_env, reward, done, info = self.env.step(action)
                if self.render:
                    self.env.render()

                # recondition on true observed state (convert to normalized)
                state1, state2 = self.obs_to_state(obs_env)  # each (7,)
                current_states = [
                    (state1 - self.mean_arm1) / self.std_arm1,
                    (state2 - self.mean_arm2) / self.std_arm2,
                ]

            # store the executed portion of the plan for logging
            full_traj.append(np.stack([segments[0], segments[1]], axis=0))  # (2, n_impl, 7)

        # concat across MPC iterations → (2, total_steps, 7)
        full_traj = np.concatenate(full_traj, axis=1) if len(full_traj) else np.zeros((2, 0, 7), dtype=np.float32)
        print("Full trajectory shape:", np.shape(full_traj))
        return full_traj


    
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

        # ---------- load MAGAIL models now that pot_dim is known ----------
        magail_models = self.load_model()

        # ---------- plan with MPC (MAGAIL uses [arm_norm, pot_norm] internally) ----------
        planned_trajs = self.reactive_mpc_plan(
            magail_models,
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
