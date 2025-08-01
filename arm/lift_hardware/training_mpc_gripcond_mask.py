# This script is used to train the Conditional ODE model for the Two Arm Lift task.
# It uses the 3-dimensional rotation vector of the arm's state and action.
# The model is conditioned on the initial grasp position of the two pot handles.

import torch
import numpy as np
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys
import pdb

def create_mpc_dataset(expert_data, planning_horizon=25, eps_wait=1e-3):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon  # we'll create one sub-trajectory starting at each time step

    # Resulting array shape: (n_traj * n_subtraj, planning_horizon, state_dim)
    result = []
    pad_masks = []
    wait_masks = []

    for traj in expert_data:
        for start_idx in range(n_subtraj):
            # If not enough steps, pad with the last step
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx].copy()
                real_len = planning_horizon
            else:
                # Need padding
                real_len = horizon - start_idx
                sub_traj = traj[start_idx:].copy()
                padding = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, padding], axis=0)
            result.append(sub_traj)

            # pad_mask: True for real frames, False for padded
            pad_mask = np.zeros(planning_horizon, dtype=bool)
            pad_mask[:real_len] = True
            pad_masks.append(pad_mask)

            # wait_mask: True if velocity ≥ eps_wait, False if nearly zero
            # We only compute vel on the _real_ portion.
            wait_mask = np.ones(planning_horizon, dtype=bool)
            if real_len > 1:
                # finite‐difference velocity between successive real frames
                vel = np.linalg.norm(sub_traj[1:real_len] - sub_traj[:real_len-1], axis=1)
                # mark any true “wait” frames as False in the mask
                wait_mask[1:real_len] = vel >= eps_wait
            # optionally you could also zero‐out the padded frames:
            wait_mask[real_len:] = False
            wait_masks.append(wait_mask)

    result = np.stack(result, axis=0)
    pad_masks = np.stack(pad_masks, axis=0)
    wait_masks = np.stack(wait_masks, axis=0)
    return result, pad_masks, wait_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Parameters
n_gradient_steps = 100_000
batch_size = 32
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
# model_size = {
#     "d_model": 512,      # twice the transformer width
#     "n_heads": 8,        # more attention heads
#     "depth":   6,        # twice the number of layers
# }
H = 200 # horizon, length of each trajectory
T = 1000 # total time steps

# Load expert data
expert_data = np.load("data/expert_actions_rotvec_sparse_1000.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]
orig1 = expert_data1
orig2 = expert_data2
orig1 = np.array(orig1)
orig2 = np.array(orig2)
expert_data1, pad_masks1, wait_masks1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2, pad_masks2, wait_masks2 = create_mpc_dataset(expert_data2, planning_horizon=H)

# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
try:
    mean = np.load("data/mean_1000.npy")
    std = np.load("data/std_1000.npy")
except FileNotFoundError:
    mean = np.mean(combined_data, axis=(0,1))
    std = np.std(combined_data, axis=(0,1))
    np.save("data/mean_1000.npy", mean)
    np.save("data/std_1000.npy", std)

# Normalize data
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std
orig1 = (orig1 - mean) / std
orig2 = (orig2 - mean) / std

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"
env = TwoArmLift()

# Preparing expert data for training
actions1 = expert_data1[:, :H, :]
actions2 = expert_data2[:, :H, :]
actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()

# Prepare conditional vectors for training
with open("data/pot_states_1000.npy", "rb") as f:
    obs = np.load(f)
# obs_init1 = np.hstack([expert_data1[:, 0, :3], expert_data1[:, 0, 6:7]])
# obs_init2 = np.hstack([expert_data2[:, 0, :3], expert_data2[:, 0, 6:7]])
# obs_final1 = np.repeat(orig1[:, -1, :3], repeats=T, axis=0)
# obs_final2 = np.repeat(orig2[:, -1, :3], repeats=T, axis=0)
# obs = np.repeat(obs, repeats=T, axis=0)
# obs1 = np.hstack([obs_init1, obs_final1, obs_init2, obs_final2, obs])
# obs2 = np.hstack([obs_init2, obs_final2, obs_init1, obs_final1, obs])

obs_init1_pos  = expert_data1[:,  0, :3]   # (n_samples, 3)
obs_init1_grip = expert_data1[:,  0,  6:7] # (n_samples, 1)
obs_init1 = np.hstack([obs_init1_pos, obs_init1_grip])  # (n_samples, 4)
obs_init2_pos  = expert_data2[:,  0, :3]
obs_init2_grip = expert_data2[:,  0,  6:7]
obs_init2 = np.hstack([obs_init2_pos, obs_init2_grip])  # (n_samples, 4)
obs_final1_pos  = np.repeat(orig1[:, -1, :3],  repeats=T, axis=0)
obs_final1_grip = np.repeat(orig1[:, -1,  6:7], repeats=T, axis=0)
obs_final1 = np.hstack([obs_final1_pos, obs_final1_grip])  # (n_samples*T, 4)
obs_final2_pos  = np.repeat(orig2[:, -1, :3],  repeats=T, axis=0)
obs_final2_grip = np.repeat(orig2[:, -1,  6:7], repeats=T, axis=0)
obs_final2 = np.hstack([obs_final2_pos, obs_final2_grip])  # (n_samples*T, 4)
obs = np.repeat(obs, repeats=T, axis=0)     # (n_samples, 6)
obs1 = np.hstack([obs_init1, obs_final1, obs_init2, obs_final2, obs[:, :3]])  # → shape (n_samples*T, 19)
obs2 = np.hstack([obs_init2, obs_final2, obs_init1, obs_final1, obs[:, 3:6]])  # → shape (n_samples*T, 19)
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# Training
end = "_lift_mpc_P%sE1_%sT_crosscondfinalpos_gripcond_mask" % (H, T)
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=300, n_models = 2, **model_size)
# action_cond_ode.train_mask([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra=end, endpoint_loss=False, pad_masks=[pad_masks1, pad_masks2], wait_masks=[wait_masks1, wait_masks2])
# action_cond_ode.save(extra=end)
action_cond_ode.load(extra=end)

# Sampling
def reactive_mpc_plan(ode_model, env, initial_states, final_states, obs, segment_length=100, total_steps=1000, n_implement=1):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
    - ode_model: the Conditional_ODE (diffusion model) instance.
    - env: your environment, which must implement reset_to() and step().
    - initial_states: a numpy array of shape (n_agents, state_size) that represent the starting states for the robots.
    - obs: a numpy array of shape (6,) representing the eef positions of the two pot handles.
    - total_steps: total length of the planned trajectory.
    - n_implement: number of steps to implement at each iteration.
    
    Returns:
    - full_traj: a numpy array of shape (n_agents, total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()      # shape: (n_agents, state_size)
    n_agents = len(current_states)

    for seg in range(total_steps // n_implement):

        base_states = current_states.copy()     
        segments = []

        for i in range(n_agents):
            cond = [base_states[i], final_states[i]]  # start with the current state and the final state for this agent
            for j in range(n_agents):
                if j != i:
                    cond.append(base_states[j])
                    cond.append(final_states[j])
            if i == 0:
                cond.append(obs[:3])
            else:
                cond.append(obs[3:6])
            # cond.append(obs)
            cond = np.hstack(cond)
            cond_tensor = torch.tensor(cond, dtype=torch.float32, device=ode_model.device).unsqueeze(0)

            sampled = ode_model.sample(
                attr=cond_tensor,
                traj_len=segment_length,
                n_samples=1,
                w=1.0,
                model_index=i
            )
            seg_i = sampled.cpu().numpy()[0]  # (segment_length, action_size)

            if seg == 0:
                take = seg_i[0:n_implement, :]
                new_state_pos  = seg_i[n_implement-1, :3]   # (n_samples, 3)
                new_state_grip = seg_i[n_implement-1, 6:7] # (n_samples, 1)
                new_state = np.hstack([new_state_pos, new_state_grip])  # (n_samples, 4)
                # new_state = seg_i[n_implement-1, :]
            else:
                take = seg_i[1:n_implement+1, :]
                new_state_pos  = seg_i[n_implement-1, :3]   # (n_samples, 3)
                new_state_grip = seg_i[n_implement-1, 6:7] # (n_samples, 1)
                new_state = np.hstack([new_state_pos, new_state_grip])  # (n_samples, 4)
                # new_state = seg_i[n_implement, :]
            segments.append(take)
            current_states[i] = new_state

        full_traj.append(np.stack(segments, axis=0))  # (n_agents, n_implement, action_size)

    # concat all segments along the time dimension
    full_traj = np.concatenate(full_traj, axis=1)     # (n_agents, total_steps, action_size)
    return full_traj


for i in range(10):
    cond_idx = i
    planned_trajs = reactive_mpc_plan(action_cond_ode, env, [obs_init1[cond_idx], obs_init2[cond_idx]], [obs_final1[cond_idx], obs_final2[cond_idx]], obs[cond_idx], segment_length=H, total_steps=T, n_implement=1)
    planned_traj1 =  planned_trajs[0] * std + mean
    np.save("samples/P%sE1_%sT_gripcond_mask/planned_traj1_%s_new.npy" % (H, T, cond_idx), planned_traj1)
    planned_traj2 = planned_trajs[1] * std + mean
    np.save("samples/P%sE1_%sT_gripcond_mask/planned_traj2_%s_new.npy" % (H, T, cond_idx), planned_traj2)