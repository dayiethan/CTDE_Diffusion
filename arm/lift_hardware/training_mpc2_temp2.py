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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Parameters
n_gradient_steps = 100_000
batch_size = 16
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 50 # horizon, length of each trajectory
T = 330 # total time steps

# Load expert data
expert_data = np.load("data/expert_actions_rotvec_sparse_330.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]
expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
try:
    mean = np.load("data/mean.npy")
    std = np.load("data/std.npy")
except FileNotFoundError:
    mean = np.mean(combined_data, axis=(0,1))
    std = np.std(combined_data, axis=(0,1))
    np.save("data/mean.npy", mean)
    np.save("data/std.npy", std)

# Normalize data
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std

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
with open("data/pot_states.npy", "rb") as f:
    obs = np.load(f)
obs_init1 = expert_data1[:, 0, :3]
obs_init2 = expert_data2[:, 0, :3]
obs = np.repeat(obs, repeats=T, axis=0)
obs1 = np.hstack([obs_init1, obs_init2, obs])
obs2 = np.hstack([obs_init2, obs_init1, obs])
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# Training
end = "_lift_mpc_P50E1_330T_crosscond_nolf_newzyxdata"
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
# action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra=end, endpoint_loss=False)
# action_cond_ode.save(extra=end)
action_cond_ode.load(extra=end)

# Sampling
def reactive_mpc_plan(
        ode_model,
        initial_states,
        obs,
        segment_length=25,
        total_steps=100,
        n_implement=5):
    """
    Plans a full trajectory by repeatedly sampling segments of length `segment_length`,
    but ensures every agent’s conditioning in each segment uses the same snapshot of
    all other agents at that segment’s start.
    """
    full_traj = []
    current_states = initial_states.copy()      # shape: (n_agents, state_size)
    n_agents = len(current_states)

    for seg in range(total_steps // n_implement):

        base_states = current_states.copy()     
        segments = []

        for i in range(n_agents):
            cond = [base_states[i]]
            for j in range(n_agents):
                if j != i:
                    cond.append(base_states[j])
            cond.append(obs)
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
                take = seg_i[0:n_implement]
                new_state = seg_i[n_implement-1, :3]
            else:
                take = seg_i[1:n_implement+1]
                new_state = seg_i[n_implement, :3]
            segments.append(take)
            current_states[i] = new_state

        full_traj.append(np.stack(segments, axis=0))  # (n_agents, n_implement, action_size)

    # concat all segments along the time dimension
    full_traj = np.concatenate(full_traj, axis=1)     # (n_agents, total_steps, action_size)
    return full_traj


for i in range(10):
    cond_idx = i
    planned_trajs = reactive_mpc_plan(action_cond_ode, [expert_data1[cond_idx, 0, :3], expert_data2[cond_idx, 0, :3]], obs[cond_idx], segment_length=H, total_steps=T, n_implement=1)
    planned_traj1 =  planned_trajs[0] * std + mean
    np.save("samples/P50E1_330T_crosscond_nolf_newzyxdata/planned_traj1_" + str(cond_idx) + ".npy", planned_traj1)
    planned_traj2 = planned_trajs[1] * std + mean
    np.save("samples/P50E1_330T_crosscond_nolf_newzyxdata/planned_traj2_" + str(cond_idx) + ".npy", planned_traj2)