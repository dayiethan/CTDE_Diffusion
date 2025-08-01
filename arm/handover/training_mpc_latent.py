# This script is used to train the Conditional ODE model for the Two Arm Handover task.
# It uses the 3-dimensional rotation vector of the arm's state and action.
# The model is conditioned on the initial grasp position of the hammer.

import torch
import numpy as np
import torch.nn as nn
from conditional_Action_DiT_latent import Conditional_ODE
import matplotlib.pyplot as plt
import sys
import pdb

class LatentEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        # x: (batch, window, features)
        b, w, f = x.shape
        x = x.view(b, w*f)
        return self.net(x)    # → (batch, latent_dim)

def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon  # we'll create one sub-trajectory starting at each time step

    # Resulting array shape: (n_traj * n_subtraj, planning_horizon, state_dim)
    latent = []
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
            _latent_list = traj[:start_idx+1]
            latent_list = np.vstack([np.repeat(_latent_list[0:1, :], repeats=horizon-len(_latent_list), axis=0), _latent_list])
            latent.append(latent_list)
            result.append(sub_traj)
    latent = np.stack(latent, axis=0)
    result = np.stack(result, axis=0)
    return result, latent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters
n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 34 # horizon, length of each trajectory
T = 340 # total time steps

# Load expert data
expert_data = np.load("data_pickup_pos/expert_actions_rotvec_200.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]
orig1 = expert_data1
orig2 = expert_data2
print(expert_data1.shape)
print(expert_data2.shape)
orig1 = np.array(orig1)
orig2 = np.array(orig2)
print(orig1.shape)
print(orig2.shape)
expert_data1, latent_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2, latent_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)
print(expert_data1.shape)
print(expert_data2.shape)

# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
mean = np.mean(combined_data, axis=(0,1))
std = np.std(combined_data, axis=(0,1))

# Normalize data
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std
latent_data1 = (latent_data1 - mean) / std
latent_data2 = (latent_data2 - mean) / std

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmHandover():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmHandover"
env = TwoArmHandover()

# Preparing expert data for training
latent1 = torch.from_numpy(latent_data1).float().to(device)
latent_dim = 8
encoder1 = LatentEncoder(input_dim=T * env.state_size, latent_dim=latent_dim).to(device)
actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()

# Prepare conditional vectors for training
with open("data_pickup_pos/hammer_states_rotvec_200.npy", "rb") as f:
    obs = np.load(f)
obs_init1 = expert_data1[:, 0, :3]
obs_init2 = expert_data2[:, 0, :3]
obs_init1_cond = expert_data1[:, 4, :3]
obs = np.repeat(obs, repeats=340, axis=0)
obs1 = np.hstack([obs_init1, obs])
obs2 = np.hstack([obs_init2, obs_init1_cond, obs])
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1] + latent_dim

# Training
action_cond_ode = Conditional_ODE(env, encoder1, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, latent1, extra="_handover_mpc_P34E5_latent")
action_cond_ode.save(extra="_handover_mpc_P34E5_largecond_latent")
action_cond_ode.load(extra="_handover_mpc_P34E5_largecond_latent")