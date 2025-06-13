import torch
import numpy as np
from utils.conditional_Action_DiT_latent import Conditional_ODE
import matplotlib.pyplot as plt
from utils.discrete import *
import sys
import pdb
import csv
from utils.mpc_util import reactive_mpc_latent_plan
import torch.nn as nn

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


def create_mpc_dataset(expert_data, planning_horizon=10):
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
# model_size = {
#     "d_model": 512,      # twice the transformer width
#     "n_heads": 8,        # more attention heads
#     "depth":   6,        # twice the number of layers
#     "lin_scale": 256,    # larger conditional embedder
# }
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 25 # horizon, length of each trajectory
HL = 100 # latent horizon, length of each latent trajectory
T = 100 # total time steps

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0) 

# Loading training trajectories
expert_data_1 = np.load('data/expert_data1_100_traj.npy')
expert_data_2 = np.load('data/expert_data2_100_traj.npy')

orig1 = expert_data_1
orig2 = expert_data_2
print(expert_data_1.shape)
print(expert_data_2.shape)

orig1 = np.array(orig1)
orig2 = np.array(orig2)
print(orig1.shape)
print(orig2.shape)

expert_data1, latent_data1 = create_mpc_dataset(expert_data_1, planning_horizon=H)
expert_data2, latent_data2 = create_mpc_dataset(expert_data_2, planning_horizon=H)
print(expert_data1.shape)
print(expert_data2.shape)
print(latent_data1.shape)
print(latent_data2.shape)

combined_data1 = np.concatenate((expert_data1, expert_data2), axis=0)
combined_data2 = np.concatenate((orig1, orig2), axis=0)
mean1 = np.mean(combined_data1, axis=(0,1))
std1 = np.std(combined_data1, axis=(0,1))
mean2 = np.mean(combined_data2, axis=(0,1))
std2 = np.std(combined_data2, axis=(0,1))
mean = (mean1 + mean2)/2
std = (std1 + std2)/2
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std
latent_data1 = (latent_data1 - mean) / std
latent_data2 = (latent_data2 - mean) / std
orig1 = (orig1 - mean) / std
orig2 = (orig2 - mean) / std

# Define environment
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"
env = TwoUnicycle()

# Setting up training data
latent1 = torch.from_numpy(latent_data1).float().to(device)
# latent2 = torch.from_numpy(latent_data2).float().to(device)
latent_dim = 8
encoder1 = LatentEncoder(input_dim=HL * env.state_size, latent_dim=latent_dim).to(device)
# encoder2 = LatentEncoder(input_dim=HL * env.state_size, latent_dim=4).to(device)
# z1 = encoder1(latent1)
# z2 = encoder2(latent2)
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs_init1_cond = expert_data1[:, 4, :]  # follower is conditioned on the leader's state 3 timesteps ahead of itself
obs_final1 = np.repeat(orig1[:, -1, :], repeats=100, axis=0)
obs_final2 = np.repeat(orig2[:, -1, :], repeats=100, axis=0)
obs1 = np.hstack([obs_init1, obs_final1, obs_init2])
obs2 = np.hstack([obs_init2, obs_final2, obs_init1_cond])
obs_temp1 = obs1
obs_temp2 = obs2
actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1] + latent_dim

actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()
sig = np.array([sigma_data1, sigma_data2])

# Training
action_cond_ode = Conditional_ODE(env, encoder1, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, latent1, subdirect="mpc/", extra="_P25E3_lf_latenttrain")
action_cond_ode.save(subdirect="mpc/", extra="_P25E3_lf_latenttrain")
action_cond_ode.load(subdirect="mpc/", extra="_P25E3_lf_latenttrain")

# Sampling
for i in range(100):
    print("Planning Sample %s" % i)
    noise_std = 0.4
    initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
    initial1 = (initial1 - mean) / std
    final1 = final_point_up + noise_std * np.random.randn(*np.shape(final_point_up))
    final1 = (final1 - mean) / std
    initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
    initial2 = (initial2 - mean) / std
    final2 = final_point_down + noise_std * np.random.randn(*np.shape(final_point_down))
    final2 = (final2 - mean) / std

    planned_trajs = reactive_mpc_latent_plan(action_cond_ode, env, [initial1, initial2], [final1, final2], encoder1, segment_length=H, total_steps=T, n_implement=3)

    planned_traj1 =  planned_trajs[0] * std + mean

    np.save("sampled_trajs/mpc_latenttrain_P25E3/mpc_traj1_%s.npy" % i, planned_traj1)

    planned_traj2 = planned_trajs[1] * std + mean

    np.save("sampled_trajs/mpc_latenttrain_P25E3/mpc_traj2_%s.npy" % i, planned_traj2)