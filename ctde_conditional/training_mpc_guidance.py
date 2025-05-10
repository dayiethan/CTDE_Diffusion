import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data, DataLoader
import numpy as np
from utils.utils import Normalizer, set_seed
from utils.conditional_Action_DiT_guidance import Conditional_ODE
import matplotlib.pyplot as plt
from utils.discrete import *
import sys
import pdb
import csv
import os
from utils.mpc_util import mpc_plan

def create_mpc_dataset(expert_data, planning_horizon=10):
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
n_gradient_steps = 10_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 10 # horizon, length of each trajectory
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
orig1 = np.array(orig1)
orig2 = np.array(orig2)
assert orig1.shape[0] == 100
assert orig2.shape[0] == 100

expert_data1 = create_mpc_dataset(expert_data_1, planning_horizon=10)
expert_data2 = create_mpc_dataset(expert_data_2, planning_horizon=10)
assert expert_data1.shape[0] == 100*T
assert expert_data2.shape[0] == 100*T

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
orig1 = (orig1 - mean) / std
orig2 = (orig2 - mean) / std
with open("data/mean.npy", "wb") as f:
    np.save(f, mean)
with open("data/std.npy", "wb") as f:
    np.save(f, std)

# Define environment
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"
env = TwoUnicycle()

# Setting up training data and conditional vectors
obs_init_leader = expert_data1[:, 0, :]
obs_init_follower = expert_data2[:, 0, :]
obs_final_leader = np.repeat(orig1[:, -1, :], repeats=100, axis=0)
obs_final_follower = np.repeat(orig2[:, -1, :], repeats=100, axis=0)
obs_leader_traj = expert_data1[:, :10, :]
obs_leader_traj = obs_leader_traj.reshape(obs_leader_traj.shape[0], -1)
obs_leader = np.hstack([obs_init_leader, obs_final_leader])
obs_follower = np.hstack([obs_init_follower, obs_final_follower, obs_leader_traj])
obs_temp_leader = obs_leader
obs_temp_follower = obs_follower
actions_leader = expert_data1
actions_follower = expert_data2
obs_leader = torch.FloatTensor(obs_leader).to(device)
obs_follower = torch.FloatTensor(obs_follower).to(device)

attr_leader = obs_leader
attr_follower = obs_follower
attr_dim_leader = attr_leader.shape[1]
attr_dim_follower = attr_follower.shape[1]

actions_leader = torch.FloatTensor(actions_leader).to(device)
actions_follower = torch.FloatTensor(actions_follower).to(device)
sigma_data_leader = actions_leader.std().item()
sigma_data_follower = actions_follower.std().item()
sig = np.array([sigma_data_leader, sigma_data_follower])


# Training
action_cond_ode = Conditional_ODE(env, attr_dim_leader, [attr_dim_follower], sigma_data_leader, [sigma_data_follower], device=device, N=100, n_followers = 1, **model_size)
# action_cond_ode.train([actions_leader, actions_follower], [attr_leader, attr_follower], int(5*n_gradient_steps), batch_size, extra="_T10_mpc_guidance")
# action_cond_ode.save(extra="_T10_mpc_guidance")
action_cond_ode.load(extra="_T10_mpc_guidance")


# Sampling preparation
noise_std = 0.
noise_leader = np.ones(np.shape(obs_temp_leader))
noise_follower = np.ones(np.shape(obs_temp_follower))
obs_temp_leader = obs_temp_leader + noise_std * noise_leader
obs_temp_follower = obs_temp_follower + noise_std * noise_follower
obs_temp_tensor_leader = torch.FloatTensor(obs_temp_leader).to(device)
obs_temp_tensor_follower = torch.FloatTensor(obs_temp_follower).to(device)
attr_test_leader = obs_temp_tensor_leader
attr_test_follower = obs_temp_tensor_follower
expert_data1 = expert_data1 * std + mean
expert_data2 = expert_data2 * std + mean
ref1 = np.mean(expert_data1, axis=0)
ref2 = np.mean(expert_data2, axis=0)
ref_agent1 = ref1[:, :]
ref_agent2 = ref2[:, :]


# Sampling
for i in range(100):
    noise_std = 0.4
    initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
    initial1 = (initial1 - mean) / std
    final1 = final_point_up + noise_std * np.random.randn(*np.shape(final_point_up))
    final1 = (final1 - mean) / std
    initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
    initial2 = (initial2 - mean) / std
    final2 = final_point_down + noise_std * np.random.randn(*np.shape(final_point_down))
    final2 = (final2 - mean) / std

    planned_traj1 = mpc_plan(action_cond_ode, env, initial1, final1, 0, segment_length=H, total_steps=T)
    leader_traj_cond = planned_traj1[:10, :]
    planned_traj1 = planned_traj1 * std + mean

    np.save("sampled_trajs/mpc_guidance_P10E1/mpc_traj1_%s.npy" % i, planned_traj1)

    planned_traj2 = mpc_plan(action_cond_ode, env, initial2, final2, 1, leader_traj_cond=leader_traj_cond, segment_length=H, total_steps=T)
    planned_traj2 = planned_traj2 * std + mean

    np.save("sampled_trajs/mpc_guidance_P10E1/mpc_traj2_%s.npy" % i, planned_traj2)