import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data, DataLoader
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys
import pdb
import csv
import os
from mpc_util import mpc_plan_splicempc

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

# num_trajectories = 10000
# points_per_trajectory = 10

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

# Use it:
expert_data1 = create_mpc_dataset(expert_data_1, planning_horizon=10)
expert_data2 = create_mpc_dataset(expert_data_2, planning_horizon=10)

print(expert_data1.shape)
print(expert_data2.shape)

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



# Setting up training data
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs_final1 = np.repeat(orig1[:, -1, :], repeats=100, axis=0)
obs_final2 = np.repeat(orig2[:, -1, :], repeats=100, axis=0)
obs1 = np.hstack([obs_init1, obs_final1])
obs2 = np.hstack([obs_init2, obs_final2])
obs_temp1 = obs1
obs_temp2 = obs2
actions1 = expert_data1
actions2 = expert_data2
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)

attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]
assert attr_dim1 == env.state_size * 2
assert attr_dim2 == env.state_size * 2

actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()
sig = np.array([sigma_data1, sigma_data2])
with open("data/sigma_data.npy", "wb") as f:
    np.save(f, sig)


# Training
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
# action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra="_T10_splicempc")
# action_cond_ode.save(extra="_T10_splicempc")
action_cond_ode.load(extra="_T10_splicempc")


# Sampling
# Sampling preparation
noise_std = 0.
noise = np.ones(np.shape(obs_temp1))
obs_temp1 = obs_temp1 + noise_std * noise
obs_temp2 = obs_temp2 + noise_std * noise
obs_temp_tensor1 = torch.FloatTensor(obs_temp1).to(device)
obs_temp_tensor2 = torch.FloatTensor(obs_temp2).to(device)
attr_test1 = obs_temp_tensor1
attr_test2 = obs_temp_tensor2
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

    planned_traj1 = mpc_plan_splicempc(action_cond_ode, env, initial1, final1, 0, segment_length=H, total_steps=T)
    planned_traj1 = planned_traj1 * std + mean

    np.save("data/splice_mpc_trajs/mpc_traj1_%s.npy" % i, planned_traj1)

    planned_traj2 = mpc_plan_splicempc(action_cond_ode, env, initial2, final2, 1, segment_length=H, total_steps=T)
    planned_traj2 = planned_traj2 * std + mean

    np.save("data/splice_mpc_trajs/mpc_traj2_%s.npy" % i, planned_traj2)

    # Plot the planned trajectory:
    # plt.figure(figsize=(22, 14))
    # plt.ylim(-7, 7)
    # plt.xlim(-1,21)
    # plt.plot(planned_traj1[:, 0], planned_traj1[:, 1], 'b.-')
    # plt.plot(planned_traj2[:, 0], planned_traj2[:, 1], 'o-', color='orange')
    # ox, oy, r = obstacle
    # circle1 = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
    # plt.gca().add_patch(circle1)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("MPC Planned Trajectory")
    # plt.savefig("figs/mpc/splice/mpc_traj_%s.png" % i)