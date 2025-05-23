# This script is used to train the Conditional ODE model for the Two Arm Lift task.
# It uses the 6-dimensional rotation matrix (first two rows) of the arm's state and action.
# The model is conditioned on the initial grasp position of the two pot handles.

import torch
import numpy as np
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 250 # horizon, length of each trajectory

# Load expert data
expert_data = np.load("data/expert_actions_rot6d_20.npy")
expert_data1 = expert_data[:, :, :10]
expert_data2 = expert_data[:, :, 10:20]


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
X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32) # Shape: (N, 10)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32) # Shape: (N, 10)

X_train2 = []
Y_train2 = []
for traj in expert_data2:
    for i in range(len(traj) - 1):
        X_train2.append(traj[i])  # Current state + goal
        Y_train2.append(traj[i + 1])  # Next state
X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 10)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 10)

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmLift():
    def __init__(self, state_size=10, action_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"
env = TwoArmLift()

# Prepping expert data for training
actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()

# Preparing conditional vectors for training
with open("data/pot_states_rot6d_20.npy", "rb") as f:
    obs = np.load(f)
obs1 = torch.FloatTensor(obs).to(device)
obs2 = torch.FloatTensor(obs).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# Training
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra="_T250_rot6d_pot_20", endpoint_loss=False)
action_cond_ode.save(extra="_T250_rot6d_pot_20")
action_cond_ode.load(extra="_T250_rot6d_pot_20")