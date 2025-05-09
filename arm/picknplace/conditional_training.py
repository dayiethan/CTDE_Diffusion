import torch
import numpy as np
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 275 # horizon, length of each trajectory

expert_data = np.load("data/expert_actions.npy")


# Compute mean and standard deviation
mean = np.mean(expert_data, axis=(0,1))
std = np.std(expert_data, axis=(0,1))

# Normalize data
expert_data = (expert_data - mean) / std

# Prepare Data for Training
X_train = []
Y_train = []
for traj in expert_data:
    for i in range(len(traj) - 1):
        X_train.append(traj[i])  # Current state + goal
        Y_train.append(traj[i + 1])  # Next state
X_train = torch.tensor(np.array(X_train), dtype=torch.float32) # Shape: (N, 7)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32) # Shape: (N, 7)

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class PicknPlace():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "PicknPlace"

env = PicknPlace()

actions = expert_data[:, :H-1, :]
with open("data/hammer_positions.npy", "rb") as f:
    obs = np.load(f)
obs = torch.FloatTensor(obs).to(device)

attr = obs
attr_dim = attr.shape[1]

actions = torch.FloatTensor(actions).to(device)
sigma_data = actions.std().item()


# Training
action_cond_ode = Conditional_ODE(env, [attr_dim], [sigma_data], device=device, N=100, n_models = 1, **model_size)
# action_cond_ode.train([actions], [attr], int(5*n_gradient_steps), batch_size, extra="_T275_rotvec_picknplace_10", endpoint_loss=False)
# action_cond_ode.save(extra="_T275_rotvec_picknplace_10")
action_cond_ode.load(extra="_T275_rotvec_picknplace_10")


# Sampling
with open("data/hammer_positions.npy", "rb") as f:
    obs = np.load(f)
cond_idx = -1    # The index of the condition you want from pot_states, this should correlate to the seed and mode that are being sampled
obs = torch.FloatTensor(obs[cond_idx]).to(device).unsqueeze(0)

n_samples = 1

sampled = action_cond_ode.sample(obs, H, n_samples, w=1., model_index=0)
sampled = sampled.cpu().detach().numpy()[0]
sampled = sampled * std + mean

np.save("data/sampled_traj.npy", sampled)