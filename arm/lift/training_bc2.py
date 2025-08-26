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
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Behavioral Cloning Network
class ImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


# Parameters
n_gradient_steps = 50_000
batch_size = 32
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 25 # horizon, length of each trajectory
T = 700 # total time steps

# Load expert data
expert_data = np.load("data/expert_actions_newslower_20.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]

# Compute mean and standard deviation
mean_arm1 = np.mean(expert_data1, axis=(0,1))
std_arm1 = np.std(expert_data1, axis=(0,1))
mean_arm2 = np.mean(expert_data2, axis=(0,1))
std_arm2 = np.std(expert_data2, axis=(0,1))

# Normalize data
expert_data1 = (expert_data1 - mean_arm1) / std_arm1
expert_data2 = (expert_data2 - mean_arm2) / std_arm2

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"
env = TwoArmLift()

with open("data/pot_start_newslower_20.npy", "rb") as f:
    pot = np.load(f)

pot_mean = np.mean(pot, axis=0)
pot_std = np.std(pot, axis=0)
pot = (pot - pot_mean) / pot_std
pot_dim = pot.shape[1]

# Build BC training data
def build_bc_data(expert_data, pot):
    X, Y = [], []
    for i in range(len(expert_data)):
        traj = expert_data[i]
        pot_start = pot[i]
        for t in range(len(traj) - 1):
            current = traj[t]
            X.append(np.hstack([current, pot_start]))  # [x_t, x_0]
            Y.append(traj[t + 1])                 # x_{t+1}
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X1, Y1 = build_bc_data(expert_data1, pot)
X2, Y2 = build_bc_data(expert_data2, pot)
X1, Y1 = torch.from_numpy(X1).to(device), torch.from_numpy(Y1).to(device)
X2, Y2 = torch.from_numpy(X2).to(device), torch.from_numpy(Y2).to(device)

input_size = X1.shape[1]
breakpoint()
model1 = ImitationNet(input_size, hidden_size=256, output_size=7).to(device)
model2 = ImitationNet(input_size, hidden_size=256, output_size=7).to(device)

params = list(model1.parameters()) + list(model2.parameters())
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# Joint training for the agents
def joint_train(models, optimizer, criterion, datasets, num_epochs=5000):
    Xs, Ys = zip(*datasets)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0
        for model, X, Y in zip(models, Xs, Ys):
            pred = model(X)
            loss += criterion(pred, Y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return models

model1, model2 = joint_train(
    [model1, model2], optimizer, criterion,
    [(X1, Y1), (X2, Y2)], num_epochs=5000
)

# Save trained models
save_path1 = "trained_models/bc/model1_retry.pth"
save_path2 = "trained_models/bc/model2_retry.pth"

torch.save(model1.state_dict(), save_path1)
torch.save(model2.state_dict(), save_path2)