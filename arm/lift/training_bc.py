import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

def create_mpc_dataset(expert_data, planning_horizon=25):
    """
    Build overlapping sub-trajectories of fixed length H.
    expert_data: (N, T, D) -> returns (N*T, H, D) with padding at ends
    Mirrors the function used in your diffusion training script.
    """
    n_traj, horizon, state_dim = expert_data.shape
    result = []
    for traj in expert_data:
        for start_idx in range(horizon):
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                sub_traj = traj[start_idx:]
                pad = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, pad], axis=0)
            result.append(sub_traj)
    return np.stack(result, axis=0)

# Training
n_gradient_steps = 50_000
batch_size = 32
hidden_layers = 256
lr = 1e-3
weight_decay = 1e-4
H = 25

# Model
class ImitationNet(nn.Module):
    def __init__(self, input_size=7, hidden_size=256, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def build_pairs_from_windows(windows):
    """
    windows: (M, H, D)
    returns inputs (M*(H-1), D) and targets (M*(H-1), D) for x_t -> x_{t+1}
    """
    X = windows[:, :-1, :].reshape(-1, windows.shape[-1])
    Y = windows[:, 1:, :].reshape(-1, windows.shape[-1])
    return X, Y

# Load and prepare data
expert_data = np.load("data/expert_actions_newslower_20.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]

# Build sliding windows (keeps parity with your diffusion preprocessing)
expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

# Load pot initial state (one per trajectory)
pot_init = np.load("data/pot_start_newslower_20.npy")  # shape: (n_traj, pot_dim)
pot_dim = pot_init.shape[1]

# We'll need n_traj and horizon from the raw expert array (before windowing)
n_traj, horizon, _ = expert_data[:, :, :7].shape  # arm1 slice just to get shapes

# Stats for the pot (over trajectories); eps like arms
eps = 1e-8
pot_mean = pot_init.mean(axis=0)
pot_std  = pot_init.std(axis=0) + eps

# For each (traj, start_idx) window we created above, attach that traj's p0 for all H steps
num_windows = n_traj * horizon  # create_mpc_dataset iterates traj -> start_idx
pot_rep     = np.repeat(pot_init, repeats=horizon, axis=0)           # (num_windows, pot_dim)
pot_windows = np.repeat(pot_rep[:, None, :], repeats=H, axis=1)      # (num_windows, H, pot_dim)

# Normalize pot features and keep arms' separate norming
pot_windows = (pot_windows - pot_mean) / pot_std

# Per-arm separate normalization (mean/std over all windows and time)
eps = 1e-8
mean1 = expert_data1.mean(axis=(0,1))
std1  = expert_data1.std(axis=(0,1)) + eps
mean2 = expert_data2.mean(axis=(0,1))
std2  = expert_data2.std(axis=(0,1)) + eps

expert_data1 = (expert_data1 - mean1) / std1
expert_data2 = (expert_data2 - mean2) / std2

# Build supervised pairs for arms
X1_arm, Y1 = build_pairs_from_windows(expert_data1)  # (N, 7), (N, 7)
X2_arm, Y2 = build_pairs_from_windows(expert_data2)  # (N, 7), (N, 7)

# Match pot features per (window, step): H-1 inputs per window
X_pot = pot_windows[:, :-1, :].reshape(-1, pot_dim)  # (N, pot_dim)

# Concatenate pot init to each arm's current state -> input size = 7 + pot_dim
X1 = np.concatenate([X1_arm, X_pot], axis=1)
X2 = np.concatenate([X2_arm, X_pot], axis=1)


# Torch datasets/loaders
ds1 = TensorDataset(torch.from_numpy(X1).float(), torch.from_numpy(Y1).float())
ds2 = TensorDataset(torch.from_numpy(X2).float(), torch.from_numpy(Y2).float())
dl1 = DataLoader(ds1, batch_size=batch_size, shuffle=True, drop_last=False)
dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=True, drop_last=False)

steps_per_epoch = max(len(dl1), len(dl2))
epochs = math.ceil(n_gradient_steps / steps_per_epoch)

input_size = 7 + pot_dim
model1 = ImitationNet(input_size=input_size, hidden_size=hidden_layers, output_size=7).to(device)
model2 = ImitationNet(input_size=input_size, hidden_size=hidden_layers, output_size=7).to(device)

params = list(model1.parameters()) + list(model2.parameters())
opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
crit = nn.MSELoss()

# Training 
def train_joint(model1, model2, dl1, dl2, epochs=100):
    for ep in range(1, epochs+1):
        model1.train(); model2.train()
        it1 = iter(dl1)
        it2 = iter(dl2)
        steps = max(len(dl1), len(dl2))
        loss_running = 0.0

        for _ in range(steps):
            try:
                xb1, yb1 = next(it1)
            except StopIteration:
                it1 = iter(dl1); xb1, yb1 = next(it1)
            try:
                xb2, yb2 = next(it2)
            except StopIteration:
                it2 = iter(dl2); xb2, yb2 = next(it2)

            xb1 = xb1.to(device); yb1 = yb1.to(device)
            xb2 = xb2.to(device); yb2 = yb2.to(device)

            pred1 = model1(xb1)
            pred2 = model2(xb2)
            loss1 = crit(pred1, yb1)
            loss2 = crit(pred2, yb2)
            loss = loss1 + loss2

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_running += loss.item()

        if ep % 100 == 0:
            print(f"[{ep:03d}/{epochs}] avg joint loss: {loss_running/steps:.6f}")

    return model1, model2

model1, model2 = train_joint(model1, model2, dl1, dl2, epochs=epochs)

torch.save(model1.state_dict(), "trained_models/bc/bc_pot_arm1.pth")
torch.save(model2.state_dict(), "trained_models/bc/bc_pot_arm2.pth")