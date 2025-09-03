import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    result = []
    for traj in expert_data:
        for start_idx in range(horizon):
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                pad = np.repeat(traj[-1][None, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([traj[start_idx:], pad], axis=0)
            result.append(sub_traj)
    return np.stack(result, axis=0)


class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h

class BigImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=7, horizon=10,
                 num_layers=8, dropout=0.1, expansion=4):
        super().__init__()
        self.horizon = horizon

        self.input = nn.Linear(input_size, hidden_size)

        # Learnable horizon/position embeddings (0..horizon-1)
        self.step_embed = nn.Embedding(horizon, hidden_size)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size, dropout=dropout, expansion=expansion)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: (batch_size, input_size)
        B = x.size(0)
        device = x.device

        # Encode input -> (B, hidden)
        h = self.input(x)  # (B, hidden_size)

        # Expand across horizon and add learnable step embeddings
        h = h.unsqueeze(1).expand(B, self.horizon, -1)              # (B, H, hidden)
        steps = torch.arange(self.horizon, device=device)           # (H,)
        h = h + self.step_embed(steps).unsqueeze(0)                 # (B, H, hidden)

        # Per-step MLP blocks
        for blk in self.blocks:
            h = blk(h)                                              # (B, H, hidden)

        # Project to outputs per step
        out = self.head(h)                                          # (B, H, output_size)
        return out

# Define initial and final points, and a single central obstacle
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# === Load raw expert trajectories (N, T, 2) ===
expert_data_1 = np.load('data/expert_data1_100_traj.npy')
expert_data_2 = np.load('data/expert_data2_100_traj.npy')

# === Build sub-trajectory datasets exactly like diffusion ===
H = 25            # planning horizon
T = 100           # rollout length
expert1 = create_mpc_dataset(expert_data_1, planning_horizon=H)  # (N*, H, 2)
expert2 = create_mpc_dataset(expert_data_2, planning_horizon=H)  # (N*, H, 2)

# === Per-agent z-score over (batch,time) ===
mean1 = expert1.mean(axis=(0,1)); std1 = expert1.std(axis=(0,1)) + 1e-8
mean2 = expert2.mean(axis=(0,1)); std2 = expert2.std(axis=(0,1)) + 1e-8
expert1_n = (expert1 - mean1) / std1
expert2_n = (expert2 - mean2) / std2

# === Training conditioning (4D): [self_init, other_init] ===
obs_init1 = expert1_n[:, 0, :]                        # (N*, 2)
obs_init2 = expert2_n[:, 0, :]
S1 = np.hstack([obs_init1, obs_init2])                # (N*, 4)
S2 = np.hstack([obs_init2, obs_init1])                # (N*, 4)

# === Targets are the full 24-step segment (absolute states), t=1..H-1 ===
Y1 = expert1_n[:, 1:H, :]                             # (N*, 24, 2)
Y2 = expert2_n[:, 1:H, :]

# === Torch tensors (keep on CPU; loaders move minibatches to GPU) ===
from torch.utils.data import TensorDataset, DataLoader
batch_size = 32
tS1, tY1 = torch.tensor(S1, dtype=torch.float32), torch.tensor(Y1, dtype=torch.float32)
tS2, tY2 = torch.tensor(S2, dtype=torch.float32), torch.tensor(Y2, dtype=torch.float32)

loader1 = DataLoader(TensorDataset(tS1, tY1), batch_size=batch_size, shuffle=True,
                     pin_memory=(device.type == 'cuda'), drop_last=True)
loader2 = DataLoader(TensorDataset(tS2, tY2), batch_size=batch_size, shuffle=True,
                     pin_memory=(device.type == 'cuda'), drop_last=True)

# Two large BC models (unchanged width/depth), but input=4 and horizon=H-1=24
model1 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=H-1)
model2 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=H-1)
print(f"BC pair params: {sum(p.numel() for p in model1.parameters()) + sum(p.numel() for p in model2.parameters()):,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3, weight_decay=1e-4)
model1, model2 = model1.to(device), model2.to(device)

# Train the Model
from torch.cuda.amp import autocast, GradScaler

def joint_train(model1, model2, optimizer, criterion, loader1, loader2, num_epochs):
    model1.train(); model2.train()
    use_cuda = (next(model1.parameters()).is_cuda or next(model2.parameters()).is_cuda)
    scaler = GradScaler(enabled=use_cuda)

    for epoch in range(num_epochs):
        it1, it2 = iter(loader1), iter(loader2)
        steps = min(len(loader1), len(loader2))
        for _ in range(steps):
            xb1, yb1 = next(it1); xb2, yb2 = next(it2)
            xb1 = xb1.to(device, non_blocking=True); yb1 = yb1.to(device, non_blocking=True)  # (B,4), (B,24,2)
            xb2 = xb2.to(device, non_blocking=True); yb2 = yb2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_cuda, dtype=torch.float16):
                p1 = model1(xb1)    # (B,24,2)
                p2 = model2(xb2)    # (B,24,2)
                l1 = criterion(p1, yb1)
                l2 = criterion(p2, yb2)
                loss = l1 + l2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: L1={l1.item():.6f}  L2={l2.item():.6f}")
    return model1, model2


target_updates = 50_000   # same optimizer steps as diffusion
steps_per_epoch = min(len(loader1), len(loader2))
num_epochs_bc = (target_updates + steps_per_epoch - 1) // steps_per_epoch
print(f"BC: steps/epoch={steps_per_epoch}, epochs={num_epochs_bc}, ~updates={steps_per_epoch*num_epochs_bc}")

trained_model1, trained_model2 = joint_train(model1, model2, optimizer, criterion,
                                             loader1, loader2, num_epochs=num_epochs_bc)

save_path1 = "trained_models/bc/bc_nofinalpos_matchtrain_50k_1.pth"
save_path2 = "trained_models/bc/bc_nofinalpos_matchtrain_50k_2.pth"
torch.save(model1.state_dict(), save_path1)
torch.save(model2.state_dict(), save_path2)

model1 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=24)
model1.load_state_dict(torch.load(save_path1, map_location='cpu'))
model1.eval()

model2 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=24)
model2.load_state_dict(torch.load(save_path2, map_location='cpu'))
model2.eval()

# put models on the same device
model1 = model1.to(device).eval()
model2 = model2.to(device).eval()

@torch.no_grad()
def reactive_mpc_plan_bc(model1, model2, init1, init2, T,
                         mean1, std1, mean2, std2, device):
    """
    Replan every step with conditioning [self_norm, other_norm].
    Model outputs a 24-step segment; we implement the first step each time.
    """
    # current states in *world* coords
    cur1 = init1.copy()
    cur2 = init2.copy()

    # normalized currents
    cur1_n = (cur1 - mean1) / std1
    cur2_n = (cur2 - mean2) / std2

    traj1, traj2 = [cur1.copy()], [cur2.copy()]

    for _ in range(T - 1):
        # 4D conditioning vectors on the correct device
        c1 = torch.tensor(np.hstack([cur1_n, cur2_n]), dtype=torch.float32, device=device).unsqueeze(0)
        c2 = torch.tensor(np.hstack([cur2_n, cur1_n]), dtype=torch.float32, device=device).unsqueeze(0)

        # predict full segment (B, 24, 2); take the first step for MPC
        out1 = model1(c1)
        out2 = model2(c2)
        step1_n = out1[:, 0, :] if out1.dim() == 3 else out1          # (1,2)
        step2_n = out2[:, 0, :] if out2.dim() == 3 else out2

        # de-normalize to absolute next states in world coords
        nxt1 = (step1_n.squeeze(0).cpu().numpy() * std1) + mean1      # (2,)
        nxt2 = (step2_n.squeeze(0).cpu().numpy() * std2) + mean2

        traj1.append(nxt1.copy()); traj2.append(nxt2.copy())

        # advance currents (world + normalized)
        cur1, cur2 = nxt1, nxt2
        cur1_n = (cur1 - mean1) / std1
        cur2_n = (cur2 - mean2) / std2

    return np.array(traj1), np.array(traj2)

# ------- run rollouts & save (match diffusion’s noise/std and T) -------
noise_std = 0.4
for i in range(100):
    print("Planning Sample", i)
    init1 = initial_point1 + noise_std * np.random.randn(2)
    init2 = initial_point2 + noise_std * np.random.randn(2)
    traj1, traj2 = reactive_mpc_plan_bc(model1, model2, init1, init2, T,
                                        mean1, std1, mean2, std2, device)
    np.save(f"sampled_trajs/bc_nofinalpos_matchtrain_50k/mpc_traj1_{i}.npy", traj1)
    np.save(f"sampled_trajs/bc_nofinalpos_matchtrain_50k/mpc_traj2_{i}.npy", traj2)
