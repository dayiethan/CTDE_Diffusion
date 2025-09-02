import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

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


# Define initial and final points, and a single central obstacle
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)
batch_size = 32
H = 25 # horizon, length of each trajectory
T = 100 # total time steps

# Expert demonstration loading
expert_data1 = np.load('data/expert_data1_100_traj.npy')
expert_data2 = np.load('data/expert_data2_100_traj.npy')
expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

mean1 = np.mean(expert_data1, axis=(0,1))
std1 = np.std(expert_data1, axis=(0,1))
mean2 = np.mean(expert_data2, axis=(0,1))
std2 = np.std(expert_data2, axis=(0,1))

expert_data1 = (expert_data1 - mean1) / std1
expert_data2 = (expert_data2 - mean2) / std2

actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
actions1 = torch.FloatTensor(actions1).float()
actions2 = torch.FloatTensor(actions2).float()
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()

obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs1 = np.hstack([obs_init1, obs_init2])
obs2 = np.hstack([obs_init2, obs_init1])
obs1 = torch.FloatTensor(obs1).float()
obs2 = torch.FloatTensor(obs2).float()
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

input_size = attr1.shape[1]

model1 = BigImitationNet(input_size, hidden_size=256, output_size=2, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)
model2 = BigImitationNet(input_size, hidden_size=256, output_size=2, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)

print(f"Total parameters: {sum(p.numel() for p in model1.parameters()) + sum(p.numel() for p in model2.parameters())}")

params = list(model1.parameters()) + list(model2.parameters())
optimizer = optim.Adam(params, lr=5e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
criterion = nn.MSELoss()

# Joint training for the agents
def joint_train(models, optimizer, criterion, datasets, num_epochs=20000):
    actions, attrs = zip(*datasets)
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for model, action, attr in zip(models, actions, attrs):
            idx = np.random.randint(0, action.shape[0], batch_size)
            true_actions = action[idx].to(device, non_blocking=True)
            input_attrs  = attr[idx].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_action = model(input_attrs)
                loss += criterion(pred_action, true_actions)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (epoch + 1) % 5000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return models


# model1, model2 = joint_train(
#     [model1, model2], optimizer, criterion,
#     [(actions1, attr1), (actions2, attr2)], num_epochs=20000
# )
         
save_path1 = "trained_models/bc/bc_nofinalpos1.pth"
save_path2 = "trained_models/bc/bc_nofinalpos2.pth"
# torch.save(model1.state_dict(), save_path1)
# torch.save(model2.state_dict(), save_path2)

# ---------------- FAST BATCHED SAMPLING (GPU-aware) ----------------
# Uses the trained bc_nofinalpos models; generates 100 rollouts in parallel.

num_trajs = 100
noise_std = 0.1
step_idx = 0  # take the first predicted step each time
model1.to(device).eval()
model2.to(device).eval()

# right under: num_trajs = 100, noise_std = 0.1, step_idx = 0 ...
dtype = torch.float32

model1.to(device).eval()
model2.to(device).eval()

# Torch versions of normalization stats on the right device + dtype
mean1_t = torch.tensor(mean1, dtype=dtype, device=device)
std1_t  = torch.tensor(std1,  dtype=dtype, device=device)
mean2_t = torch.tensor(mean2, dtype=dtype, device=device)
std2_t  = torch.tensor(std2,  dtype=dtype, device=device)

def norm1_t(x):   return (x - mean1_t) / (std1_t + 1e-8)
def norm2_t(x):   return (x - mean2_t) / (std2_t + 1e-8)
def denorm1_t(x): return x * (std1_t + 1e-8) + mean1_t
def denorm2_t(x): return x * (std2_t + 1e-8) + mean2_t

# Batch of noisy initials — force float32 here
curr1 = torch.tensor(initial_point1, dtype=dtype, device=device).unsqueeze(0).repeat(num_trajs, 1)
curr2 = torch.tensor(initial_point2, dtype=dtype, device=device).unsqueeze(0).repeat(num_trajs, 1)
curr1 = curr1 + noise_std * torch.randn_like(curr1)
curr2 = curr2 + noise_std * torch.randn_like(curr2)

# Preallocate as float32
out1 = torch.empty(num_trajs, T, 2, device=device, dtype=dtype)
out2 = torch.empty(num_trajs, T, 2, device=device, dtype=dtype)
out1[:, 0, :] = curr1
out2[:, 0, :] = curr2

with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
    for t in range(1, T):
        # Build attributes in normalized space (still float32)
        a1 = torch.cat([norm1_t(curr1), norm2_t(curr2)], dim=1)
        a2 = torch.cat([norm2_t(curr2), norm1_t(curr1)], dim=1)

        # (Optional extra safety) match the current param dtype anyway:
        p_dtype = next(model1.parameters()).dtype
        a1 = a1.to(p_dtype)
        a2 = a2.to(p_dtype)

        pred_seq1 = model1(a1)   # OK now (no Double vs Half)
        pred_seq2 = model2(a2)

        curr1 = denorm1_t(pred_seq1[:, step_idx, :].to(dtype))
        curr2 = denorm2_t(pred_seq2[:, step_idx, :].to(dtype))

        out1[:, t, :] = curr1
        out2[:, t, :] = curr2


# Move once to CPU for plotting
out1_np = out1.detach().cpu().numpy()
out2_np = out2.detach().cpu().numpy()

# Plot same style as training_bc.py
plt.figure(figsize=(20, 8))
for i in range(num_trajs):
    traj1 = out1_np[i]
    traj2 = out2_np[i]
    plt.plot(traj1[:, 0], traj1[:, 1], 'b-', alpha=0.5)
    plt.plot(traj2[:, 0], traj2[:, 1], 'C1-', alpha=0.5)
    plt.scatter(traj1[0, 0], traj1[0, 1], c='green', s=10)
    plt.scatter(traj1[-1, 0], traj1[-1, 1], c='red', s=10)
    plt.scatter(traj2[0, 0], traj2[0, 1], c='green', s=10)
    plt.scatter(traj2[-1, 0], traj2[-1, 1], c='red', s=10)

ox, oy, r = obstacle
plt.gca().add_patch(plt.Circle((ox, oy), r, color='gray', alpha=0.3))

plt.xlabel('X'); plt.ylabel('Y'); plt.grid(True)
plt.show()
# -------------------------------------------------------------------
