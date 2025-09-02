# training_magail_ctde.py

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1) Device & performance tweak
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 2) Hyperparameters
state_dim   = 4       # e.g. [x, y, other_agent_x, other_agent_y]
action_dim  = 2       # e.g. next [x, y]
hidden_size = 64
batch_size = 32
target_updates = 50_000
H = 25
T = 100
noise_std = 0.4
lr_G        = 1e-4
lr_D        = 1e-4

# 3) Models
class GenNet(nn.Module):
    def __init__(self, width=688, depth=10):
        super().__init__()
        layers = [nn.Linear(state_dim, width), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU(inplace=True)]
        layers += [nn.Linear(width, action_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DiscNet(nn.Module):
    def __init__(self, width=960, depth=10, spectral=True):
        super().__init__()
        def lin(i,o):
            m = nn.Linear(i,o)
            return nn.utils.spectral_norm(m) if spectral else m
        L = [lin(state_dim+action_dim, width), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(depth-1):
            L += [lin(width,width), nn.LeakyReLU(0.2, inplace=True)]
        L += [lin(width,1)]
        self.net = nn.Sequential(*L)
    def forward(self, x, a):  # x: (B,4), a: (B,2)
        return self.net(torch.cat([x,a], dim=1)).squeeze(1)

def count_params(m): return sum(p.numel() for p in m.parameters())
    
def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    result = []
    for traj in expert_data:
        for start_idx in range(horizon):
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                sub_traj = np.concatenate([traj[start_idx:], 
                                           np.repeat(traj[-1][None, :], end_idx - horizon, axis=0)], axis=0)
            result.append(sub_traj)
    return np.stack(result, axis=0)

# Load raw experts
expert1_raw = np.load('data/expert_data1_100_traj.npy')  # (N,T,2)
expert2_raw = np.load('data/expert_data2_100_traj.npy')

# Build H-step windows (exactly like diffusion)
expert1 = create_mpc_dataset(expert1_raw, planning_horizon=H)   # (N*,H,2)
expert2 = create_mpc_dataset(expert2_raw, planning_horizon=H)

# Per-agent z-score (over batch & time)
mean1, std1 = np.mean(expert1, axis=(0,1)), np.std(expert1, axis=(0,1))
mean2, std2 = np.mean(expert2, axis=(0,1)), np.std(expert2, axis=(0,1))
expert1_n = (expert1 - mean1) / std1
expert2_n = (expert2 - mean2) / std2

# Conditioning & targets:
# inputs S* = [self_init, other_init] at t=0 (normalized), shape (N*,4)
# targets A* = absolute next state at t=1 (normalized), shape (N*,2)
obs_init1 = expert1_n[:, 0, :]                # (N*,2)
obs_init2 = expert2_n[:, 0, :]
S1 = np.hstack([obs_init1, obs_init2])        # agent1 condition
S2 = np.hstack([obs_init2, obs_init1])        # agent2 condition
A1 = expert1_n[:, 1, :]                       # absolute next state
A2 = expert2_n[:, 1, :]

# Torch datasets/loaders
tS1, tA1 = torch.tensor(S1, dtype=torch.float32), torch.tensor(A1, dtype=torch.float32)
tS2, tA2 = torch.tensor(S2, dtype=torch.float32), torch.tensor(A2, dtype=torch.float32)
loader1 = DataLoader(TensorDataset(tS1, tA1), batch_size=batch_size, shuffle=True, drop_last=True,
                     pin_memory=(device.type=="cuda"))
loader2 = DataLoader(TensorDataset(tS2, tA2), batch_size=batch_size, shuffle=True, drop_last=True,
                     pin_memory=(device.type=="cuda"))

steps_per_epoch = min(len(loader1), len(loader2))
n_epochs = (target_updates + steps_per_epoch - 1) // steps_per_epoch
print(f"MAGAIL: steps/epoch={steps_per_epoch}, epochs={n_epochs}, ~updates={steps_per_epoch*n_epochs}")

# ------------------------- models ---------------------------
G1 = GenNet(width=688, depth=10).to(device)
G2 = GenNet(width=688, depth=10).to(device)
D  = DiscNet(width=960, depth=10, spectral=True).to(device)

print(f"G total params: {count_params(G1)+count_params(G2):,}")
print(f"D params:       {count_params(D):,}")

optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=1e-4)
optD = optim.Adam(D.parameters(), lr=2e-4)  # TTUR helps stability at this scale
bce  = nn.BCEWithLogitsLoss()

# ----------------------- training loop ----------------------
# G1.train(); G2.train(); D.train()
# for epoch in range(1, n_epochs+1):
#     lossD_acc = lossG_acc = 0.0
#     for (x1,y1),(x2,y2) in zip(loader1, loader2):
#         x1,y1 = x1.to(device), y1.to(device)
#         x2,y2 = x2.to(device), y2.to(device)

#         # ---- D step
#         with torch.no_grad():
#             f1 = G1(x1); f2 = G2(x2)
#         r1 = D(x1,y1);  rf1 = D(x1,f1)
#         r2 = D(x2,y2);  rf2 = D(x2,f2)
#         lossD = 0.5*(bce(r1, torch.ones_like(r1)) + bce(rf1, torch.zeros_like(rf1))) \
#               + 0.5*(bce(r2, torch.ones_like(r2)) + bce(rf2, torch.zeros_like(rf2)))
#         optD.zero_grad(set_to_none=True); lossD.backward(); optD.step()

#         # ---- G step
#         f1 = G1(x1); f2 = G2(x2)
#         lossG = bce(D(x1,f1), torch.ones_like(f1[:,0])) + bce(D(x2,f2), torch.ones_like(f2[:,0]))
#         optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()

#         lossD_acc += lossD.item(); lossG_acc += lossG.item()

#     if epoch % 5 == 0:
#         print(f"Epoch {epoch:4d} | lossD={lossD_acc/len(loader1):.4f} | lossG={lossG_acc/len(loader1):.4f}")

# ----------------------- save artifacts ---------------------
save_path_G1 = "trained_models/magail/G_nofinalpos_matchtrain_50k_1.pth"
save_path_G2 = "trained_models/magail/G_nofinalpos_matchtrain_50k_2.pth"
save_path_D  = "trained_models/magail/D_nofinalpos_matchtrain_50k.pth"

# torch.save(G1.state_dict(), save_path_G1)
# torch.save(G2.state_dict(), save_path_G2)
# torch.save(D.state_dict(),  save_path_D)
# np.savez("trained_models/magail/norm_stats.npz",
#          mean1=mean1, std1=std1, mean2=mean2, std2=std2)
print("Training complete and models saved under trained_models/magail/")

# --------------------- reactive MPC sampling ----------------
G1 = GenNet(width=688, depth=10).to(device)
G2 = GenNet(width=688, depth=10).to(device)
G1.load_state_dict(torch.load(save_path_G1, map_location=device))
G2.load_state_dict(torch.load(save_path_G2, map_location=device))
G1.eval(); G2.eval()

@torch.no_grad()
def reactive_mpc_plan_magail(G1, G2, init1, init2, T, mean1, std1, mean2, std2):
    """
    Replan every step:
      cond1 = [self_current1_norm, other_current2_norm], pred next (normalized) absolute state
    """
    cur1_n = (init1 - mean1) / std1
    cur2_n = (init2 - mean2) / std2
    traj1, traj2 = [init1.copy()], [init2.copy()]
    for t in range(T-1):
        c1 = torch.tensor(np.hstack([cur1_n, cur2_n]), dtype=torch.float32, device=device).unsqueeze(0)
        c2 = torch.tensor(np.hstack([cur2_n, cur1_n]), dtype=torch.float32, device=device).unsqueeze(0)
        nxt1 = G1(c1).cpu().numpy()[0] * std1 + mean1
        nxt2 = G2(c2).cpu().numpy()[0] * std2 + mean2
        traj1.append(nxt1.copy()); traj2.append(nxt2.copy())
        cur1_n = (nxt1 - mean1) / std1
        cur2_n = (nxt2 - mean2) / std2
    return np.vstack(traj1), np.vstack(traj2)

# rollouts (same initial centers and noise as diffusion)
# for i in range(100):
#     print("Planning Sample", i)
#     init1 = np.array([0.0, 0.0])  + noise_std * np.random.randn(2)
#     init2 = np.array([20.0, 0.0]) + noise_std * np.random.randn(2)
#     traj1, traj2 = reactive_mpc_plan_magail(G1, G2, init1, init2, T, mean1, std1, mean2, std2)
#     np.save(f"sampled_trajs/magail_nofinalpos_matchtrain_50k/mpc_traj1_{i}.npy", traj1)
#     np.save(f"sampled_trajs/magail_nofinalpos_matchtrain_50k/mpc_traj2_{i}.npy", traj2)

fixed_init1 = np.array([0.0, 0.0], dtype=np.float32)
fixed_init2 = np.array([20.0, 0.0], dtype=np.float32)


for i in range(100):
    print("Planning Sample (fixed init)", i)
    traj1, traj2 = reactive_mpc_plan_magail(G1, G2, fixed_init1, fixed_init2, T, mean1, std1, mean2, std2)
    np.save(f"sampled_trajs/magail_nofinalpos_matchtrain_50k_fixed/mpc_traj1_{i}.npy", traj1)
    np.save(f"sampled_trajs/magail_nofinalpos_matchtrain_50k_fixed/mpc_traj2_{i}.npy", traj2)

print("Sampling complete for both random and fixed-initial sets.")