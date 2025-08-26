import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
batch_size   = 32
hidden_size  = 256
lr_G         = 1e-4
lr_D         = 1e-4
target_steps = 50_000   # ~ same order of optimizer steps as your BC/diffusion runs
eps          = 1e-8

# Load data
expert = np.load("data/expert_actions_newslower_20.npy")  # (N, T, 14)
arm1   = expert[:, :, :7].astype(np.float32)
arm2   = expert[:, :, 7:14].astype(np.float32)

pot_init = np.load("data/pot_start_newslower_20.npy").astype(np.float32)  # (N, pot_dim)
pot_mean = np.mean(pot_init, axis=0)
pot_std = np.std(pot_init, axis=0)
pot_norm = (pot_init - pot_mean) / pot_std
pot_dim  = pot_init.shape[1]

# Normalization
mean1 = np.mean(arm1, axis=(0,1)); std1 = np.std(arm1, axis=(0,1)) + eps
mean2 = np.mean(arm2, axis=(0,1)); std2 = np.std(arm2, axis=(0,1)) + eps

arm1_n = (arm1 - mean1) / std1
arm2_n = (arm2 - mean2) / std2

# Build state-action pairs for MAGAIL
def build_sa_per_arm(arm_norm: np.ndarray, pot_norm: np.ndarray):
    # arm_norm: (N, T, 7), pot_norm: (N, pot_dim)
    S, A = [], []
    N, T, D = arm_norm.shape
    for i in range(N):
        p0 = pot_norm[i]  # static per trajectory
        for t in range(T - 1):
            s = np.hstack([arm_norm[i, t], p0])  # (7 + pot_dim,)
            a = arm_norm[i, t + 1]               # predict next (7,)
            S.append(s); A.append(a)
    return np.array(S, np.float32), np.array(A, np.float32)

S1, A1 = build_sa_per_arm(arm1_n, pot_norm)
S2, A2 = build_sa_per_arm(arm2_n, pot_norm)

state_dim  = 7 + pot_dim
action_dim = 7

# Torch datasets / loaders
tS1, tA1 = torch.from_numpy(S1), torch.from_numpy(A1)
tS2, tA2 = torch.from_numpy(S2), torch.from_numpy(A2)
loader1  = DataLoader(TensorDataset(tS1, tA1), batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=max(1, os.cpu_count()//2), pin_memory=(device.type=="cuda"),
                      persistent_workers=True)
loader2  = DataLoader(TensorDataset(tS2, tA2), batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=max(1, os.cpu_count()//2), pin_memory=(device.type=="cuda"),
                      persistent_workers=True)

# Models
class GenNet(nn.Module):
    def __init__(self, s_dim, a_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, h), nn.ReLU(),
            nn.Linear(h, h),     nn.ReLU(),
            nn.Linear(h, a_dim)
        )
    def forward(self, x):
        return self.net(x)

class DiscNet(nn.Module):
    def __init__(self, s_dim, a_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, h), nn.LeakyReLU(0.2),
            nn.Linear(h, h),             nn.LeakyReLU(0.2),
            nn.Linear(h, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1)).squeeze(1)

G1 = GenNet(state_dim, action_dim, hidden_size).to(device)
G2 = GenNet(state_dim, action_dim, hidden_size).to(device)
D  = DiscNet(state_dim, action_dim, hidden_size).to(device)

optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=lr_G, betas=(0.5, 0.999))
optD = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
bce_logits = nn.BCEWithLogitsLoss()

# Training
steps_per_epoch = max(len(loader1), len(loader2))
n_epochs = math.ceil(target_steps / steps_per_epoch)

it1 = iter(loader1)
it2 = iter(loader2)

for epoch in range(1, n_epochs + 1):
    G1.train(); G2.train(); D.train()
    lossD_sum = 0.0; lossG_sum = 0.0

    it1 = iter(loader1); it2 = iter(loader2)
    for _ in range(steps_per_epoch):
        try: s1, a1 = next(it1)
        except StopIteration: it1 = iter(loader1); s1, a1 = next(it1)
        try: s2, a2 = next(it2)
        except StopIteration: it2 = iter(loader2); s2, a2 = next(it2)

        s1 = s1.to(device, non_blocking=True); a1 = a1.to(device, non_blocking=True)
        s2 = s2.to(device, non_blocking=True); a2 = a2.to(device, non_blocking=True)

        with torch.no_grad():
            f1 = G1(s1)
            f2 = G2(s2)

        r1 = D(s1, a1); f1l = D(s1, f1)
        r2 = D(s2, a2); f2l = D(s2, f2)

        lossD1 = 0.5*(bce_logits(r1, torch.ones_like(r1)) + bce_logits(f1l, torch.zeros_like(f1l)))
        lossD2 = 0.5*(bce_logits(r2, torch.ones_like(r2)) + bce_logits(f2l, torch.zeros_like(f2l)))
        lossD  = lossD1 + lossD2
        optD.zero_grad(set_to_none=True)
        lossD.backward()
        optD.step()

        f1 = G1(s1); f2 = G2(s2)
        g1 = bce_logits(D(s1, f1), torch.ones_like(r1))
        g2 = bce_logits(D(s2, f2), torch.ones_like(r2))
        lossG = g1 + g2
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()

        lossD_sum += lossD.item(); lossG_sum += lossG.item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"[{epoch:04d}/{n_epochs}] lossD={lossD_sum/steps_per_epoch:.4f}  lossG={lossG_sum/steps_per_epoch:.4f}")

# Save models
out_dir = "trained_models/magail"
os.makedirs(out_dir, exist_ok=True)
torch.save(G1.state_dict(), os.path.join(out_dir, "G1.pth"))
torch.save(G2.state_dict(), os.path.join(out_dir, "G2.pth"))
torch.save(D.state_dict(),  os.path.join(out_dir, "D.pth"))
print(f"Saved to {out_dir}")