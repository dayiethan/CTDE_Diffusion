# training_magail_ctde_v2.py

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
state_dim   = 4       # e.g. [x, y, goal_x, goal_y]
action_dim  = 2       # e.g. next [x, y]
hidden_size = 64
batch_size  = 64
n_epochs    = 2000
lr_G        = 1e-4
lr_D        = 1e-4

# 3) Models
class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,   hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DiscNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size,             hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1)).squeeze(1)

# Individual generators
G1 = GenNet().to(device)
G2 = GenNet().to(device)
G3 = GenNet().to(device)
# Shared discriminator
D  = DiscNet().to(device)

optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()) + list(G3.parameters()), lr=lr_G)
optD = optim.Adam(D.parameters(), lr=lr_D)
bce_logits = nn.BCEWithLogitsLoss()

# 4) Load expert data & build (s, a) pairs
datas = []
for idx in range(1, 4):
    datas.append(np.load(f'data/expert_data{idx}_400_traj_06_noise.npy'))

def build_sa(expert):
    S, A = [], []
    for traj in expert:
        goal = traj[-1]
        for t in range(len(traj) - 1):
            s = np.hstack([traj[t], goal])  # (state_dim,)
            a = traj[t+1]                   # (action_dim,)
            S.append(s); A.append(a)
    return np.stack(S), np.stack(A)

S1, A1 = build_sa(datas[0])
S2, A2 = build_sa(datas[1])
S3, A3 = build_sa(datas[2])

# 5) Joint normalization for states & actions
state_data  = np.concatenate([S1, S2, S3], axis=0)
action_data = np.concatenate([A1, A2, A3], axis=0)
state_mean, state_std   = state_data.mean(0),  state_data.std(0)  + 1e-6
action_mean, action_std = action_data.mean(0), action_data.std(0) + 1e-6
S1 = (S1 - state_mean) / state_std
S2 = (S2 - state_mean) / state_std
S3 = (S3 - state_mean) / state_std
A1 = (A1 - action_mean) / action_std
A2 = (A2 - action_mean) / action_std
A3 = (A3 - action_mean) / action_std

# 6) Torch datasets & loaders
tS1, tA1 = torch.tensor(S1, dtype=torch.float32), torch.tensor(A1, dtype=torch.float32)
tS2, tA2 = torch.tensor(S2, dtype=torch.float32), torch.tensor(A2, dtype=torch.float32)
tS3, tA3 = torch.tensor(S3, dtype=torch.float32), torch.tensor(A3, dtype=torch.float32)
loader1 = DataLoader(TensorDataset(tS1, tA1), batch_size=batch_size, shuffle=True, drop_last=True)
loader2 = DataLoader(TensorDataset(tS2, tA2), batch_size=batch_size, shuffle=True, drop_last=True)
loader3 = DataLoader(TensorDataset(tS3, tA3), batch_size=batch_size, shuffle=True, drop_last=True)

# 7) Training loop (centralized training structure)
for epoch in range(1, n_epochs + 1):
    G1.train(); G2.train(); G3.train(); D.train()
    lossD_sum = 0.0
    lossG_sum = 0.0

    for (x1, a1), (x2, a2), (x3, a3) in zip(loader1, loader2, loader3):
        x1, a1 = x1.to(device), a1.to(device)
        x2, a2 = x2.to(device), a2.to(device)
        x3, a3 = x3.to(device), a3.to(device)

        # —— Discriminator update across all agents ——
        with torch.no_grad():
            fake1 = G1(x1)
            fake2 = G2(x2)
            fake3 = G3(x3)
        real_logit1 = D(x1, a1)
        fake_logit1 = D(x1, fake1)
        real_logit2 = D(x2, a2)
        fake_logit2 = D(x2, fake2)
        real_logit3 = D(x3, a3)
        fake_logit3 = D(x3, fake3)
        lossD1 = 0.5*(bce_logits(real_logit1, torch.ones_like(real_logit1)) + bce_logits(fake_logit1, torch.zeros_like(fake_logit1)))
        lossD2 = 0.5*(bce_logits(real_logit2, torch.ones_like(real_logit2)) + bce_logits(fake_logit2, torch.zeros_like(fake_logit2)))
        lossD3 = 0.5*(bce_logits(real_logit3, torch.ones_like(real_logit3)) + bce_logits(fake_logit3, torch.zeros_like(fake_logit3)))
        lossD = lossD1 + lossD2 + lossD3
        optD.zero_grad()
        lossD.backward()
        optD.step()

        # —— Generator update for each agent ——
        fake1 = G1(x1)
        fake2 = G2(x2)
        fake3 = G3(x3)
        lossG1 = bce_logits(D(x1, fake1), torch.ones_like(fake1[:, 0]))
        lossG2 = bce_logits(D(x2, fake2), torch.ones_like(fake2[:, 0]))
        lossG3 = bce_logits(D(x3, fake3), torch.ones_like(fake3[:, 0]))
        lossG = lossG1 + lossG2 + lossG3
        optG.zero_grad()
        lossG.backward()
        optG.step()

        lossD_sum += lossD.item()
        lossG_sum += lossG.item()

    if epoch % 100 == 0:
        avgD = lossD_sum / len(loader1)
        avgG = lossG_sum / len(loader1)
        print(f"Epoch {epoch:4d} | lossD: {avgD:.4f} | lossG: {avgG:.4f}")

# 8) Save models
torch.save(G1.state_dict(), "trained_models/magail_ctde/G1.pth")
torch.save(G2.state_dict(), "trained_models/magail_ctde/G2.pth")
torch.save(G3.state_dict(), "trained_models/magail_ctde/G3.pth")
torch.save(D.state_dict(),  "trained_models/magail_ctde/D.pth")
print("CTDE MAGAIL training complete; models saved under trained_models/magail_ctde/")
