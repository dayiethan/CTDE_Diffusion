import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import einops
from typing import Optional
import os
from matplotlib import pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# hidden_size = 384 | 768 | 1024 | 1152
# depth =       12  | 24  | 28
# patch_size =  2   | 4   | 8
# n_heads =     6   | 12  | 16  (hidden_size can be divided by n_heads)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiscreteCondEmbedder(nn.Module):
    def __init__(self,
        attr_dim: int, hidden_size: int, num_bins: int = 100):
        super().__init__()
        self.num_bins, self.attr_dim = num_bins, attr_dim
        self.embedding = nn.Embedding(attr_dim * num_bins, 128)
        self.attn = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.linear = nn.Linear(128 * attr_dim, hidden_size)
    
    def forward(self, attr: torch.Tensor, mask: torch.Tensor = None):
        '''
        attr: (batch_size, attr_dim)
        mask: (batch_size, attr_dim) 0 or 1, 0 means ignoring
        '''
        offset = torch.arange(self.attr_dim, device=attr.device, dtype=torch.long)[None,] * self.num_bins
        # e.g. attr=[12, 42, 7] -> [12, 142, 207]
        emb = self.embedding(attr + offset) # (b, attr_dim, 128)
        if mask is not None: emb *= mask.unsqueeze(-1) # (b, attr_dim, 128)
        emb, _ = self.attn(emb, emb, emb) # (b, attr_dim, 128)
        return self.linear(einops.rearrange(emb, 'b c d -> b (c d)')) # (b, hidden_size)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))
    def forward(self, x: torch.Tensor):
        return self.mlp(x)

class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x,x,x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class Finallayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)
    
class DiT1d(nn.Module):
    def __init__(self,
        x_dim: int, attr_dim: int,
        d_model: int = 384, n_heads: int = 6, depth: int = 12, dropout: float = 0.1):
        super().__init__()
        self.x_dim, self.attr_dim, self.d_model, self.n_heads, self.depth = x_dim, attr_dim, d_model, n_heads, depth
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_emb = TimeEmbedding(d_model)
        self.attr_proj = DiscreteCondEmbedder(attr_dim, d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.pos_emb_cache = None
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = Finallayer1d(d_model, x_dim)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, 
        x: torch.Tensor, t: torch.Tensor, 
        attr: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        '''
        Input:
            - x:    (batch, horizon, x_dim)
            - t:    (batch, 1)
            - attr: (batch, attr_dim)
            - mask: (batch, attr_dim)
        
        Output:
            - y:    (batch, horizon, x_dim)
        '''
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.x_proj(x) + self.pos_emb_cache[None,]
        t = self.t_emb(t)
        if attr is not None:
            t += self.attr_proj(attr, mask)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return x

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)


# Parse expert data from single_uni_full_traj.csv
import csv
all_points = []
with open('data/single_uni_full_traj_up.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[2]), float(row[3])
        all_points.append([x, y])

with open('data/single_uni_full_traj_down.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[2]), float(row[3])
        all_points.append([x, y])


all_points_rev = list(reversed(all_points))

num_trajectories = 2000
points_per_trajectory = 100

expert_data = [
    all_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory = expert_data[0]
x = [point[0] for point in first_trajectory]
y = [point[1] for point in first_trajectory]

expert_data_rev = [
    all_points_rev[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory_rev = expert_data_rev[0]
x_rev = [point[0] for point in first_trajectory_rev]
y_rev = [point[1] for point in first_trajectory_rev]

# expert_data = expert_data + list(reversed(expert_data_rev))

expert_data = np.array(expert_data)
expert_data_rev = np.array(expert_data_rev)

# Compute mean and standard deviation
combined_data = np.concatenate((expert_data, expert_data_rev), axis=0)
mean = np.mean(combined_data, axis=(0,1))
std = np.std(combined_data, axis=(0,1))

# Normalize data
expert_data = (expert_data - mean) / std
expert_data_rev = (expert_data_rev - mean) / std

# Prepare Data for Training
# Create input-output pairs (state + goal -> next state)
X_train = []
Y_train = []

for traj in expert_data:
    for i in range(len(traj) - 1):
        X_train.append(np.hstack([traj[i], final_point_up]))  # Current state + goal
        Y_train.append(traj[i + 1])  # Next state

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)  # Shape: (N, 4)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)  # Shape: (N, 2)

X_train_rev = []
Y_train_rev = []

for traj in expert_data_rev:
    for i in range(len(traj) - 1):
        X_train_rev.append(np.hstack([traj[i], final_point_down]))  # Current state + goal
        Y_train_rev.append(traj[i + 1])  # Next state

X_train_rev = torch.tensor(np.array(X_train_rev), dtype=torch.float32)  # Shape: (N, 4)
Y_train_rev = torch.tensor(np.array(Y_train_rev), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizers
betas = torch.tensor([0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999])
denoiser1 = DiT1d(x_dim=2, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)
denoiser2 = DiT1d(x_dim=2, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)
state_dim = 4   # e.g., state vector of size 10
# action_dim = 4   # e.g., action vector of size 5
max_steps = len(betas) # Maximum diffusion steps
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, 0)
num_epochs = 3000
batch_size = 64
lr = 1e-3

losses = np.zeros(num_epochs)
optimizer1 = torch.optim.Adam(denoiser1.parameters(), lr)
optimizer2 = torch.optim.Adam(denoiser2.parameters(), lr)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs)


# for epoch in range(num_epochs):
#     t = torch.randint(0, max_steps, (batch_size, 1))
#     integers = torch.randint(0, num_trajectories, (batch_size,))
#     x01 = torch.tensor(expert_data[integers]).float()
#     x02 = torch.tensor(expert_data_rev[integers]).float()
#     eps1 = torch.randn_like(x01)
#     eps2 = torch.randn_like(x02)
#     x_noised1  = x01
#     x_noised2  = x02
#     x_noised1 = x_noised1*torch.sqrt(alphas_bar[t]).unsqueeze(-1) + eps1*torch.sqrt(1-alphas_bar[t]).unsqueeze(-1)
#     x_noised2 = x_noised2*torch.sqrt(alphas_bar[t]).unsqueeze(-1) + eps2*torch.sqrt(1-alphas_bar[t]).unsqueeze(-1)

#     pred1 = denoiser1(x_noised1, t.float())
#     pred2 = denoiser2(x_noised2, t.float())

#     loss = F.mse_loss(pred1, eps1) + F.mse_loss(pred2, eps2)

#     # if loss.detach().item() < 3:
#     #     print("Loss:",loss.detach().item())
#     #     print("Epoch:",epoch)
#     #     torch.save(denoiser1.state_dict(), 'checkpoints_new/unet1_diff_tran_epoch'+str(epoch)+'.pth')
#     #     torch.save(denoiser2.state_dict(), 'checkpoints_new/unet2_diff_tran_epoch'+str(epoch)+'.pth')

#     optimizer1.zero_grad()
#     optimizer2.zero_grad()
#     loss.backward()
#     optimizer1.step()
#     optimizer2.step()
#     scheduler1.step()
#     scheduler2.step()
#     losses[epoch] = loss.detach().item()

#     if epoch%100 == 0:
#         print("Epoch:",epoch)
#         print("Loss:",losses[epoch])

#     if (epoch+1)%1000 == 0:
#         torch.save(denoiser1.state_dict(), 'checkpoints_new/unet1_diff_tran_epoch'+str(epoch)+'_multimode_temp.pth')
#         torch.save(denoiser2.state_dict(), 'checkpoints_new/unet2_diff_tran_epoch'+str(epoch)+'_multimode_temp.pth')

#     if losses[epoch] < 3:
#         lr = 1e-4


denoiser1 = DiT1d(x_dim=2, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)
denoiser2 = DiT1d(x_dim=2, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)
denoiser1.load_state_dict(torch.load("checkpoints_new/unet1_diff_tran_epoch2999_multimode_temp.pth"))
denoiser2.load_state_dict(torch.load("checkpoints_new/unet2_diff_tran_epoch2999_multimode_temp.pth"))

def compute_action_diff(alphas_bar, alphas, betas, denoiser):
    u_out = torch.randn((1, 100, 2))  # Initialize with standard normal noise
    for t in range(len(alphas_bar)-1, -1, -1):  # Loop from T-1 to 0
        if t > 0:
            z = torch.randn_like(u_out)
        else:
            z = 0
        alpha_t = alphas[t]
        alpha_bar_t = alphas_bar[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        beta_t = betas[t]
        sigma_t = 0.5*torch.sqrt(beta_t)
        with torch.no_grad():
            eps_theta = denoiser(u_out, torch.tensor([[t]], dtype=torch.float32))
            u_out = (1 / sqrt_alpha_t) * (u_out - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) + sigma_t * z
    return u_out

print(alphas_bar.shape)
print(alphas.shape)
print(betas.shape)

u_out1 = compute_action_diff(alphas_bar, alphas, betas, denoiser1)
u_out2 = compute_action_diff(alphas_bar, alphas, betas, denoiser2)

traj1 = u_out1.squeeze().detach().numpy()
traj2 = u_out2.squeeze().detach().numpy()

traj1 = traj1 * std + mean
traj2 = traj2 * std + mean

expert_data = expert_data * std + mean
expert_data_rev = expert_data_rev * std + mean


# Plot the Expert and Generated Trajectories with a Single Central Obstacle
plt.figure(figsize=(20, 8))
for traj in expert_data[1::100]:  # Plot a few expert trajectories
    first_trajectory = traj
    x = [point[0] for point in first_trajectory]
    y = [point[1] for point in first_trajectory]
    plt.plot(x, y, 'b--')
for traj in expert_data_rev[1::100]:  # Plot a few expert trajectories
    first_trajectory = traj
    x = [point[0] for point in first_trajectory]
    y = [point[1] for point in first_trajectory]
    plt.plot(x, y, 'g--')

# Plot the generated trajectory
plt.plot(traj1[:, 0], traj1[:, 1], 'r-', label='Generated')
plt.plot(traj2[:, 0], traj2[:, 1], 'y-', label='Generated')

# Plot the single central obstacle as a circle
ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point_up[0], initial_point_up[1], c='red', s=100, label='Start/End')
plt.scatter(final_point_up[0], final_point_up[1], c='red', s=100, label='Start/End')

plt.legend()
plt.title('Smooth Imitation Learning: Expert vs Generated Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('figs/two_agents_shared/expert_vs_generated_trajectories_multimode4.png')
plt.show()

# # Plot the Training Loss
# plt.figure()
# plt.plot(losses, label='Up')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig('figs/two_agents_shared/loss_graph.png')
# plt.show()