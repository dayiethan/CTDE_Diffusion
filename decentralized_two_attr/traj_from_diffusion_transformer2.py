import math
import torch
import einops
import torch.nn as nn
from typing import Optional
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

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

# class TimeEmbedding(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#         self.mlp = nn.Sequential(
#             nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim))
#     def forward(self, x: torch.Tensor):
#         device = x.device
#         half_dim = self.dim // 8
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return self.mlp(emb)
    
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

#real distribution

#load data

trajectory = np.loadtxt("data/full_traj.csv",delimiter=",", dtype=float)

max_traj_array = np.max(trajectory, axis=0)

np.savetxt("data/max_traj_array.csv", max_traj_array, delimiter=",")

trajectory = trajectory/max_traj_array

trajectory = (trajectory).reshape(-1, 100, 10)

print(trajectory.shape)

batch_size = 2

# betas = torch.tensor([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
# betas = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
betas = torch.tensor([0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999])

denoiser = DiT1d(x_dim=6, attr_dim=2, d_model=384, n_heads=6, depth=12, dropout=0.1)

state_dim = 6   # e.g., state vector of size 10
action_dim = 4   # e.g., action vector of size 5
max_steps = len(betas) # Maximum diffusion steps

alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, 0)


folder_path = "checkpoints"

if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# load the model

denoiser.load_state_dict(torch.load("checkpoints/unet_diff_tran_final.pth"))

def compute_action_diff(alphas_bar, alphas, betas, denoiser, attr):
    alpha_bar = torch.prod(1 - betas)
    
    # Initialize noisy trajectory
    u0 = torch.randn((1, 100, 6)) * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar)
    u_out = u0

    for t in range(len(alphas_bar), 0, -1):
        if t > 1:
            z = torch.randn_like(u0) 
        else:
            z = 0
        sigma_sq = betas[t-1] * (1 - alphas_bar[t-1] / alphas[t-1]) / (1 - alphas_bar[t-1])

        with torch.no_grad():
            # Conditioning each step on the attribute tensor
            u_out = (1 / np.sqrt(alphas[t-1])) * (
                u_out - (1 - alphas[t-1]) * denoiser(u_out, torch.tensor([[t-1]]).float(), attr=attr) / np.sqrt(1 - alphas_bar[t-1])
            ) + torch.sqrt(sigma_sq) * z

    return u_out


# u_out = compute_action_diff(alphas_bar, alphas, betas, denoiser)

def main():
    n_x = 6
    n_u = 4
    n_g = 1
    x0 = np.array([0.0, 0.0, 0.0, 20.0, 0.0, np.pi])

    time_horizon = 10
    dt = 0.1
    timesteps = int(time_horizon / dt)

    traj = np.zeros((101, 10))
    traj[0, 4:] = x0

    max_traj_array = np.loadtxt("data/max_traj_array.csv", delimiter=",")
    print(max_traj_array)

    action_normalization_arr = max_traj_array[0:4]
    state_normalization_arr = max_traj_array[4:]

    # Define attributes for both agents (one attribute per agent)
    attr = torch.tensor([[50, 75]])  # For example, 50 for Agent 1 and 75 for Agent 2

    # Generate the trajectory using the conditioned denoiser
    u_out = compute_action_diff(alphas_bar, alphas, betas, denoiser, attr)
    traj = u_out.squeeze().detach().numpy()

    print(traj.shape)

    # Denormalize the generated trajectory
    for i in range(6):
        traj[:, i] = traj[:, i] * state_normalization_arr[i]

    # Plotting trajectories
    plt.plot(traj[:, 0], traj[:, 1], label="Agent 1")
    plt.plot(traj[:, 3], traj[:, 4], label="Agent 2")
    plt.legend()
    plt.show()

    # Animation setup
    fig = plt.figure()
    ax = plt.axes(xlim=(-1.5, 20.5), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2, color='blue', label="Agent 1")
    line2, = ax.plot([], [], lw=2, color='orange', label="Agent 2")
    plt.legend(frameon=False)

    # Turn off top and right splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Animation function
    def animate(n):
        line.set_xdata(traj[:n, 0])
        line.set_ydata(traj[:n, 1])
        line2.set_xdata(traj[:n, 3])
        line2.set_ydata(traj[:n, 4])
        return line, line2

    # Save animation
    anim = FuncAnimation(fig, animate, frames=traj.shape[0], interval=40)
    anim.save('figs/test_trajectory_animation.gif')
    plt.show()

if __name__ == '__main__':
    main()
