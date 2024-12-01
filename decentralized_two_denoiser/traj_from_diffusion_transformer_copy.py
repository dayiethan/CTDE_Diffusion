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

trajectory1 = np.loadtxt("data/single_uni_full_traj_up.csv",delimiter=",", dtype=float)
trajectory2 = np.loadtxt("data/single_uni_full_traj_down.csv",delimiter=",", dtype=float)

max_traj_array1 = np.max(trajectory1, axis=0)
max_traj_array2 = np.max(trajectory2, axis=0)

np.savetxt("data/max_traj_array1.csv", max_traj_array1, delimiter=",")
np.savetxt("data/max_traj_array2.csv", max_traj_array2, delimiter=",")

trajectory1 = trajectory1/max_traj_array1
trajectory2 = trajectory2/max_traj_array2

trajectory1 = (trajectory1).reshape(1000, 100, 5)
trajectory2 = (trajectory2).reshape(1000, 100, 5)

batch_size = 2

# betas = torch.tensor([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
# betas = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
betas = torch.tensor([0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999])

denoiser1 = DiT1d(x_dim=2, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)
denoiser2 = DiT1d(x_dim=2, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)

state_dim= 2   # e.g., state vector of size 10
action_dim = 3   # e.g., action vector of size 5
max_steps = len(betas) # Maximum diffusion steps

alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, 0)

folder_path = "checkpoints_shared"

if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# load the model

denoiser1.load_state_dict(torch.load("checkpoints_shared/unet1_diff_tran_final.pth"))
denoiser2.load_state_dict(torch.load("checkpoints_shared/unet2_diff_tran_final.pth"))

def compute_action_diff(alphas_bar, alphas, betas, denoiser):
    u_out = torch.randn((1, 100, 2))  # Initialize with standard normal noise
    for t in range(len(alphas_bar) - 1, -1, -1):
        if t > 0:
            z = torch.randn_like(u_out)
        else:
            z = 0
        alpha_t = alphas[t]
        alpha_bar_t = alphas_bar[t]
        beta_t = betas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sigma_t = torch.sqrt(beta_t)
        with torch.no_grad():
            eps_theta = denoiser(u_out, torch.tensor([[t]], dtype=torch.float32))
            u_out = (1 / sqrt_alpha_t) * (u_out - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) + sigma_t * z
    return u_out



u_out1 = compute_action_diff(alphas_bar, alphas, betas, denoiser1)
u_out2 = compute_action_diff(alphas_bar, alphas, betas, denoiser2)

def wrap_to_pi(angle):
    """
    Wrap an angle to the range (-pi, pi].
    
    Parameters:
    - angle (float): The input angle in radians.

    Returns:
    - float: The wrapped angle.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def two_unicycle_dynamics(x, u, dt):
    """
    Simulate the discrete-time dynamics of a unicycle.

    Parameters:
    - x: Current state [x, y, theta]
    - u: Control input [v, w] (linear velocity, angular velocity)
    - dt: Time step

    Returns:
    - New state after applying the dynamics.
    """
    v1, w1, v2, w2 = u
    x[0] += v1 * np.cos(x[2]) * dt
    x[1] += v1 * np.sin(x[2]) * dt
    x[2] += w1 * dt 
    x[2] = wrap_to_pi(x[2])
    
    x[3] += v2 * np.cos(x[5]) * dt
    x[4] += v2 * np.sin(x[5]) * dt
    x[5] += w2 * dt
    x[5] = wrap_to_pi(x[5])
    
    return x

def main():
    n_x = 6
    n_u = 4
    n_g = 1
    x01 = np.array([0.0,0.0])
    x02 = np.array([20.0,0.0])

    time_horizon = 10
    dt = 0.1
    timesteps = int(time_horizon/dt)

    traj1 = np.zeros((101,2))
    traj2 = np.zeros((101,2))

    traj1[0,:] = x01
    traj2[0,:] = x02

    # max_traj_array1 = np.loadtxt("data/max_traj_array1.csv", delimiter=",")
    # max_traj_array2 = np.loadtxt("data/max_traj_array2.csv", delimiter=",")

    # state_normalization_arr1 = max_traj_array1[:]
    # state_normalization_arr2 = max_traj_array2[:]

    traj1 = u_out1.squeeze().detach().numpy()
    traj2 = u_out2.squeeze().detach().numpy()

    # Load global_mean and global_std
    global_mean = np.load("data/global_mean.npy")
    global_std = np.load("data/global_std.npy")

    # Denormalize the generated trajectories
    traj1 = traj1 * global_std + global_mean
    traj2 = traj2 * global_std + global_mean

    # for i in range(2):
    #     traj1[:,i] = traj1[:,i]*state_normalization_arr1[i]
    #     traj2[:,i] = traj2[:,i]*state_normalization_arr2[i]

    plt.plot(traj1[:,0],traj1[:,1],label="Agent 1")
    plt.plot(traj2[:,0],traj2[:,1],label="Agent 2")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(xlim=(-1.5, 20.5), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2, color = 'blue',label="Agent 1")
    line2, = ax.plot([], [], lw=2, color = 'orange',label="Agent 2")
    plt.legend(frameon=False)
    #turn of top and right splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    def animate(n):
        line.set_xdata(traj1[:n, 0])
        line.set_ydata(traj1[:n, 1])
        line2.set_xdata(traj2[:n, 0])
        line2.set_ydata(traj2[:n, 1])
        return line,line2

    anim = FuncAnimation(fig, animate, frames=traj1.shape[0], interval=40)
    anim.save('figs/two_agents_shared/test_trajectory_animation.gif')
    plt.show()

    #animate trajectory


if(__name__ == '__main__'):
    main()
