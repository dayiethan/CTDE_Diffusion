import math
import torch
import einops
import torch.nn as nn
from typing import Optional
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt


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

trajectory = np.loadtxt("data/full_traj.csv",delimiter=",", dtype=float)

max_traj_array = np.max(trajectory, axis=0)

np.savetxt("data/max_traj_array.csv", max_traj_array, delimiter=",")

trajectory = trajectory/max_traj_array

trajectory = (trajectory).reshape(-1, 200, 10)

print(trajectory.shape)

batch_size = 2

# betas = torch.tensor([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
# betas = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
betas = torch.tensor([0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.999])

denoiser1 = DiT1d(x_dim=6, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)
denoiser2 = DiT1d(x_dim=6, attr_dim=1, d_model=384, n_heads=6, depth=12, dropout=0.1)

# denoiser.load_state_dict(torch.load('checkpoints/unet_diff_tran_epoch499.pth'))

state_dim = 6   # e.g., state vector of size 10
action_dim = 4   # e.g., action vector of size 5
max_steps = len(betas) # Maximum diffusion steps

alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, 0)

nb_epochs = 2000
batch_size = 32
lr = 1e-3

losses = np.zeros(nb_epochs)
optimizer1 = torch.optim.Adam(denoiser1.parameters(), lr)
optimizer2 = torch.optim.Adam(denoiser2.parameters(), lr)

folder_path = "checkpoints"

if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for epoch in tqdm(range(nb_epochs), desc="Training Progress"):

    # t = random.randint(0, len(alphas_bar)-1)
    # t_arr = t*torch.ones(batch_size,).int()

    t = torch.randint(0, max_steps, (batch_size, 1))

    # choose random integers from 0 to 100 of size batch_size
    integers = torch.randint(0, trajectory.shape[0], (batch_size,))
    
    # choose the corresponding trajectory
    x0 = torch.tensor(trajectory[integers,:,4:]).float()

    # x0[:, :2] += torch.randn_like(x0[:, :2])

    eps = torch.randn_like(x0)

    x_noised  = x0

    x_noised = x_noised*torch.sqrt(alphas_bar[t]).unsqueeze(-1) + eps*torch.sqrt(1-alphas_bar[t]).unsqueeze(-1)

    pred1 = denoiser1(x_noised, t.float())
    pred2 = denoiser2(x_noised, t.float())

    loss = torch.linalg.vector_norm(eps - pred1) + torch.linalg.vector_norm(eps - pred2)

    if loss.detach().item() < 3:
        print("Loss:",loss.detach().item())
        print("Epoch:",epoch)
        torch.save(denoiser1.state_dict(), 'checkpoints/unet1_diff_tran_epoch'+str(epoch)+'.pth')
        torch.save(denoiser2.state_dict(), 'checkpoints/unet2_diff_tran_epoch'+str(epoch)+'.pth')

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    losses[epoch] = loss.detach().item()

    if epoch%100 == 0:
        print("Epoch:",epoch)
        print("Loss:",losses[epoch])

    if (epoch+1)%500 == 0:
        torch.save(denoiser1.state_dict(), 'checkpoints/unet1_diff_tran_epoch'+str(epoch)+'.pth')
        torch.save(denoiser2.state_dict(), 'checkpoints/unet2_diff_tran_epoch'+str(epoch)+'.pth')

    if losses[epoch] < 3:
        lr = 1e-4

plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
# plt.ylim(0, 100)
plt.show()

torch.save(denoiser1.state_dict(), 'checkpoints/unet1_diff_tran_final.pth')
torch.save(denoiser2.state_dict(), 'checkpoints/unet2_diff_tran_final.pth')