# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:24:01 2024

@author: Jean-Baptiste

Taken from https://github.com/ZibinDong/AlignDiff-ICLR2024/blob/main/utils/dit_utils.py

Prediction of the states only

conditional DiT based on the style of trajectories:
"""

import os
import math
import time
import torch
import einops
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Optional
import matplotlib.pyplot as plt


#%% Diffusion Transformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ContinuousCondEmbedder(nn.Module):
    """Modified from DiscreteCondEmbedder to embed the initial state,
    a continuous variable instead of a 1-hot vector
    The embedding transforms the discrete 1-hot into a continuous vector, don't need that here.
    Just a regular affine layer to make the initial state of the right dimension."""
    
    def __init__(self, attr_dim: int, hidden_size: int):
        super().__init__()
        self.attr_dim = attr_dim
        self.embedding = nn.Linear(attr_dim, int(attr_dim*128)) # 1 layer affine to transform initial state into embedding vector
        self.attn = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.linear = nn.Linear(128 * attr_dim, hidden_size)
    
    def forward(self, attr: torch.Tensor, mask: torch.Tensor = None):
        '''
        attr: (batch_size, attr_dim)
        mask: (batch_size, attr_dim) 0 or 1, 0 means ignoring
        '''
        emb = self.embedding(attr).reshape((-1, self.attr_dim, 128)) # (b, attr_dim, 128)
        if mask is not None: emb *= mask.unsqueeze(-1) # (b, attr_dim, 128)
        emb, _ = self.attn(emb, emb, emb) # (b, attr_dim, 128)
        return self.linear(einops.rearrange(emb, 'b c d -> b (c d)')) # (b, hidden_size)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))
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
    def __init__(self, x_dim: int, attr_dim: int, d_model: int = 384, 
                 n_heads: int = 6, depth: int = 12, dropout: float = 0.1):
        super().__init__()
        self.attr_dim = attr_dim # dimension of the attributes
        self.x_dim, self.d_model, self.n_heads, self.depth = x_dim, d_model, n_heads, depth
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_emb = TimeEmbedding(d_model)
        self.attr_proj = ContinuousCondEmbedder(attr_dim, d_model)
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
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                attr: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        '''
        Input:  x: (batch, horizon, x_dim)     t:  (batch, 1)
             attr: (batch, attr_dim)         mask: (batch, attr_dim)
        
        Output: y: (batch, horizon, x_dim)
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
    
    
    
    
    
    
#%% Diffusion ODE
    
class Conditional_ODE():
    def __init__(self, env, attr_dim: int, sigma_data: float,
        sigma_min: float = 0.002, sigma_max: float = 80,
        rho: float = 7, p_mean: float = -1.2, p_std: float = 1.2, 
        d_model: int = 384, n_heads: int = 6, depth: int = 12,
        device: str = "cpu", N: int = 5, lr: float = 2e-4):
        """Predicts the sequence of actions to apply conditioned on the initial state
        Diffusion trained according to EDM: "Elucidating the Design Space of Diffusion-Based Generative Models"
        """
        
        self.task = env.name
        self.specs = f"{d_model}_{n_heads}_{depth}"
        self.filename = "Cond_ODE_" + self.task + "_specs_" + self.specs
        self.state_size = env.state_size
        self.action_size = env.action_size
        assert attr_dim == self.state_size, "The attribute is the conditionement on the state"
        
        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        
        self.device = device
        self.F = DiT1d(self.action_size, attr_dim=attr_dim, d_model=d_model, n_heads=n_heads, depth=depth, dropout=0.1).to(device)
        self.F.train()
        # Exponential Moving Average (ema)
        self.F_ema = deepcopy(self.F).requires_grad_(False)
        self.F_ema.eval()
        self.optim = torch.optim.AdamW(self.F.parameters(), lr=lr, weight_decay=1e-4)
        self.set_N(N) # number of noise scales
       
        print(f'Initialized Action Conditional ODE model with {count_parameters(self.F)} parameters.')
        
    def ema_update(self, decay=0.999):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay*p_ema.data + (1-decay)*p.data

    def set_N(self, N):
        self.N = N
        self.sigma_s = (self.sigma_max**(1/self.rho)+torch.arange(N, device=self.device)/(N-1)*\
            (self.sigma_min**(1/self.rho)-self.sigma_max**(1/self.rho)))**self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        if self.t_s is not None:
            self.coeff1 = (self.dot_sigma_s/self.sigma_s + self.dot_scale_s/self.scale_s)
            self.coeff2 = self.dot_sigma_s/self.sigma_s*self.scale_s
            
    def c_skip(self, sigma): return self.sigma_data**2/(self.sigma_data**2+sigma**2)
    def c_out(self, sigma): return sigma*self.sigma_data/(self.sigma_data**2+sigma**2).sqrt()
    def c_in(self, sigma): return 1/(self.sigma_data**2+sigma**2).sqrt()
    def c_noise(self, sigma): return 0.25*(sigma).log()
    def loss_weighting(self, sigma): return (self.sigma_data**2+sigma**2)/((sigma*self.sigma_data)**2)
    def sample_noise_distribution(self, N):
        log_sigma = torch.randn((N,1,1),device=self.device)*self.p_std + self.p_mean
        return log_sigma.exp()
    
    def D(self, x, sigma, condition = None, mask = None, use_ema = False):
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.F_ema if use_ema else self.F
        return c_skip*x + c_out*F(c_in*x, c_noise.squeeze(-1), condition, mask)
    
    def update(self, x, condition):
        """Updates the DiT module given a trajectory batch x: (batch, horizon, state_size)
        and their corresponding attributes condition: (batch, attr_dim) """
        sigma = self.sample_noise_distribution(x.shape[0])
        eps = torch.randn_like(x) * sigma
        loss_mask = torch.ones_like(x)
        
        mask = (torch.rand(*condition.shape, device=self.device) > 0.2).int()
        pred = self.D(x + eps, sigma, condition, mask)            
        loss = (loss_mask * self.loss_weighting(sigma) * (pred - x)**2).mean()
        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.F.parameters(), 10.)
        self.optim.step()
        self.ema_update()
        return loss.item(), grad_norm.item()
    
    
    def train(self, x_normalized:torch.Tensor, attributes:torch.Tensor, n_gradient_steps:int,
              batch_size:int = 32, extra:str="", time_limit=None):
        """Trains the DiT module from NORMALIZED data x_normalized: (nb trajs, horizon, action_size)
        The attributes are the initial states of each trajectory
        time_limit in seconds"""
                
        print('Begins training of the Diffusion Transformer ' + self.filename + extra)
        if time_limit is not None:
            t0 = time.time()
            print(f"Training limited to {time_limit:.0f}s")
            
        N_trajs = x_normalized.shape[0]
        loss_avg = 0.
        
        pbar = tqdm(range(n_gradient_steps))
        for step in range(n_gradient_steps):
            
            idx = np.random.randint(0, N_trajs, batch_size) # sample a random batch of trajectories
            x = x_normalized[idx].clone()
            attr = attributes[idx].clone()
            loss, grad_norm = self.update(x, attr)
            loss_avg += loss
            if (step+1) % 10 == 0:
                pbar.set_description(f'step: {step+1} loss: {loss_avg / 10.:.4f} grad_norm: {grad_norm:.4f} ')
                pbar.update(10)
                loss_avg = 0.
                self.save(extra)
                if time_limit is not None and time.time() - t0 > time_limit:
                    print(f"Time limit reached at {time.time() - t0:.0f}s")
                    break
                
        print('\nTraining completed!')        
        
    
    @torch.no_grad()
    def sample(self, attr, traj_len, n_samples: int, w:float = 1.5, N:int = None):
        """Samples 'n_samples' trajectories of length 'traj_len' conditioned on
        initial state s0=attr.
        Heun's 2nd order sampling from the EDM paper"""
        
        if N is not None and N != self.N: self.set_N(N)
        x = torch.randn((n_samples, traj_len, self.action_size), device=self.device) * self.sigma_s[0] * self.scale_s[0]
        
        # Doubling x, attr since we sample eps(conditioned) - eps(unconditioned) see Section 4 of AlignDiff
        attr_mask = torch.ones_like(attr) # Is it the right mask?
        attr = attr.repeat(2, 1)
        attr_mask = attr_mask.repeat(2, 1)
        attr_mask[n_samples:] = 0
        
        for i in range(self.N):
            with torch.no_grad():
                D = self.D(x.repeat(2,1,1)/self.scale_s[i], torch.ones((2*n_samples,1,1),device=self.device)*self.sigma_s[i], attr, attr_mask, use_ema=True)
                D = w*D[:n_samples] + (1-w)*D[n_samples:]
            delta = self.coeff1[i]*x - self.coeff2[i]*D
            dt = self.t_s[i]-self.t_s[i+1] if i != self.N-1 else self.t_s[i]
            x = x - delta*dt
            
        return x    
    
    
    def save(self, extra:str = ""):
        torch.save({'model': self.F.state_dict(), 'model_ema': self.F_ema.state_dict()}, "trained_models/"+ self.filename+extra+"2.pt")
        
    
    def load(self, extra:str = ""):    
        name = "trained_models/" + self.filename + extra + "2.pt"
        if os.path.isfile(name):
            print("Loading " + name)
            checkpoint = torch.load(name, map_location=self.device, weights_only=True)
            self.F.load_state_dict(checkpoint['model'])
            self.F_ema.load_state_dict(checkpoint['model_ema'])
            return True # loaded
        else:
            print("File " + name + " doesn't exist. Not loading anything.")
            return False # not loaded




#%%


class Conditional_Planner():
    def __init__(self, env, ode: Conditional_ODE, normalizer):
        """Planner enables trjaectory prediction """
        self.env = env
        self.ode = ode
        self.normalizer = normalizer
        self.device = ode.device
        self.state_size = env.state_size
        self.action_size = env.action_size
      
        
    @torch.no_grad()
    def traj(self, s0, traj_len, N:int=None):
        """Returns n_samples action sequences of length traj_len starting from
        UNnormalized state s0."""
        
        if type(s0) == np.ndarray:
            nor_s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)[None,]
        else: # s0 = Tensor
            nor_s0 = s0.clone()
            s0 = s0.squeeze().numpy()
        nor_s0 = self.normalizer.normalize(nor_s0)
            
        assert nor_s0.shape == (1, self.state_size), "Only works for a single state"
        
        action_pred = self.ode.sample(attr=nor_s0, traj_len=traj_len-1, n_samples=1, N=N)
        traj_pred, traj_reward = self._open_loop(s0, action_pred.numpy())
        return traj_pred, action_pred, traj_reward   
          
    
    @torch.no_grad()
    def best_traj(self, s0, traj_len, n_samples_per_s0=1, N:int=None):
        """Returns 1 trajectory of length traj_len starting from each
        UNnormalized states s0.
        For each s0  n_samples_per_s0 are generated, the one with the longest survival is chosen"""
        
        if s0.shape == (self.state_size,):
            s0.reshape((1, self.state_size))
        assert s0.shape[1] == self.state_size
        
        if type(s0) == np.ndarray:
            nor_s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        else: # s0 = Tensor
            nor_s0 = s0.clone()
            s0 = s0.numpy()
        
        N_s0 = nor_s0.shape[0] # 
        nor_s0 = self.normalizer.normalize(nor_s0)
        nor_s0 = nor_s0.repeat_interleave(n_samples_per_s0, dim=0)
        n_samples = nor_s0.shape[0] # total number of samples = N_s0 * n_samples_per_s0
        
        # Sample all the sequences of actions at once: faster
        Actions_pred = self.ode.sample(attr=nor_s0, traj_len=traj_len-1, n_samples=n_samples, N=N)
        
        Best_Trajs = np.zeros((N_s0, traj_len, self.state_size))
        Best_rewards = np.zeros(N_s0)
        Best_Actions = np.zeros((N_s0, traj_len-1, self.action_size))
        
        for s_id in range(N_s0): # index of the s0
            highest_reward = -np.inf # look for the trajectory with highest reward for each s0
            i = s_id*n_samples_per_s0
            Sampled_Trajs = []
            
            for sample_id in range(n_samples_per_s0):    
                traj, reward = self._open_loop(s0[s_id], Actions_pred[i + sample_id].numpy())
                Sampled_Trajs.append(traj)
                if reward > highest_reward:
                    highest_reward = reward
                    id_highest_reward = sample_id
            
            t = Sampled_Trajs[id_highest_reward].shape[0]
            Best_Trajs[s_id, :t] = Sampled_Trajs[id_highest_reward].copy()
            Best_rewards[s_id] = highest_reward
            Best_Actions[s_id] = Actions_pred[i + id_highest_reward]
        
        return Best_Trajs, Best_Actions, Best_rewards


    def _open_loop(self, s0, Actions):
        """Applies the sequence of actions in open-loop on the initial state s0"""
        assert s0.shape[0] == self.state_size
        assert Actions.shape[1] == self.action_size
        
        N_steps = Actions.shape[0]
        Traj = np.zeros((N_steps+1, self.state_size))
        Traj[0] = self.env.reset_to(s0)
        traj_reward = 0.
        
        for t in range(N_steps):
            Traj[t+1], reward, done, _, _ = self.env.step(Actions[t])
            traj_reward += reward
            if done: break
        
        return Traj[:t+2], traj_reward




