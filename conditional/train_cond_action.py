# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 08:44:08 2024

@author: Jean-Baptiste Bouvier

Training the Action Conditional DiT for the Walker
Predicts sequences of actions conditioned on a normalized initial state
"""

import torch
import numpy as np

# from walker2d import WalkerEnv
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE, Conditional_Planner


#%% Hyperparameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gradient_steps = 10_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 300 # horizon, length of each trajectory
set_seed(0) 

env = WalkerEnv() 
    
  
#%% Dataset
    
N_trajs = 1000 # number of trajectories for training
data = np.load(f"datasets/walker_{N_trajs}trajs_500steps.npz")
obs = data["Trajs"][:, 0] # only keep the initial states of trajectories
actions = data["Actions"][:, :H-1] # cut the length of trajectories to H

obs = torch.FloatTensor(obs).to(device)
normalizer = Normalizer(obs)
attr = normalizer.normalize(obs) # Conditioned on the normalized initial states
attr_dim = attr.shape[1]
assert attr_dim == env.state_size

actions = torch.FloatTensor(actions).to(device)
sigma_data = actions.std().item()


#%% Training

print("Conditional Action Diffusion Transformer without projections")
action_cond_ode = Conditional_ODE(env, attr_dim, sigma_data, device=device, N=5, **model_size)
action_cond_ode.load()
action_cond_ode.train(actions, attr, int(5*n_gradient_steps), batch_size, extra="")



#%% Evaluation

N_s0 = 8 # number of different initial states sampled to evaluate each model
N_samples = 6 # number of trajectories sampled pre initial state to evaluate each model



plot_height = False

### Test trajectories, unseen in training
data = np.load("datasets/walker_28trajs_500steps.npz")
loaded_trajs = data["Trajs"][:, :H] # cut the lenght of trajectories to H
loaded_actions = data["Actions"][:, :H]

traj_id = 1 # index of the load traj to test in {0, 1, ..., N_eval_trajs-1}
true_actions = loaded_actions[traj_id]
true_traj = np.zeros((H, env.state_size))
true_traj[0] = env.reset_to(loaded_trajs[traj_id, 0])
loaded_reward = 0

for t in range(H-1):
    true_traj[t+1], reward, done, _, _ = env.step(true_actions[t])
    loaded_reward += reward
    if done: break

labels = ["loaded"]
list_of_rewards = [loaded_reward]
list_of_rewards_std = [0.]
list_of_survival = [1.]
list_of_survival_std = [0.]

#%%

action_planner = Conditional_Planner(env, action_cond_ode, normalizer)
s0 = true_traj[0].reshape((1, env.state_size))
traj, actions, reward = action_planner.best_traj(s0=s0, traj_len=H, n_samples_per_s0=4)
survival = traj[0].shape[0]/H
print(f"Open-loop actions from DiT conditioned on first state, reward: {reward[0]:.1f}  survival: {survival:.2f}")
ID_traj, ID_actions = ID.closest_admissible_traj(traj[0])[:2]
print(f"Inverse dynamics traj diverges at step {ID_traj.shape[0]:}")

traj_comparison(env, true_traj, "true", traj[0], "sampled",
                traj_3=ID_traj, label_3="ID",
                title="Action DiT conditioned on first state", 
                plot_height=plot_height)

mean_rwd, std_rwd, mean_survival, std_survival = open_loop_stats(env, action_planner, ID, N_s0=N_s0, H=H, N_samples=N_samples)
labels.append("action")
list_of_rewards.append(mean_rwd)
list_of_rewards_std.append(std_rwd)
list_of_survival.append(mean_survival)
list_of_survival_std.append(std_survival)
print("\n")







