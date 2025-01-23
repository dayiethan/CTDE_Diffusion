import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE, Conditional_Planner
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 100 # horizon, length of each trajectory

# loading the trajectories

trajectory = np.loadtxt("data/full_traj_obstacle.csv",delimiter=",", dtype=float)
mean = trajectory.mean(axis=0)
std = trajectory.mean(axis=0)
max_traj_array = np.max(trajectory, axis=0)
np.savetxt("data/max_traj_array_rand.csv", max_traj_array, delimiter=",")
np.savetxt("data/traj_mean.csv", mean, delimiter=",")
np.savetxt("data/traj_std.csv", std, delimiter=",")
# trajectory = trajectory/max_traj_array
trajectory = (trajectory - mean) / std
trajectory = (trajectory).reshape(-1, 100, 10)

N_trajs = trajectory.shape[0] # number of trajectories for training

# define an enviornment objcet which has attrubutess like name, state_size, action_size etc

class TwoUnicycle():
    def __init__(self, state_size=10, action_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"

env = TwoUnicycle()

obs = trajectory[:, 0, :] # only keep the initial states of trajectories
actions = trajectory[:, :H-1, :] # cut the length of trajectories to H

obs = torch.FloatTensor(obs).to(device)
attr = obs # Conditioned on the normalized initial states
attr_dim = attr.shape[1]
assert attr_dim == env.state_size

actions = torch.FloatTensor(actions).to(device)
sigma_data = actions.std().item()

# Training

print("Conditional Action Diffusion Transformer without projections")
action_cond_ode = Conditional_ODE(env, attr_dim, sigma_data, device=device, N=20, lr=1e-4, **model_size)
# action_cond_ode.load()
action_cond_ode.train(actions, attr, int(5*n_gradient_steps), batch_size, extra="")
action_cond_ode.save()

attr = attr[0].unsqueeze(0)

print(attr)

traj_len = 100
n_samples = 1

sampled = action_cond_ode.sample(attr, traj_len, n_samples)

sampled = sampled.cpu().detach().numpy()
# sampled = sampled * max_traj_array
sampled = sampled * std + mean

print(sampled.shape)

plt.plot(sampled[0, :, 4], sampled[0, :, 5], color='blue')
plt.plot(sampled[0, :, 7], sampled[0, :, 8], color='orange')

plt.show()


