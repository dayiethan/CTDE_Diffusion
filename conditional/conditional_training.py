import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE, Conditional_Planner
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gradient_steps = 1_000_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 100 # horizon, length of each trajectory

# loading the trajectories

trajectory = np.loadtxt("data/trajs_noise1.csv",delimiter=",", dtype=float)
mean = trajectory.mean(axis=0)
std = trajectory.std(axis=0)
max_traj_array = np.max(trajectory, axis=0)
np.savetxt("data/max_traj_array_rand.csv", max_traj_array, delimiter=",")
np.savetxt("data/traj_mean.csv", mean, delimiter=",")
np.savetxt("data/traj_std.csv", std, delimiter=",")
# trajectory = trajectory/max_traj_array
trajectory = (trajectory - mean) / std
trajectory = (trajectory).reshape(-1, 100, 10)

# trajectory = trajectory * std + mean
# plt.figure(figsize=(20, 8))
# for traj in trajectory:
#     plt.plot(traj[0, 4], traj[0, 5], 'go', markersize=8)
#     plt.plot(traj[0, 7], traj[0, 8], 'go', markersize=8)
# plt.savefig("test0.5.png")

# plt.figure(figsize=(20, 8))
# for traj in trajectory07:
#     plt.plot(traj[0, 4], traj[0, 5], 'bo', markersize=8)
#     plt.plot(traj[0, 7], traj[0, 8], 'bo', markersize=8)
# plt.savefig("test0.7.png")

# for traj in trajectory[:]:  # Plot a few expert trajectories
#     first_trajectory = traj
#     x1 = [point[4] for point in first_trajectory]
#     y1 = [point[5] for point in first_trajectory]
#     x2 = [point[7] for point in first_trajectory]
#     y2 = [point[8] for point in first_trajectory]
#     plt.plot(x1, y1, 'b--')
#     plt.plot(x2, y2, 'r--')

# import sys
# sys.exit()

N_trajs = trajectory.shape[0] # number of trajectories for training

# define an enviornment objcet which has attrubutess like name, state_size, action_size etc

class TwoUnicycle():
    def __init__(self, state_size=10, action_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"

env = TwoUnicycle()

obs_init = trajectory[:, 0, :] # only keep the initial states of trajectories
obs_final = trajectory[:, -1, :]
obs = np.hstack([obs_init, obs_final])
obs_temp = obs
actions = trajectory[:, :H-1, :] # cut the length of trajectories to H

obs = torch.FloatTensor(obs).to(device)

attr = obs # Conditioned on the normalized initial states
attr_dim = attr.shape[1]
assert attr_dim == env.state_size * 2

actions = torch.FloatTensor(actions).to(device)
sigma_data = actions.std().item()

# Training

print("Conditional Action Diffusion Transformer without projections")
action_cond_ode = Conditional_ODE(env, attr_dim, sigma_data, device=device, N=100, **model_size)
# action_cond_ode.load(extra="clamp_noise1")
action_cond_ode.train(actions, attr, int(5*n_gradient_steps), batch_size, extra="clamp_noise1")
action_cond_ode.save(extra="clamp_noise1")

noise_std = 0.05
noise = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0])
obs_temp = obs_temp + noise_std * noise
obs_temp_tensor = torch.FloatTensor(obs_temp).to(device)  # ensure it's a tensor
attr_test = obs_temp_tensor

trajectory = trajectory * std + mean
for i in range(10):
    attr_t = attr_test[i*10].unsqueeze(0)
    attr_n = attr_t.cpu().detach().numpy()[0]

    traj_len = 100
    n_samples = 1

    sampled = action_cond_ode.sample(attr_t, traj_len, n_samples, w=1.)

    sampled = sampled.cpu().detach().numpy()
    sampled = sampled * std + mean

    init_state = attr_n[:10]    # shape (10,)
    final_state = attr_n[10:]   # shape (10,)

    init_state = init_state * std + mean
    final_state = final_state * std + mean

    attr_n = np.concatenate([init_state, final_state])

    plt.figure(figsize=(20, 8))
    for traj in trajectory:
        plt.plot(traj[0, 4], traj[0, 5], 'go', markersize=8)
        plt.plot(traj[0, 7], traj[0, 8], 'go', markersize=8)
    plt.plot(attr_n[4], attr_n[5], 'bo')
    plt.plot(attr_n[7], attr_n[8], 'o', color='orange')
    plt.plot(attr_n[14], attr_n[15], 'bo')
    plt.plot(attr_n[17], attr_n[18], 'o', color='orange')
    plt.plot(sampled[0, :, 4], sampled[0, :, 5], color='blue')
    plt.plot(sampled[0, :, 7], sampled[0, :, 8], color='orange')
    plt.savefig("figs/noise1_vary0.05/plot%s.png" % i)


