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

# env = WalkerEnv() 
    
  
#%% Dataset
    
N_trajs = 20 # number of trajectories for training

initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)


# Parse expert data from single_uni_full_traj.csv
import csv
all_points1 = []    # want modes 1, 2, 4, 6
all_points2 = []    # want modes 1, 2, 3, 5
with open('data/mode6_agent1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points1.append([x, y])
all_points1 = all_points1[:5000]
with open('data/mode4_agent1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points1.append([x, y])
all_points1 = all_points1[:10000]
# with open('data/mode3_agent1.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         x, y = float(row[0]), float(row[1])
#         all_points1.append([x, y])
with open('data/mode2_agent1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points1.append([x, y])
all_points1 = all_points1[:15000]
# with open('data/mode5_agent1.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         x, y = float(row[0]), float(row[1])
#         all_points1.append([x, y])
with open('data/mode1_agent1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points1.append([x, y])
all_points1 = all_points1[:20000]


with open('data/mode5_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:5000]
with open('data/mode3_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:10000]
with open('data/mode2_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:15000]
# with open('data/mode4_agent2.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         x, y = float(row[0]), float(row[1])
#         all_points2.append([x, y])
with open('data/mode1_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:20000]
# with open('data/mode6_agent2.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         x, y = float(row[0]), float(row[1])
#         all_points2.append([x, y])


num_trajectories = 200
points_per_trajectory = 100

expert_data1 = [
    all_points1[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory1 = expert_data1[0]
x1 = [point[0] for point in first_trajectory1]
y1 = [point[1] for point in first_trajectory1]

expert_data2 = [
    all_points2[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory2 = expert_data2[0]
x2 = [point[0] for point in first_trajectory2]
y2 = [point[1] for point in first_trajectory2]

# expert_data = expert_data + list(reversed(expert_data_rev))

expert_data1 = np.array(expert_data1)
expert_data2 = np.array(expert_data2)
actions1 = np.diff(expert_data1, axis=1)
actions2 = np.diff(expert_data2, axis=1)

# data = np.load(f"datasets/walker_{N_trajs}trajs_500steps.npz")
obs1 = expert_data1[:, 0] # only keep the initial states of trajectories
obs2 = expert_data2[:, 0] # only keep the initial states of trajectories

obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
normalizer1 = Normalizer(obs1)
normalizer2 = Normalizer(obs2)
attr1 = normalizer1.normalize(obs1) # Conditioned on the normalized initial states
attr2 = normalizer2.normalize(obs2) # Conditioned on the normalized initial states
attr_dim = attr1.shape[1]
data1 = torch.FloatTensor(expert_data1).to(device)
data2 = torch.FloatTensor(expert_data2).to(device)
# assert attr_dim == env.state_size

actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()


#%% Training

print("Conditional Action Diffusion Transformer without projections")
action_cond_ode1 = Conditional_ODE("exp1", 2, attr_dim, sigma_data1, device=device, N=5, **model_size)
action_cond_ode2 = Conditional_ODE("exp2", 2, attr_dim, sigma_data2, device=device, N=5, **model_size)
# action_cond_ode1.load()
# action_cond_ode1.train(data1, attr1, int(5*n_gradient_steps), batch_size, extra="")
# action_cond_ode2.train(data2, attr2, int(5*n_gradient_steps), batch_size, extra="")
action_cond_ode1.load()
action_cond_ode2.load()


#%% Evaluation

# N_s0 = 8 # number of different initial states sampled to evaluate each model
# N_samples = 6 # number of trajectories sampled pre initial state to evaluate each model



# plot_height = False

# ### Test trajectories, unseen in training
# data = np.load("datasets/walker_28trajs_500steps.npz")
# loaded_trajs = data["Trajs"][:, :H] # cut the lenght of trajectories to H
# loaded_actions = data["Actions"][:, :H]

# traj_id = 1 # index of the load traj to test in {0, 1, ..., N_eval_trajs-1}
# true_actions = loaded_actions[traj_id]
# true_traj = np.zeros((H, 2))
# true_traj[0] = env.reset_to(loaded_trajs[traj_id, 0])
# loaded_reward = 0

# for t in range(H-1):
#     true_traj[t+1], reward, done, _, _ = env.step(true_actions[t])
#     loaded_reward += reward
#     if done: break

# labels = ["loaded"]
# list_of_rewards = [loaded_reward]
# list_of_rewards_std = [0.]
# list_of_survival = [1.]
# list_of_survival_std = [0.]

#%%

true_traj1 = np.array(first_trajectory1)
action_planner1 = Conditional_Planner(2, 2, action_cond_ode1, normalizer1)
s0 = true_traj1[0].reshape((1, 2))
traj1, actions, reward = action_planner1.best_traj(s0=s0, traj_len=H, n_samples_per_s0=4)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 8))

plt.plot(traj1[:, 0], traj1[:, 1], 'r-', label='Generated')
# plt.plot(traj2[:, 0], traj2[:, 1], 'y-', label='Generated')

ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

# Mark start and end points
plt.scatter(initial_point_up[0], initial_point_up[1], c='red', s=100, label='Start/End')
plt.scatter(final_point_up[0], final_point_up[1], c='red', s=100, label='Start/End')

# plt.legend()
# plt.title('Smooth Imitation Learning: Expert vs Generated Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('expert_vs_generated_trajectories.png')
# survival = traj[0].shape[0]/H
# print(f"Open-loop actions from DiT conditioned on first state, reward: {reward[0]:.1f}  survival: {survival:.2f}")
# ID_traj, ID_actions = ID.closest_admissible_traj(traj[0])[:2]
# print(f"Inverse dynamics traj diverges at step {ID_traj.shape[0]:}")

# traj_comparison(env, true_traj, "true", traj[0], "sampled",
#                 traj_3=ID_traj, label_3="ID",
#                 title="Action DiT conditioned on first state", 
#                 plot_height=plot_height)

# mean_rwd, std_rwd, mean_survival, std_survival = open_loop_stats(env, action_planner, ID, N_s0=N_s0, H=H, N_samples=N_samples)
# labels.append("action")
# list_of_rewards.append(mean_rwd)
# list_of_rewards_std.append(std_rwd)
# list_of_survival.append(mean_survival)
# list_of_survival_std.append(std_survival)
# print("\n")
