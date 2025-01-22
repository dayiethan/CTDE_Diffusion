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
  
#%% Dataset
    
N_trajs = 20 # number of trajectories for training
# Parse expert data from single_uni_full_traj.csv
# Define initial and final points, and a single central obstacle
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
all_points1 = all_points1[:500]
with open('data/mode4_agent1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points1.append([x, y])
all_points1 = all_points1[:1000]
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
all_points1 = all_points1[:1500]
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
all_points1 = all_points1[:2000]


with open('data/mode5_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:500]
with open('data/mode3_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:1000]
with open('data/mode2_agent2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x, y = float(row[0]), float(row[1])
        all_points2.append([x, y])
all_points2 = all_points2[:1500]
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
all_points2 = all_points2[:2000]
# with open('data/mode6_agent2.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         x, y = float(row[0]), float(row[1])
#         all_points2.append([x, y])


num_trajectories = 20
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


start1 = expert_data1[:, 0] # only keep the initial states of trajectories
end1 = expert_data1[:, -1] # only keep the final states of trajectories
obs1 = np.hstack((start1, end1)) # concatenate initial and final states
start2 = expert_data2[:, 0] # only keep the initial states of trajectories
end2 = expert_data2[:, -1] # only keep the final states of trajectories
obs2 = np.hstack((start2, end2)) # concatenate initial and final states 
# actions = data["Actions"][:, :H-1] # cut the length of trajectories to H

obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
normalizer1 = Normalizer(obs1)
normalizer2 = Normalizer(obs2)
attr1 = normalizer1.normalize(obs1) # Conditioned on the normalized initial states
attr2 = normalizer2.normalize(obs2) # Conditioned on the normalized initial states
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# actions = torch.FloatTensor(actions).to(device)
# sigma_data = actions.std().item()
exp = np.concatenate((expert_data1, expert_data2), axis=0)
data = torch.FloatTensor(exp).to(device)
sigma_exp = data.std().item()


#%% Training

print("Conditional Action Diffusion Transformer without projections")
action_cond_ode = Conditional_ODE(env, attr_dim1, sigma_exp, device=device, N=5, **model_size)
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







