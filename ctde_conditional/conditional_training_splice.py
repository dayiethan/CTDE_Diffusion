import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys
import pdb
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 100 # horizon, length of each trajectory

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0) 

# Loading training trajectories
all_points1 = []    # want modes 1, 2, 4, 6
all_points2 = []    # want modes 1, 2, 3, 5
with open('data/trajs_noise1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x1, y1 = float(row[4]), float(row[5])
        x2, y2 = float(row[7]), float(row[8])
        all_points1.append([x1, y1])
        all_points2.append([x2, y2])

num_trajectories = 10000
points_per_trajectory = 10

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


expert_data1 = np.array(expert_data1)
expert_data2 = np.array(expert_data2)



# Unspliced trajectories to get final positions
orig1 = [
    all_points1[i * 100:(i + 1) * 100]
    for i in range(1000)
]
orig2 = [
    all_points2[i * 100:(i + 1) * 100]
    for i in range(1000)
]
orig1 = np.array(orig1)
orig2 = np.array(orig2)

combined_data1 = np.concatenate((expert_data1, expert_data2), axis=0)
combined_data2 = np.concatenate((orig1, orig2), axis=0)
mean1 = np.mean(combined_data1, axis=(0,1))
std1 = np.std(combined_data1, axis=(0,1))
mean2 = np.mean(combined_data2, axis=(0,1))
std2 = np.std(combined_data2, axis=(0,1))
mean = (mean1 + mean2)/2
std = (std1 + std2)/2
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std
orig1 = (orig1 - mean) / std
orig2 = (orig2 - mean) / std



# Prepare Data for Training
X_train1 = []
Y_train1 = []

for traj in expert_data1:
    for i in range(len(traj) - 1):
        X_train1.append(np.hstack([traj[i], final_point_up]))  # Current state + goal
        Y_train1.append(traj[i + 1])  # Next state

X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32)  # Shape: (N, 4)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32)  # Shape: (N, 2)

X_train2 = []
Y_train2 = []

for traj in expert_data2:
    for i in range(len(traj) - 1):
        X_train2.append(np.hstack([traj[i], final_point_down]))  # Current state + goal
        Y_train2.append(traj[i + 1])  # Next state

X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 4)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 2)

# Define environment
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"
env = TwoUnicycle()



# Setting up training data
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs_final1 = np.repeat(orig1[:, -1, :], repeats=10, axis=0)
obs_final2 = np.repeat(orig2[:, -1, :], repeats=10, axis=0)
obs1 = np.hstack([obs_init1, obs_final1])
obs2 = np.hstack([obs_init2, obs_final2])
obs_temp1 = obs1
obs_temp2 = obs2
actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)

attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]
assert attr_dim1 == env.state_size * 2
assert attr_dim2 == env.state_size * 2

actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()


# Training
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra="_T10")
action_cond_ode.save(extra="_T10")
# action_cond_ode.load(extra="_T10")



# Sampling preparation
noise_std = 0.
noise = np.ones(np.shape(obs_temp1))
obs_temp1 = obs_temp1 + noise_std * noise
obs_temp2 = obs_temp2 + noise_std * noise
obs_temp_tensor1 = torch.FloatTensor(obs_temp1).to(device)
obs_temp_tensor2 = torch.FloatTensor(obs_temp2).to(device)
attr_test1 = obs_temp_tensor1
attr_test2 = obs_temp_tensor2
expert_data1 = expert_data1 * std + mean
expert_data2 = expert_data2 * std + mean
ref1 = np.mean(expert_data1, axis=0)
ref2 = np.mean(expert_data2, axis=0)
ref_agent1 = ref1[:, :]
ref_agent2 = ref2[:, :]



# Sampling
for i in range(10):
    attr_t1 = attr_test1[i*10].unsqueeze(0)
    attr_t2 = attr_test2[i*10].unsqueeze(0)
    attr_n1 = attr_t1.cpu().detach().numpy()[0]
    attr_n2 = attr_t2.cpu().detach().numpy()[0]

    traj_len = 100
    n_samples = 1

    sampled1 = action_cond_ode.sample(attr_t1, traj_len, n_samples, w=1., model_index = 0)
    sampled2 = action_cond_ode.sample(attr_t2, traj_len, n_samples, w=1., model_index = 1)

    sampled1 = sampled1.cpu().detach().numpy()
    sampled2 = sampled2.cpu().detach().numpy()
    sampled1 = sampled1 * std + mean
    sampled2 = sampled2 * std + mean
    test1 = np.mean(sampled1, axis=0)
    test2 = np.mean(sampled2, axis=0)
    test_agent1 = test1[:, :]
    test_agent2 = test2[:, :]

    # sys.setrecursionlimit(10000)
    # fast_frechet = FastDiscreteFrechetMatrix(euclidean)
    # frechet1 = fast_frechet.distance(ref_agent1,test_agent1)
    # frechet2 = fast_frechet.distance(ref_agent2,test_agent2)
    # print(frechet1, frechet2)

    init_state1 = attr_n1[:2]
    final_state1 = attr_n1[2:]
    init_state2 = attr_n2[:2]
    final_state2 = attr_n2[2:]

    init_state1 = init_state1 * std + mean
    final_state1 = final_state1 * std + mean
    init_state2 = init_state2 * std + mean
    final_state2 = final_state2 * std + mean

    attr_n1 = np.concatenate([init_state1, final_state1])
    attr_n2 = np.concatenate([init_state2, final_state2])

    plt.figure(figsize=(20, 8))
    for traj in expert_data1:
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        plt.plot(traj[-1, 0], traj[0, 1], 'go', markersize=8)
    for traj in expert_data2:
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        plt.plot(traj[-1, 0], traj[0, 1], 'go', markersize=8)
    plt.plot(attr_n1[0], attr_n1[1], 'bo')
    plt.plot(attr_n2[0], attr_n2[1], 'o', color='orange')
    plt.plot(attr_n1[2], attr_n1[3], 'bo')
    plt.plot(attr_n2[2], attr_n2[3], 'o', color='orange')
    plt.plot(sampled1[0, :, 0], sampled1[0, :, 1], color='blue')
    plt.plot(sampled2[0, :, 0], sampled2[0, :, 1], color='orange')
    # plt.legend(loc="upper right", fontsize=14)
    plt.savefig("figs/T10/plot%s.png" % i)


