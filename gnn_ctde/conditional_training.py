import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 250 # horizon, length of each trajectory

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

expert_data = np.load("data/expert_actions_rotvec.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]

states = np.load("data/expert_states_rotvec.npy")
states1 = states[:, :, :7]
states2 = states[:, :, 7:14]


# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
mean = np.mean(combined_data, axis=(0,1))
std = np.std(combined_data, axis=(0,1))

# Normalize data
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std

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

# define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

env = TwoArmLift()

obs_init1 = states1[:, 0, :]
obs_init2 = states2[:, 0, :]
obs_final1 = states1[:, -1, :]
obs_final2 = states2[:, -1, :]
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
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra="_T250_rotvec")
action_cond_ode.save(extra="_T250_rotvec")
action_cond_ode.load(extra="_T250_rotvec")

# import pdb
# breakpoint()
noise_std = 0.05
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

for i in range(10):
    attr_t1 = attr_test1[i*10].unsqueeze(0)
    attr_t2 = attr_test2[i*10].unsqueeze(0)
    attr_n1 = attr_t1.cpu().detach().numpy()[0]
    attr_n2 = attr_t2.cpu().detach().numpy()[0]

    traj_len = 250
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

    sys.setrecursionlimit(10000)
    fast_frechet = FastDiscreteFrechetMatrix(euclidean)
    frechet1 = fast_frechet.distance(ref_agent1,test_agent1)
    frechet2 = fast_frechet.distance(ref_agent2,test_agent2)
    print(frechet1, frechet2)

    init_state1 = attr_n1[:env.state_size]
    final_state1 = attr_n1[env.state_size:]
    init_state2 = attr_n2[:env.state_size]
    final_state2 = attr_n2[env.state_size:]

    init_state1 = init_state1 * std + mean
    final_state1 = final_state1 * std + mean
    init_state2 = init_state2 * std + mean
    final_state2 = final_state2 * std + mean

    attr_n1 = np.concatenate([init_state1, final_state1])
    attr_n2 = np.concatenate([init_state2, final_state2])

    plt.figure(figsize=(20, 8))
    plt.scatter(expert_data1[:, 0, 0], expert_data1[:, 0, 1], color='green')
    plt.scatter(expert_data2[:, 0, 0], expert_data2[:, 0, 1], color='green')
    plt.scatter(expert_data1[:, -1, 0], expert_data1[:, -1, 1], color='green')
    plt.scatter(expert_data2[:, -1, 0], expert_data2[:, -1, 1], color='green')
    plt.plot(attr_n1[0], attr_n1[1], 'bo')
    plt.plot(attr_n2[0], attr_n2[1], 'o', color='orange')
    plt.plot(attr_n1[2], attr_n1[3], 'bo')
    plt.plot(attr_n2[2], attr_n2[3], 'o', color='orange')
    plt.plot(sampled1[0, :, 0], sampled1[0, :, 1], color='blue', label=f"Agent 1 Traj (Frechet: {frechet1:.2f})")
    plt.plot(sampled2[0, :, 0], sampled2[0, :, 1], color='orange', label=f"Agent 2 Traj (Frechet: {frechet2:.2f})")
    plt.legend(loc="upper right", fontsize=14)
    plt.savefig("figs/T250_rotvec/plot%s.png" % i)


