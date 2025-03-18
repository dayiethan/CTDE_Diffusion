import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data, DataLoader
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
H = 10 # horizon, length of each trajectory

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
with open("data/mean.npy", "wb") as f:
    np.save(f, mean)
with open("data/std.npy", "wb") as f:
    np.save(f, std)



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
sig = np.array([sigma_data1, sigma_data2])
with open("data/sigma_data.npy", "wb") as f:
    np.save(f, sig)

# Define the GNN model
class TwoAgentGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoAgentGNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Output separate embeddings for each agent

# Example input dimensions
input_dim = 4
hidden_dim = 16
output_dim = 4

# Instantiate the GNN
gnn = TwoAgentGNN(input_dim, hidden_dim, output_dim).to(device)

# Stack observations to form a batch of agent pairs
x_batch = torch.cat([obs1, obs2], dim=0)  # Shape: (20000, 4)

# Define edge connections (same for all batches)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Shape: (2, 2)

# Expand edge_index for batch processing
batch_size = obs1.shape[0]  # 10000
edge_index = edge_index.repeat(1, batch_size)  # Shape: (2, 2 * batch_size)

# Adjust node indices for batched graph (ensuring unique node indices per batch)
offsets = torch.arange(batch_size) * 2  # Offsets for each batch
edge_index = edge_index + offsets.repeat_interleave(2).unsqueeze(0)

# Forward pass through GNN
with torch.no_grad():
    embeddings = gnn(x_batch, edge_index)  # Shape: (20000, 4)

# Reshape to separate agents
embeddings1, embeddings2 = embeddings.chunk(2, dim=0)  # Both shapes: (10000, 4)

# combine actions_1 and actions_2 into a single tensor
actions = torch.cat([actions1, actions2], dim=0)
embeddings = torch.cat([embeddings1, embeddings2], dim=0)

sigma_data = actions.std().item()

# Training
action_cond_ode = Conditional_ODE(env, [attr_dim1], [sigma_data], device=device, N=100, n_models = 1, **model_size)
action_cond_ode.train([actions], [embeddings], int(5*n_gradient_steps), batch_size, extra="_T10_2")
action_cond_ode.save(extra="_T10_2_gnn")
# action_cond_ode.load(extra="_T10_2")




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

sys.exit()

# Sampling
for i in range(10):
    _i = np.random.randint(0, 10000)
    attr_t1 = attr_test1[_i].unsqueeze(0)
    attr_t2 = attr_test2[_i].unsqueeze(0)
    attr_n1 = attr_t1.cpu().detach().numpy()[0]
    attr_n2 = attr_t2.cpu().detach().numpy()[0]

    traj_len = 10
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
    plt.savefig("figs/T10_2/plot%s.png" % (i+10))


