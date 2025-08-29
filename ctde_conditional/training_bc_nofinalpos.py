import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Behavioral Cloning Network
class ImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)
   
class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h

class BigImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=7, horizon=10,
                 num_layers=8, dropout=0.1, expansion=4):
        super().__init__()
        self.horizon = horizon

        self.input = nn.Linear(input_size, hidden_size)

        # Learnable horizon/position embeddings (0..horizon-1)
        self.step_embed = nn.Embedding(horizon, hidden_size)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size, dropout=dropout, expansion=expansion)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: (batch_size, input_size)
        B = x.size(0)
        device = x.device

        # Encode input -> (B, hidden)
        h = self.input(x)  # (B, hidden_size)

        # Expand across horizon and add learnable step embeddings
        h = h.unsqueeze(1).expand(B, self.horizon, -1)              # (B, H, hidden)
        steps = torch.arange(self.horizon, device=device)           # (H,)
        h = h + self.step_embed(steps).unsqueeze(0)                 # (B, H, hidden)

        # Per-step MLP blocks
        for blk in self.blocks:
            h = blk(h)                                              # (B, H, hidden)

        # Project to outputs per step
        out = self.head(h)                                          # (B, H, output_size)
        return out
    
def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon  # we'll create one sub-trajectory starting at each time step

    # Resulting array shape: (n_traj * n_subtraj, planning_horizon, state_dim)
    result = []

    for traj in expert_data:
        for start_idx in range(n_subtraj):
            # If not enough steps, pad with the last step
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                # Need padding
                sub_traj = traj[start_idx:]
                padding = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, padding], axis=0)
            result.append(sub_traj)

    result = np.stack(result, axis=0)
    return result


# Define initial and final points, and a single central obstacle
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)
batch_size = 32
H = 25 # horizon, length of each trajectory
T = 100 # total time steps

# Expert demonstration loading
expert_data1 = np.load('data/expert_data1_100_traj.npy')
expert_data2 = np.load('data/expert_data2_100_traj.npy')
expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

mean1 = np.mean(expert_data1, axis=(0,1))
std1 = np.std(expert_data1, axis=(0,1))
mean2 = np.mean(expert_data2, axis=(0,1))
std2 = np.std(expert_data2, axis=(0,1))

expert_data1 = (expert_data1 - mean1) / std1
expert_data2 = (expert_data2 - mean2) / std2

actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
actions1 = torch.FloatTensor(actions1).float()
actions2 = torch.FloatTensor(actions2).float()
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()

obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs1 = np.hstack([obs_init1, obs_init2])
obs2 = np.hstack([obs_init2, obs_init1])
obs1 = torch.FloatTensor(obs1).float()
obs2 = torch.FloatTensor(obs2).float()
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

input_size = attr1.shape[1]

model1 = BigImitationNet(input_size, hidden_size=256, output_size=2, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)
model2 = BigImitationNet(input_size, hidden_size=256, output_size=2, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)

print(f"Total parameters: {sum(p.numel() for p in model1.parameters()) + sum(p.numel() for p in model2.parameters())}")

params = list(model1.parameters()) + list(model2.parameters())
optimizer = optim.Adam(params, lr=5e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
criterion = nn.MSELoss()

# Joint training for the agents
def joint_train(models, optimizer, criterion, datasets, num_epochs=20000):
    actions, attrs = zip(*datasets)
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for model, action, attr in zip(models, actions, attrs):
            idx = np.random.randint(0, action.shape[0], batch_size)
            true_actions = action[idx].to(device, non_blocking=True)
            input_attrs  = attr[idx].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_action = model(input_attrs)
                loss += criterion(pred_action, true_actions)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (epoch + 1) % 5000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return models


model1, model2 = joint_train(
    [model1, model2], optimizer, criterion,
    [(actions1, attr1), (actions2, attr2)], num_epochs=20000
)
         
save_path1 = "trained_models/bc/bc_nofinalpos1.pth"
save_path2 = "trained_models/bc/bc_nofinalpos2.pth"
torch.save(model1.state_dict(), save_path1)
torch.save(model2.state_dict(), save_path2)

# model1 = ImitationNet(input_size=4, hidden_size=64, output_size=2)
# model1.load_state_dict(torch.load(save_path1, map_location='cpu'))
# model1.eval()

# model2 = ImitationNet(input_size=4, hidden_size=64, output_size=2)
# model2.load_state_dict(torch.load(save_path2, map_location='cpu'))
# model2.eval()

# # Generate a New Trajectory Using the Trained Model
# noise_std = 0.1
# generated_trajectories1 = []
# generated_trajectories2 = []

# for i in range(100):
#     initial1 = initial_point1 + noise_std * np.random.randn(*np.shape(initial_point1))
#     final1 = final_point1 + noise_std * np.random.randn(*np.shape(final_point1))
#     initial2 = initial_point2 + noise_std * np.random.randn(*np.shape(initial_point2))
#     final2 = final_point2 + noise_std * np.random.randn(*np.shape(final_point2))
#     with torch.no_grad():
#         state1 = np.hstack([initial1, final1])  # Initial state + goal
#         state1 = torch.tensor(state1, dtype=torch.float32).unsqueeze(0)
#         traj1 = [initial1]

#         state2 = np.hstack([initial2, final2])  # Initial state + goal
#         state2 = torch.tensor(state2, dtype=torch.float32).unsqueeze(0)
#         traj2 = [initial2]

#         for _ in range(100 - 1):  # 100 steps total
#             next_state1 = model1(state1).numpy().squeeze()
#             traj1.append(next_state1)
#             state1 = torch.tensor(np.hstack([next_state1, final1]), dtype=torch.float32).unsqueeze(0)

#             next_state2 = model2(state2).numpy().squeeze()
#             traj2.append(next_state2)
#             state2 = torch.tensor(np.hstack([next_state2, final2]), dtype=torch.float32).unsqueeze(0)

#     generated_trajectories1.append(np.array(traj1))
#     # np.save(f"sampled_trajs/bc/mpc_traj1_{i}.npy", np.array(traj1))
#     generated_trajectories2.append(np.array(traj2))
#     # np.save(f"sampled_trajs/bc/mpc_traj2_{i}.npy", np.array(traj2))


# # Plotting
# plt.figure(figsize=(20, 8))
# for i in range(len(generated_trajectories1)):
#     traj1 = generated_trajectories1[i]
#     traj2 = generated_trajectories2[i]
#     plt.plot(traj1[:, 0], traj1[:, 1], 'b-', alpha=0.5)
#     plt.plot(traj2[:, 0], traj2[:, 1], 'C1-', alpha=0.5)
#     plt.scatter(traj1[0, 0], traj1[0, 1], c='green', s=10)  # Start point
#     plt.scatter(traj1[-1, 0], traj1[-1, 1], c='red', s=10)  # End point
#     plt.scatter(traj2[0, 0], traj2[0, 1], c='green', s=10)  # Start point
#     plt.scatter(traj2[-1, 0], traj2[-1, 1], c='red', s=10)  # End point

# ox, oy, r = obstacle
# circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
# plt.gca().add_patch(circle)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()