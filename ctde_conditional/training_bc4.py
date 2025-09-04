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

# Define initial and final points, and a single central obstacle
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Expert demonstration loading
expert_data_1 = np.load('data/expert_data1_100_traj.npy')
expert_data_2 = np.load('data/expert_data2_100_traj.npy')
# import numpy as np, matplotlib.pyplot as plt

# for xy in expert_data_1: plt.plot(xy[:,0], xy[:,1], color='blue')
# for xy in expert_data_2: plt.plot(xy[:,0], xy[:,1], color='orange')
# plt.axis("equal"); plt.xlabel("x"); plt.ylabel("y"); plt.show()

X_train1 = []
Y_train1 = []
X_train2 = []
Y_train2 = []

for i in range(len(expert_data_1)):
    for j in range(len(expert_data_1[i]) - 1):
        X_train1.append(np.hstack([expert_data_1[i][j], expert_data_2[i][j]]))  # Current state + goal
        Y_train1.append(expert_data_1[i][j + 1])  # Next state
X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32)  # Shape: (N, 4)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32)  # Shape: (N, 2)

for i in range(len(expert_data_2)):
    for j in range(len(expert_data_2[i]) - 1):
        X_train2.append(np.hstack([expert_data_2[i][j], expert_data_1[i][j]]))  # Current state + goal
        Y_train2.append(expert_data_2[i][j + 1])  # Next state
X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 4)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizers
model1 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=24)
model2 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=24)
print(f"Total parameters: {sum(p.numel() for p in model1.parameters()) + sum(p.numel() for p in model2.parameters())}")
criterion = nn.MSELoss()  # Mean Squared Error Loss
all_params = []
all_params += list(model1.parameters())
all_params += list(model2.parameters())
optimizer = optim.Adam(all_params, lr=0.001, weight_decay=1e-4)

model1, model2 = model1.to(device), model2.to(device)
X_train1, Y_train1 = X_train1.to(device), Y_train1.to(device)
X_train2, Y_train2 = X_train2.to(device), Y_train2.to(device)

# Train the Model
def train_model(model, optimizer, criterion, X_train, Y_train, num_epochs=5000):
    losses = []

    for epoch in range(num_epochs):
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model, losses

def joint_train(model1, model2, optimizer, criterion, X_train1, Y_train1, X_train2, Y_train2, num_epochs=20000):
    losses1 = []
    losses2 = []

    for epoch in range(num_epochs):
        loss_total = 0.0

        predictions1 = model1(X_train1)
        loss1 = criterion(predictions1, Y_train1)
        loss_total += loss1

        predictions2 = model2(X_train2)
        loss2 = criterion(predictions2, Y_train2)
        loss_total += loss2

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        losses1.append(loss1.item())
        losses2.append(loss2.item())

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')

    return model1, model2, losses1, losses2

# trained_model1, losses1 = train_model(model1, optimizer1, criterion, X_train1, Y_train1)
# trained_model2, losses2 = train_model(model2, optimizer2, criterion, X_train2, Y_train2)
trained_model1, trained_model2, losses1, losses2 = joint_train(model1, model2, optimizer, criterion, X_train1, Y_train1, X_train2, Y_train2)

                      
save_path1 = "trained_models/bc/crosscond_1.pth"
save_path2 = "trained_models/bc/crosscond_2.pth"
torch.save(model1.state_dict(), save_path1)
torch.save(model2.state_dict(), save_path2)

model1 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=24)
model1.load_state_dict(torch.load(save_path1, map_location='cpu'))
model1.eval()

model2 = BigImitationNet(input_size=4, hidden_size=256, output_size=2, horizon=24)
model2.load_state_dict(torch.load(save_path2, map_location='cpu'))
model2.eval()

# Generate a New Trajectory Using the Trained Model
noise_std = 0.1
generated_trajectories1 = []
generated_trajectories2 = []

for i in range(100):
    print(f"Generating trajectory {i + 1}")
    initial1 = initial_point1 + noise_std * np.random.randn(*np.shape(initial_point1))
    final1 = final_point1 + noise_std * np.random.randn(*np.shape(final_point1))
    initial2 = initial_point2 + noise_std * np.random.randn(*np.shape(initial_point2))
    final2 = final_point2 + noise_std * np.random.randn(*np.shape(final_point2))
    with torch.no_grad():
        # keep single 2D state as input
        state1 = torch.tensor(initial1, dtype=torch.float32).unsqueeze(0)
        state2 = torch.tensor(initial2, dtype=torch.float32).unsqueeze(0)
        traj1  = [initial1.copy()]
        traj2  = [initial2.copy()]

        for _ in range(100 - 1):  # total steps
            out1 = model1(state1)                   # (1, 24, 2) or (1, 2) if horizon=1
            out2 = model2(state2)

            step1 = out1[:, 0, :] if out1.dim() == 3 else out1   # take FIRST step only
            step2 = out2[:, 0, :] if out2.dim() == 3 else out2

            next_state1 = step1.squeeze(0).detach().cpu().numpy()  # (2,)
            next_state2 = step2.squeeze(0).detach().cpu().numpy()

            traj1.append(next_state1.copy())
            traj2.append(next_state2.copy())

            state1 = torch.from_numpy(next_state1).float().unsqueeze(0)  # back to (1,2)
            state2 = torch.from_numpy(next_state2).float().unsqueeze(0)


    generated_trajectories1.append(np.array(traj1))
    np.save(f"sampled_trajs/bc_crosscond/mpc_traj1_{i}.npy", np.array(traj1))
    generated_trajectories2.append(np.array(traj2))
    np.save(f"sampled_trajs/bc_crosscond/mpc_traj2_{i}.npy", np.array(traj2))


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