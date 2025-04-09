import os
import math
import time
import torch
import einops
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from typing import Optional
import matplotlib.pyplot as plt
from torch_geometric.nn import GraphConv
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

class TwoAgentTrajectoryGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, horizon, action_dim):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        traj = self.decoder(x)
        traj = traj.view(-1, self.horizon, self.action_dim)
        return traj

# class TwoAgentTrajectoryGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embedding_dim, action_dim, horizon):
#         super().__init__()
#         self.horizon = horizon
#         self.action_dim = action_dim

#         # Encoder: GNN that processes initial state as node feature
#         self.conv1 = GraphConv(input_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, embedding_dim)

#         # Decoder: GNN that takes (embedding + current state) and predicts delta or next state
#         self.decoder_gnn = GraphConv(embedding_dim + action_dim, hidden_dim)
#         self.decoder_out = nn.Linear(hidden_dim, action_dim)

#     def forward(self, s0, edge_index):
#         """
#         s0: Initial states as node features (2 * batch_size, input_dim)
#         edge_index: Edge indices for the batched 2-agent graphs
#         """
#         # Encoder GNN
#         x = self.conv1(s0, edge_index)
#         x = F.relu(x)
#         embeddings = self.conv2(x, edge_index)
#         embeddings = F.relu(embeddings)  # shape: (2 * batch_size, embedding_dim)

#         # Rollout over horizon
#         traj = [s0]
#         state = s0

#         for t in range(self.horizon - 1):
#             decoder_input = torch.cat([embeddings], dim=-1)  # shape: (2 * batch_size, emb + action_dim)
#             h = self.decoder_gnn(decoder_input, edge_index)
#             h = F.relu(h)
#             delta = self.decoder_out(h)

#             next_state = state + delta  # or delta if you want absolute prediction
#             traj.append(next_state)
#             state = next_state

#         return torch.stack(traj, dim=1)  # (2 * batch_size, horizon, action_dim)

class GNNBaselineTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def train(self, x_normalized_list, attributes_list, n_gradient_steps, batch_size=32, time_limit=None):
        print("Training GNN baseline for trajectory prediction")
        assert len(x_normalized_list) == 2 and len(attributes_list) == 2, "Only supports 2-agent setup"

        N = x_normalized_list[0].shape[0]
        horizon = x_normalized_list[0].shape[1]
        action_dim = x_normalized_list[0].shape[2]
        attr_dim = attributes_list[0].shape[1]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        if time_limit is not None:
            t0 = time.time()
            print(f"Training time limit: {time_limit:.0f}s")

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=self.device)
        edge_index = edge_index.repeat(1, batch_size)
        offsets = (torch.arange(batch_size) * 2).to(self.device)
        edge_index = edge_index + offsets.repeat_interleave(2).unsqueeze(0)

        pbar = tqdm(range(n_gradient_steps))
        loss_avg = 0.0

        for step in range(n_gradient_steps):
            idx = np.random.randint(0, N, batch_size)

            attr1 = attributes_list[0][idx].to(self.device)
            attr2 = attributes_list[1][idx].to(self.device)
            x1 = x_normalized_list[0][idx].to(self.device)
            x2 = x_normalized_list[1][idx].to(self.device)

            node_features = torch.stack([attr1, attr2], dim=1).view(-1, attr_dim)
            true_traj = torch.stack([x1, x2], dim=1).view(-1, horizon, action_dim)

            pred_traj = self.model(node_features, edge_index)

            loss = loss_fn(pred_traj, true_traj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            if (step + 1) % 10 == 0:
                pbar.set_description(f"step {step+1} | loss {loss_avg / 10:.4f}")
                pbar.update(10)
                loss_avg = 0.0

            if time_limit is not None and time.time() - t0 > time_limit:
                print(f"Time limit reached at {time.time() - t0:.0f}s")
                break

        print("GNN baseline training complete!")

    def sample_trajectories(self, attr1, attr2):
        self.model.eval()
        with torch.no_grad():
            node_features = torch.cat([attr1, attr2], dim=0).to(self.device)  # (2, attr_dim)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=self.device)
            traj = self.model(node_features, edge_index)  # (2, horizon, action_dim)
        return traj


    def save(self, path='trained_models/gnn_baseline.pt'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"GNN baseline model saved to {path}")

    def load(self, path='trained_models/gnn_baseline.pt'):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            print(f"GNN baseline model loaded from {path}")
            return True
        else:
            print(f"File {path} does not exist. Skipping model loading.")
            return False