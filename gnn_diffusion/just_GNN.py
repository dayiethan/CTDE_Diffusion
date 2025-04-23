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

class NAgentGNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.relu(x)

class AgentPolicy(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, horizon, action_dim):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )

    def forward(self, embedding):
        traj = self.net(embedding)
        return traj.view(-1, self.horizon, self.action_dim)

class GNNWithAgentPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, horizon, action_dim):
        super().__init__()
        self.encoder = NAgentGNNEncoder(input_dim, hidden_dim, embedding_dim)
        self.policy = AgentPolicy(embedding_dim, hidden_dim, horizon, action_dim)

    def forward(self, node_features, edge_index, agent_idx):
        embeddings = self.encoder(node_features, edge_index)
        selected_embeddings = embeddings[agent_idx]
        return self.policy(selected_embeddings)

class GNNBaselineTrainerDecentralized:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def train(self, x_normalized_list, attributes_list, n_gradient_steps, batch_size=32, time_limit=None):
        print("Training Decentralized GNN policy for trajectory prediction")

        num_agents = len(x_normalized_list)
        N = x_normalized_list[0].shape[0]
        horizon = x_normalized_list[0].shape[1]
        action_dim = x_normalized_list[0].shape[2]
        attr_dim = attributes_list[0].shape[1]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        if time_limit is not None:
            t0 = time.time()
            print(f"Training time limit: {time_limit:.0f}s")

        pbar = tqdm(range(n_gradient_steps))
        loss_avg = 0.0

        for step in range(n_gradient_steps):
            idx = np.random.randint(0, N, batch_size)
            attrs = [attributes_list[i][idx].to(self.device) for i in range(num_agents)]
            trues = [x_normalized_list[i][idx].to(self.device) for i in range(num_agents)]

            node_features = torch.stack(attrs, dim=1).view(-1, attr_dim)

            edge_index = torch.combinations(torch.arange(num_agents), r=2).T
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            edge_index = edge_index.repeat(1, batch_size)
            offsets = (torch.arange(batch_size) * num_agents).to(self.device)
            edge_index = edge_index + offsets.repeat_interleave(num_agents * (num_agents - 1)).unsqueeze(0)

            optimizer.zero_grad()
            total_loss = 0.0

            for agent in range(num_agents):
                agent_idx = torch.arange(agent, batch_size * num_agents, num_agents).to(self.device)
                pred_traj = self.model(node_features, edge_index, agent_idx)
                loss = loss_fn(pred_traj, trues[agent])
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            loss_avg += total_loss.item()
            if (step + 1) % 10 == 0:
                pbar.set_description(f"step {step+1} | loss {loss_avg / 10:.4f}")
                pbar.update(10)
                loss_avg = 0.0

            if time_limit is not None and time.time() - t0 > time_limit:
                print(f"Time limit reached at {time.time() - t0:.0f}s")
                break

        print("Decentralized GNN training complete!")

    def sample_trajectory(self, attr_list, agent=0):
        self.model.eval()
        with torch.no_grad():
            node_features = torch.cat(attr_list, dim=0).to(self.device)
            num_agents = len(attr_list)
            edge_index = torch.combinations(torch.arange(num_agents), r=2).T
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(self.device)
            agent_idx = torch.tensor([agent], dtype=torch.long, device=self.device)
            traj = self.model(node_features, edge_index, agent_idx)
        return traj

    def save(self, path='trained_models/gnn_decentralized.pt'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"GNN decentralized model saved to {path}")

    def load(self, path='trained_models/gnn_decentralized.pt'):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            print(f"GNN decentralized model loaded from {path}")
            return True
        else:
            print(f"File {path} does not exist. Skipping model loading.")
            return False

# class TwoAgentGNNEncoder(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(TwoAgentGNNEncoder, self).__init__()
#         self.conv1 = GraphConv(input_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, output_dim)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return F.relu(x)  # Output separate embeddings for each agent

# class AgentPolicy(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, horizon, action_dim):
#         super().__init__()
#         self.horizon = horizon
#         self.action_dim = action_dim

#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, horizon * action_dim)
#         )

#     def forward(self, embedding):  # embedding: (batch_size, embedding_dim)
#         traj = self.net(embedding)  # (batch_size, horizon * action_dim)
#         return traj.view(-1, self.horizon, self.action_dim)

# class GNNWithAgentPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embedding_dim, horizon, action_dim):
#         super().__init__()
#         self.encoder = TwoAgentGNNEncoder(input_dim, hidden_dim, embedding_dim)
#         self.policy = AgentPolicy(embedding_dim, hidden_dim, horizon, action_dim)

#     def forward(self, node_features, edge_index, agent_idx):
#         """
#         node_features: (batch_size * num_agents, input_dim)
#         edge_index: (2, num_edges)
#         agent_idx: list or tensor of length batch_size with indices of the agent node to predict
#         """
#         embeddings = self.encoder(node_features, edge_index)  # (batch_size * num_agents, embedding_dim)
#         selected_embeddings = embeddings[agent_idx]            # (batch_size, embedding_dim)
#         return self.policy(selected_embeddings)                # (batch_size, horizon, action_dim)

# class GNNBaselineTrainerDecentralized:
#     def __init__(self, model, device='cpu'):
#         self.model = model
#         self.device = device
#         self.model.to(self.device)

#     def train(self, x_normalized_list, attributes_list, n_gradient_steps, batch_size=32, time_limit=None):
#         print("Training Decentralized GNN policy for trajectory prediction")
#         assert len(x_normalized_list) == 2 and len(attributes_list) == 2, "Only supports 2-agent setup"

#         N = x_normalized_list[0].shape[0]
#         horizon = x_normalized_list[0].shape[1]
#         action_dim = x_normalized_list[0].shape[2]
#         attr_dim = attributes_list[0].shape[1]

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#         loss_fn = nn.MSELoss()

#         if time_limit is not None:
#             t0 = time.time()
#             print(f"Training time limit: {time_limit:.0f}s")

#         pbar = tqdm(range(n_gradient_steps))
#         loss_avg = 0.0

#         for step in range(n_gradient_steps):
#             idx = np.random.randint(0, N, batch_size)

#             attr1 = attributes_list[0][idx].to(self.device)
#             attr2 = attributes_list[1][idx].to(self.device)
#             x1 = x_normalized_list[0][idx].to(self.device)
#             x2 = x_normalized_list[1][idx].to(self.device)

#             # Stack node features for both agents: shape (batch_size * 2, attr_dim)
#             node_features = torch.stack([attr1, attr2], dim=1).view(-1, attr_dim)

#             # Create edge_index for batch of 2-node graphs
#             edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=self.device)
#             edge_index = edge_index.repeat(1, batch_size)
#             offsets = (torch.arange(batch_size) * 2).to(self.device)
#             edge_index = edge_index + offsets.repeat_interleave(2).unsqueeze(0)

#             optimizer.zero_grad()
#             total_loss = 0.0

#             # Compute loss separately for both agents
#             for agent in range(2):
#                 agent_idx = torch.arange(agent, batch_size * 2, 2).to(self.device)  # Select agent nodes
#                 true_traj = [x1, x2][agent]  # Ground truth for current agent

#                 pred_traj = self.model(node_features, edge_index, agent_idx)
#                 loss = loss_fn(pred_traj, true_traj)
#                 total_loss += loss

#             total_loss.backward()
#             optimizer.step()

#             loss_avg += total_loss.item()
#             if (step + 1) % 10 == 0:
#                 pbar.set_description(f"step {step+1} | loss {loss_avg / 10:.4f}")
#                 pbar.update(10)
#                 loss_avg = 0.0

#             if time_limit is not None and time.time() - t0 > time_limit:
#                 print(f"Time limit reached at {time.time() - t0:.0f}s")
#                 break

#         print("Decentralized GNN training complete!")

#     def sample_trajectory(self, attr1, attr2, agent=0):
#         self.model.eval()
#         with torch.no_grad():
#             node_features = torch.cat([attr1, attr2], dim=0).to(self.device)  # (2, attr_dim)
#             edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=self.device)
#             agent_idx = torch.tensor([agent], dtype=torch.long, device=self.device)
#             traj = self.model(node_features, edge_index, agent_idx)  # (1, horizon, action_dim)
#         return traj

#     def save(self, path='trained_models/gnn_decentralized.pt'):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         torch.save(self.model.state_dict(), path)
#         print(f"GNN decentralized model saved to {path}")

#     def load(self, path='trained_models/gnn_decentralized.pt'):
#         if os.path.isfile(path):
#             self.model.load_state_dict(torch.load(path, map_location=self.device))
#             self.model.to(self.device)
#             print(f"GNN decentralized model loaded from {path}")
#             return True
#         else:
#             print(f"File {path} does not exist. Skipping model loading.")
#             return False


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