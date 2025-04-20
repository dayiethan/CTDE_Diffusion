import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt

class DQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_states, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out = nn.Linear(64,n_actions)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = self.out(x)
        return x

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def eps_decay(itr,num_episodes, eps_start, eps_end):
    if itr>=num_episodes*100:
        return eps_end
    else:
        return (eps_end - eps_start)*itr/(num_episodes*100) + eps_start

def action(state, eps, n_actions, policy_net):
    p = np.random.random()
    if p < eps:
        a = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
    else:
        a = policy_net(state).max(1)[1].view(1, 1)

    return a


def update_network(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma):
    if len(memory) < batch_size:
        return

    batch_data = memory.sample(batch_size)

    batch = Transition(*zip(*batch_data))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)


    Q_s_a = policy_net(state_batch).gather(1, action_batch)

    y_js = torch.zeros(batch_size)

    with torch.no_grad():
        for i in range(batch_size):
            if batch.done[i]:
                y_js[i] = reward_batch[i]
            else:
                y_js[i] = reward_batch[i] + gamma*target_net(next_state_batch[i]).max()

    criterion = nn.MSELoss()

    loss = criterion(Q_s_a, y_js.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_network(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions,
                    n_states, batch_size, eps_start, eps_end, gamma, Transition):
    reward_array = np.zeros(num_episodes)
    itr = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        s = env.reset()
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        done = False
        time = 0
        while not done:
            eps = eps_decay(itr,num_episodes, eps_start, eps_end)
            a = action(s, eps, n_actions, policy_net)
            s_next, r, done = env.step(a.item())

            r = torch.tensor([r])
            reward_array[i_episode] += gamma**(time)*r
            time += 1

            s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)

            memory.push(s, a, s_next, r, done)

            # Move to the next state
            s = s_next

            # Perform one step of the optimization (on the policy network)
            update_network(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma)

            itr+=1

            if itr%1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

    return policy_net, reward_array

def train_network_without_targ(num_episodes, policy_net, target_net, optimizer, memory, env,
                                n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition):
    reward_array = np.zeros(num_episodes)
    itr = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        s = env.reset()
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        done = False
        time = 0
        while not done:
            eps = eps_decay(itr,num_episodes, eps_start, eps_end)
            a = action(s, eps, n_actions, policy_net)
            s_next, r, done = env.step(a.item())
            r = torch.tensor([r])

            reward_array[i_episode] += gamma**(time)*r
            time += 1

            s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)

            memory.push(s, a, s_next, r, done)

            # Move to the next state
            s = s_next

            # Perform one step of the optimization (on the policy network)
            update_network(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma)

            itr+=1

            #if itr%1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, reward_array
