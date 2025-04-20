import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt

from DQN import *
import discreteaction_pendulum

def main():
    batch_size = 64
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    num_episodes = 100

    env = discreteaction_pendulum.Pendulum()

    n_actions = env.num_actions
    n_states = env.num_states

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
    memory = ReplayMemory(1000000, Transition)

    policy_net, reward_array = train_network(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition)

    plt.plot(reward_array)
    plt.savefig('figures/learning_curve.png')

    policy = lambda s: (policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)).item()

    env.video(policy, filename='figures/test_discreteaction_pendulum_1.gif')

if __name__ == '__main__':
    main()
