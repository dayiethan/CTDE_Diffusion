import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from discrete import *
import sys
import pdb
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mpc_plan(ode_model, env, initial_state, fixed_goal, model_i, segment_length=H, total_steps=T):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being trained
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_state = initial_state.copy()
    n_segments = total_steps // segment_length
    for seg in range(n_segments):
        cond = np.hstack([current_state, fixed_goal])
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
        sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=model_i)
        segment = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

        if seg == 0:
            full_traj.extend(segment)
        else:
            full_traj.extend(segment[1:])

        current_state = segment[-1]
    return np.array(full_traj)

def mpc_plan_multi(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments.
    Each agent’s condition is built as:
      [ own current state, own goal, other_agent_1 current state, other_agent_1 goal, ... ]
      
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size) with each agent's current state.
      - fixed_goals: numpy array of shape (n_agents, state_size) with each agent's final desired state.
      - segment_length: number of timesteps planned per segment.
      - total_steps: total number of timesteps for the full trajectory.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every segment
    full_segments = []
    n_segments = total_steps // segment_length

    # Loop over planning segments.
    for seg in range(n_segments):
        seg_trajectories = []
        # For each agent, build its own condition and sample a trajectory segment.
        for i in range(n_agents):
            # Start with agent i's current state and goal.
            cond = [current_states[i], fixed_goals[i]]
            # Append other agents' current state and goal.
            for j in range(n_agents):
                if j != i:
                    cond.append(current_states[j])
                    cond.append(fixed_goals[j])
            # Create a 1D condition vector for agent i.
            cond_vector = np.hstack(cond)
            # Convert to tensor.
            cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
            # Sample a segment for agent i.
            sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
            seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)
            seg_trajectories.append(seg_i)
            # Update current state for agent i (using the last state from the segment)
            current_states[i] = seg_i[-1]
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    # Optionally, transpose so that time is the first dimension:
    # full_traj = np.transpose(full_traj, (1, 0, 2))
    return full_traj