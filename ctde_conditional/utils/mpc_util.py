import torch
import numpy as np
from utils.conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.discrete import *
import sys
import pdb
import csv
from utils.gmm import expert_likelihood
from joblib import dump, load
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splice_plan(ode_model, env, initial_state, fixed_goal, model_i, segment_length=10, total_steps=100):
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

def splice_plan_multi(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
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

def splice_plan_safe(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments with a safety filter.
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
    gmm = load("expert_gmm.pkl")

    # Loop over planning segments.
    for seg in range(n_segments):
        valid_segment = False
        while not valid_segment:
          seg_trajectories = []
          current_states_temp = initial_states.copy()
          # For each agent, build its own condition and sample a trajectory segment.
          likely_vec = np.zeros(n_agents*2)
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
              likely_vec[i*2] = seg_i[-1][0]
              likely_vec[i*2 + 1] = seg_i[-1][1]
              seg_trajectories.append(seg_i)
              # Update current state for agent i (using the last state from the segment)
              current_states_temp[i] = seg_i[-1]
              # current_states[i] = seg_i[-1]
          prob = expert_likelihood(gmm, likely_vec)
          print(prob)
          if prob > 0.045:
              print("valid")
              valid_segment = True
              for i in range(n_agents):
                  current_states[i] = current_states_temp[i]
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    return full_traj

def splice_plan_mode_safe(ode_model, env, initial_states, fixed_goals, mode, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments with a safety filter.
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
    gmm = load("expert_gmm.pkl")

    # Loop over planning segments.
    for seg in range(n_segments):
        valid_segment = False
        while not valid_segment:
          seg_trajectories = []
          current_states_temp = initial_states.copy()
          # For each agent, build its own condition and sample a trajectory segment.
          likely_vec = np.zeros(n_agents*2)
          for i in range(n_agents):
              # Start with agent i's current state and goal.
              cond = [current_states[i], fixed_goals[i]]
              # Append other agents' current state and goal.
              for j in range(n_agents):
                  if j != i:
                      cond.append(current_states[j])
                      cond.append(fixed_goals[j])
              cond.append(mode)
              # Create a 1D condition vector for agent i.
              cond_vector = np.hstack(cond)
              # Convert to tensor.
              cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
              # Sample a segment for agent i.
              sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
              seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)
              likely_vec[i*2] = seg_i[-1][0]
              likely_vec[i*2 + 1] = seg_i[-1][1]
              seg_trajectories.append(seg_i)
              # Update current state for agent i (using the last state from the segment)
              current_states_temp[i] = seg_i[-1]
              # current_states[i] = seg_i[-1]
          prob = expert_likelihood(gmm, likely_vec)
          print(prob)
          if prob > 0.045:
              print("valid")
              valid_segment = True
              for i in range(n_agents):
                  current_states[i] = current_states_temp[i]
          else:
              mode = np.random.randint(0, 6) + 1
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    return full_traj

def splice_plan_mode_multi(ode_model, env, initial_states, fixed_goals, mode, segment_length=10, total_steps=100):
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
            cond.append(mode)
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

def collision_cost(traj, obstacles, safety_margin=0.5):
    # traj: (segment_length, state_size) trajectory
    # obstacle: tuple (ox, oy, r)
    # Compute cost as, for example, inverse of the distance to the obstacle at each timestep.
    costs = []
    for obstacle in obstacles:
      cost = 0
      ox, oy, r = obstacle
      for state in traj:
        x, y = state[:2]
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        # If within the safety margin, add a high penalty
        if dist < safety_margin:
            cost += 1e3
        else:
            cost += 1.0 / dist  # lower cost for further states
      costs.append(cost)
    return max(costs)

def splice_plan_multi_safe(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100, n_candidates=5):
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # update each segment
    full_segments = []
    n_segments = total_steps // segment_length
    gmm = load('expert_gmm.pkl')

    for seg in range(n_segments):
        seg_trajectories = []
        obstacles = []
        for i in range(n_agents):
            best_traj = None
            best_cost = float('inf')
            # Sample multiple candidates
            for _ in range(n_candidates):
                cond = [current_states[i], fixed_goals[i]]
                for j in range(n_agents):
                    if j != i:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                        obstacle = (current_states[j][0], current_states[j][1], 2)
                        obstacles.append(obstacle)
                cond_vector = np.hstack(cond)
                cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                candidate = sampled.cpu().detach().numpy()[0]
                breakpoint()
                cost = collision_cost(candidate, obstacles)
                if cost < best_cost:
                    best_cost = cost
                    best_traj = candidate
            seg_trajectories.append(best_traj)
            current_states[i] = best_traj[-1]
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)
    full_traj = np.concatenate(full_segments, axis=1)
    return full_traj


def splice_plan_multi_true(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
    """
    True MPC: At each step, plan a full segment but only execute the first step.
    
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size).
      - fixed_goals: numpy array of shape (n_agents, state_size).
      - segment_length: how many steps we plan ahead (default 10).
      - total_steps: how many total steps to run.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every step
    full_traj = []

    for step in range(total_steps):
        next_states = []
        # For each agent, plan a 10-step trajectory, but we'll only take the first action.
        for i in range(n_agents):
            # Build the condition for agent i
            cond1 = np.hstack([current_states[0], fixed_goals[0]])
            cond2 = np.hstack([current_states[1], fixed_goals[1]])
            cond1_tensor = torch.tensor(cond1, dtype=torch.float32, device=device).unsqueeze(0)
            cond2_tensor = torch.tensor(cond2, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Sample a full segment
            sampled = ode_model.sample(attr=[cond1_tensor, cond2_tensor], traj_len=segment_length, n_samples=1, w=1., model_index=i)
            seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)

            # Take only the first step
            next_state_i = seg_i[1]
            next_states.append(next_state_i)

        # Update current states
        current_states = np.array(next_states)
        # Save the executed states
        full_traj.append(current_states)

    full_traj = np.stack(full_traj, axis=1)  # Shape: (total_steps, n_agents, state_size)
    return full_traj


def mpc_plan(ode_model, env, initial_state, fixed_goal, model_i, leader_traj_cond = None, segment_length=10, total_steps=100):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_state = initial_state.copy()
    n_segments = total_steps // segment_length

    for seg in range(100):
        if leader_traj_cond is not None:
            cond = np.hstack([current_state, fixed_goal, leader_traj_cond.flatten()])
        else:
            cond = np.hstack([current_state, fixed_goal])
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
        sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=model_i)
        segment = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

        next_state_i = segment[1]
        full_traj.append(next_state_i)

        current_state = next_state_i
    return np.array(full_traj)


def reactive_mpc_plan(ode_model, env, initial_states, fixed_goals, model_i, segment_length=25, total_steps=100, n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i and j != 0:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                cond.append(current_states[0])
                cond.append(fixed_goals[0])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)

def reactive_mpc_latent_plan(ode_model, env, initial_states, fixed_goals, encoders, segment_length=10, latent_length=10, total_steps=100, n_implement=1):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    Conditioned on the latent space representation of the history of the agents.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    n_agents = len(initial_states)
    # full trajectories per agent
    full_traj = [ [] for _ in range(n_agents) ]
    # current state & history buffer per agent
    current = [s.copy() for s in initial_states]
    history = [ [s.copy()] for s in initial_states ]

    steps = total_steps // n_implement
    for seg_idx in range(steps):
        # plan one segment for each agent
        next_states = []
        for i in range(n_agents):
            # 1) build latent window from history[i]
            hist = np.stack(history[i], axis=0)              # (t, state_dim)
            if hist.shape[0] < latent_length:
                pad = np.repeat(hist[0:1], latent_length - hist.shape[0], axis=0)
                latent_win = np.concatenate([pad, hist], axis=0)
            else:
                latent_win = hist[-latent_length:]
            # encode → z (1, latent_dim)
            with torch.no_grad():
                z = encoders[i](
                    torch.from_numpy(latent_win[None]).float().to(device)
                )  # shape: (1, latent_dim)
            z_np = z.cpu().numpy().reshape(-1)  # (latent_dim,)

            # 2) build cond vector per your training spec
            if i == 0:
                # leader cond: [curr_leader, goal_leader, curr_follower, z]
                cond_list = [
                    current[0], fixed_goals[0],
                    current[1]
                ]
            else:
                # follower cond: [curr_foll, goal_foll, leader_5ahead, z]
                # compute leader's 5-ahead index in history[0]
                idx5 = min(len(history[0]) - 1, seg_idx * n_implement + 5)
                leader_5 = history[0][idx5]
                cond_list = [
                    current[1], fixed_goals[1],
                    leader_5
                ]
            cond_list.append(z_np)

            cond = np.hstack(cond_list)   # (cond_dim,)
            cond_t = torch.from_numpy(cond[None]).float().to(device)  # (1,cond_dim)

            # 3) sample full segment
            sampled = ode_model.sample(
                attr=cond_t,
                traj_len=segment_length,
                n_samples=1,
                w=1.0,
                model_index=i
            )
            seg_traj = sampled.cpu().numpy()[0]  # (segment_length, state_dim)

            # 4) implement the right slice
            if seg_idx == 0:
                take = seg_traj[0:n_implement]
            else:
                # skip the first point so we don't repeat
                take = seg_traj[1: 1 + n_implement]

            # advance
            next_state = take[-1]
            full_traj[i].append(take)
            next_states.append(next_state)

        # commit next_states → current & history
        for i in range(n_agents):
            current[i] = next_states[i].copy()
            history[i].append(next_states[i].copy())

    # concatenate each agent's segments into (total_steps, state_dim)
    full_traj = [ np.concatenate(blocks, axis=0) for blocks in full_traj ]
    # stack agents → shape (n_agents, total_steps, state_dim)
    return np.stack(full_traj, axis=0)