import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from discrete import *
from just_GNN import *
import sys
import pdb
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
T = 100 # Trajectory horizon
H = 100 # Planning horizon
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)


# Define environment
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"
env = TwoUnicycle()

with open("data/mean.npy", "rb") as f:
    mean = np.load(f)
with open("data/std.npy", "rb") as f:
    std = np.load(f)
with open("data/sigma_data.npy", "rb") as f:
    sig = np.load(f)


two_agents_gnn = GNNWithAgentPolicy(input_dim=4, hidden_dim=256, embedding_dim=4, horizon=H, action_dim=2)
trainer = GNNBaselineTrainerDecentralized(two_agents_gnn, device=device)

# trainer.train([actions1, actions2], [attr1, attr2], n_gradient_steps=100000)
# trainer.save()

trainer.load(path='trained_models/gnn_baseline_H_100_decentralized.pt')

def mpc_plan(ode_model, env, initial_states, fixed_goals, model_i, segment_length=H, total_steps=T):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()
    n_segments = total_steps // segment_length
    for seg in range(n_segments):
        cond1 = np.hstack([current_states[0], fixed_goals[0]])
        cond2 = np.hstack([current_states[1], fixed_goals[1]])
        cond1_tensor = torch.tensor(cond1, dtype=torch.float32, device=device).unsqueeze(0)
        cond2_tensor = torch.tensor(cond2, dtype=torch.float32, device=device).unsqueeze(0)
        sampled = ode_model.sample(attr=[cond1_tensor, cond2_tensor], traj_len=segment_length, n_samples=1, w=1., model_index=model_i)
        segment = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

        if seg == 0:
            full_traj.extend(segment)
        else:
            full_traj.extend(segment[1:])

        current_state = segment[-1]
    return np.array(full_traj)

def mpc_plan_multi(trainer, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
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

        cond1 = np.hstack([current_states[0], fixed_goals[0]])
        cond2 = np.hstack([current_states[1], fixed_goals[1]])
        cond1_tensor = torch.tensor(cond1, dtype=torch.float32, device=device).unsqueeze(0)
        cond2_tensor = torch.tensor(cond2, dtype=torch.float32, device=device).unsqueeze(0)

        # sampled_traj = trainer.sample_trajectories(cond1_tensor, cond2_tensor)

        # sampled1 = sampled_traj[0,:,:]
        # sampled2 = sampled_traj[1,:,:]

        sampled1 = trainer.sample_trajectory([cond1_tensor, cond2_tensor], agent=0)[0,:,:]
        sampled2 = trainer.sample_trajectory([cond1_tensor, cond2_tensor], agent=1)[0,:,:]

        current_states[0] = sampled1[-1,:]
        current_states[1] = sampled2[-1,:]

        seg_trajectories.append(sampled1)
        seg_trajectories.append(sampled2)

        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    # Optionally, transpose so that time is the first dimension:
    # full_traj = np.transpose(full_traj, (1, 0, 2))
    return full_traj


# --- 2. MPC Planning and Video Generation ---

for i in range(100):
    noise_std = 0.2
    initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
    initial1 = (initial1 - mean) / std
    final1 = final_point_up + noise_std * np.random.randn(*np.shape(final_point_up))
    final1 = (final1 - mean) / std
    initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
    initial2 = (initial2 - mean) / std
    final2 = final_point_down + noise_std * np.random.randn(*np.shape(final_point_down))
    final2 = (final2 - mean) / std

    # planned_traj1 = mpc_plan(action_cond_ode, env, [initial1, initial2], [final1, final2], 0, segment_length=H, total_steps=T)
    # planned_traj1 = planned_traj1 * std + mean

    # planned_traj2 = mpc_plan(action_cond_ode, env, [initial1, initial2], [final1, final2], 1, segment_length=H, total_steps=T)
    # planned_traj2 = planned_traj2 * std + mean

    planned_trajs = mpc_plan_multi(trainer, env, [initial2, initial1], [final2, final1], segment_length=H, total_steps=T)

    planned_traj1 = planned_trajs[0] * std + mean
    planned_traj2 = planned_trajs[1] * std + mean

    initial1 = planned_traj1[-1,:]
    initial2 = planned_traj2[-1,:]

    # # Save the planned trajectories to a CSV file:

    save_folder = "data/H_100_noise"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # convert to numpy array

    planned_traj1 = np.array(planned_traj1)
    planned_traj2 = np.array(planned_traj2)

    np.savetxt(os.path.join(save_folder, f"mlp_gnn_planned_noise_0_2_traj1_{i}.csv"), planned_traj1, delimiter=",")
    np.savetxt(os.path.join(save_folder, f"mlp_gnn_planned_noise_0_2_traj2_{i}.csv"), planned_traj2, delimiter=",")


    # # Plot the planned trajectory:
    # plt.figure(figsize=(22, 14))
    # plt.ylim(-7, 7)
    # plt.xlim(-1,21)
    # plt.plot(planned_traj1[:, 0], planned_traj1[:, 1], 'b.-')
    # plt.plot(planned_traj2[:, 0], planned_traj2[:, 1], 'o-', color='orange')
    # ox, oy, r = obstacle
    # circle1 = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
    # plt.gca().add_patch(circle1)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("MPC Planned Trajectory")
    # plt.savefig("figs/mpc_just_gnn/mpc_traj_%s.png" % i)
    # plt.show()

    # # Generate a video of the planning process:
    # fig, ax = plt.subplots(figsize=(22, 14))
    # ax.set_xlim(-1, 21)
    # ax.set_ylim(-7, 7)
    # circle2 = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
    # ax.add_patch(circle2)
    # line, = ax.plot([], [], 'b-', lw=2, label="Traj 1")
    # markers, = ax.plot([], [], 'bo', markersize=8)
    # line2, = ax.plot([], [], 'r-', lw=2, label="Traj 2")
    # markers2, = ax.plot([], [], 'ro', markersize=8)
    # title = ax.text(0.5, 1.05, "MPC Planning", transform=ax.transAxes, ha="center")


    # def init():
    #     line.set_data([], [])
    #     return line, title

    # def update(frame):
    #     # Update the first trajectory.
    #     line.set_data(planned_traj1[:frame, 0], planned_traj1[:frame, 1])
    #     if frame >= 10:
    #         indices = np.arange(0, frame, 10)
    #         if indices[-1] != frame - 1:
    #             indices = np.append(indices, frame - 1)
    #     else:
    #         indices = [0]
    #     markers.set_data(planned_traj1[indices, 0], planned_traj1[indices, 1])
        
    #     # Update the second trajectory.
    #     line2.set_data(planned_traj2[:frame, 0], planned_traj2[:frame, 1])
    #     if frame >= 10:
    #         indices2 = np.arange(0, frame, 10)
    #         if indices2[-1] != frame - 1:
    #             indices2 = np.append(indices2, frame - 1)
    #     else:
    #         indices2 = [0]
    #     markers2.set_data(planned_traj2[indices2, 0], planned_traj2[indices2, 1])
        
    #     title.set_text(f"Step {frame}")
    #     return line, markers, line2, markers2, title



    # ani = animation.FuncAnimation(fig, update, frames=len(planned_traj1), init_func=init,
    #                             blit=True, interval=50)


    # ani.save("figs/mpc/mpc_ani_%s.mp4" % i, writer="ffmpeg", fps=12)
    # plt.close()
    print("MPC planning and video generation complete.")


