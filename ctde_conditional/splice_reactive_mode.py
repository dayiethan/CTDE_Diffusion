import torch
import numpy as np
from utils.utils import Normalizer, set_seed
from utils.conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.discrete import *
import sys
import pdb
import csv
from utils.mpc_util import splice_plan_mode_safe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
T = 100 # Trajectory horizon
H = 10 # Planning horizon
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

with open("data/mean_reactive.npy", "rb") as f:
    mean = np.load(f)
with open("data/std_reactive.npy", "rb") as f:
    std = np.load(f)
with open("data/sigma_data_reactive.npy", "rb") as f:
    sig = np.load(f)


# Training
action_cond_ode = Conditional_ODE(env, [9, 9], sig.tolist(), device=device, N=100, n_models = 2, lin_scale = 128, **model_size)
action_cond_ode.load(extra="_T10_reactivemode")


# --- 2. MPC Planning and Video Generation ---

for i in range(10):
    noise_std = 0.4
    initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
    initial1 = (initial1 - mean) / std
    final1 = final_point_up + noise_std * np.random.randn(*np.shape(final_point_up))
    final1 = (final1 - mean) / std
    initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
    initial2 = (initial2 - mean) / std
    final2 = final_point_down + noise_std * np.random.randn(*np.shape(final_point_down))
    final2 = (final2 - mean) / std
    mode = 6

    planned_trajs = splice_plan_mode_safe(action_cond_ode, env, [initial1, initial2], [final1, final2], mode, segment_length=H, total_steps=T)
    planned_traj1 = planned_trajs[0] * std + mean
    planned_traj2 = planned_trajs[1] * std + mean

    # Plot the planned trajectory:
    plt.figure(figsize=(22, 14))
    plt.ylim(-7, 7)
    plt.xlim(-1,21)
    plt.plot(planned_traj1[:, 0], planned_traj1[:, 1], 'b.-')
    plt.plot(planned_traj2[:, 0], planned_traj2[:, 1], 'o-', color='red')
    ox, oy, r = obstacle
    circle1 = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
    plt.gca().add_patch(circle1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MPC Planned Trajectory")
    plt.savefig("figs/mpc/reactive_mode/mode%s/mpc_traj_%s.png" % (mode, i))
    plt.show()

    # Generate a video of the planning process:
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-7, 7)
    circle2 = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
    ax.add_patch(circle2)
    line, = ax.plot([], [], 'b-', lw=2, label="Traj 1")
    markers, = ax.plot([], [], 'bo', markersize=8)
    line2, = ax.plot([], [], 'r-', lw=2, label="Traj 2")
    markers2, = ax.plot([], [], 'ro', markersize=8)
    title = ax.text(0.5, 1.05, "MPC Planning", transform=ax.transAxes, ha="center")


    def init():
        line.set_data([], [])
        return line, title

    def update(frame):
        # Update the first trajectory.
        line.set_data(planned_traj1[:frame, 0], planned_traj1[:frame, 1])
        if frame >= 10:
            indices = np.arange(0, frame, 10)
            if indices[-1] != frame - 1:
                indices = np.append(indices, frame - 1)
        else:
            indices = [0]
        markers.set_data(planned_traj1[indices, 0], planned_traj1[indices, 1])
        
        # Update the second trajectory.
        line2.set_data(planned_traj2[:frame, 0], planned_traj2[:frame, 1])
        if frame >= 10:
            indices2 = np.arange(0, frame, 10)
            if indices2[-1] != frame - 1:
                indices2 = np.append(indices2, frame - 1)
        else:
            indices2 = [0]
        markers2.set_data(planned_traj2[indices2, 0], planned_traj2[indices2, 1])
        
        title.set_text(f"Step {frame}")
        return line, markers, line2, markers2, title



    ani = animation.FuncAnimation(fig, update, frames=len(planned_traj1), init_func=init,
                                blit=True, interval=50)


    ani.save("figs/mpc/reactive_mode/mode%s/mpc_ani_%s.mp4" % (mode, i), writer="ffmpeg", fps=12)
    plt.close()
    print("MPC planning and video generation complete.")