import torch
import numpy as np
import matplotlib.pyplot as plt
from conditional_Action_DiT import Conditional_ODE
from discrete import *
import sys

from diffuser.models.ma_temporal import ConvAttentionDeconv
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.utils.training import Trainer
from diffuser.datasets.sequence import SequenceDataset  
from diffuser.utils.evaluator import MADEvaluator
from diffuser.utils.swap_rendering import SwapRenderer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoUnicycle():
    def __init__(self, state_size=10, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"

env = TwoUnicycle()


conv_model = ConvAttentionDeconv(
    horizon=100,
    transition_dim=env.state_size,
    dim=128,
    history_horizon=0,
    dim_mults=(1, 2, 4, 8),
    n_agents=2,
    returns_condition=False,
    env_ts_condition=False,
    condition_dropout=0.1,
    kernel_size=5,
    residual_attn=False,
    use_layer_norm=False,
    max_path_length=100,
    use_temporal_attention=False
)

diff_model = GaussianDiffusion(
        model=conv_model,
        n_agents=2,                  # adjust as necessary
        horizon=100,
        history_horizon=0,
        observation_dim=env.state_size,
        action_dim=env.action_size,
        use_inv_dyn=True,
        discrete_action=False,
        n_timesteps=1000,            # can adjust
        clip_denoised=True,
        predict_epsilon=True,
        action_weight=10.0,
        hidden_dim=256,
        loss_discount=1.0,
        returns_condition=False,
        condition_guidance_w=1.2
    )

data = torch.load("checkpoints/checkpoint/state_99000.pt")
diff_model.load_state_dict(data['model'])
diff_model.to(device)

diff_model.eval()
diff_model.set_ddim_scheduler(n_ddim_steps=15)

horizon = diff_model.horizon + diff_model.history_horizon
batch_size = 16  # or any batch size you are using
dummy_x = torch.zeros(batch_size, horizon, diff_model.n_agents, diff_model.observation_dim, device=device)
dummy_masks = torch.zeros(batch_size, horizon, diff_model.n_agents, diff_model.observation_dim, device=device, dtype=torch.bool)


dummy_condition = {
    "x": dummy_x,
    "masks": dummy_masks,
}
with torch.no_grad():
    samples = diff_model.conditional_sample(
        cond=dummy_condition,
        verbose=True,
        return_diffusion=False
    )

import matplotlib.pyplot as plt
import numpy as np

# Convert the sample to numpy (ensure it's on CPU)
samples_np = samples.cpu().numpy()

# Choose one trajectory from the batch and one agent
# samples_np shape: (batch_size, time_horizon, n_agents, state_dim)
trajectory = samples_np[0, :, 0, :]  # shape (100, 10)
time = np.arange(trajectory.shape[0])

plt.figure(figsize=(20, 8))
trajectory1_a1 = samples_np[0, :, 0, :]
for traj in trajectory:
    plt.plot(traj[0, 4], traj[0, 5], 'go', markersize=8)
    plt.plot(traj[0, 7], traj[0, 8], 'go', markersize=8)
plt.savefig("test0.5.png")

plt.figure(figsize=(20, 8))
for traj in trajectory:
    plt.plot(traj[0, 4], traj[0, 5], 'bo', markersize=8)
    plt.plot(traj[0, 7], traj[0, 8], 'bo', markersize=8)
plt.savefig("test0.7.png")

plt.figure(figsize=(10, 6))
for i in range(trajectory.shape[1]):
    plt.plot(time, trajectory[:, i], label=f"Feature {i}")
plt.xlabel("Time Step")
plt.ylabel("Feature Value")
plt.title("State Features Over Time for Agent 0 (Sample 0)")
plt.legend()
plt.savefig("fig")
