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
        action_weight=20.0,
        hidden_dim=256,
        loss_discount=0.99,
        returns_condition=False,
        condition_guidance_w=3.0
    )

data = torch.load("/mnt/data1/chendazhong/CTDE_Diffusion/madiff/madiff_model.pt")
diff_model.load_state_dict(data['model'])
diff_model.to(device)

diff_model.eval()
diff_model.set_ddim_scheduler(n_ddim_steps=50)

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
    print("Sample shape:", samples.shape)
    print("First sample values:", samples[0])

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
trajectory1_a2 = samples_np[0, :, 1, :]
plt.plot(trajectory1_a1[:, 7], trajectory1_a1[:, 8], color='blue')
plt.plot(trajectory1_a2[:, 4], trajectory1_a2[:, 5], color='orange')
print("Sample trajectory shape:", samples_np.shape)

print("First agent trajectory first 5 steps:\n", samples_np[0, :5, 0, :])
print("Second agent trajectory first 5 steps:\n", samples_np[0, :5, 1, :])

plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("/mnt/data1/chendazhong/CTDE_Diffusion/madiff/fig2.png")
