# CTDE Conditional

This directory contains the implementation of a conditional diffusion model for multi-agent systems using the Centralized Training with Decentralized Execution (CTDE) paradigm.

## Subdirectories

- **`data/`**: Contains expert demonstrations and stored means and variances.
- **`figs/`**: Resulting plots of sampled trajectories.
- **`sampled_trajs/`**: Stores of sampled trajectories for plotting.
- **`trained_models/`**: Contains the trained models for the various implementations (see below).

## figs/ Descriptions
Some of the overlap between categories in the 'splice/' and not is due to those in the 'splice/' directory planning for the whole trajectory where as those not in the directory are only planning for a single 10 timestep horizon.
- **`splice/baseline`**: Simple plan 10 timesteps, execute 10 timesteps
- **`splice/reactive`**: Plan 10 timesteps, execute 10 timesteps with conditioning on both agents' positions
- **`splice/reactive_collision`**: Plan 10 timesteps, execute 10 timesteps with collision avoidance (bad)
- **`splice/reactive_mode`**: Plan 10 timesteps, execute 10 timesteps with conditioning on mode
- **`splice/reactive_safety`**: Plan 10 timesteps, execute 10 timesteps with safety filter based on how likely pair of positions are to appear in the expert demonstration distribution
- **`T10`**: Planning only 10 timesteps after splicing up expert demonstrations
- **`T10_reactive`**: Planning only 10 timesteps and conditioning on own state and other agent's state
- **`T10_reactivemode`**: Planning only 10 timesteps and conditioning on both agents' state as well as the mode (labeled data)
- **`T100`**: Full horizon (100 timestpes) planning
- **`T100_noise0.05`**: Full horizon (100 timestpes) planning with noised sampled atat 0.05 noise level

## Main Files
- **`training_mpc_guidance.py`**: Training and sampling of MPC approach (10 timestep planning, 1 timestep execution) with guidance function in denoising process to avoid collision
- **`training_mpc_lf_latent.py`**: Training and sampling of mpc approach(varying timestep planning, varying timestep execution) while conditioned as a leader/follower approach on the latent representation of the leader's history
- **`training_mpc_lf.py`**: Training and sampling of mpc approach(varying timestep planning, varying timestep execution) while conditioned as a leader/follower approach
- **`training_mpc.py`**: Training and sampling of MPC approach (10 timestep planning, 1 timestep execution)
- **`training_reactive.py`**: Training and sampling of splice approach(10 timestep planning, 10 timestep execution) while conditioned on additional elements (other agent's state and/or expert demonstration mode)
- **`training_splice.py`**: Training and sampling of splice approach (10 timestep planning, 10 timestep execution)
- **`training.py`**: Training and sampling of full horizon planning (original)
- **`splice_baseline.py`**: Sampling of splice approach to generate a full 100 timestep trajectory
- **`splice_reactive_mode.py`**: Sampling of splice approach with additional conditioning on other agent's state and expert demonstration mode to generate a full 100 timestep trajectory
- **`splice_reactive.py`**: Sampling of splice approach with additional conditioning on other agent's state to generate a full 100 timestep trajectory