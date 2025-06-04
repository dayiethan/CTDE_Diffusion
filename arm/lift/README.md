# Handover

This directory contains the implementation of a conditional diffusion model for multi-agent systems using the Centralized Training with Decentralized Execution (CTDE) paradigm.

## Subdirectories

- **`data/`**: Contains datasets used directly for training
- **`rollouts_pot/`**: Contains the raw data from demonstrations
- **`trained_models/`**: Contains the trained models for the various implementations (see below)


## trained_models/ Descriptions
P#E# means that it is trained for a certain number of planning timesteps and a certain number of execution timesteps.
- **`_lift_mpc_P25E2_10ksteps`**: MPC planning for 25 timesteps, execute 2 timesteps wtih 10000 training steps
- **`_lift_mpc_P25E2_50ksteps`**: MPC planning for 25 timesteps, execute 2 timesteps wtih 50000 training steps
- **`_lift_mpc_P25E5`**: MPC planning for 25 timesteps, execute 5 timesteps wtih 100000 training steps
- **`_T250_rot6d_pot_20`**: Full horizon planning using 6d rotation matrix conditioned on the pot handle positions with 20 demonstrations
- **`_T250_rot6d_pot_100`**: Full horizon planning using 6d rotation matrix conditioned on the pot handle positions with 100 demonstrations
- **`_T250_rot6d_pot_400`**: Full horizon planning using 6d rotation matrix conditioned on the pot handle positions with 400 demonstrations
- **`_T250_rot6d`**: Full horizon planning using 6d rotation matrix conditioned on the pot position
- **`_T250_rotvec_pot_20`**: Full horizon planning using 3d rotation vector conditioned on the pot handle positions with 20 demonstrations
- **`_T250_rotvec_pot_100`**: Full horizon planning using 3d rotation vector conditioned on the pot handle positions with 100 demonstrations
- **`_T250_rotvec_pot_400`**: Full horizon planning using 3d rotation vector conditioned on the pot handle positions with 400 demonstrations
- **`_T250_rotvec`**: Full horizon planning using 3d rotation vector conditioned on the pot position



## Main Files
- **`conditional_Action_DiT.py`**: Sets up the traditional condtional DiT
- **`conditional_training_rot6d.py`**: Training script for full horizon planning using 6d rotation matrix
- **`conditional_training_rotvec.py`**: Training script for full horizon planning using 3d rotation vector
- **`demo.py`**: Script that generates expert demonstrations
- **`discrete.py`**: For computing Frechet distance
- **`env.py`**: Sets up the handover environment for getting demonstrations
- **`parse_data.ipynb`**: Generates the useable datasets for training from the rollout expert demonstrations
- **`sampled_mpc.py`**: Samples models in MPC fashion
- **`salmpled_rot6d.py`**: Samples models that use 6d rotation matrix
- **`sampled_rotvec.py`**: Samples models that use 3d rotation vector
- **`training_mpc.py`**: Trains models in MPC fashion
- **`transform_utils.py`**: Transforming between rotation representations
