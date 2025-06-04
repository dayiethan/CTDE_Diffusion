# Handover

This directory contains the implementation of a conditional diffusion model for multi-agent systems using the Centralized Training with Decentralized Execution (CTDE) paradigm.

## Subdirectories

- **`data_pickup_pos/`**: Contains datasets used directly for training. Uses the pickup position of the hammer
- **`rollouts_pickup_pos/`**: Contains the raw data from demonstrations
- **`trained_models/`**: Contains the trained models for the various implementations (see below)


## trained_models/ Descriptions
P#E# means that it is trained for a certain number of planning timesteps and a certain number of execution timesteps.
- **`_handover_mpc_P34E5_#`**: # are various iterations of the planning for 34 timesteps, execute 5 timesteps
- **`_handover_mpc_P34E5_largecond_actionH`**: Conditioned on (own pos, hammer pos) for leader, (own pos, leader pos 5 steps ahead, hammer pos) for follwer with action size as H rather than H-1
- **`_handover_mpc_P34E5_largecond_latent`**: Conditioned on (own pos, hammer pos) for leader, (own pos, leader pos 5 steps ahead, hammer pos, latent history representation of leader) for follwer
- **`_handover_mpc_P34E5_largecond`**: Conditioned on (own pos, hammer pos) for leader, (own pos, leader pos 5 steps ahead, hammer pos) for follwer
- **`_T340_rot6d_hammer_100`**: Full horizon planning using 6d rotation matrix with 100 demonstrations
- **`_T340_rot6d_hammer_200`**: Full horizon planning using 6d rotation matrix with 200 demonstrations
- **`_T340_rotvec_hammer_100`**: Full horizon planning using 3d rotation vector with 100 demonstrations
- **`_T340_rotvec_hammer_200`**: Full horizon planning using 3d rotation vector with 200 demonstrations
- **`_T340_rotvec_hammer_pickup_pos_20`**: Full horizon planning using 3d rotation vector with 20 demonstrations conditioned on the actual pickup position of the hammer which works better than just the position of the hammer
- **`_T340_rotvec_hammer_pickup_pos_200`**: Full horizon planning using 3d rotation vector with 200 demonstrations conditioned on the actual pickup position of the hammer which works better than just the position of the hammer

## Main Files
- **`conditional_Action_DiT_latent.py`**: Sets up the condtional DiT when also encoding the history of the leader as a latent space to use as part of the conditional vector for the follower
- **`conditional_Action_DiT.py`**: Sets up the traditional condtional DiT
- **`conditional_training_rot6d.py`**: Training script for full horizon planning using 6d rotation matrix
- **`conditional_training_rotvec.py`**: Training script for full horizon planning using 3d rotation vector
- **`demo.py`**: Script that generates expert demonstrations
- **`env.py`**: Sets up the handover environment for getting demonstrations
- **`parse_data.ipynb`**: Generates the useable datasets for training from the rollout expert demonstrations
- **`sampled_mpc.py`**: Samples models in MPC fashion
- **`salmpled_rot6d.py`**: Samples models that use 6d rotation matrix
- **`sampled_rotvec.py`**: Samples models that use 3d rotation vector
- **`training_mpc_latent.py`**: Trains models in MPC fashion making use of latent space representation of leader to condition follower
- **`training_mpc.py`**: Trains models in MPC fashion
- **`transform_utils.py`**: Transforming between rotation representations
