# Multi-Agent Diffusion Models

This repository contains code for multi-agent diffusion models used in path planning around obstacles. The project is organized into two main folders: `base` and `random`.

## Folder Structure

### base
The `base` folder contains the original working 2-agent diffusion model path planning code. This model is designed to navigate around a central obstacle using similar training data. The key components of this folder include:

- **data/**: Includes the training data used for the model.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **traj_from_diffusion_transformer.py**: Generates a trajectory from a trained 2-agent diffusion model.
- **diffusion_transformer.py**: Uses expert demonstrations to train the 2-agent diffusion model.

### decentralized_two_attr
The `decentralized` folder contains the 2-agent diffusion model path planning code with the goal of training two agents' policies against another so that they can be executed in a decentralized manner. This approach tries to use the attribute function provided by the AlignDiff code. The key components of this folder include:

- **data/**: Includes the training data used for the model.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **traj_from_diffusion_transformer.py**: Generates a trajectory from a trained 2-agent diffusion model.
- **diffusion_transformer.py**: Uses expert demonstrations to train the 2-agent diffusion model.

### decentralized_two_denoiser
The `decentralized` folder contains the 2-agent diffusion model path planning code with the goal of training two agents' policies against another so that they can be executed in a decentralized manner. This approach tries to use two separate denoisers, one for each agent. The key components of this folder include:

- **data/**: Includes the training data used for the model.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **traj_from_diffusion_transformer.py**: Generates a trajectory from a trained 2-agent diffusion model.
- **diffusion_transformer.py**: Uses expert demonstrations to train the 2-agent diffusion model.

### random
The `random` folder contains ongoing work that utilizes expert demonstration data that includes a random array of initial and final positions for the agents. The goal of this work is to find optimal trajectories for two agents starting at given initial and final conditions. The key components of this folder include:

- **data/**: Includes the expert demonstration data.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **traj_from_diffusion_transformer_cond.py**: Generates a trajectory from a trained diffusion model with random initial and final positions.
- **diffusion_transformer_cond.py**: Uses expert demonstrations to train the diffusion model with random initial and final positions.