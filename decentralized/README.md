# Decentralized

This directory contains the implementation of a diffusion model for multi-agent systems using the Centralized Training with Decentralized Execution (CTDE) paradigm.

## Subdirectories

- **`checking/`**: Sanity checking experiments with various numbers of demonstrations.
- **`multi_mode/`**: Providing every agent with multiple different modes of execution as demonstration data.
- **`multi_mode_alternating/`**: Providing every agent with multiple different modes while also training them in an alternating fashion at every timestep.
- **`same_policy/`**: Training a single policy and using it to execute both agents.
- **`single_mode/`**: Giving each agent only a single mode in the expert demonstrations.
- **`vary_init/`**: Training with multiple modes and alternating training but varying the initial condition during execution/sampling.