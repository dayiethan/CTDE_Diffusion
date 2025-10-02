# Quick Start Guide: Pi Zero on KINOVA Robosuite

This guide will help you run Pi Zero with natural language commands like **"help other robot move pot"** on your two-arm KINOVA robosuite environment.

## Prerequisites

1. **Python 3.10+** (recommended)
2. **CUDA-capable GPU** (optional but recommended for training)
3. **Robosuite** environment already set up

## Installation

### Step 1: Install Dependencies

```bash
# From the piZero directory
cd c:\Documents\Berkeley\BAIR_ICON\CTDE_Diffusion\arm\lift\piZero

# Install requirements
pip install -r requirements.txt

# Install OpenPi from source (if not available on PyPI)
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
pip install -e .
cd ..
```

### Step 2: Verify Installation

```bash
python -c "import openpi; import robosuite; print('All imports successful!')"
```

## Usage Options

You have **two ways** to use Pi Zero with your robosuite environment:

### Option A: Direct Inference (No Training - Using Pre-trained Pi Zero)

If you just want to test Pi Zero's zero-shot capabilities:

```bash
# This will use the base Pi0 model without fine-tuning
# NOTE: This likely won't work well without training on your specific task!
python evaluate_pi0.py \
  --config_name pi0_kinova_two_arm \
  --checkpoint_repo "physical-intelligence/pi0-base" \
  --task_description "help other robot move pot" \
  --num_episodes 5 \
  --render
```

**Expected Result:** The model will try to execute the task but will likely fail because it hasn't been trained on your specific robot configuration.

### Option B: Train on Your Data (Recommended)

This is the proper workflow:

#### 1. Convert Your Demonstration Data

```bash
# Convert your rollout pickle files to HuggingFace format
python convert_to_pi0_format.py \
  --rollout_dir ../rollouts/newslower \
  --output_dir ./kinova_dataset \
  --task_description "help other robot move pot"
```

This creates a dataset with:
- **Observations**: Robot states + camera images
- **Actions**: 14-dim actions (7 per robot)
- **Prompts**: Natural language task descriptions

#### 2. Upload Dataset to HuggingFace (Optional but Recommended)

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login to HuggingFace
huggingface-cli login

# Upload your dataset
huggingface-cli upload ./kinova_dataset YOUR_USERNAME/kinova_pot_lifting --repo-type=dataset
```

#### 3. Configure Training

Edit `config.py` and update the `repo_id` in the training configs:

```python
# Line 199, 230, 250, 279, 298 - change:
repo_id="YOUR_REPO_ID"
# to:
repo_id="YOUR_USERNAME/kinova_pot_lifting"  # or local path: "./kinova_dataset"
```

#### 4. Add Config to OpenPi

You need to add your configs to OpenPi's config file:

```bash
# Find OpenPi's config.py location
python -c "import openpi; print(openpi.__file__)"

# Copy the configs from piZero/config.py to openpi/training/config.py
# Add the _CONFIGS list items to OpenPi's config
```

**OR** use the configs directly by importing them (easier):

Create a file `train_pi0.py`:

```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

from openpi.training import train
import sys
sys.path.append('.')
import config  # Import your local config

# Train with LoRA (recommended - faster and cheaper)
train.main(config.get_config("pi0_kinova_two_arm_lora"))
```

#### 5. Train the Model

```bash
# Train with LoRA (recommended - 2-4 hours on GPU)
python train_pi0.py

# Or if you added configs to OpenPi:
python -m openpi.training.train pi0_kinova_two_arm_lora
```

Training will:
- Fine-tune Pi0 on your demonstrations
- Save checkpoints periodically
- Train for ~10,000 steps (configurable)

#### 6. Evaluate Your Trained Model

```bash
python evaluate_pi0.py \
  --config_name pi0_kinova_two_arm_lora \
  --checkpoint_repo "./output/pi0_kinova_two_arm_lora/checkpoints" \
  --task_description "help other robot move pot" \
  --num_episodes 10 \
  --render
```

## Using Natural Language Commands

Pi Zero accepts natural language task descriptions. You can use various phrasings:

```bash
# Original task description
--task_description "Pick up the pot with both robots and lift it together"

# Alternative phrasings
--task_description "help other robot move pot"
--task_description "lift the pot collaboratively"
--task_description "grasp handles and raise pot"
--task_description "two robots cooperate to lift pot"
```

**Important:** The model generalizes better to similar phrasings if your training data includes varied language descriptions.

## Advanced: Dual-Policy Setup

If you want **separate policies for each robot** (more complex but potentially better coordination):

```bash
# Train Robot 0 policy
python -m openpi.training.train pi0_kinova_robot0

# Train Robot 1 policy
python -m openpi.training.train pi0_kinova_robot1

# Evaluate with both policies
python evaluate_dual_robot_pi0.py \
  --robot0_model "./output/pi0_kinova_robot0/checkpoints" \
  --robot1_model "./output/pi0_kinova_robot1/checkpoints" \
  --num_episodes 10 \
  --render
```

## What You Need for This to Work

### Minimum Requirements:

1. **Demonstration Data** (~50-100 successful rollouts)
   - Must include camera observations
   - Must include successful task completions
   - Should cover diverse initial conditions

2. **Hardware**:
   - GPU with 16GB+ VRAM (for training)
   - OR CPU with 32GB+ RAM (much slower)

3. **Time**:
   - Data conversion: ~5 minutes
   - Training (LoRA): 2-4 hours on GPU
   - Training (Full): 8-12 hours on GPU

### Your Current Setup:

Based on your files, you have:
- ✅ Environment: `TwoArmLiftRole` (robosuite)
- ✅ Rollout data: `rollouts/newslower/*.pkl`
- ✅ Controller config: `kinova.json`
- ✅ Two cameras: `robot0_eye_in_hand`, `robot1_eye_in_hand`
- ✅ Action space: 14-dim (7 per robot)

**You're ready to go! Start with Option B above.**

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'openpi'"

**Solution:** Install OpenPi from source:
```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && pip install -e .
```

### Issue: "Camera images not found in observation"

**Solution:** Ensure your environment is set up with cameras:
```python
camera_names=["robot0_eye_in_hand", "robot1_eye_in_hand"]
use_camera_obs=True
```

### Issue: "JAX out of memory"

**Solution:** Reduce batch size or limit JAX memory:
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

### Issue: "Actions don't match expected format"

**Solution:** Check your controller config matches the expected 14-dim action space (7 per robot).

## Next Steps

1. **Collect More Data**: More demonstrations = better performance
2. **Data Augmentation**: Add varied language descriptions during data collection
3. **Hyperparameter Tuning**: Adjust `action_horizon`, learning rate, etc.
4. **Deployment**: Once working in sim, deploy to real robots

## Support

For issues specific to:
- **Pi Zero**: Check [OpenPi GitHub](https://github.com/Physical-Intelligence/openpi)
- **Robosuite**: Check [Robosuite Docs](https://robosuite.ai/)
- **This Code**: Open an issue or check the comments in the code

Good luck! 🚀

