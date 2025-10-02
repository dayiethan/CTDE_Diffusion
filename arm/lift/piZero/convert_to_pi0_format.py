#!/usr/bin/env python3
"""
Convert KINOVA demonstration data from pickle format to Pi0/HuggingFace format.

This script processes your rollout pickle files and creates a HuggingFace dataset
compatible with Pi0 training.
"""

import numpy as np
import pickle as pkl
import torch
from datasets import Dataset, DatasetDict
import os
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import json

def load_rollout_data(rollout_path: str) -> Dict[str, Any]:
    """Load a single rollout pickle file."""
    with open(rollout_path, "rb") as f:
        return pkl.load(f)

def extract_state_from_obs(obs_dict: Dict[str, Any]) -> np.ndarray:
    """
    Extract state vector from observation dictionary.
    
    You can customize this based on what state representation you want:
    - Option 1: Simple (pos + quat + gripper)
    - Option 2: Full 20-dim with rot6d (from your parse_data.ipynb)
    """
    # Option 1: Simple state (16-dim)
    state_components = [
        obs_dict.get("robot0_eef_pos", np.zeros(3)),
        obs_dict.get("robot0_eef_quat_site", np.zeros(4)), 
        [obs_dict.get("robot0_gripper_pos", 0.0)],
        obs_dict.get("robot1_eef_pos", np.zeros(3)),
        obs_dict.get("robot1_eef_quat_site", np.zeros(4)),
        [obs_dict.get("robot1_gripper_pos", 0.0)],
    ]
    
    state = np.concatenate([np.atleast_1d(comp).flatten() for comp in state_components])
    return state.astype(np.float32)

def process_rollout_to_pi0_format(
    rollout: Dict[str, Any], 
    task_description: str = "Pick up the pot with both robots and lift it together"
) -> List[Dict[str, Any]]:
    """
    Convert a single rollout to list of Pi0 dataset entries.
    
    Each timestep becomes one dataset entry.
    """
    pi0_entries = []
    
    observations = rollout["observations"]
    actions = np.array(rollout["actions"])
    camera0_obs = rollout.get("camera0_obs", [])
    camera1_obs = rollout.get("camera1_obs", [])
    
    # Handle missing camera data
    has_cameras = len(camera0_obs) > 0 and len(camera1_obs) > 0
    
    for t in range(len(observations)):
        obs = observations[t]
        action = actions[t] if t < len(actions) else actions[-1]
        
        # Extract state
        state = extract_state_from_obs(obs)
        
        # Get camera images
        if has_cameras and t < len(camera0_obs):
            camera0_img = camera0_obs[t]
            camera1_img = camera1_obs[t] if t < len(camera1_obs) else camera0_obs[t]
        else:
            # Dummy images if no camera data
            camera0_img = np.zeros((256, 256, 3), dtype=np.uint8)
            camera1_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create Pi0 entry
        entry = {
            "observation.state": state,
            "observation.images.base_0_rgb": camera0_img,
            "observation.images.left_wrist_0_rgb": camera0_img,  # Duplicate robot0 cam
            "observation.images.right_wrist_0_rgb": camera1_img,
            "action": action.astype(np.float32),
            "prompt": task_description,
        }
        
        pi0_entries.append(entry)
    
    return pi0_entries

def convert_dataset(
    rollout_dir: str,
    output_dir: str,
    task_description: str = "Pick up the pot with both robots and lift it together",
    train_split: float = 0.8,
    val_split: float = 0.1
):
    """
    Convert all rollout files to Pi0 HuggingFace dataset format.
    """
    print(f"Converting rollouts from {rollout_dir} to Pi0 format...")
    
    # Find all pickle files
    rollout_files = []
    for file in os.listdir(rollout_dir):
        if file.endswith('.pkl'):
            rollout_files.append(os.path.join(rollout_dir, file))
    
    print(f"Found {len(rollout_files)} rollout files")
    
    # Process all rollouts
    all_entries = []
    for file_path in tqdm(rollout_files, desc="Processing rollouts"):
        try:
            rollout = load_rollout_data(file_path)
            entries = process_rollout_to_pi0_format(rollout, task_description)
            all_entries.extend(entries)
            print(f"  {os.path.basename(file_path)}: {len(entries)} timesteps")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print(f"Total dataset size: {len(all_entries)} timesteps")
    
    # Shuffle and split data
    np.random.seed(42)
    indices = np.random.permutation(len(all_entries))
    
    n_train = int(len(indices) * train_split)
    n_val = int(len(indices) * val_split)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create dataset splits
    train_data = [all_entries[i] for i in train_indices]
    val_data = [all_entries[i] for i in val_indices]
    test_data = [all_entries[i] for i in test_indices]
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_data)} timesteps")
    print(f"  Val: {len(val_data)} timesteps") 
    print(f"  Test: {len(test_data)} timesteps")
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    # Save metadata
    metadata = {
        "task_description": task_description,
        "action_dim": 14,
        "state_dim": len(all_entries[0]["observation.state"]),
        "image_size": [256, 256, 3],
        "num_episodes": len(rollout_files),
        "total_timesteps": len(all_entries),
        "splits": {
            "train": len(train_data),
            "validation": len(val_data), 
            "test": len(test_data),
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to: {output_dir}")
    print(f"Metadata saved to: {output_dir}/metadata.json")
    
    return dataset_dict

def main():
    parser = argparse.ArgumentParser(description="Convert KINOVA rollouts to Pi0 format")
    parser.add_argument("--rollout_dir", type=str, required=True,
                      help="Directory containing rollout pickle files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for HuggingFace dataset")
    parser.add_argument("--task_description", type=str,
                      default="Pick up the pot with both robots and lift it together",
                      help="Task description for the dataset")
    parser.add_argument("--train_split", type=float, default=0.8,
                      help="Training split ratio")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Validation split ratio")
    
    args = parser.parse_args()
    
    convert_dataset(
        rollout_dir=args.rollout_dir,
        output_dir=args.output_dir,
        task_description=args.task_description,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    print("\n✅ Conversion complete!")
    print("\nNext steps:")
    print("1. Upload dataset to HuggingFace Hub:")
    print(f"   huggingface-cli upload {args.output_dir} your_username/kinova_pot_lifting --repo-type=dataset")
    print("2. Update config.py with your repo ID")
    print("3. Start training with Pi0!")

if __name__ == "__main__":
    main()
