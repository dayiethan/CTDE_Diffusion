import pickle as pkl
import matplotlib.pyplot as plt
import os
import time
import torch
import numpy as np

# Import the ViT model, make sure the corrected code above is saved in image_encoder.py
from image_encoder import create_vit_encoder

def load_and_process_and_display_images(filepath, vit_encoder):
    """
    Loads a .pkl file, processes images with a ViT, and displays them.

    Args:
        filepath (str): The path to the .pkl file.
        vit_encoder (VisionTransformerEncoder): The ViT model instance.
    """
    try:
        with open(filepath, 'rb') as f:
            rollout = pkl.load(f)

        if 'camera0_obs' not in rollout or not rollout['camera0_obs']:
            print(f"No 'camera0_obs' data found or it's empty in '{filepath}'.")
            return
        
        plt.ion()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        
        im0 = ax0.imshow(rollout['camera0_obs'][0])
        ax0.set_title("Robot 0 Camera Image")
        ax0.axis('off')
        
        im1 = ax1.imshow(rollout['camera1_obs'][0])
        ax1.set_title("Robot 1 Camera Image")
        ax1.axis('off')
        
        fig.tight_layout()
        plt.show(block=False)

        num_frames = min(len(rollout['camera0_obs']), len(rollout['camera1_obs']))
        for i in range(num_frames):
            try:
                # Get the raw image data (NumPy arrays)
                img0_np = rollout['camera0_obs'][i]
                img1_np = rollout['camera1_obs'][i]

                # Check for invalid image shapes for both cameras
                if img0_np.shape[0] == 0 or img0_np.shape[1] == 0 or img1_np.shape[0] == 0 or img1_np.shape[1] == 0:
                    print(f"Due to invalid image shape, skipping frame {i}: camera0_shape={img0_np.shape}, camera1_shape={img1_np.shape}")
                    continue

                # Convert NumPy arrays to PyTorch tensors
                img0_tensor = torch.from_numpy(img0_np).permute(2, 0, 1).unsqueeze(0).float()
                img1_tensor = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).float()
                
                # Use the ViT model for a forward pass, and disable gradient computation
                with torch.no_grad(): 
                    latent_features0 = vit_encoder(img0_tensor)
                    latent_features1 = vit_encoder(img1_tensor)

                # print(f"Frame {i}: Latent feature shape for camera 0: {latent_features0.shape}")
                # print(f"Frame {i}: Latent feature shape for camera 1: {latent_features1.shape}")
                print(f"Frame {i} Latent feature for camera 0:{latent_features0}")
                print(f"Frame {i} Latent feature for camera 1:{latent_features1}")

                # Update the plot
                im0.set_data(img0_np)
                im1.set_data(img1_np)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.01)

            except Exception as e:
                print(f"An error occurred while processing frame {i}: {e}")
                continue

        plt.ioff()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
vit_encoder = create_vit_encoder()
directory = "rollouts/newslower"
filename = "rollout_seed0_mode2.pkl"
file_path = os.path.join(directory, filename)
load_and_process_and_display_images(file_path, vit_encoder)