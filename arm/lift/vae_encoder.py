import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
import math
import numpy as np
from typing import Optional


class VAEEncoder(nn.Module):
    """
    Wrapper around Stable Diffusion's pretrained AutoencoderKL encoder that outputs
    fixed-size latent vectors (B, latent_dim) for plug-and-play use like image_encoder.

    This encoder is designed to be a drop-in replacement for VisionTransformerEncoder
    in robotics applications, particularly for processing robot camera observations.

    Features:
    - Expects input x in [0, 1] or [-1, 1], shape (B, 3, H, W)
    - Encodes to VAE latent space (B, 4, H/8, W/8) 
    - Uses adaptive pooling and projection to output (B, latent_dim)
    - VAE weights are frozen by default; only projection layer trains
    - Supports different pooling strategies for better feature extraction
    - Compatible with robot camera image sizes (256x256)
    """
    def __init__(self, 
                 img_size: int = 256,
                 latent_dim: int = 128, 
                 pretrained_model: str = "stabilityai/sd-vae-ft-ema",
                 freeze_vae: bool = True,
                 pooling_strategy: str = "adaptive_avg",
                 use_cls_token: bool = False,
                 dropout: float = 0.1):
        """
        Args:
            img_size: Expected input image size (assumed square)
            latent_dim: Output feature dimension
            pretrained_model: HuggingFace model name for VAE
            freeze_vae: Whether to freeze VAE encoder weights
            pooling_strategy: How to pool VAE features ('adaptive_avg', 'adaptive_max', 'attention')
            use_cls_token: Whether to add a learnable CLS token (similar to ViT)
            dropout: Dropout rate for projection layers
        """
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.pooling_strategy = pooling_strategy
        self.use_cls_token = use_cls_token
        self.freeze_vae = freeze_vae
        
        # Load pretrained VAE encoder
        self.vae = AutoencoderKL.from_pretrained(pretrained_model)
        self.expected_latent_channels = 4
        
        # Calculate VAE latent spatial dimensions (typically 8x downsampling)
        self.latent_spatial_size = img_size // 8
        
        # Different pooling strategies
        if pooling_strategy == "adaptive_avg":
            self.pool_size = (8, 8)  # Smaller pool size for better spatial resolution
            self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
            pooled_features = self.expected_latent_channels * self.pool_size[0] * self.pool_size[1]
            
        elif pooling_strategy == "adaptive_max":
            self.pool_size = (8, 8)
            self.pool = nn.AdaptiveMaxPool2d(self.pool_size)
            pooled_features = self.expected_latent_channels * self.pool_size[0] * self.pool_size[1]
            
        elif pooling_strategy == "attention":
            # Attention-based pooling similar to ViT
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.expected_latent_channels,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            pooled_features = self.expected_latent_channels
            
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # CLS token for attention-based processing (optional, similar to ViT)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.expected_latent_channels))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Projection layers with dropout for regularization
        self.projection = nn.Sequential(
            nn.Linear(pooled_features, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize projection layers
        self.init_weights()
        
        # Optionally freeze VAE weights
        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

    def init_weights(self):
        """Initialize projection layer weights similar to ViT"""
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VAE encoder
        
        Args:
            x: Input images (B, 3, H, W) in range [0, 1] or [-1, 1]
            
        Returns:
            latent_features: (B, latent_dim) feature vectors
        """
        batch_size = x.shape[0]
        
        # Ensure input is in correct range for VAE ([-1, 1])
        if x.min() >= 0:
            x = x * 2.0 - 1.0
        
        # Encode through VAE (frozen)
        if self.freeze_vae:
            with torch.no_grad():
                latent = self.vae.encode(x).latent_dist.sample()
        else:
            latent = self.vae.encode(x).latent_dist.sample()
        
        # Verify latent shape
        if latent.shape[1] != self.expected_latent_channels:
            raise RuntimeError(f"Unexpected latent channels: {latent.shape[1]} (expected {self.expected_latent_channels})")
        
        # Apply pooling strategy
        if self.pooling_strategy in ["adaptive_avg", "adaptive_max"]:
            pooled = self.pool(latent)  # (B, 4, pool_h, pool_w)
            pooled_flat = pooled.view(batch_size, -1)  # (B, 4*pool_h*pool_w)
            
        elif self.pooling_strategy == "attention":
            # Reshape for attention: (B, spatial_tokens, channels)
            B, C, H, W = latent.shape
            latent_tokens = latent.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
            
            if self.use_cls_token:
                # Add CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
                latent_tokens = torch.cat([cls_tokens, latent_tokens], dim=1)  # (B, 1+H*W, C)
                
                # Self-attention
                attended, _ = self.attention_pool(latent_tokens, latent_tokens, latent_tokens)
                pooled_flat = attended[:, 0]  # Use CLS token output (B, C)
            else:
                # Global average after attention
                attended, _ = self.attention_pool(latent_tokens, latent_tokens, latent_tokens)
                pooled_flat = attended.mean(dim=1)  # (B, C)
        
        # Project to final latent dimension
        latent_features = self.projection(pooled_flat)
        
        return latent_features

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate VAE feature maps for visualization/analysis
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            latent: VAE latent feature maps (B, 4, H/8, W/8)
        """
        if x.min() >= 0:
            x = x * 2.0 - 1.0
            
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            
        return latent


def create_vae_encoder(**kwargs) -> VAEEncoder:
    """
    Factory function to create VAE encoder with default parameters
    matching the project's image processing needs.
    """
    defaults = {
        'img_size': 256,
        'latent_dim': 128,
        'pretrained_model': "stabilityai/sd-vae-ft-ema",
        'freeze_vae': True,
        'pooling_strategy': "adaptive_avg",
        'use_cls_token': False,
        'dropout': 0.1
    }
    defaults.update(kwargs)
    return VAEEncoder(**defaults)


# Additional utility functions for compatibility with existing code

def create_vae_encoder_with_attention(**kwargs) -> VAEEncoder:
    """Create VAE encoder with attention pooling (more similar to ViT)"""
    defaults = {
        'pooling_strategy': "attention",
        'use_cls_token': True,
    }
    defaults.update(kwargs)
    return create_vae_encoder(**defaults)


class VAEImageProcessor:
    """
    Utility class for batch processing images with VAE encoder,
    similar to how images are processed in parse_data.ipynb
    """
    def __init__(self, vae_encoder: VAEEncoder, device: str = 'cuda', batch_size: int = 8):
        self.vae_encoder = vae_encoder.to(device).eval()
        self.device = device
        self.batch_size = batch_size
    
    def process_image_sequence(self, images: np.ndarray) -> np.ndarray:
        """
        Process a sequence of images (e.g., from robot camera observations)
        
        Args:
            images: numpy array of shape (T, H, W, 3) with values in [0, 255]
            
        Returns:
            latents: numpy array of shape (T, latent_dim)
        """
        latents = []
        
        print(f"Processing {len(images)} frames with VAE encoder...")
        
        # Process in batches for efficiency
        for i in range(0, len(images), self.batch_size):
            batch_end = min(i + self.batch_size, len(images))
            batch_imgs = images[i:batch_end]
            
            print(f"  Batch {i//self.batch_size + 1}/{(len(images)-1)//self.batch_size + 1}: frames {i}-{batch_end}")
            
            # Convert to tensor and normalize
            batch_tensor = torch.from_numpy(batch_imgs).float() / 255.0  # [0, 1]
            batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(self.device)  # (B, 3, H, W)
            
            with torch.no_grad():
                batch_latents = self.vae_encoder(batch_tensor).cpu().numpy()
                latents.extend(batch_latents)
        
        return np.array(latents)


