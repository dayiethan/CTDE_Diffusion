import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

class ImagePatchAndEmbed(nn.Module):
    """
        This is our first class in the ViT encoder architecture, 
        this class is responsible for patching our images and embedding 
        them into a higher dimentional space so that our model can learn
        more nuanced features. 

        @param image_dim: the dimension of our input image / trajectory - defaults to 256.
        @param patch_size: 16, the size of each patch - 16 is standard in research from my understanding.
        @param input_channels: 3, 3 channels for RGB.
        @param embedding_dimension: 768, the dimension of our embedding space - 768 is standard in research from my 
                                    understanding, we get the number through 16 * 16 * 3.

    """
    def __init__(self, image_dim = 256, patch_size = 16, input_channels = 3, embedding_dimension = 768):
        super().__init__()
        self.image_dim = image_dim
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedding_dimension = embedding_dimension
        self.number_of_patches = (image_dim // patch_size) ** 2 
        self.patch_dimensions = patch_size * patch_size * input_channels
        self.projection = nn.Linear(self.patch_dimensions, embedding_dimension)

        self.positional_embedding = nn.Parameter(torch.randn(1, self.number_of_patches, embedding_dimension))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        height, width = x.shape[2], x.shape[3]
        x = x.reshape(batch_size, self.input_channels, height // self.patch_size, width, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, self.number_of_patches, self.patch_dimensions)
        x = self.projection(x)
        x = x + self.positional_embedding
        return x


class MultiHeadAttention(nn.Module):
    """
        First implementation of multi-head attention for our encoder.
        @param embed_dim: the dimension of our embedding space - defaults to 768.
        @param n_heads: the number of heads for our attention mechanism - defaults to 12.
        @param dropout: the dropout rate for our attention mechanism - defaults to 0.1.
    """
    def __init__(self, embedding_dimension=768, number_of_heads=12, dropout=0.1):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        
        self.qkv = nn.Linear(embedding_dimension, embedding_dimension * 3)
        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, number_of_patches, embedding_dimension = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, number_of_patches, 3, self.number_of_heads, self.head_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dimension ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, number_of_patches, embedding_dimension)
        return self.proj(out)

class TransformerBlock(nn.Module):
    """
        Super basic transformer block for our encoder, simply does our 
        multi-head attention step and an Multi-layer-perceptron.
    """
    def __init__(self, embedding_dimension=768, number_of_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.attn = MultiHeadAttention(embedding_dimension, number_of_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        mlp_hidden_dim = int(embedding_dimension * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embedding_dimension),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embedding_dimension=768, depth=12, number_of_heads=12, mlp_ratio=4.0, dropout=0.1, latent_dim=128):
        super().__init__()
        self.patch_embed = ImagePatchAndEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.number_of_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dimension))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embedding_dimension))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dimension, number_of_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_dimension)
        self.head = nn.Linear(embedding_dimension, latent_dim)
        self.init_weights()
    
    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        latent_features = self.head(x[:, 0])
        return latent_features

def create_vit_encoder(**kwargs):
    return VisionTransformerEncoder(embedding_dimension=768, depth=12, number_of_heads=12, **kwargs)


