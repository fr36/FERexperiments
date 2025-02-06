import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MAE_ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, mask_ratio=0.75):
        super(MAE_ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3  # assuming RGB images
        
        # Patch Embedding Layer
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Encoding
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer Encoder Layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        
        # Masking
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))  # mask token for the masked patches

        # Final Decoder (Reconstruction Head)
        self.decoder = nn.Linear(dim, self.patch_dim)  # For reconstructing original patches

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        
        # Step 1: Patch Embedding
        x = self.patch_embedding(x)  # (batch_size, dim, 14, 14)
        x = x.flatten(2)  # (batch_size, dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, dim)
        
        # Step 2: Add Class Token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches+1, dim)
        
        # Step 3: Add Positional Embedding
        x = x + self.position_embedding
        
        # Step 4: Masking
        # We randomly select patches to mask
        mask = torch.rand(x.shape[1]) < self.mask_ratio  # (num_patches + 1,)
        mask[0] = 1  # Never mask the class token
        mask = mask.expand(x.size(0), -1)  # (batch_size, num_patches+1)
        
        # Replace masked patches with the mask token
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        
        # Step 5: Transformer Encoder
        x_encoded = self.encoder(x_masked)  # (batch_size, num_patches+1, dim)
        
        # Step 6: Decoder (Reconstruction)
        # We need to only reconstruct the masked patches (not class token)
        x_decoded = self.decoder(x_encoded[:, 1:])  # Skip the class token, (batch_size, num_patches, patch_dim)
        
        # Reshape to match the original patch size (batch_size, num_patches, patch_dim)
        x_decoded = x_decoded.view(x.size(0), -1, self.patch_size, self.patch_size, 3)  # (batch_size, num_patches, patch_size, patch_size, 3)
        
        return x_decoded, mask

    def reconstruct(self, x):
        """
        仅用于重建功能（非训练过程）。
        """
        x_decoded, mask = self.forward(x)
        return x_decoded
