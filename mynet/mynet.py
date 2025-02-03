import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                            kernel_size=patch_size, 
                            stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    

class MAE_ViT(nn.Module):
    def __init__(self, num_stages=2, num_classes=7):
        super().__init__()
        # Shared Components
        self.patch_embed = PatchEmbedding()
        self.pos_embed = nn.Parameter(torch.randn(1, 196, 768))  # (1, num_patches, dim)
        
        # Stage1 (MAE)
        self.encoder = nn.Sequential(*[ViTBlock(768, 12) for _ in range(12)])
        self.decoder = nn.Sequential(*[ViTBlock(768, 12) for _ in range(4)])
        self.mask_token = nn.Parameter(torch.randn(1, 1, 768))
        self.register_buffer('all_indices', torch.arange(196))  # 假设总patch数为196
        
        # Stage2 (Supervised)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )
        
        self.stage = 1  # 1: pretrain, 2: finetune

    def random_masking(self, x, mask_ratio=0.75):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # 生成随机排序索引
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 关键修复：保存恢复索引
        
        # 掩码可见部分
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_visible, ids_keep, ids_restore

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, L, E)
        x = x + self.pos_embed
        
        
        if self.stage == 1:
            # 步骤1：掩码处理并保存索引
            x_visible, ids_keep, ids_restore = self.random_masking(x)
            
            # 步骤2：编码器处理可见patch
            latent = self.encoder(x_visible)  # [B, 49, 768]
            
            # 步骤3：重建完整序列
            mask_tokens = self.mask_token.repeat(x.shape[0], 196 - ids_keep.shape[1], 1)  # [B, 147, 768]
            x_full = torch.cat([latent, mask_tokens], dim=1)  # [B, 196, 768]
            
            # 步骤4：恢复原始patch顺序
            x_full = torch.gather(
                x_full, 
                dim=1,
                index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            )  # [B, 196, 768]
            # 步骤5：解码器处理完整序列
            x_recon = self.decoder(x_full)  # [B, 196, 768]
            return x_recon
        
        else:
            # Supervised Finetuning
            x = self.encoder(x)
            cls_token = x.mean(dim=1)  # Global average pooling
            return self.cls_head(cls_token)