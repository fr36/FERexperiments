import torch
import torch.nn as nn
import numpy as np

class MAEViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12, 
                 decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, mlp_ratio=4.,
                 mask_ratio=0.75, num_classes=7):
        super().__init__()
        
        # 图像分块相关参数
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_depth)
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, 
                                    patch_size * patch_size * in_chans)
        
        # Classification head (for fine-tuning)
        self.fc = nn.Linear(embed_dim, num_classes)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # 初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                          int(self.num_patches**.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
    def random_masking(self, x):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                              index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Patch Embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Masking
        x, mask, ids_restore = self.random_masking(x)
        
        # Encoder
        x = self.encoder(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = torch.zeros_like(x[:, :1, :]).repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle
        x = torch.gather(x_, dim=1, 
                        index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        
        # Decoder
        x = self.decoder(x, x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        return x
    
    def forward_classification(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Encoder
        x = self.encoder(x)
        
        # Classification head
        x = x.mean(dim=1)  # global average pooling
        x = self.fc(x)
        
        return x
    
    def forward(self, x, mask_ratio=None, is_train=True):
        if is_train and mask_ratio is not None:
            # MAE training
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            return pred, mask
        else:
            # Classification fine-tuning
            return self.forward_classification(x)