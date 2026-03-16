import torch
import torch.nn as nn
from .rstb import ResidualSwinTransformerBlock

class SwinIRLight(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, upscale=2, img_range=1., resi_connection='1conv'):
        super().__init__()
        
        self.img_range = img_range
        self.mean = torch.zeros(1, 3, 1, 1)
        
        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualSwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)

        # 3. High-quality image reconstruction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # Upsampling for Super-Resolution
        if upscale > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, in_chans * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        else:
            self.upsample = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x):
        # Normalize
        self.mean = self.mean.to(x.device)
        x = (x - self.mean) * self.img_range
        
        # Shallow feature extraction
        x = self.conv_first(x)
        res = x
        
        # Deep feature extraction
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, (H, W))
        x = self.norm(x)
        
        # To image format
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Reconstruction
        x = self.conv_after_body(x) + res
        x = self.upsample(x)
        
        # Un-normalize
        x = x / self.img_range + self.mean
        return x

def create_model(upscale=2):
    return SwinIRLight(upscale=upscale)
