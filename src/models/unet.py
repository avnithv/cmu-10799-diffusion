"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        self.time_embed_dim =  4 * base_channels
        self.time_embedding = TimestepEmbedding(time_embed_dim=self.time_embed_dim)
        self.encoder = nn.ModuleList(
            (nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1), )
        )

        res = 64
        cur_ch = base_channels
        skip_ch = [base_channels]
        for level in range(len(channel_mult)):
            ch = base_channels * channel_mult[level]
            
            layers = nn.ModuleList()

            if level > 0:
                layers.append(Downsample(cur_ch))
                skip_ch.append(cur_ch)

            for rb in range(num_res_blocks):
                if rb == 0:
                    layers.append(ResBlock(cur_ch, ch, self.time_embed_dim, dropout, use_scale_shift_norm))
                else:
                    layers.append(ResBlock(ch, ch, self.time_embed_dim, dropout, use_scale_shift_norm))

                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))

                skip_ch.append(ch)


            self.encoder.extend(layers)

            cur_ch = ch
            if level < len(channel_mult) - 1:
                res //= 2
        

        self.middle = nn.ModuleList(
            (
                ResBlock(cur_ch, cur_ch, self.time_embed_dim, dropout, use_scale_shift_norm),
                AttentionBlock(cur_ch, num_heads=num_heads),
                ResBlock(cur_ch, cur_ch, self.time_embed_dim, dropout, use_scale_shift_norm)
            )
        )

        self.decoder = nn.ModuleList()

        for level in reversed(range(len(channel_mult))):
            ch = base_channels * channel_mult[level]
            
            layers = nn.ModuleList()

            if level < len(channel_mult) - 1:
                layers.append(Upsample(cur_ch))

            for rb in range(num_res_blocks + 1):
                skip_con = skip_ch[-1]
                del skip_ch[-1]
                
                if rb == 0:
                    layers.append(ResBlock(cur_ch + skip_con, ch, self.time_embed_dim, dropout, use_scale_shift_norm))
                else:
                    layers.append(ResBlock(ch + skip_con, ch, self.time_embed_dim, dropout, use_scale_shift_norm))

                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))

            self.decoder.extend(layers)

            cur_ch = ch
            if level > 0:
                res *= 2

        assert res == 64
        assert len(skip_ch) == 0

        self.head = nn.Sequential(
            GroupNorm32(32, cur_ch),
            nn.SiLU(),
            nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement the forward pass of the unet
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """

        t_emb = self.time_embedding(t)
        skip_cons = []
        for module in self.encoder:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
            else:
                x = module(x)

            if isinstance(module, AttentionBlock):
                skip_cons[-1] = x
            else:
                skip_cons.append(x)


        for module in self.middle:
            if isinstance(module, ResBlock):
                x = module(x, t_emb)
            else:
                x = module(x)

        for module in self.decoder:
            if isinstance(module, ResBlock):
                x_skip = skip_cons[-1]
                del skip_cons[-1]
                x = torch.cat([x, x_skip], dim=1)

                x = module(x, t_emb)
            else:
                x = module(x)

        x = self.head(x)
        return x


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
