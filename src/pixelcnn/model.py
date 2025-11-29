"""
PixelCNN implementation with proper autoregressive masking.

This module implements a PixelCNN model that generates images pixel by pixel
using masked convolutions to ensure autoregressive generation.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d(nn.Conv2d):
    """Masked convolution layer for autoregressive generation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_type: str = "A",
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        **kwargs
    ):
        """
        Initialize masked convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            mask_type: Type of mask ("A" for first layer, "B" for subsequent layers)
            stride: Convolution stride
            padding: Convolution padding
            dilation: Convolution dilation
            groups: Convolution groups
            bias: Whether to use bias
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.mask_type = mask_type
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self._create_mask()
    
    def _create_mask(self) -> None:
        """Create the autoregressive mask."""
        h, w = self.kernel_size
        center_h, center_w = h // 2, w // 2
        
        # Create mask
        mask = torch.ones_like(self.weight)
        
        if self.mask_type == "A":
            # Type A mask: exclude center pixel
            mask[:, :, center_h, center_w:] = 0
            mask[:, :, center_h + 1:, :] = 0
        else:
            # Type B mask: include center pixel
            mask[:, :, center_h, center_w + 1:] = 0
            mask[:, :, center_h + 1:, :] = 0
        
        self.mask.copy_(mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked convolution."""
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlock(nn.Module):
    """Residual block with masked convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mask_type: str = "B"
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            mask_type: Type of mask for convolutions
        """
        super().__init__()
        
        self.conv1 = MaskedConv2d(
            in_channels, out_channels, kernel_size, mask_type=mask_type, padding=kernel_size//2
        )
        self.conv2 = MaskedConv2d(
            out_channels, out_channels, kernel_size, mask_type=mask_type, padding=kernel_size//2
        )
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.shortcut(x)
        
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        
        return F.relu(out + residual)


class PixelCNN(nn.Module):
    """
    PixelCNN model for autoregressive image generation.
    
    This model generates images pixel by pixel using masked convolutions
    to ensure proper autoregressive ordering.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 12,
        num_residual_blocks: int = 5,
        dropout: float = 0.5,
        num_classes: int = 256
    ):
        """
        Initialize PixelCNN model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            hidden_channels: Number of hidden channels
            num_layers: Number of convolutional layers
            num_residual_blocks: Number of residual blocks
            dropout: Dropout probability
            num_classes: Number of classes for each pixel (256 for 8-bit images)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # First layer with type A mask
        self.conv1 = MaskedConv2d(
            in_channels, hidden_channels, kernel_size=7, mask_type="A", padding=3
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, mask_type="B")
            for _ in range(num_residual_blocks)
        ])
        
        # Additional convolutional layers
        self.conv_layers = nn.ModuleList([
            MaskedConv2d(hidden_channels, hidden_channels, kernel_size=3, mask_type="B", padding=1)
            for _ in range(num_layers - num_residual_blocks - 1)
        ])
        
        # Output layers
        self.conv_out = MaskedConv2d(hidden_channels, hidden_channels, kernel_size=1, mask_type="B")
        self.dropout = nn.Dropout2d(dropout)
        self.conv_final = nn.Conv2d(hidden_channels, in_channels * num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PixelCNN.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels * num_classes, height, width)
        """
        # First layer
        out = F.relu(self.conv1(x))
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Additional conv layers
        for layer in self.conv_layers:
            out = F.relu(layer(out))
        
        # Output layers
        out = F.relu(self.conv_out(out))
        out = self.dropout(out)
        out = self.conv_final(out)
        
        return out
    
    def generate(
        self,
        shape: Tuple[int, int, int],
        num_samples: int = 1,
        device: torch.device = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate images autoregressively.
        
        Args:
            shape: Shape of images to generate (channels, height, width)
            num_samples: Number of samples to generate
            device: Device to generate on
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated images tensor of shape (num_samples, channels, height, width)
        """
        if device is None:
            device = next(self.parameters()).device
            
        channels, height, width = shape
        images = torch.zeros(num_samples, channels, height, width, device=device)
        
        self.eval()
        with torch.no_grad():
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        # Get current pixel logits
                        logits = self(images)
                        pixel_logits = logits[:, c * self.num_classes:(c + 1) * self.num_classes, h, w]
                        
                        # Apply temperature
                        pixel_logits = pixel_logits / temperature
                        
                        # Apply top-k filtering
                        if top_k is not None:
                            top_k = min(top_k, pixel_logits.size(-1))
                            topk_logits, topk_indices = torch.topk(pixel_logits, top_k)
                            pixel_logits = torch.full_like(pixel_logits, float('-inf'))
                            pixel_logits.scatter_(-1, topk_indices, topk_logits)
                        
                        # Apply top-p filtering
                        if top_p is not None:
                            sorted_logits, sorted_indices = torch.sort(pixel_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                            pixel_logits[indices_to_remove] = float('-inf')
                        
                        # Sample pixel value
                        probs = F.softmax(pixel_logits, dim=-1)
                        pixel_values = torch.multinomial(probs, 1).squeeze(-1)
                        
                        # Normalize to [-1, 1] range
                        images[:, c, h, w] = (pixel_values.float() / (self.num_classes - 1)) * 2 - 1
        
        return images
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Loss tensor
        """
        # Convert input to class indices
        x_scaled = (x + 1) / 2  # Scale from [-1, 1] to [0, 1]
        x_indices = (x_scaled * (self.num_classes - 1)).long().clamp(0, self.num_classes - 1)
        
        # Get predictions
        logits = self(x)
        
        # Reshape for cross-entropy loss
        batch_size, channels, height, width = x.shape
        logits = logits.view(batch_size, self.num_classes, channels, height, width)
        
        # Compute loss
        loss = F.cross_entropy(logits, x_indices, reduction='mean')
        
        return loss
