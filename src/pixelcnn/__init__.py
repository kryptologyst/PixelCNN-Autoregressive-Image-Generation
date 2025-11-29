"""
PixelCNN: Autoregressive Image Generation with Masked Convolutions

This package implements PixelCNN, an autoregressive generative model that generates
images pixel by pixel using masked convolutions to ensure proper autoregressive ordering.

Key Components:
- model: PixelCNN architecture with masked convolutions
- data: Data loading and preprocessing utilities
- training: PyTorch Lightning training module
- evaluation: Comprehensive evaluation metrics
- sampling: Various sampling strategies and visualization tools
"""

from .model import PixelCNN, MaskedConv2d, ResidualBlock
from .data import CIFAR10Dataset, CelebADataset, create_dataloaders, get_transforms
from .training import PixelCNNTrainer, train_pixelcnn, set_seed
from .evaluation import PixelCNNEvaluator, compute_model_comparison
from .sampling import PixelCNNSampler, create_interactive_sampler

__version__ = "1.0.0"
__author__ = "PixelCNN Implementation Team"

__all__ = [
    "PixelCNN",
    "MaskedConv2d", 
    "ResidualBlock",
    "CIFAR10Dataset",
    "CelebADataset",
    "create_dataloaders",
    "get_transforms",
    "PixelCNNTrainer",
    "train_pixelcnn",
    "set_seed",
    "PixelCNNEvaluator",
    "compute_model_comparison",
    "PixelCNNSampler",
    "create_interactive_sampler"
]
