"""
Sampling and visualization utilities for PixelCNN.
"""

from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image
import io
import base64

from .model import PixelCNN
from .data import denormalize_image


class PixelCNNSampler:
    """Sampler for PixelCNN with various generation strategies."""
    
    def __init__(self, model: PixelCNN, device: torch.device = None):
        """
        Initialize sampler.
        
        Args:
            model: Trained PixelCNN model
            device: Device to run sampling on
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
    
    def generate_samples(
        self,
        num_samples: int = 16,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples with specified parameters.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of images to generate (channels, height, width)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            seed: Random seed for reproducibility
            
        Returns:
            Generated samples tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        with torch.no_grad():
            samples = self.model.generate(
                shape=image_size,
                num_samples=num_samples,
                device=self.device,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Denormalize and clamp
        samples = denormalize_image(samples)
        samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def generate_with_progressive_sampling(
        self,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        save_intermediate: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a single image with progressive sampling visualization.
        
        Args:
            image_size: Size of image to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            save_intermediate: Whether to save intermediate steps
            
        Returns:
            Tuple of (final_image, intermediate_images)
        """
        channels, height, width = image_size
        image = torch.zeros(1, channels, height, width, device=self.device)
        intermediate_images = []
        
        self.model.eval()
        with torch.no_grad():
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        # Get current pixel logits
                        logits = self.model(image)
                        pixel_logits = logits[:, c * self.model.num_classes:(c + 1) * self.model.num_classes, h, w]
                        
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
                        image[:, c, h, w] = (pixel_values.float() / (self.model.num_classes - 1)) * 2 - 1
                        
                        # Save intermediate image
                        if save_intermediate and (h * width + w) % 10 == 0:  # Save every 10 pixels
                            intermediate = denormalize_image(image.clone())
                            intermediate = torch.clamp(intermediate, 0, 1)
                            intermediate_images.append(intermediate)
        
        # Final image
        final_image = denormalize_image(image)
        final_image = torch.clamp(final_image, 0, 1)
        
        return final_image, intermediate_images
    
    def interpolate_latent_space(
        self,
        start_image: torch.Tensor,
        end_image: torch.Tensor,
        num_steps: int = 10,
        temperature: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Interpolate between two images in the latent space.
        
        Args:
            start_image: Starting image
            end_image: Ending image
            num_steps: Number of interpolation steps
            temperature: Sampling temperature
            
        Returns:
            List of interpolated images
        """
        interpolated_images = []
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            
            # Linear interpolation
            interpolated = (1 - alpha) * start_image + alpha * end_image
            
            # Generate from interpolated image
            with torch.no_grad():
                generated = self.model.generate(
                    shape=start_image.shape[1:],
                    num_samples=1,
                    device=self.device,
                    temperature=temperature
                )
                
                # Blend with interpolation
                final_image = (1 - alpha) * start_image + alpha * generated
                final_image = denormalize_image(final_image)
                final_image = torch.clamp(final_image, 0, 1)
                
                interpolated_images.append(final_image)
        
        return interpolated_images
    
    def create_sampling_grid(
        self,
        num_samples: int = 16,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None
    ) -> plt.Figure:
        """Create a grid of generated samples."""
        samples = self.generate_samples(
            num_samples=num_samples,
            image_size=image_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed
        )
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(grid_size * grid_size):
            if i < num_samples:
                axes[i].imshow(samples[i].permute(1, 2, 0).cpu().numpy())
                axes[i].set_title(f'Sample {i+1}')
            else:
                axes[i].axis('off')
            
            axes[i].axis('off')
        
        plt.suptitle(f'Generated Samples (T={temperature}, Top-k={top_k}, Top-p={top_p})')
        plt.tight_layout()
        
        return fig
    
    def create_temperature_comparison(
        self,
        temperatures: List[float] = [0.5, 1.0, 1.5, 2.0],
        num_samples: int = 4,
        image_size: Tuple[int, int, int] = (3, 32, 32)
    ) -> plt.Figure:
        """Create comparison of samples at different temperatures."""
        fig, axes = plt.subplots(len(temperatures), num_samples, figsize=(12, 3 * len(temperatures)))
        
        if len(temperatures) == 1:
            axes = axes.reshape(1, -1)
        
        for i, temp in enumerate(temperatures):
            samples = self.generate_samples(
                num_samples=num_samples,
                image_size=image_size,
                temperature=temp
            )
            
            for j in range(num_samples):
                axes[i, j].imshow(samples[j].permute(1, 2, 0).cpu().numpy())
                axes[i, j].set_title(f'T={temp}')
                axes[i, j].axis('off')
        
        plt.suptitle('Temperature Comparison')
        plt.tight_layout()
        
        return fig
    
    def create_progressive_animation(
        self,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """Create animation of progressive image generation."""
        _, intermediate_images = self.generate_with_progressive_sampling(
            image_size=image_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            save_intermediate=True
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_title('Progressive Generation')
        ax.axis('off')
        
        def animate(frame):
            ax.clear()
            ax.imshow(intermediate_images[frame][0].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'Step {frame + 1}/{len(intermediate_images)}')
            ax.axis('off')
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(intermediate_images),
            interval=200, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        
        return anim
    
    def analyze_pixel_dependencies(
        self,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        num_samples: int = 100
    ) -> plt.Figure:
        """Analyze pixel dependencies in generated images."""
        samples = self.generate_samples(
            num_samples=num_samples,
            image_size=image_size,
            temperature=1.0
        )
        
        # Convert to numpy for analysis
        samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()
        
        # Compute pixel correlations
        correlations = []
        for i in range(samples_np.shape[1] - 1):  # height
            for j in range(samples_np.shape[2] - 1):  # width
                # Correlation between adjacent pixels
                corr = np.corrcoef(
                    samples_np[:, i, j, 0].flatten(),
                    samples_np[:, i, j+1, 0].flatten()
                )[0, 1]
                correlations.append(corr)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sample images
        for i in range(4):
            ax = axes[i//2, i%2]
            ax.imshow(samples_np[i])
            ax.set_title(f'Sample {i+1}')
            ax.axis('off')
        
        plt.suptitle('Pixel Dependency Analysis')
        plt.tight_layout()
        
        return fig


def create_interactive_sampler(
    model: PixelCNN,
    device: torch.device = None
) -> PixelCNNSampler:
    """Create an interactive sampler for the model."""
    return PixelCNNSampler(model, device)


def save_samples_as_images(
    samples: torch.Tensor,
    save_dir: str,
    prefix: str = "sample"
) -> List[str]:
    """
    Save generated samples as image files.
    
    Args:
        samples: Generated samples tensor
        save_dir: Directory to save images
        prefix: Prefix for filenames
        
    Returns:
        List of saved file paths
    """
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    
    for i, sample in enumerate(samples):
        # Convert to PIL Image
        sample_np = sample.permute(1, 2, 0).cpu().numpy()
        sample_np = (sample_np * 255).astype(np.uint8)
        image = Image.fromarray(sample_np)
        
        # Save image
        filename = f"{prefix}_{i:04d}.png"
        filepath = os.path.join(save_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths
