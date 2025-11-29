"""
Evaluation metrics and utilities for PixelCNN.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import precision_recall_curve, auc
import lpips
from clean_fid import fid

from .model import PixelCNN
from .data import denormalize_image


class PixelCNNEvaluator:
    """Evaluator for PixelCNN models."""
    
    def __init__(self, device: torch.device = None):
        """
        Initialize evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize metrics
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.is_metric = InceptionScore().to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)
        
        # LPIPS model for diversity
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
    def evaluate_model(
        self,
        model: PixelCNN,
        test_loader: torch.utils.data.DataLoader,
        num_generated_samples: int = 1000,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate PixelCNN model comprehensively.
        
        Args:
            model: Trained PixelCNN model
            test_loader: Test data loader
            num_generated_samples: Number of samples to generate for evaluation
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        metrics = {}
        
        # Generate samples
        print("Generating samples for evaluation...")
        generated_samples = self._generate_samples(
            model, num_generated_samples, temperature, top_k, top_p
        )
        
        # Get real samples
        real_samples = self._get_real_samples(test_loader, num_generated_samples)
        
        # Compute metrics
        print("Computing evaluation metrics...")
        
        # FID
        fid_score = self._compute_fid(generated_samples, real_samples)
        metrics["fid"] = fid_score
        
        # Inception Score
        is_score = self._compute_inception_score(generated_samples)
        metrics["inception_score"] = is_score
        
        # LPIPS diversity
        lpips_diversity = self._compute_lpips_diversity(generated_samples)
        metrics["lpips_diversity"] = lpips_diversity
        
        # Precision and Recall
        precision, recall = self._compute_precision_recall(generated_samples, real_samples)
        metrics["precision"] = precision
        metrics["recall"] = recall
        
        # Perceptual distance
        perceptual_distance = self._compute_perceptual_distance(generated_samples, real_samples)
        metrics["perceptual_distance"] = perceptual_distance
        
        return metrics
    
    def _generate_samples(
        self,
        model: PixelCNN,
        num_samples: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> torch.Tensor:
        """Generate samples from the model."""
        samples = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                
                batch_samples = model.generate(
                    shape=(3, 32, 32),
                    num_samples=current_batch_size,
                    device=self.device,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Denormalize and clamp
                batch_samples = denormalize_image(batch_samples)
                batch_samples = torch.clamp(batch_samples, 0, 1)
                
                samples.append(batch_samples)
        
        return torch.cat(samples, dim=0)
    
    def _get_real_samples(
        self,
        test_loader: torch.utils.data.DataLoader,
        num_samples: int
    ) -> torch.Tensor:
        """Get real samples from test loader."""
        samples = []
        count = 0
        
        for batch_images, _ in test_loader:
            if count >= num_samples:
                break
            
            batch_images = batch_images.to(self.device)
            batch_images = denormalize_image(batch_images)
            batch_images = torch.clamp(batch_images, 0, 1)
            
            samples.append(batch_images)
            count += batch_images.size(0)
        
        samples = torch.cat(samples, dim=0)
        return samples[:num_samples]
    
    def _compute_fid(self, generated: torch.Tensor, real: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance."""
        self.fid_metric.reset()
        
        # Update with real and generated samples
        self.fid_metric.update(real, real=True)
        self.fid_metric.update(generated, real=False)
        
        return self.fid_metric.compute().item()
    
    def _compute_inception_score(self, generated: torch.Tensor) -> float:
        """Compute Inception Score."""
        self.is_metric.reset()
        self.is_metric.update(generated)
        
        return self.is_metric.compute().item()
    
    def _compute_lpips_diversity(self, generated: torch.Tensor) -> float:
        """Compute LPIPS diversity within generated samples."""
        if generated.size(0) < 2:
            return 0.0
        
        # Sample pairs for diversity computation
        num_pairs = min(1000, generated.size(0) * (generated.size(0) - 1) // 2)
        distances = []
        
        with torch.no_grad():
            for _ in range(num_pairs):
                # Randomly sample two different images
                idx1, idx2 = torch.randperm(generated.size(0))[:2]
                img1, img2 = generated[idx1:idx1+1], generated[idx2:idx2+1]
                
                # Compute LPIPS distance
                distance = self.lpips_model(img1, img2)
                distances.append(distance.item())
        
        return np.mean(distances)
    
    def _compute_precision_recall(
        self,
        generated: torch.Tensor,
        real: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute Precision and Recall using manifold estimation."""
        # Use Inception features for manifold estimation
        from torchvision.models import inception_v3
        inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
        inception.eval()
        
        def get_features(images):
            with torch.no_grad():
                # Resize to 299x299 for Inception
                images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
                features = inception(images_resized)
                return features.cpu().numpy()
        
        # Get features
        real_features = get_features(real)
        generated_features = get_features(generated)
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import pairwise_distances
        
        real_distances = pairwise_distances(real_features, real_features)
        generated_distances = pairwise_distances(generated_features, generated_features)
        cross_distances = pairwise_distances(generated_features, real_features)
        
        # Compute precision and recall
        precision = np.mean([np.min(cross_distances[i]) for i in range(len(generated_features))])
        recall = np.mean([np.min(cross_distances[:, i]) for i in range(len(real_features))])
        
        return precision, recall
    
    def _compute_perceptual_distance(
        self,
        generated: torch.Tensor,
        real: torch.Tensor
    ) -> float:
        """Compute average perceptual distance between generated and real samples."""
        distances = []
        
        with torch.no_grad():
            for i in range(min(100, generated.size(0))):
                # Find closest real sample
                gen_img = generated[i:i+1]
                real_distances = []
                
                for j in range(min(100, real.size(0))):
                    real_img = real[j:j+1]
                    distance = self.lpips_model(gen_img, real_img)
                    real_distances.append(distance.item())
                
                distances.append(min(real_distances))
        
        return np.mean(distances)
    
    def generate_comparison_grid(
        self,
        model: PixelCNN,
        real_images: torch.Tensor,
        num_samples: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> plt.Figure:
        """Generate comparison grid of real vs generated images."""
        model.eval()
        
        # Generate samples
        generated = model.generate(
            shape=(3, 32, 32),
            num_samples=num_samples,
            device=self.device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Denormalize
        generated = denormalize_image(generated)
        generated = torch.clamp(generated, 0, 1)
        
        real_images = denormalize_image(real_images[:num_samples])
        real_images = torch.clamp(real_images, 0, 1)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # Real images
            axes[0, i].imshow(real_images[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title("Real")
            axes[0, i].axis('off')
            
            # Generated images
            axes[1, i].imshow(generated[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title("Generated")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training curves."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def compute_model_comparison(
    models: Dict[str, PixelCNN],
    test_loader: torch.utils.data.DataLoader,
    evaluator: PixelCNNEvaluator
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple PixelCNN models.
    
    Args:
        models: Dictionary of model names to PixelCNN models
        test_loader: Test data loader
        evaluator: Evaluator instance
        
    Returns:
        Dictionary of metrics for each model
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluator.evaluate_model(model, test_loader)
        results[model_name] = metrics
    
    return results


def create_metrics_report(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a comprehensive metrics report."""
    model_names = list(results.keys())
    metrics = list(results[model_names[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i >= 4:
            break
            
        values = [results[model][metric] for model in model_names]
        
        bars = axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
