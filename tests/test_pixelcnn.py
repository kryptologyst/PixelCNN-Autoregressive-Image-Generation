"""
Unit tests for PixelCNN implementation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pixelcnn.model import PixelCNN, MaskedConv2d, ResidualBlock
from pixelcnn.data import CIFAR10Dataset, get_transforms
from pixelcnn.training import PixelCNNTrainer, set_seed
from pixelcnn.evaluation import PixelCNNEvaluator
from pixelcnn.sampling import PixelCNNSampler


class TestMaskedConv2d:
    """Test masked convolution layer."""
    
    def test_mask_creation(self):
        """Test that masks are created correctly."""
        conv = MaskedConv2d(3, 64, kernel_size=7, mask_type="A")
        assert conv.mask is not None
        assert conv.mask.shape == conv.weight.shape
        
        # Type A mask should exclude center pixel
        center_h, center_w = 3, 3  # kernel_size // 2
        assert conv.mask[0, 0, center_h, center_w] == 0
    
    def test_mask_type_b(self):
        """Test Type B mask."""
        conv = MaskedConv2d(3, 64, kernel_size=7, mask_type="B")
        
        # Type B mask should include center pixel
        center_h, center_w = 3, 3
        assert conv.mask[0, 0, center_h, center_w] == 1
    
    def test_forward_pass(self):
        """Test forward pass with masking."""
        conv = MaskedConv2d(3, 64, kernel_size=7, mask_type="A", padding=3)
        x = torch.randn(2, 3, 32, 32)
        
        output = conv(x)
        assert output.shape == (2, 64, 32, 32)


class TestResidualBlock:
    """Test residual block."""
    
    def test_residual_connection(self):
        """Test that residual connection works."""
        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        
        output = block(x)
        assert output.shape == x.shape
    
    def test_channel_change(self):
        """Test residual block with channel change."""
        block = ResidualBlock(32, 64)
        x = torch.randn(2, 32, 32, 32)
        
        output = block(x)
        assert output.shape == (2, 64, 32, 32)


class TestPixelCNN:
    """Test PixelCNN model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = PixelCNN(in_channels=3, hidden_channels=64, num_layers=6)
        assert model.in_channels == 3
        assert model.hidden_channels == 64
        assert model.num_classes == 256
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        x = torch.randn(2, 3, 32, 32)
        
        output = model(x)
        expected_shape = (2, 3 * 256, 32, 32)
        assert output.shape == expected_shape
    
    def test_compute_loss(self):
        """Test loss computation."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        x = torch.randn(2, 3, 32, 32)
        
        loss = model.compute_loss(x)
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_generation(self):
        """Test autoregressive generation."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        model.eval()
        
        with torch.no_grad():
            samples = model.generate(
                shape=(3, 16, 16),  # Smaller size for faster testing
                num_samples=2,
                temperature=1.0
            )
        
        assert samples.shape == (2, 3, 16, 16)
        assert torch.all(samples >= -1) and torch.all(samples <= 1)


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_transforms(self):
        """Test data transforms."""
        transform = get_transforms(image_size=32, is_training=True)
        assert isinstance(transform, nn.Module)
        
        # Test transform on random image
        x = torch.randn(3, 32, 32)
        transformed = transform(x)
        assert transformed.shape == (3, 32, 32)
    
    def test_cifar10_dataset(self):
        """Test CIFAR-10 dataset loading."""
        transform = get_transforms(image_size=32, is_training=False)
        dataset = CIFAR10Dataset(root="./test_data", train=True, download=False, transform=transform)
        
        # This will fail if data doesn't exist, which is expected in CI
        # In real testing, you'd download the data first
        try:
            sample, label = dataset[0]
            assert sample.shape == (3, 32, 32)
            assert isinstance(label, int)
        except FileNotFoundError:
            pytest.skip("CIFAR-10 data not available for testing")


class TestTraining:
    """Test training utilities."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.randn(5)
        np_rand = np.random.randn(5)
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.randn(5)
        np_rand2 = np.random.randn(5)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model_config = {
            "in_channels": 3,
            "hidden_channels": 32,
            "num_layers": 4,
            "num_residual_blocks": 2,
            "dropout": 0.1,
            "num_classes": 256
        }
        
        training_config = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_val": 1.0
        }
        
        data_config = {
            "dataset_name": "cifar10",
            "batch_size": 16,
            "num_workers": 0,  # Use 0 for testing
            "image_size": 32,
            "data_root": "./test_data",
            "use_augmentation": False
        }
        
        trainer = PixelCNNTrainer(model_config, training_config, data_config)
        assert trainer.model is not None
        assert trainer.learning_rate == 1e-3


class TestEvaluation:
    """Test evaluation utilities."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        device = torch.device("cpu")  # Use CPU for testing
        evaluator = PixelCNNEvaluator(device)
        
        assert evaluator.device == device
        assert evaluator.fid_metric is not None
        assert evaluator.is_metric is not None
    
    def test_metrics_computation(self):
        """Test metrics computation with dummy data."""
        device = torch.device("cpu")
        evaluator = PixelCNNEvaluator(device)
        
        # Create dummy data
        real_samples = torch.randn(10, 3, 32, 32)
        generated_samples = torch.randn(10, 3, 32, 32)
        
        # Test FID computation
        fid_score = evaluator._compute_fid(generated_samples, real_samples)
        assert isinstance(fid_score, float)
        assert fid_score >= 0
        
        # Test LPIPS diversity
        diversity = evaluator._compute_lpips_diversity(generated_samples)
        assert isinstance(diversity, float)
        assert diversity >= 0


class TestSampling:
    """Test sampling utilities."""
    
    def test_sampler_initialization(self):
        """Test sampler initialization."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        sampler = PixelCNNSampler(model)
        
        assert sampler.model == model
        assert sampler.device == next(model.parameters()).device
    
    def test_sample_generation(self):
        """Test sample generation."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        sampler = PixelCNNSampler(model)
        
        samples = sampler.generate_samples(
            num_samples=4,
            image_size=(3, 16, 16),
            temperature=1.0
        )
        
        assert samples.shape == (4, 3, 16, 16)
        assert torch.all(samples >= 0) and torch.all(samples <= 1)
    
    def test_temperature_comparison(self):
        """Test temperature comparison."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        sampler = PixelCNNSampler(model)
        
        temperatures = [0.5, 1.0, 1.5]
        
        for temp in temperatures:
            samples = sampler.generate_samples(
                num_samples=2,
                temperature=temp
            )
            assert samples.shape == (2, 3, 32, 32)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training_step(self):
        """Test a single training step."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy batch
        batch = torch.randn(4, 3, 32, 32)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        
        # Save model
        torch.save(model.state_dict(), "test_model.pth")
        
        # Load model
        new_model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)
        new_model.load_state_dict(torch.load("test_model.pth"))
        
        # Test that models are equivalent
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1, output2)
        
        # Clean up
        Path("test_model.pth").unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
