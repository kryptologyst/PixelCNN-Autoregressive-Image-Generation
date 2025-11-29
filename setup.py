#!/usr/bin/env python3
"""
Setup script for PixelCNN project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up PixelCNN project...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create necessary directories
    directories = [
        "data", "checkpoints", "logs", "assets/samples", 
        "generated_samples", "test_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install pre-commit hooks
    if run_command("pre-commit install", "Installing pre-commit hooks"):
        print("✅ Pre-commit hooks installed")
    
    # Run tests to verify installation
    if run_command("python -m pytest tests/ -v", "Running tests"):
        print("✅ All tests passed")
    else:
        print("⚠️  Some tests failed (this might be expected if data is not available)")
    
    # Create a simple test script
    test_script = """
import torch
from src.pixelcnn import PixelCNN, set_seed

# Test basic functionality
set_seed(42)
model = PixelCNN(in_channels=3, hidden_channels=32, num_layers=4)

# Test forward pass
x = torch.randn(2, 3, 32, 32)
output = model(x)
print(f"✅ Forward pass successful: {output.shape}")

# Test generation
model.eval()
with torch.no_grad():
    samples = model.generate(shape=(3, 16, 16), num_samples=2)
    print(f"✅ Generation successful: {samples.shape}")

print("🎉 PixelCNN setup verification completed successfully!")
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    if run_command("python test_setup.py", "Verifying setup"):
        print("✅ Setup verification completed")
        os.remove("test_setup.py")
    else:
        print("❌ Setup verification failed")
        sys.exit(1)
    
    print("\n🎉 PixelCNN project setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Download CIFAR-10 dataset: python scripts/train.py --config configs/default.yaml")
    print("2. Train a model: python scripts/train.py --config configs/default.yaml")
    print("3. Generate samples: python scripts/train.py --sample-only --resume checkpoints/last.ckpt")
    print("4. Launch demo: streamlit run demo/app.py")
    print("5. Run notebook: jupyter notebook notebooks/pixelcnn_example.ipynb")


if __name__ == "__main__":
    main()
