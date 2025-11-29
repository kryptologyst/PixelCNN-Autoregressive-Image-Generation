"""
Main training script for PixelCNN.
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pixelcnn.training import train_pixelcnn, set_seed
from pixelcnn.evaluation import PixelCNNEvaluator
from pixelcnn.sampling import PixelCNNSampler
from pixelcnn.data import create_dataloaders


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PixelCNN model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate model")
    parser.add_argument("--sample-only", action="store_true", help="Only generate samples")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(42)
    
    if args.eval_only or args.sample_only:
        # Load model for evaluation/sampling
        from pixelcnn.training import PixelCNNTrainer
        
        # Load checkpoint
        if args.resume:
            trainer_module = PixelCNNTrainer.load_from_checkpoint(args.resume)
            model = trainer_module.model
        else:
            raise ValueError("Must provide checkpoint path for evaluation/sampling")
        
        model.to(device)
        
        if args.eval_only:
            # Evaluate model
            print("Evaluating model...")
            evaluator = PixelCNNEvaluator(device)
            _, _, test_loader = create_dataloaders(**config["data"])
            
            metrics = evaluator.evaluate_model(
                model, test_loader, **config["evaluation"]
            )
            
            print("Evaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        if args.sample_only:
            # Generate samples
            print("Generating samples...")
            sampler = PixelCNNSampler(model, device)
            
            # Generate sample grid
            fig = sampler.create_sampling_grid(**config["sampling"])
            fig.savefig("assets/samples/generated_samples.png", dpi=300, bbox_inches='tight')
            print("Samples saved to assets/samples/generated_samples.png")
            
            # Generate temperature comparison
            fig = sampler.create_temperature_comparison()
            fig.savefig("assets/samples/temperature_comparison.png", dpi=300, bbox_inches='tight')
            print("Temperature comparison saved to assets/samples/temperature_comparison.png")
    
    else:
        # Train model
        print("Training PixelCNN model...")
        
        # Create directories
        Path("checkpoints").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("assets/samples").mkdir(parents=True, exist_ok=True)
        
        # Train model
        trainer_module = train_pixelcnn(
            model_config=config["model"],
            training_config=config["training"],
            data_config=config["data"],
            logger_config=config["logger"]
        )
        
        print("Training completed!")
        
        # Generate final samples
        print("Generating final samples...")
        sampler = PixelCNNSampler(trainer_module.model, device)
        
        fig = sampler.create_sampling_grid(**config["sampling"])
        fig.savefig("assets/samples/final_samples.png", dpi=300, bbox_inches='tight')
        print("Final samples saved to assets/samples/final_samples.png")


if __name__ == "__main__":
    main()
