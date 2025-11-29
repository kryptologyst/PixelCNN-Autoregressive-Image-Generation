"""
Training utilities and Lightning module for PixelCNN.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from .model import PixelCNN
from .data import create_dataloaders, denormalize_image


class PixelCNNTrainer(pl.LightningModule):
    """PyTorch Lightning module for PixelCNN training."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        data_config: Dict[str, Any]
    ):
        """
        Initialize PixelCNN trainer.
        
        Args:
            model_config: Model configuration parameters
            training_config: Training configuration parameters
            data_config: Data configuration parameters
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store configs
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Initialize model
        self.model = PixelCNN(**model_config)
        
        # Training parameters
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.gradient_clip_val = training_config.get("gradient_clip_val", 1.0)
        
        # Sampling parameters
        self.sample_temperature = training_config.get("sample_temperature", 1.0)
        self.sample_top_k = training_config.get("sample_top_k", None)
        self.sample_top_p = training_config.get("sample_top_p", None)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, _ = batch
        
        # Compute loss
        loss = self.model.compute_loss(images)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images, _ = batch
        
        # Compute loss
        loss = self.model.compute_loss(images)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Generate samples
        if self.current_epoch % 5 == 0:  # Generate samples every 5 epochs
            self.generate_and_log_samples()
    
    def generate_and_log_samples(self, num_samples: int = 16) -> None:
        """Generate and log sample images."""
        self.model.eval()
        with torch.no_grad():
            # Generate samples
            samples = self.model.generate(
                shape=(3, 32, 32),  # CIFAR-10 size
                num_samples=num_samples,
                device=self.device,
                temperature=self.sample_temperature,
                top_k=self.sample_top_k,
                top_p=self.sample_top_p
            )
            
            # Denormalize images
            samples = denormalize_image(samples)
            samples = torch.clamp(samples, 0, 1)
            
            # Create grid
            grid = vutils.make_grid(samples, nrow=4, normalize=False)
            
            # Log to wandb if available
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({
                    "generated_samples": wandb.Image(grid),
                    "epoch": self.current_epoch
                })
            
            # Log to tensorboard if available
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image(
                    "generated_samples", grid, self.current_epoch
                )
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }
    
    def configure_callbacks(self) -> list:
        """Configure training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints",
            filename="pixelcnn-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            min_delta=1e-4,
            patience=20,
            verbose=True,
            mode="min"
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        return callbacks


def train_pixelcnn(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_config: Dict[str, Any],
    logger_config: Optional[Dict[str, Any]] = None
) -> PixelCNNTrainer:
    """
    Train PixelCNN model.
    
    Args:
        model_config: Model configuration parameters
        training_config: Training configuration parameters
        data_config: Data configuration parameters
        logger_config: Logger configuration parameters
        
    Returns:
        Trained PixelCNN trainer
    """
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(**data_config)
    
    # Initialize trainer
    trainer_module = PixelCNNTrainer(model_config, training_config, data_config)
    
    # Configure logger
    logger = None
    if logger_config and logger_config.get("use_wandb", False):
        logger = WandbLogger(
            project=logger_config.get("project_name", "pixelcnn"),
            name=logger_config.get("run_name", "pixelcnn_run"),
            log_model=True
        )
    elif logger_config and logger_config.get("use_tensorboard", True):
        logger = TensorBoardLogger(
            save_dir=logger_config.get("log_dir", "./logs"),
            name="pixelcnn"
        )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 100),
        accelerator="auto",
        devices="auto",
        precision=training_config.get("precision", "16-mixed"),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        logger=logger,
        callbacks=trainer_module.configure_callbacks(),
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train model
    trainer.fit(trainer_module, train_loader, val_loader)
    
    return trainer_module


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
