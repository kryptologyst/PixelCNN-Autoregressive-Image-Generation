# PixelCNN Autoregressive Image Generation

A production-ready implementation of PixelCNN for autoregressive image generation using masked convolutions.

## Overview

PixelCNN is an autoregressive generative model that generates images pixel by pixel, where each pixel's value depends on the previously generated pixels above and to the left. This implementation provides:

- **Proper masked convolutions** ensuring autoregressive ordering
- **Residual blocks** for improved training stability
- **Multiple sampling strategies** (temperature, top-k, top-p)
- **Comprehensive evaluation metrics** (FID, Inception Score, LPIPS)
- **Interactive demo** with Streamlit
- **Production-ready structure** with proper logging and checkpointing

## Features

### Model Architecture
- Masked convolutions with Type A and Type B masks
- Residual blocks for deeper networks
- Configurable architecture parameters
- Support for different image sizes and channels

### Training
- PyTorch Lightning integration
- Automatic mixed precision training
- Gradient clipping and learning rate scheduling
- Comprehensive logging with WandB/TensorBoard
- Model checkpointing and early stopping

### Evaluation
- Fréchet Inception Distance (FID)
- Inception Score (IS)
- LPIPS diversity metrics
- Precision and Recall estimation
- Perceptual distance computation

### Sampling
- Autoregressive generation with various strategies
- Temperature scaling for controlling randomness
- Top-k and top-p (nucleus) sampling
- Progressive generation visualization
- Interactive sampling interface

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/PixelCNN-Autoregressive-Image-Generation.git
cd PixelCNN-Autoregressive-Image-Generation

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Training a Model

1. **Basic training with default configuration:**
```bash
python scripts/train.py --config configs/default.yaml
```

2. **Training with custom parameters:**
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --device cuda \
    --resume checkpoints/last.ckpt
```

3. **Training on CelebA dataset:**
```bash
# Modify configs/default.yaml to set dataset_name: "celeba"
python scripts/train.py --config configs/default.yaml
```

### Generating Samples

1. **Generate samples from trained model:**
```bash
python scripts/train.py \
    --sample-only \
    --resume checkpoints/best_model.ckpt
```

2. **Evaluate model:**
```bash
python scripts/train.py \
    --eval-only \
    --resume checkpoints/best_model.ckpt
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive sample generation
- Temperature and sampling parameter controls
- Progressive generation visualization
- Model comparison tools

## Configuration

The project uses YAML configuration files for easy parameter management:

```yaml
# Model configuration
model:
  in_channels: 3
  hidden_channels: 64
  num_layers: 12
  num_residual_blocks: 5
  dropout: 0.5
  num_classes: 256

# Training configuration
training:
  max_epochs: 100
  learning_rate: 1e-3
  weight_decay: 1e-4
  gradient_clip_val: 1.0
  precision: "16-mixed"

# Data configuration
data:
  dataset_name: "cifar10"
  batch_size: 64
  num_workers: 4
  image_size: 32
  use_augmentation: true
```

## Project Structure

```
0366_PixelCNN_implementation/
├── src/pixelcnn/           # Main package
│   ├── __init__.py
│   ├── model.py           # PixelCNN architecture
│   ├── data.py            # Data loading utilities
│   ├── training.py        # Training module
│   ├── evaluation.py      # Evaluation metrics
│   └── sampling.py        # Sampling utilities
├── configs/               # Configuration files
│   └── default.yaml
├── scripts/               # Training scripts
│   └── train.py
├── demo/                  # Interactive demo
│   └── app.py
├── tests/                 # Unit tests
│   └── test_pixelcnn.py
├── assets/                # Generated samples and visualizations
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── requirements.txt       # Dependencies
├── .gitignore
└── README.md
```

## Usage Examples

### Basic Model Creation
```python
from pixelcnn import PixelCNN

# Create model
model = PixelCNN(
    in_channels=3,
    hidden_channels=64,
    num_layers=12,
    num_residual_blocks=5
)

# Generate samples
samples = model.generate(
    shape=(3, 32, 32),
    num_samples=16,
    temperature=1.0
)
```

### Training with PyTorch Lightning
```python
from pixelcnn import PixelCNNTrainer, train_pixelcnn

# Train model
trainer_module = train_pixelcnn(
    model_config=model_config,
    training_config=training_config,
    data_config=data_config
)
```

### Evaluation
```python
from pixelcnn import PixelCNNEvaluator

# Evaluate model
evaluator = PixelCNNEvaluator(device)
metrics = evaluator.evaluate_model(model, test_loader)

print(f"FID: {metrics['fid']:.4f}")
print(f"Inception Score: {metrics['inception_score']:.4f}")
```

### Sampling Strategies
```python
from pixelcnn import PixelCNNSampler

sampler = PixelCNNSampler(model, device)

# Standard sampling
samples = sampler.generate_samples(
    num_samples=16,
    temperature=1.0
)

# Top-k sampling
samples = sampler.generate_samples(
    num_samples=16,
    temperature=1.0,
    top_k=50
)

# Top-p (nucleus) sampling
samples = sampler.generate_samples(
    num_samples=16,
    temperature=1.0,
    top_p=0.9
)
```

## Datasets

### CIFAR-10
- **Size**: 32x32 RGB images
- **Classes**: 10 object categories
- **Samples**: 50,000 training, 10,000 test
- **Usage**: Default dataset for quick experimentation

### CelebA
- **Size**: 64x64 RGB images (resized from 178x218)
- **Content**: Celebrity face images
- **Samples**: ~200,000 images
- **Usage**: Higher resolution training

### Custom Datasets
The data loading utilities support custom datasets by implementing the same interface as the provided datasets.

## Evaluation Metrics

### Fréchet Inception Distance (FID)
- Measures the distance between real and generated image distributions
- Lower values indicate better quality
- Uses Inception v3 features

### Inception Score (IS)
- Measures both quality and diversity
- Higher values indicate better performance
- Based on Inception v3 classifier predictions

### LPIPS Diversity
- Measures perceptual diversity within generated samples
- Uses learned perceptual similarity metrics
- Higher values indicate more diverse samples

### Precision and Recall
- Estimates precision (quality) and recall (diversity)
- Uses manifold estimation in feature space
- Provides balanced quality-diversity assessment

## Model Architecture Details

### Masked Convolutions
- **Type A Mask**: Excludes center pixel (first layer)
- **Type B Mask**: Includes center pixel (subsequent layers)
- Ensures autoregressive ordering (left-to-right, top-to-bottom)

### Residual Blocks
- Skip connections for improved gradient flow
- Batch normalization and ReLU activations
- Configurable number of residual blocks

### Output Layer
- Predicts discrete pixel values (0-255 for 8-bit images)
- Uses cross-entropy loss for training
- Supports different color depths

## Training Tips

### Hyperparameters
- **Learning Rate**: Start with 1e-3, reduce on plateau
- **Batch Size**: Use largest batch size that fits in memory
- **Gradient Clipping**: Helps with training stability
- **Dropout**: Use 0.5 for regularization

### Data Augmentation
- Horizontal flipping (for non-symmetric datasets)
- Color jittering for robustness
- Careful with augmentation that breaks autoregressive ordering

### Monitoring
- Watch validation loss for overfitting
- Generate samples periodically to check quality
- Use FID for objective quality assessment

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Poor Sample Quality**
   - Increase model capacity
   - Adjust temperature during sampling
   - Check data preprocessing

3. **Training Instability**
   - Use gradient clipping
   - Reduce learning rate
   - Add more regularization

### Performance Optimization

1. **GPU Acceleration**
   - Use CUDA when available
   - Enable mixed precision training
   - Use data parallel training

2. **Data Loading**
   - Increase number of workers
   - Use pin_memory for GPU training
   - Preprocess data offline when possible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black ruff

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{pixelcnn_implementation,
  title={PixelCNN: Autoregressive Image Generation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/PixelCNN-Autoregressive-Image-Generation}
}
```

## Acknowledgments

- Original PixelCNN paper by van den Oord et al.
- PyTorch Lightning for training framework
- Streamlit for interactive demo
- Various open-source libraries for evaluation metrics

## Model Card

### Intended Use
This model is designed for:
- Research in autoregressive generative modeling
- Educational purposes for understanding PixelCNN
- Generating synthetic images for data augmentation
- Exploring autoregressive image generation

### Limitations
- Slow generation due to autoregressive nature
- Limited to fixed image sizes during training
- May generate artifacts in complex scenes
- Training can be computationally expensive

### Bias and Fairness
- Model inherits biases from training data
- Generated images may reflect dataset biases
- Consider dataset diversity when training
- Evaluate generated samples for fairness

### Safety Considerations
- Generated images should not be used for malicious purposes
- Consider watermarking generated content
- Be aware of potential misuse in deepfake generation
- Follow responsible AI practices
# PixelCNN-Autoregressive-Image-Generation
