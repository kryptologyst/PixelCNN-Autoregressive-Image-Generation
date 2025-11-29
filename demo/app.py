"""
Interactive demo for PixelCNN using Streamlit.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pixelcnn.model import PixelCNN
from pixelcnn.sampling import PixelCNNSampler
from pixelcnn.data import denormalize_image


@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device):
    """Load trained PixelCNN model."""
    from pixelcnn.training import PixelCNNTrainer
    
    trainer_module = PixelCNNTrainer.load_from_checkpoint(checkpoint_path)
    model = trainer_module.model.to(device)
    model.eval()
    
    return model


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Denormalize and clamp
    image = denormalize_image(tensor)
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy and PIL
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    return Image.fromarray(image_np)


def create_sample_grid(samples: torch.Tensor, nrow: int = 4) -> Image.Image:
    """Create a grid of samples."""
    num_samples = samples.size(0)
    ncol = min(nrow, num_samples)
    nrow = (num_samples + ncol - 1) // ncol
    
    # Create grid
    grid_size = (nrow * 32, ncol * 32, 3)
    grid = np.zeros(grid_size, dtype=np.uint8)
    
    for i, sample in enumerate(samples):
        row = i // ncol
        col = i % ncol
        
        sample_img = tensor_to_image(sample)
        grid[row*32:(row+1)*32, col*32:(col+1)*32] = np.array(sample_img)
    
    return Image.fromarray(grid)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="PixelCNN Demo",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 PixelCNN Interactive Demo")
    st.markdown("Generate images pixel by pixel using autoregressive modeling")
    
    # Sidebar controls
    st.sidebar.header("Generation Controls")
    
    # Model selection
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoints/last.ckpt",
        help="Path to trained model checkpoint"
    )
    
    # Generation parameters
    num_samples = st.sidebar.slider("Number of Samples", 1, 16, 8)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    
    # Sampling strategy
    sampling_method = st.sidebar.selectbox(
        "Sampling Method",
        ["Standard", "Top-k", "Top-p", "Both"]
    )
    
    top_k = None
    top_p = None
    
    if sampling_method in ["Top-k", "Both"]:
        top_k = st.sidebar.slider("Top-k", 1, 100, 50)
    
    if sampling_method in ["Top-p", "Both"]:
        top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
    
    # Seed control
    use_seed = st.sidebar.checkbox("Use Fixed Seed", value=False)
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Seed", value=42, min_value=0)
    
    # Load model
    if st.sidebar.button("Load Model") or Path(checkpoint_path).exists():
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(checkpoint_path, device)
            sampler = PixelCNNSampler(model, device)
            
            st.sidebar.success("Model loaded successfully!")
            
            # Generate samples
            if st.sidebar.button("Generate Samples"):
                with st.spinner("Generating samples..."):
                    samples = sampler.generate_samples(
                        num_samples=num_samples,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed
                    )
                
                # Display samples
                st.header("Generated Samples")
                
                # Create grid
                grid_img = create_sample_grid(samples)
                st.image(grid_img, caption=f"Generated samples (T={temperature})")
                
                # Individual samples
                st.subheader("Individual Samples")
                cols = st.columns(4)
                for i, sample in enumerate(samples):
                    with cols[i % 4]:
                        sample_img = tensor_to_image(sample)
                        st.image(sample_img, caption=f"Sample {i+1}")
                
                # Download button
                buf = io.BytesIO()
                grid_img.save(buf, format="PNG")
                buf.seek(0)
                
                st.download_button(
                    label="Download Grid",
                    data=buf.getvalue(),
                    file_name=f"pixelcnn_samples_t{temperature}.png",
                    mime="image/png"
                )
            
            # Progressive generation
            if st.sidebar.button("Progressive Generation"):
                with st.spinner("Generating progressive sample..."):
                    final_image, intermediate_images = sampler.generate_with_progressive_sampling(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                
                st.header("Progressive Generation")
                
                # Show final image
                final_img = tensor_to_image(final_image[0])
                st.image(final_img, caption="Final Generated Image")
                
                # Show intermediate steps
                st.subheader("Generation Steps")
                step_cols = st.columns(4)
                for i, intermediate in enumerate(intermediate_images[:16]):  # Show first 16 steps
                    with step_cols[i % 4]:
                        intermediate_img = tensor_to_image(intermediate[0])
                        st.image(intermediate_img, caption=f"Step {i+1}")
            
            # Temperature comparison
            if st.sidebar.button("Temperature Comparison"):
                with st.spinner("Generating temperature comparison..."):
                    temperatures = [0.5, 1.0, 1.5, 2.0]
                    comparison_samples = []
                    
                    for temp in temperatures:
                        samples = sampler.generate_samples(
                            num_samples=4,
                            temperature=temp,
                            seed=seed
                        )
                        comparison_samples.append(samples)
                
                st.header("Temperature Comparison")
                
                for i, temp in enumerate(temperatures):
                    st.subheader(f"Temperature = {temp}")
                    temp_grid = create_sample_grid(comparison_samples[i])
                    st.image(temp_grid)
            
            # Model info
            st.sidebar.header("Model Information")
            st.sidebar.info(f"""
            **Model Architecture:**
            - Hidden Channels: {model.hidden_channels}
            - Residual Blocks: {len(model.residual_blocks)}
            - Total Layers: {model.num_layers}
            - Dropout: {model.dropout.p if hasattr(model.dropout, 'p') else 'N/A'}
            """)
            
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    
    else:
        st.warning("Please provide a valid checkpoint path to load the model.")
    
    # Information section
    st.header("About PixelCNN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **PixelCNN** is an autoregressive generative model that generates images pixel by pixel.
        
        **Key Features:**
        - Uses masked convolutions to ensure autoregressive ordering
        - Generates high-quality images one pixel at a time
        - Supports various sampling strategies (temperature, top-k, top-p)
        - Can be trained on various datasets (CIFAR-10, CelebA, etc.)
        """)
    
    with col2:
        st.markdown("""
        **Sampling Parameters:**
        - **Temperature**: Controls randomness (lower = more conservative)
        - **Top-k**: Limits sampling to top-k most likely tokens
        - **Top-p**: Nucleus sampling, keeps tokens with cumulative probability ≤ p
        
        **Generation Process:**
        1. Start with empty image
        2. Generate pixels left-to-right, top-to-bottom
        3. Each pixel depends on previously generated pixels
        4. Use masked convolutions to enforce this dependency
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with PyTorch, PyTorch Lightning, and Streamlit")


if __name__ == "__main__":
    main()
