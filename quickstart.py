#!/usr/bin/env python3
"""
Quick start script for PixelCNN.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description="Quick start PixelCNN")
    parser.add_argument("--action", choices=["setup", "train", "sample", "demo", "test"], 
                      default="setup", help="Action to perform")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--checkpoint", default="checkpoints/last.ckpt", help="Checkpoint path")
    
    args = parser.parse_args()
    
    if args.action == "setup":
        print("🚀 Running setup...")
        subprocess.run([sys.executable, "setup.py"], check=True)
        
    elif args.action == "train":
        print("🏋️ Starting training...")
        subprocess.run([
            sys.executable, "scripts/train.py",
            "--config", args.config
        ], check=True)
        
    elif args.action == "sample":
        print("🎨 Generating samples...")
        subprocess.run([
            sys.executable, "scripts/train.py",
            "--sample-only",
            "--resume", args.checkpoint
        ], check=True)
        
    elif args.action == "demo":
        print("🎪 Launching demo...")
        subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
        
    elif args.action == "test":
        print("🧪 Running tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)


if __name__ == "__main__":
    main()
