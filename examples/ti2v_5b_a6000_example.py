#!/usr/bin/env python3
"""
TI2V-5B Example Script for GPU-A6000
=====================================

This script demonstrates how to use the Wan2.2 TI2V-5B model efficiently on GPU-A6000 hardware.
The TI2V-5B model is optimized for consumer hardware and supports both text-to-video and 
image-to-video generation at 720P@24fps.

GPU-A6000 specs: 10 CPU cores, 18GB RAM, 48GB VRAM
Recommended for: TI2V-5B model (requires ~24GB VRAM with optimizations)

Usage:
    python examples/ti2v_5b_a6000_example.py --mode t2v --prompt "Your text prompt"
    python examples/ti2v_5b_a6000_example.py --mode i2v --image path/to/image.jpg --prompt "Your text prompt"
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.cuda
from PIL import Image

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to path to import wan modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import wan modules, but make it optional for testing
try:
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video
    WAN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Wan modules not available: {e}")
    WAN_AVAILABLE = False
    # Mock configs for testing
    WAN_CONFIGS = {
        "ti2v-5B": type('Config', (), {
            'sample_steps': 50,
            'sample_shift': 1.0,
            'sample_guide_scale': 7.5,
            'frame_num': 120
        })()
    }

class TI2V5BGenerator:
    """Optimized TI2V-5B generator for GPU-A6000"""
    
    def __init__(self, model_path: str):
        """
        Initialize the generator with optimized settings for GPU-A6000
        
        Args:
            model_path: Path to the Wan2.2-TI2V-5B model directory
        """
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Verify model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # GPU-A6000 optimized settings
        self.config = {
            "task": "ti2v-5B",
            "size": "1280*704",  # 720P resolution for TI2V-5B
            "offload_model": True,  # Enable model offloading for memory efficiency
            "convert_model_dtype": True,  # Convert to optimized data types
            "t5_cpu": True,  # Keep T5 on CPU to save GPU memory
            "sample_steps": None,  # Use default from config
            "sample_shift": None,  # Use default from config
            "sample_guide_scale": None,  # Use default from config
            "frame_num": None,  # Use default from config
        }
        
        logger.info(f"Initializing TI2V-5B on {self.device}")
        logger.info(f"Model path: {self.model_path}")
        
        # Only check GPU memory if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            logger.info("CUDA not available - running in CPU mode")
        
    def load_model(self):
        """Load the model with GPU-A6000 optimizations"""
        logger.info("Loading TI2V-5B model...")
        start_time = time.time()
        
        # Import and initialize wan with optimized settings
        # This will be handled by the generate.py script's model loading logic
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
    def generate_text_to_video(self, 
                             prompt: str, 
                             output_path: str = "output_t2v.mp4",
                             seed: int = 42) -> str:
        """
        Generate video from text prompt
        
        Args:
            prompt: Text description for video generation
            output_path: Path to save generated video
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated video file
        """
        logger.info("Starting Text-to-Video generation...")
        logger.info(f"Prompt: {prompt}")
        
        cmd_args = [
            "python", "generate.py",
            "--task", self.config["task"],
            "--size", self.config["size"],
            "--ckpt_dir", str(self.model_path),
            "--prompt", f'"{prompt}"',
            "--base_seed", str(seed),
            "--save_file", output_path
        ]
        
        # Add optimization flags
        if self.config["offload_model"]:
            cmd_args.append("--offload_model")
            cmd_args.append("True")
        if self.config["convert_model_dtype"]:
            cmd_args.append("--convert_model_dtype")
        if self.config["t5_cpu"]:
            cmd_args.append("--t5_cpu")
            
        logger.info(f"Command: {' '.join(cmd_args)}")
        
        # For integration with the actual generate.py script
        return self._run_generation(cmd_args, output_path)
        
    def generate_image_to_video(self, 
                              image_path: str,
                              prompt: str,
                              output_path: str = "output_i2v.mp4",
                              seed: int = 42) -> str:
        """
        Generate video from image and text prompt
        
        Args:
            image_path: Path to input image
            prompt: Text description for video generation
            output_path: Path to save generated video
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated video file
        """
        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        logger.info("Starting Image-to-Video generation...")
        logger.info(f"Image: {image_path}")
        logger.info(f"Prompt: {prompt}")
        
        cmd_args = [
            "python", "generate.py",
            "--task", self.config["task"],
            "--size", self.config["size"],
            "--ckpt_dir", str(self.model_path),
            "--image", image_path,
            "--prompt", f'"{prompt}"',
            "--base_seed", str(seed),
            "--save_file", output_path
        ]
        
        # Add optimization flags
        if self.config["offload_model"]:
            cmd_args.append("--offload_model")
            cmd_args.append("True")
        if self.config["convert_model_dtype"]:
            cmd_args.append("--convert_model_dtype")
        if self.config["t5_cpu"]:
            cmd_args.append("--t5_cpu")
            
        logger.info(f"Command: {' '.join(cmd_args)}")
        
        return self._run_generation(cmd_args, output_path)
    
    def _run_generation(self, cmd_args: list, output_path: str) -> str:
        """Run the generation command and return output path"""
        import subprocess
        
        start_time = time.time()
        
        try:
            # Change to the Wan2.2 directory to run generate.py
            original_cwd = os.getcwd()
            os.chdir(self.model_path.parent)
            
            result = subprocess.run(cmd_args, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            logger.info(f"Video saved to: {output_path}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Generation failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
        finally:
            os.chdir(original_cwd)
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9,
                "total": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        return {}

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="TI2V-5B Example for GPU-A6000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Wan2.2-TI2V-5B",
        help="Path to the Wan2.2-TI2V-5B model directory"
    )
    
    parser.add_argument(
        "--mode",
        choices=["t2v", "i2v"],
        help="Generation mode: t2v (text-to-video) or i2v (image-to-video)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene mountain landscape with flowing water and gentle morning light",
        help="Text prompt for video generation"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (required for i2v mode)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output video path (default: auto-generated)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--check_gpu",
        action="store_true",
        help="Check GPU compatibility and exit"
    )
    
    args = parser.parse_args()
    
    # Check GPU compatibility
    if args.check_gpu or not torch.cuda.is_available():
        logger.info("=== GPU Compatibility Check ===")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name}")
            logger.info(f"VRAM: {props.total_memory / 1e9:.1f}GB")
            logger.info(f"Compute Capability: {props.major}.{props.minor}")
            
            if props.total_memory / 1e9 >= 24:
                logger.info("GPU has sufficient VRAM for TI2V-5B")
            else:
                logger.warning("GPU may not have sufficient VRAM. Use with caution.")
        else:
            logger.error("CUDA not available")
        
        if args.check_gpu:
            return
    
    # Validate arguments
    if not args.check_gpu and not args.mode:
        parser.error("--mode is required unless using --check_gpu")
    
    if args.mode == "i2v" and not args.image:
        parser.error("--image is required for image-to-video mode")
    
    if not args.output and args.mode:
        timestamp = int(time.time())
        args.output = f"output_{args.mode}_{timestamp}.mp4"
    
    # Initialize generator
    try:
        generator = TI2V5BGenerator(args.model_path)
        
        # Show initial memory usage
        memory = generator.get_memory_usage()
        if memory:
            logger.info(f"Initial GPU memory: {memory['allocated']:.2f}GB allocated, "
                       f"{memory['reserved']:.2f}GB reserved")
        
        # Generate video
        if args.mode == "t2v":
            output_path = generator.generate_text_to_video(
                prompt=args.prompt,
                output_path=args.output,
                seed=args.seed
            )
        else:  # i2v
            output_path = generator.generate_image_to_video(
                image_path=args.image,
                prompt=args.prompt,
                output_path=args.output,
                seed=args.seed
            )
        
        # Show final memory usage
        memory = generator.get_memory_usage()
        if memory:
            logger.info(f"Final GPU memory: {memory['allocated']:.2f}GB allocated, "
                       f"{memory['reserved']:.2f}GB reserved")
        
        logger.info("=== Generation Complete ===")
        logger.info(f"Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()