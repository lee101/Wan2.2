#!/usr/bin/env python3
"""
Simple example usage of the Ultimate FastVideo LoRA Optimizer.

This demonstrates how to use the optimizer for quick video generation with LoRA models.
"""

import os
import sys

# Add the current directory to path so we can import the optimizer
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ultimate_lora_t2v import UltimateFastVideoLoRAOptimizer

def main():
    # Configuration
    MODEL_PATH = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
    LORA_PATH = None  # Set this to your LoRA path, e.g., "path/to/your_lora.safetensors"
    LORA_HIGH_PATH = None  # Optional high noise LoRA
    
    # Example prompts
    prompts = [
        "A majestic eagle soaring through mountain valleys at golden hour",
        "Ocean waves crashing on rocky shores under a starlit sky",
        "A peaceful Japanese garden with cherry blossoms in spring",
    ]
    
    print("🚀 Ultimate FastVideo LoRA Optimizer Example")
    print("=" * 50)
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model path not found: {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your WAN2.2-Fun model")
        return
    
    # Initialize optimizer
    print("🔄 Initializing optimizer...")
    
    optimizer = UltimateFastVideoLoRAOptimizer(
        model_path=MODEL_PATH,
        lora_path=LORA_PATH,
        lora_high_path=LORA_HIGH_PATH,
        lora_weight=0.7,  # Adjust as needed
        lora_high_weight=0.6,  # Adjust as needed
        verbose=True
    )
    
    # Example 1: Single video generation with balanced quality
    print("\n📹 Example 1: Single video generation")
    try:
        video = optimizer.generate(
            prompt=prompts[0],
            quality="balanced",  # Good speed/quality balance
            video_length=49,     # Shorter for faster demo
            sample_size=[480, 832],
            fps=16,
            seed=42,
            save_videos=True,
            output_dir="example_outputs"
        )
        print("✅ Single video generated successfully!")
        
    except Exception as e:
        print(f"❌ Single video generation failed: {e}")
    
    # Example 2: Fast batch generation
    print("\n📹 Example 2: Batch generation with fast quality")
    try:
        videos = optimizer.generate(
            prompt=prompts,  # Multiple prompts
            quality="fast",  # Faster generation
            video_length=33, # Even shorter for batch demo
            sample_size=[384, 640],  # Smaller resolution
            fps=16,
            seed=123,
            save_videos=True,
            output_dir="example_batch_outputs"
        )
        print(f"✅ Batch generation completed! Generated {len(videos)} videos")
        
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
    
    # Example 3: Lightning speed demo
    print("\n⚡ Example 3: Lightning speed demo")
    try:
        video = optimizer.generate(
            prompt="A quick demo of lightning-fast generation",
            quality="lightning",  # Maximum speed
            video_length=17,      # Very short
            sample_size=[320, 576],  # Small resolution
            fps=8,                # Lower FPS
            seed=999,
            save_videos=True,
            output_dir="lightning_demo"
        )
        print("⚡ Lightning demo completed!")
        
    except Exception as e:
        print(f"❌ Lightning demo failed: {e}")
    
    # Example 4: Custom settings
    print("\n🛠️ Example 4: Custom settings")
    try:
        custom_settings = {
            'steps': 20,
            'guidance_scale': 7.5,
            'token_merge_ratio': 0.3,
            'taylorseer_threshold': 0.10,
            'enable_fp8': True,
            'enable_compilation': False
        }
        
        video = optimizer.generate(
            prompt="A video with custom optimization settings",
            quality="balanced",
            custom_settings=custom_settings,
            video_length=33,
            sample_size=[480, 832],
            seed=456,
            save_videos=True,
            output_dir="custom_outputs"
        )
        print("🛠️ Custom settings example completed!")
        
    except Exception as e:
        print(f"❌ Custom settings example failed: {e}")
    
    print("\n🎉 All examples completed!")
    print("\nGenerated videos can be found in:")
    print("  - example_outputs/")
    print("  - example_batch_outputs/")
    print("  - lightning_demo/")
    print("  - custom_outputs/")
    
    print("\n💡 Tips for best results:")
    print("  - Use 'balanced' or 'fast' quality for daily use")
    print("  - Try 'lightning' for quick previews")
    print("  - Use 'quality' or 'maximum' for final outputs")
    print("  - Adjust LoRA weights between 0.3-1.0 for different effects")
    print("  - Use batch processing for multiple similar prompts")

if __name__ == "__main__":
    main()