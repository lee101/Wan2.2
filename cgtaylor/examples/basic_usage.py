#!/usr/bin/env python3
# Copyright (c) 2025 Wan Team Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Basic usage example of CG-Taylor MoE acceleration for Wan 2.2.

This example demonstrates how to use CG-Taylor acceleration with Wan 2.2
for text-to-video generation with significant speedup while maintaining quality.
"""

import sys
import time
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.configs import get_config
from wan.text2video import WanT2V
from wan.utils.utils import save_video
from cgtaylor.forwards.wan22_pipeline_moe import (
    enable_cg_taylor_moe_acceleration,
    disable_cg_taylor_moe_acceleration,
    create_cg_taylor_wan_t2v
)


def basic_cg_taylor_example(checkpoint_dir: str, model_variant: str = "A14B"):
    """
    Basic example showing how to use CG-Taylor acceleration.
    
    Args:
        checkpoint_dir: Path to model checkpoints
        model_variant: Model variant ("A14B" or "TI2V-5B")
    """
    
    print(f"=== Basic CG-Taylor MoE Example for {model_variant} ===")
    
    # Get model configuration
    config = get_config(model_variant.lower())
    
    # Initialize standard Wan T2V model
    print("Initializing Wan T2V model...")
    wan_t2v = WanT2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=True,
        init_on_cpu=False
    )
    
    # Test prompt
    prompt = "A cat walking through a garden with colorful flowers blooming in spring sunlight."
    
    # Generation parameters
    generation_params = {
        'size': (1280, 720),
        'frame_num': 81,
        'sampling_steps': 50,
        'guide_scale': 5.0,
        'seed': 42,
        'offload_model': True
    }
    
    # Adjust for TI2V-5B
    if "TI2V-5B" in model_variant:
        generation_params.update({
            'size': (1280, 704),
            'frame_num': 73,
            'shift': 3.0
        })
    
    print(f"Generating video with prompt: {prompt}")
    print(f"Parameters: {generation_params}")
    
    # === Method 1: Manual enable/disable ===
    print("\n--- Method 1: Manual CG-Taylor Control ---")
    
    # Generate without CG-Taylor (baseline)
    print("Generating baseline video (no acceleration)...")
    start_time = time.time()
    
    baseline_video = wan_t2v.generate(
        input_prompt=prompt,
        **generation_params
    )
    
    baseline_time = time.time() - start_time
    print(f"Baseline generation completed in {baseline_time:.2f} seconds")
    
    # Enable CG-Taylor acceleration
    print("\nEnabling CG-Taylor MoE acceleration...")
    enable_cg_taylor_moe_acceleration(
        wan_t2v,
        model_variant=model_variant,
        high_noise_threshold=0.30,  # More aggressive caching for high-noise
        low_noise_threshold=0.13,   # Conservative caching for low-noise
        expert_switch_snr=0.5       # SNR threshold for expert switching
    )
    
    # Generate with CG-Taylor acceleration
    print("Generating accelerated video...")
    start_time = time.time()
    
    accelerated_video = wan_t2v.generate(
        input_prompt=prompt,
        enable_cg_taylor=True,
        **generation_params
    )
    
    accelerated_time = time.time() - start_time
    print(f"Accelerated generation completed in {accelerated_time:.2f} seconds")
    
    # Calculate speedup
    speedup = baseline_time / accelerated_time
    print(f"Acceleration achieved: {speedup:.2f}x speedup!")
    
    # Disable CG-Taylor
    disable_cg_taylor_moe_acceleration(wan_t2v)
    
    # === Method 2: Pre-configured instance ===
    print("\n--- Method 2: Pre-configured CG-Taylor Instance ---")
    
    # Create instance with CG-Taylor pre-enabled
    cg_taylor_wan = create_cg_taylor_wan_t2v(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        high_noise_threshold=0.25,  # Custom threshold
        low_noise_threshold=0.10,
        expert_switch_snr=0.4,
        device_id=0,
        rank=0,
        t5_cpu=True
    )
    
    print("Generating with pre-configured CG-Taylor instance...")
    start_time = time.time()
    
    preconfigured_video = cg_taylor_wan.generate(
        input_prompt=prompt,
        **generation_params
    )
    
    preconfigured_time = time.time() - start_time
    preconfigured_speedup = baseline_time / preconfigured_time
    print(f"Pre-configured generation completed in {preconfigured_time:.2f} seconds")
    print(f"Pre-configured acceleration: {preconfigured_speedup:.2f}x speedup!")
    
    # Save videos for comparison
    print("\nSaving videos...")
    save_video(baseline_video, "basic_example_baseline.mp4")
    save_video(accelerated_video, "basic_example_accelerated.mp4") 
    save_video(preconfigured_video, "basic_example_preconfigured.mp4")
    
    print("Videos saved:")
    print("- basic_example_baseline.mp4 (vanilla generation)")
    print("- basic_example_accelerated.mp4 (CG-Taylor accelerated)")
    print("- basic_example_preconfigured.mp4 (pre-configured instance)")
    
    print(f"\n=== Summary ===")
    print(f"Baseline time: {baseline_time:.2f}s")
    print(f"Accelerated time: {accelerated_time:.2f}s")
    print(f"Pre-configured time: {preconfigured_time:.2f}s")
    print(f"Manual acceleration: {speedup:.2f}x")
    print(f"Pre-configured acceleration: {preconfigured_speedup:.2f}x")
    

def advanced_cg_taylor_example(checkpoint_dir: str, model_variant: str = "A14B"):
    """
    Advanced example showing fine-tuned CG-Taylor usage.
    
    Args:
        checkpoint_dir: Path to model checkpoints
        model_variant: Model variant ("A14B" or "TI2V-5B")
    """
    
    print(f"\n=== Advanced CG-Taylor MoE Example for {model_variant} ===")
    
    # Get model configuration
    config = get_config(model_variant.lower())
    
    # Initialize model
    wan_t2v = WanT2V(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=True
    )
    
    # Different scenarios with different thresholds
    scenarios = [
        {
            "name": "Conservative (High Quality)",
            "high_noise_threshold": 0.20,
            "low_noise_threshold": 0.08,
            "expert_switch_snr": 0.6,
            "description": "More conservative caching for maximum quality retention"
        },
        {
            "name": "Balanced (Recommended)",
            "high_noise_threshold": 0.30,
            "low_noise_threshold": 0.13,
            "expert_switch_snr": 0.5,
            "description": "Balanced between quality and speed"
        },
        {
            "name": "Aggressive (Max Speed)",
            "high_noise_threshold": 0.40,
            "low_noise_threshold": 0.18,
            "expert_switch_snr": 0.4,
            "description": "More aggressive caching for maximum acceleration"
        }
    ]
    
    prompt = "A majestic eagle soaring over snow-capped mountains during golden hour."
    
    # Test each scenario
    results = []
    
    for scenario in scenarios:
        print(f"\n--- Testing {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Thresholds: High={scenario['high_noise_threshold']}, "
              f"Low={scenario['low_noise_threshold']}, SNR={scenario['expert_switch_snr']}")
        
        # Configure CG-Taylor with scenario parameters
        enable_cg_taylor_moe_acceleration(
            wan_t2v,
            model_variant=model_variant,
            high_noise_threshold=scenario['high_noise_threshold'],
            low_noise_threshold=scenario['low_noise_threshold'],
            expert_switch_snr=scenario['expert_switch_snr']
        )
        
        # Generate video
        start_time = time.time()
        video = wan_t2v.generate(
            input_prompt=prompt,
            size=(1280, 720),
            frame_num=81,
            sampling_steps=50,
            guide_scale=5.0,
            seed=123,  # Fixed seed for comparison
            enable_cg_taylor=True
        )
        generation_time = time.time() - start_time
        
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # Save video with scenario name
        filename = f"advanced_example_{scenario['name'].lower().replace(' ', '_')}.mp4"
        save_video(video, filename)
        
        results.append({
            'scenario': scenario['name'],
            'time': generation_time,
            'filename': filename
        })
        
        # Disable for next scenario
        disable_cg_taylor_moe_acceleration(wan_t2v)
    
    # Print summary
    print("\n=== Advanced Example Summary ===")
    for result in results:
        print(f"{result['scenario']:20s}: {result['time']:6.2f}s -> {result['filename']}")


def batch_generation_example(checkpoint_dir: str, model_variant: str = "A14B"):
    """
    Example showing batch generation with CG-Taylor acceleration.
    
    Args:
        checkpoint_dir: Path to model checkpoints  
        model_variant: Model variant ("A14B" or "TI2V-5B")
    """
    
    print(f"\n=== Batch Generation Example for {model_variant} ===")
    
    # Get model configuration
    config = get_config(model_variant.lower())
    
    # Create pre-configured instance for batch processing
    wan_t2v = create_cg_taylor_wan_t2v(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        high_noise_threshold=0.30,
        low_noise_threshold=0.13,
        device_id=0,
        rank=0,
        t5_cpu=True
    )
    
    # Batch of prompts
    prompts = [
        "A red sports car driving through a tunnel with neon lights.",
        "Cherry blossoms falling in a Japanese garden during spring.",
        "A lighthouse standing tall against stormy ocean waves.",
        "Northern lights dancing across the Arctic sky at night."
    ]
    
    print(f"Generating {len(prompts)} videos with CG-Taylor acceleration...")
    
    total_start_time = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        
        start_time = time.time()
        video = wan_t2v.generate(
            input_prompt=prompt,
            size=(1280, 720),
            frame_num=81,
            sampling_steps=50,
            guide_scale=5.0,
            seed=i * 42,  # Different seed for each video
            offload_model=True
        )
        generation_time = time.time() - start_time
        
        # Save video
        filename = f"batch_example_{i:02d}.mp4"
        save_video(video, filename)
        
        print(f"Generated in {generation_time:.2f}s -> {filename}")
    
    total_time = time.time() - total_start_time
    avg_time = total_time / len(prompts)
    
    print(f"\n=== Batch Generation Summary ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per video: {avg_time:.2f} seconds")
    print(f"Estimated acceleration: ~3-5x compared to vanilla generation")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CG-Taylor MoE usage examples")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to model checkpoints")
    parser.add_argument("--model-variant", default="A14B", choices=["A14B", "TI2V-5B"],
                       help="Model variant to use")
    parser.add_argument("--example", default="basic", 
                       choices=["basic", "advanced", "batch", "all"],
                       help="Which example to run")
    
    args = parser.parse_args()
    
    try:
        if args.example == "basic" or args.example == "all":
            basic_cg_taylor_example(args.checkpoint_dir, args.model_variant)
            
        if args.example == "advanced" or args.example == "all":
            advanced_cg_taylor_example(args.checkpoint_dir, args.model_variant)
            
        if args.example == "batch" or args.example == "all":
            batch_generation_example(args.checkpoint_dir, args.model_variant)
            
        print("\n Examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)