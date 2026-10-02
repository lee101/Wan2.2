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
Test script for CG-Taylor MoE acceleration on Wan 2.2.

This script compares vanilla Wan 2.2 generation with CG-Taylor accelerated
generation to validate quality preservation and measure acceleration gains.
"""

import argparse
import logging
import time
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import wan modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wan.configs import get_config
from wan.text2video import WanT2V
from wan.utils.utils import save_video
from cgtaylor.forwards.wan22_pipeline_moe import (
    enable_cg_taylor_moe_acceleration,
    disable_cg_taylor_moe_acceleration
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cg_taylor_test.log')
        ]
    )


def calculate_video_metrics(video1, video2):
    """
    Calculate basic quality metrics between two videos.
    
    Args:
        video1: First video tensor [C, F, H, W]
        video2: Second video tensor [C, F, H, W]
    
    Returns:
        dict: Dictionary containing quality metrics
    """
    # Convert to numpy for easier computation
    v1 = video1.cpu().numpy()
    v2 = video2.cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((v1 - v2) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Structural Similarity (simplified approximation)
    mean1, mean2 = np.mean(v1), np.mean(v2)
    var1, var2 = np.var(v1), np.var(v2)
    covar = np.mean((v1 - mean1) * (v2 - mean2))
    
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
           ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'l1_diff': np.mean(np.abs(v1 - v2)),
        'max_diff': np.max(np.abs(v1 - v2))
    }


def run_generation_test(wan_t2v, prompt, config, use_cg_taylor=False, **generation_kwargs):
    """
    Run a single generation test.
    
    Args:
        wan_t2v: WanT2V instance
        prompt: Text prompt for generation
        config: Test configuration
        use_cg_taylor: Whether to use CG-Taylor acceleration
        **generation_kwargs: Additional generation parameters
    
    Returns:
        tuple: (generated_video, generation_time, additional_info)
    """
    torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # Generate video
    with torch.no_grad():
        video = wan_t2v.generate(
            input_prompt=prompt,
            enable_cg_taylor=use_cg_taylor,
            **generation_kwargs
        )
    
    generation_time = time.time() - start_time
    
    # Get memory usage
    memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    info = {
        'generation_time': generation_time,
        'memory_used_gb': memory_used,
        'video_shape': video.shape,
        'use_cg_taylor': use_cg_taylor
    }
    
    return video, generation_time, info


def test_cg_taylor_acceleration(checkpoint_dir, model_variant="A14B", test_prompts=None):
    """
    Main test function for CG-Taylor acceleration.
    
    Args:
        checkpoint_dir: Path to model checkpoints
        model_variant: Model variant to test
        test_prompts: List of test prompts (optional)
    """
    
    logging.info(f"Starting CG-Taylor MoE acceleration test for {model_variant}")
    
    # Default test prompts
    if test_prompts is None:
        test_prompts = [
            "A cat walking through a garden with colorful flowers blooming.",
            "Ocean waves crashing against a rocky coastline during sunset.",
            "A butterfly landing on a flower in slow motion.",
        ]
    
    # Get model configuration
    config = get_config(model_variant.lower())
    
    # Test parameters
    test_params = {
        'size': (1280, 720) if "A14B" in model_variant else (1280, 704),
        'frame_num': 81 if "A14B" in model_variant else 24 * 3 + 1,  # Approximate for TI2V-5B
        'sampling_steps': 50,
        'guide_scale': 5.0,
        'seed': 42,  # Fixed seed for reproducibility
        'offload_model': True
    }
    
    # Override for TI2V-5B
    if "TI2V-5B" in model_variant:
        test_params.update({
            'size': (1280, 704),
            'frame_num': 73,  # 24fps * 3s + 1
            'shift': 3.0
        })
    
    # Initialize model
    logging.info("Initializing Wan T2V model...")
    try:
        wan_t2v = WanT2V(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=0,
            rank=0,
            t5_cpu=True,
            init_on_cpu=False
        )
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        return False
    
    # Test results storage
    results = {
        'vanilla': [],
        'cg_taylor': [],
        'quality_metrics': [],
        'acceleration_ratios': []
    }
    
    # Run tests for each prompt
    for i, prompt in enumerate(test_prompts):
        logging.info(f"\n=== Testing prompt {i+1}/{len(test_prompts)} ===")
        logging.info(f"Prompt: {prompt}")
        
        # Test 1: Vanilla generation
        logging.info("Running vanilla generation...")
        try:
            vanilla_video, vanilla_time, vanilla_info = run_generation_test(
                wan_t2v, prompt, config, use_cg_taylor=False, **test_params
            )
            results['vanilla'].append((vanilla_video, vanilla_time, vanilla_info))
            logging.info(f"Vanilla generation completed in {vanilla_time:.2f}s")
        except Exception as e:
            logging.error(f"Vanilla generation failed: {e}")
            continue
        
        # Enable CG-Taylor acceleration
        logging.info("Enabling CG-Taylor MoE acceleration...")
        enable_cg_taylor_moe_acceleration(
            wan_t2v, 
            model_variant=model_variant,
            high_noise_threshold=0.30 if "A14B" in model_variant else 0.25,
            low_noise_threshold=0.13 if "A14B" in model_variant else 0.10
        )
        
        # Test 2: CG-Taylor accelerated generation
        logging.info("Running CG-Taylor accelerated generation...")
        try:
            cg_taylor_video, cg_taylor_time, cg_taylor_info = run_generation_test(
                wan_t2v, prompt, config, use_cg_taylor=True, **test_params
            )
            results['cg_taylor'].append((cg_taylor_video, cg_taylor_time, cg_taylor_info))
            logging.info(f"CG-Taylor generation completed in {cg_taylor_time:.2f}s")
        except Exception as e:
            logging.error(f"CG-Taylor generation failed: {e}")
            continue
        
        # Calculate quality metrics
        logging.info("Calculating quality metrics...")
        quality_metrics = calculate_video_metrics(vanilla_video, cg_taylor_video)
        results['quality_metrics'].append(quality_metrics)
        
        # Calculate acceleration ratio
        acceleration_ratio = vanilla_time / cg_taylor_time
        results['acceleration_ratios'].append(acceleration_ratio)
        
        logging.info(f"Acceleration ratio: {acceleration_ratio:.2f}x")
        logging.info(f"Quality metrics: PSNR={quality_metrics['psnr']:.2f}dB, "
                    f"SSIM={quality_metrics['ssim']:.4f}, MSE={quality_metrics['mse']:.6f}")
        
        # Save videos for comparison
        output_dir = Path("cg_taylor_test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        try:
            save_video(vanilla_video, str(output_dir / f"prompt_{i+1}_vanilla.mp4"))
            save_video(cg_taylor_video, str(output_dir / f"prompt_{i+1}_cg_taylor.mp4"))
            logging.info(f"Videos saved to {output_dir}")
        except Exception as e:
            logging.warning(f"Failed to save videos: {e}")
        
        # Disable CG-Taylor for next iteration
        disable_cg_taylor_moe_acceleration(wan_t2v)
    
    # Generate summary report
    generate_test_report(results, model_variant, test_prompts)
    
    return True


def generate_test_report(results, model_variant, test_prompts):
    """
    Generate a comprehensive test report.
    
    Args:
        results: Test results dictionary
        model_variant: Model variant tested
        test_prompts: List of test prompts
    """
    
    logging.info("\n" + "="*60)
    logging.info("CG-TAYLOR MOE ACCELERATION TEST REPORT")
    logging.info("="*60)
    logging.info(f"Model Variant: {model_variant}")
    logging.info(f"Test Prompts: {len(test_prompts)}")
    
    if not results['acceleration_ratios']:
        logging.warning("No successful test runs completed!")
        return
    
    # Calculate averages
    avg_acceleration = np.mean(results['acceleration_ratios'])
    avg_psnr = np.mean([m['psnr'] for m in results['quality_metrics']])
    avg_ssim = np.mean([m['ssim'] for m in results['quality_metrics']])
    avg_mse = np.mean([m['mse'] for m in results['quality_metrics']])
    
    vanilla_times = [info[1] for info in results['vanilla']]
    cg_taylor_times = [info[1] for info in results['cg_taylor']]
    
    logging.info("\n--- PERFORMANCE SUMMARY ---")
    logging.info(f"Average Acceleration Ratio: {avg_acceleration:.2f}x")
    logging.info(f"Average Vanilla Time: {np.mean(vanilla_times):.2f}s")
    logging.info(f"Average CG-Taylor Time: {np.mean(cg_taylor_times):.2f}s")
    logging.info(f"Time Savings: {np.mean(vanilla_times) - np.mean(cg_taylor_times):.2f}s per generation")
    
    logging.info("\n--- QUALITY SUMMARY ---")
    logging.info(f"Average PSNR: {avg_psnr:.2f}dB")
    logging.info(f"Average SSIM: {avg_ssim:.4f}")
    logging.info(f"Average MSE: {avg_mse:.6f}")
    
    # Quality thresholds (adjust based on requirements)
    psnr_threshold = 30.0  # dB
    ssim_threshold = 0.95
    
    quality_passed = avg_psnr >= psnr_threshold and avg_ssim >= ssim_threshold
    
    logging.info("\n--- QUALITY ASSESSMENT ---")
    logging.info(f"PSNR Threshold: {psnr_threshold}dB ({'PASS' if avg_psnr >= psnr_threshold else 'FAIL'})")
    logging.info(f"SSIM Threshold: {ssim_threshold} ({'PASS' if avg_ssim >= ssim_threshold else 'FAIL'})")
    logging.info(f"Overall Quality: {'PASS' if quality_passed else 'FAIL'}")
    
    logging.info("\n--- INDIVIDUAL RESULTS ---")
    for i, (acceleration, quality) in enumerate(zip(results['acceleration_ratios'], results['quality_metrics'])):
        logging.info(f"Prompt {i+1}: {acceleration:.2f}x speedup, "
                    f"PSNR={quality['psnr']:.2f}dB, SSIM={quality['ssim']:.4f}")
    
    # Final assessment
    target_acceleration = 3.0  # Target from requirements
    acceleration_passed = avg_acceleration >= target_acceleration
    
    logging.info("\n--- FINAL ASSESSMENT ---")
    logging.info(f"Target Acceleration: {target_acceleration}x ({'PASS' if acceleration_passed else 'FAIL'})")
    logging.info(f"Quality Retention: {'PASS' if quality_passed else 'FAIL'}")
    logging.info(f"Overall Result: {'SUCCESS' if (acceleration_passed and quality_passed) else 'NEEDS_IMPROVEMENT'}")
    
    logging.info("="*60)


def main():
    """Main entry point for the test script."""
    
    parser = argparse.ArgumentParser(description="Test CG-Taylor MoE acceleration on Wan 2.2")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to model checkpoints")
    parser.add_argument("--model-variant", default="A14B", choices=["A14B", "TI2V-5B"], 
                       help="Model variant to test")
    parser.add_argument("--test-prompts", nargs="+", help="Custom test prompts")
    parser.add_argument("--output-dir", default="cg_taylor_test_outputs", 
                       help="Directory to save test outputs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    setup_logging()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Check checkpoint directory
    if not Path(args.checkpoint_dir).exists():
        logging.error(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
        return 1
    
    # Run tests
    try:
        success = test_cg_taylor_acceleration(
            args.checkpoint_dir, 
            args.model_variant,
            args.test_prompts
        )
        return 0 if success else 1
    except Exception as e:
        logging.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())