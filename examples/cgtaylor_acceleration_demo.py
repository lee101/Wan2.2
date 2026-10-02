#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
CG-Taylor Acceleration Demo for Wan 2.2
Demonstrates usage of CG-Taylor acceleration with Wan 2.2 video generation
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from wan.configs.wan_t2v_A14B import t2v_A14B
from wan.text2video_accelerated import WanT2VAccelerated
from wan.text2video import WanT2V


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cgtaylor_demo.log')
        ]
    )


def benchmark_comparison(model_accelerated, model_standard, prompt, **kwargs):
    """Compare accelerated vs standard generation"""
    logging.info("=" * 60)
    logging.info("BENCHMARK COMPARISON")
    logging.info("=" * 60)
    
    # Warm up GPU
    logging.info("Warming up GPU...")
    _ = model_standard.generate("test", size=(640, 360), frame_num=17, sampling_steps=10)
    torch.cuda.empty_cache()
    
    # Standard generation
    logging.info("Running standard generation...")
    start_time = time.time()
    video_standard = model_standard.generate(prompt, **kwargs)
    standard_time = time.time() - start_time
    
    torch.cuda.empty_cache()
    
    # Accelerated generation
    logging.info("Running accelerated generation...")
    start_time = time.time()
    video_accelerated = model_accelerated.generate(prompt, **kwargs)
    accelerated_time = time.time() - start_time
    
    # Get acceleration stats
    stats = model_accelerated.get_acceleration_stats()
    
    # Results
    speedup = standard_time / accelerated_time if accelerated_time > 0 else 0
    
    logging.info(f"\n{'='*50}")
    logging.info(f"BENCHMARK RESULTS")
    logging.info(f"{'='*50}")
    logging.info(f"Standard generation time: {standard_time:.2f}s")
    logging.info(f"Accelerated generation time: {accelerated_time:.2f}s")
    logging.info(f"Actual speedup: {speedup:.2f}x")
    logging.info(f"CG-Taylor stats: {stats}")
    
    return {
        'standard_time': standard_time,
        'accelerated_time': accelerated_time,
        'speedup': speedup,
        'stats': stats,
        'video_standard': video_standard,
        'video_accelerated': video_accelerated
    }


def demo_basic_usage():
    """Demonstrate basic CG-Taylor acceleration usage"""
    logging.info("=" * 60)
    logging.info("BASIC USAGE DEMO")
    logging.info("=" * 60)
    
    # Model configuration
    config = t2v_A14B.copy()
    checkpoint_dir = "path/to/wan22/models"  # Update this path
    
    # Create accelerated model
    logging.info("Creating accelerated Wan T2V model...")
    model = WanT2VAccelerated(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_cgtaylor=True,
        device_id=0
    )
    
    # Generation parameters
    prompt = "A majestic eagle soaring through mountain valleys at sunset"
    generation_params = {
        'size': (1280, 720),
        'frame_num': 81,
        'sampling_steps': 40,
        'guide_scale': 4.0,
        'shift': 12.0,
        'seed': 42
    }
    
    # Generate video
    logging.info(f"Generating video with prompt: '{prompt}'")
    start_time = time.time()
    
    video = model.generate(prompt, **generation_params)
    
    generation_time = time.time() - start_time
    stats = model.get_acceleration_stats()
    
    logging.info(f"\nGeneration completed in {generation_time:.2f}s")
    logging.info(f"Video shape: {video.shape if video is not None else 'None'}")
    logging.info(f"Acceleration statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value}")
    
    return model, video


def demo_configuration_tuning():
    """Demonstrate configuration tuning for different scenarios"""
    logging.info("=" * 60)
    logging.info("CONFIGURATION TUNING DEMO")
    logging.info("=" * 60)
    
    config = t2v_A14B.copy()
    checkpoint_dir = "path/to/wan22/models"
    
    model = WanT2VAccelerated(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_cgtaylor=True,
        device_id=0
    )
    
    # Demo 1: Conservative settings for high quality
    logging.info("Demo 1: Conservative settings for high quality")
    model.set_acceleration_config(
        high_noise_config={
            'confidence_threshold': 0.10,  # Very conservative
            'cache_window': 3,
            'order': 2
        },
        low_noise_config={
            'confidence_threshold': 0.08,  # Even more conservative
            'cache_window': 2,
            'order': 2
        }
    )
    
    video1 = model.generate(
        "A serene lake reflection at dawn",
        size=(1280, 720),
        frame_num=49,
        sampling_steps=50
    )
    stats1 = model.get_acceleration_stats()
    logging.info(f"Conservative settings stats: {stats1}")
    
    # Demo 2: Aggressive settings for speed
    logging.info("\nDemo 2: Aggressive settings for maximum speed")
    model.set_acceleration_config(
        high_noise_config={
            'confidence_threshold': 0.35,  # More aggressive
            'cache_window': 6,
            'order': 3
        },
        low_noise_config={
            'confidence_threshold': 0.20,  # More aggressive
            'cache_window': 4,
            'order': 3
        }
    )
    
    video2 = model.generate(
        "A bustling city street at night",
        size=(960, 540),
        frame_num=49,
        sampling_steps=30
    )
    stats2 = model.get_acceleration_stats()
    logging.info(f"Aggressive settings stats: {stats2}")
    
    return model, [video1, video2], [stats1, stats2]


def demo_memory_efficiency():
    """Demonstrate memory-efficient generation"""
    logging.info("=" * 60)
    logging.info("MEMORY EFFICIENCY DEMO")
    logging.info("=" * 60)
    
    config = t2v_A14B.copy()
    checkpoint_dir = "path/to/wan22/models"
    
    model = WanT2VAccelerated(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_cgtaylor=True,
        device_id=0
    )
    
    # Monitor memory usage
    def log_memory():
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    logging.info("Memory usage before generation:")
    log_memory()
    
    # Generate with memory-efficient settings
    video = model.generate(
        "A peaceful forest scene with falling leaves",
        size=(1280, 720),
        frame_num=81,
        sampling_steps=40,
        offload_model=True  # Enable model offloading
    )
    
    logging.info("Memory usage after generation:")
    log_memory()
    
    stats = model.get_acceleration_stats()
    logging.info(f"Memory-efficient generation stats: {stats}")
    
    return model, video


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='CG-Taylor Acceleration Demo for Wan 2.2')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to Wan 2.2 model checkpoints')
    parser.add_argument('--demo', type=str, choices=['basic', 'benchmark', 'config', 'memory', 'all'],
                       default='basic', help='Demo to run')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        logging.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        logging.error("Please download Wan 2.2 models and specify the correct path")
        return
    
    # Update config with correct checkpoint path
    global t2v_A14B
    # This is a demo - you would need to properly set up the model paths
    
    logging.info("Starting CG-Taylor Acceleration Demo")
    logging.info(f"Using device: cuda:{args.device}")
    logging.info(f"Checkpoint directory: {args.checkpoint_dir}")
    
    if args.demo in ['basic', 'all']:
        try:
            model, video = demo_basic_usage()
            logging.info("Basic usage demo completed successfully")
        except Exception as e:
            logging.error(f"Basic usage demo failed: {e}")
    
    if args.demo in ['benchmark', 'all']:
        try:
            # Create both models for comparison
            config = t2v_A14B.copy()
            model_accelerated = WanT2VAccelerated(
                config=config,
                checkpoint_dir=args.checkpoint_dir,
                enable_cgtaylor=True,
                device_id=args.device
            )
            model_standard = WanT2V(
                config=config,
                checkpoint_dir=args.checkpoint_dir,
                device_id=args.device
            )
            
            results = benchmark_comparison(
                model_accelerated,
                model_standard,
                "A dragon flying over a medieval castle",
                size=(1280, 720),
                frame_num=49,
                sampling_steps=30
            )
            logging.info("Benchmark comparison completed successfully")
        except Exception as e:
            logging.error(f"Benchmark comparison failed: {e}")
    
    if args.demo in ['config', 'all']:
        try:
            model, videos, stats_list = demo_configuration_tuning()
            logging.info("Configuration tuning demo completed successfully")
        except Exception as e:
            logging.error(f"Configuration tuning demo failed: {e}")
    
    if args.demo in ['memory', 'all']:
        try:
            model, video = demo_memory_efficiency()
            logging.info("Memory efficiency demo completed successfully")
        except Exception as e:
            logging.error(f"Memory efficiency demo failed: {e}")
    
    logging.info("All demos completed!")


if __name__ == "__main__":
    main()