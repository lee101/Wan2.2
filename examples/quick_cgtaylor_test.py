#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Quick test script for CG-Taylor acceleration
Simple validation that the acceleration system works
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from wan.acceleration import WanCGTaylorAdapter, TaylorConfig


def test_taylor_cache():
    """Test the Taylor cache functionality"""
    print("Testing Taylor cache functionality...")
    
    config = TaylorConfig(
        order=2,
        confidence_threshold=0.2,
        cache_window=5
    )
    
    # Create a mock config object for the adapter
    class MockConfig:
        def __init__(self):
            self.num_train_timesteps = 1000
    
    adapter = WanCGTaylorAdapter(MockConfig(), boundary=0.875)
    
    # Test expert selection
    expert_name, cache, config = adapter.select_expert_cache(900)  # High noise
    assert expert_name == "high_noise", f"Expected high_noise, got {expert_name}"
    
    expert_name, cache, config = adapter.select_expert_cache(400)  # Low noise
    assert expert_name == "low_noise", f"Expected low_noise, got {expert_name}"
    
    print("Expert selection works correctly")
    
    # Test feature caching
    dummy_features = torch.randn(1, 100, 512)
    adapter.update_cache(900, dummy_features)
    adapter.update_cache(890, dummy_features * 1.1)
    adapter.update_cache(880, dummy_features * 1.2)
    
    # Test prediction
    predicted = adapter.predict_features(870)
    print(f"Taylor prediction: {'successful' if predicted is not None else 'not ready'}")
    
    # Test statistics
    stats = adapter.get_acceleration_stats()
    print(f"Statistics: {stats}")
    
    return True


def test_integration():
    """Test integration components"""
    print("\nTesting integration components...")
    
    try:
        from wan.text2video_accelerated import WanT2VAccelerated
        print("WanT2VAccelerated import successful")
    except ImportError as e:
        print(f"WanT2VAccelerated import failed: {e}")
        return False
    
    try:
        from wan.acceleration import WanCGTaylorAdapter
        print("WanCGTaylorAdapter import successful")
    except ImportError as e:
        print(f"WanCGTaylorAdapter import failed: {e}")
        return False
    
    return True


def test_memory_management():
    """Test memory management features"""
    print("\nTesting memory management...")
    
    config = TaylorConfig(memory_efficient=True, cache_window=3)
    
    class MockConfig:
        num_train_timesteps = 1000
    
    adapter = WanCGTaylorAdapter(MockConfig())
    
    # Test optimization for different resolutions
    adapter.optimize_for_resolution((1280, 720), 81, batch_size=1)
    print("Resolution optimization completed")
    
    # Test cache clearing
    adapter.clear_caches()
    print("Cache clearing completed")
    
    return True


def main():
    """Run quick tests"""
    print("=" * 50)
    print("CG-Taylor Acceleration Quick Test")
    print("=" * 50)
    
    # Suppress some logging for cleaner output
    logging.getLogger().setLevel(logging.WARNING)
    
    success = True
    
    try:
        success &= test_taylor_cache()
    except Exception as e:
        print(f"Taylor cache test failed: {e}")
        success = False
    
    try:
        success &= test_integration()
    except Exception as e:
        print(f"Integration test failed: {e}")
        success = False
    
    try:
        success &= test_memory_management()
    except Exception as e:
        print(f"Memory management test failed: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("All quick tests PASSED!")
        print("\nTo use CG-Taylor acceleration:")
        print("1. Use WanT2VAccelerated instead of WanT2V")
        print("2. Set enable_cgtaylor=True (default)")
        print("3. Optionally tune configuration with set_acceleration_config()")
        print("\nExample:")
        print("from wan.text2video_accelerated import WanT2VAccelerated")
        print("model = WanT2VAccelerated(config, checkpoint_dir)")
        print("video = model.generate('your prompt')")
    else:
        print("Some tests FAILED!")
        print("Check the error messages above for details.")
    
    print("=" * 50)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)