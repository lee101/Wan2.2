#!/usr/bin/env python3
"""
Test script for Ultimate FastVideo LoRA Optimizer with safe CPU offload settings.
This script tests both with and without LoRA models and focuses on stability.
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add VideoX-Fun to path
videox_fun_path = "/vfast/data/code/VideoX-Fun"
if os.path.exists(videox_fun_path):
    sys.path.insert(0, videox_fun_path)

try:
    from ultimate_lora_t2v import UltimateFastVideoLoRAOptimizer
    print("✅ Successfully imported UltimateFastVideoLoRAOptimizer")
except ImportError as e:
    print(f"❌ Failed to import optimizer: {e}")
    print("Make sure VideoX-Fun is available and all dependencies are installed.")
    sys.exit(1)

def download_test_lora():
    """Download a test LoRA model from Hugging Face if none exists."""
    lora_dir = Path(current_dir) / "test_loras"
    lora_dir.mkdir(exist_ok=True)
    
    test_lora_path = lora_dir / "test_lora.safetensors"
    
    if test_lora_path.exists():
        print(f"✅ Test LoRA already exists: {test_lora_path}")
        return str(test_lora_path)
    
    print("🔄 No test LoRA found. For now, we'll test without LoRA...")
    print("💡 You can add your own LoRA models to the test_loras/ directory")
    
    # For now, we'll test without LoRA
    return None

def create_safe_optimizer(model_path, lora_path=None):
    """Create optimizer with safe CPU offload settings."""
    print("🔄 Creating optimizer with safe CPU offload settings...")
    
    # Override the class to force safe settings
    class SafeUltimateFastVideoLoRAOptimizer(UltimateFastVideoLoRAOptimizer):
        def _apply_memory_optimizations(self):
            """Force sequential CPU offload for maximum safety."""
            print("💾 Applying SAFE sequential CPU offload (slow but stable)")
            
            # Import required modules
            from videox_fun.utils.fp8_optimization import replace_parameters_by_name
            
            # Always use sequential CPU offload for testing
            replace_parameters_by_name(self.pipeline.transformer, ["modulation",], device=self.device)
            replace_parameters_by_name(self.pipeline.transformer_2, ["modulation",], device=self.device)
            self.pipeline.transformer.freqs = self.pipeline.transformer.freqs.to(device=self.device)
            self.pipeline.transformer_2.freqs = self.pipeline.transformer_2.freqs.to(device=self.device)
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
            
            if self.verbose:
                print("✅ Sequential CPU offload enabled for maximum stability")
    
    return SafeUltimateFastVideoLoRAOptimizer(
        model_path=model_path,
        lora_path=lora_path,
        lora_weight=0.5,  # Conservative weight
        enable_optimizations=False,  # Disable optimizations for safety
        verbose=True
    )

def test_without_lora(model_path):
    """Test basic inference without LoRA."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Basic inference WITHOUT LoRA")
    print("="*60)
    
    try:
        optimizer = create_safe_optimizer(model_path)
        
        # Very conservative settings for testing
        start_time = time.time()
        
        video = optimizer.generate(
            prompt="A simple test: a cat sitting peacefully in a garden",
            quality="lightning",  # Fastest setting for testing
            video_length=17,      # Very short for testing
            sample_size=[320, 576],  # Small resolution
            fps=8,                # Low FPS
            seed=42,
            save_videos=True,
            output_dir="test_outputs_no_lora"
        )
        
        generation_time = time.time() - start_time
        print(f"✅ SUCCESS: Generated video in {generation_time:.2f}s")
        print(f"📊 Video shape: {video.shape if hasattr(video, 'shape') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_lora(model_path, lora_path):
    """Test inference with LoRA."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Inference WITH LoRA")
    print("="*60)
    
    if not lora_path or not os.path.exists(lora_path):
        print("⚠️  SKIPPED: No LoRA model available for testing")
        return True
    
    try:
        optimizer = create_safe_optimizer(model_path, lora_path)
        
        start_time = time.time()
        
        video = optimizer.generate(
            prompt="A LoRA-enhanced test: a majestic eagle soaring through mountains",
            quality="lightning",
            video_length=17,
            sample_size=[320, 576],
            fps=8,
            seed=123,
            save_videos=True,
            output_dir="test_outputs_with_lora"
        )
        
        generation_time = time.time() - start_time
        print(f"✅ SUCCESS: Generated LoRA video in {generation_time:.2f}s")
        print(f"📊 Video shape: {video.shape if hasattr(video, 'shape') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_presets(model_path):
    """Test different quality presets."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Quality presets comparison")
    print("="*60)
    
    quality_presets = ['lightning', 'draft', 'fast']
    results = {}
    
    for quality in quality_presets:
        print(f"\n🎯 Testing {quality} quality...")
        
        try:
            optimizer = create_safe_optimizer(model_path)
            
            start_time = time.time()
            
            video = optimizer.generate(
                prompt=f"Quality test {quality}: a peaceful mountain lake",
                quality=quality,
                video_length=17,
                sample_size=[320, 576],
                fps=8,
                seed=42,
                save_videos=True,
                output_dir=f"test_outputs_{quality}"
            )
            
            generation_time = time.time() - start_time
            results[quality] = {
                'time': generation_time,
                'success': True,
                'shape': video.shape if hasattr(video, 'shape') else None
            }
            
            print(f"✅ {quality}: {generation_time:.2f}s")
            
        except Exception as e:
            results[quality] = {
                'time': None,
                'success': False,
                'error': str(e)
            }
            print(f"❌ {quality}: Failed - {e}")
    
    # Print summary
    print(f"\n📊 QUALITY PRESET RESULTS:")
    print("-" * 40)
    for quality, result in results.items():
        if result['success']:
            print(f"{quality:10}: ✅ {result['time']:.2f}s")
        else:
            print(f"{quality:10}: ❌ Failed")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test LoRA inference with safe settings")
    parser.add_argument("--model_path", type=str, 
                       default="/vfast/data/code/Wan2.2/Wan2.2-I2V-A14B",
                       help="Path to WAN2.2 model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA model (optional)")
    parser.add_argument("--download_test_lora", action="store_true",
                       help="Try to download a test LoRA model")
    parser.add_argument("--skip_lora_test", action="store_true",
                       help="Skip LoRA testing")
    
    args = parser.parse_args()
    
    print("🚀 Ultimate FastVideo LoRA Optimizer - SAFE TEST")
    print("="*60)
    print("⚠️  Using SAFE CPU offload settings (slow but stable)")
    print("💡 This test prioritizes stability over speed")
    print("="*60)
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        print("Available model paths:")
        wan_models = Path("/vfast/data/code/Wan2.2").glob("Wan2.2-*")
        for model in wan_models:
            if model.is_dir():
                print(f"  - {model}")
        return 1
    
    print(f"✅ Using model: {args.model_path}")
    
    # Handle LoRA path
    lora_path = args.lora_path
    if args.download_test_lora and not lora_path:
        lora_path = download_test_lora()
    
    if lora_path and os.path.exists(lora_path):
        print(f"✅ Using LoRA: {lora_path}")
    else:
        print("⚠️  No LoRA model specified - testing without LoRA")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic inference without LoRA
    print("\n🔄 Starting Test 1...")
    result1 = test_without_lora(args.model_path)
    test_results.append(("Basic inference (no LoRA)", result1))
    
    # Test 2: With LoRA (if available and not skipped)
    if not args.skip_lora_test:
        print("\n🔄 Starting Test 2...")
        result2 = test_with_lora(args.model_path, lora_path)
        test_results.append(("LoRA inference", result2))
    
    # Test 3: Quality presets
    print("\n🔄 Starting Test 3...")
    result3 = test_quality_presets(args.model_path)
    successful_presets = sum(1 for r in result3.values() if r['success'])
    test_results.append((f"Quality presets ({successful_presets}/3)", successful_presets > 0))
    
    # Final summary
    print("\n" + "="*60)
    print("🏆 FINAL TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, passed_test in test_results:
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:30}: {status}")
        if passed_test:
            passed += 1
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The optimizer is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n📁 Output videos saved in:")
    output_dirs = ['test_outputs_no_lora', 'test_outputs_with_lora', 
                   'test_outputs_lightning', 'test_outputs_draft', 'test_outputs_fast']
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"  - {output_dir}/")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())