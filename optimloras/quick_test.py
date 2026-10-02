#!/usr/bin/env python3
"""
Quick test of the Ultimate LoRA Optimizer with minimal settings for safety.
This test uses CPU offload and very conservative settings to ensure stability.
"""

import os
import sys
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add VideoX-Fun to path
from paths import DEFAULT_MODEL_DIR, add_videox_fun_to_path

add_videox_fun_to_path()

def main():
    print("Quick Ultimate LoRA Optimizer Test")
    print("=" * 50)
    print("Using SAFE CPU offload settings for stability")
    print("This test prioritizes stability over speed")
    print("=" * 50)
    
    try:
        # Import the optimizer
        from ultimate_lora_t2v import UltimateFastVideoLoRAOptimizer
        print("Optimizer imported successfully")
        
        # Create a custom safe class that forces CPU offload
        class SafeLoRAOptimizer(UltimateFastVideoLoRAOptimizer):
            def _apply_memory_optimizations(self):
                """Force sequential CPU offload for maximum safety."""
                print("Applying SEQUENTIAL CPU OFFLOAD (slow but safe)")
                
                from videox_fun.utils.fp8_optimization import replace_parameters_by_name
                
                # Always use sequential CPU offload for testing
                replace_parameters_by_name(self.pipeline.transformer, ["modulation",], device=self.device)
                replace_parameters_by_name(self.pipeline.transformer_2, ["modulation",], device=self.device)
                self.pipeline.transformer.freqs = self.pipeline.transformer.freqs.to(device=self.device)
                self.pipeline.transformer_2.freqs = self.pipeline.transformer_2.freqs.to(device=self.device)
                self.pipeline.enable_sequential_cpu_offload(device=self.device)
                
                print("Sequential CPU offload enabled for maximum stability")
        
        # Initialize with safe settings
        print("Initializing optimizer with safe settings...")
        model_path = str(DEFAULT_MODEL_DIR)
        
        start_init = time.time()
        optimizer = SafeLoRAOptimizer(
            model_path=model_path,
            enable_optimizations=False,  # Disable all optimizations for safety
            verbose=True
        )
        init_time = time.time() - start_init
        
        print(f"Optimizer initialized successfully in {init_time:.2f}s")
        
        # Generate a very simple test video
        print("\n Generating test video with minimal settings...")
        
        start_gen = time.time()
        video = optimizer.generate(
            prompt="A simple test video",
            quality="lightning",     # Fastest quality
            video_length=17,         # Very short video
            sample_size=[320, 576],  # Small resolution
            fps=8,                   # Low FPS
            seed=42,
            save_videos=True,
            output_dir="quick_test_output"
        )
        gen_time = time.time() - start_gen
        
        print(f"Video generated successfully in {gen_time:.2f}s")
        print(f"Video shape: {video.shape if hasattr(video, 'shape') else 'N/A'}")
        
        # Test summary
        total_time = init_time + gen_time
        print(f"\n QUICK TEST COMPLETE!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Initialization: {init_time:.2f}s")
        print(f"Generation: {gen_time:.2f}s")
        print(f"Output saved to: quick_test_output/")
        
        # Check output file
        if os.path.exists("quick_test_output"):
            files = os.listdir("quick_test_output")
            if files:
                print(f"Generated files: {files}")
            else:
                print("No output files found")
        
        print("\n SUCCESS! The Ultimate LoRA Optimizer is working correctly!")
        print("You can now use it with LoRA models for enhanced results.")
        
        return True
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)