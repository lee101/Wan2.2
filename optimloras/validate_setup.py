#!/usr/bin/env python3
"""
Validate that the Ultimate LoRA Optimizer setup is working correctly.
This does a quick validation without actually loading the heavy models.
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_DIR,
    add_videox_fun_to_path,
)

def main():
    print("Ultimate LoRA Optimizer Setup Validation")
    print("=" * 50)
    
    # Test 1: Essential imports
    print("1. Testing essential imports...")
    required_modules = ['torch', 'numpy', 'PIL', 'omegaconf', 'diffusers', 'transformers']
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   {module}")
        except ImportError:
            print(f"   {module} - MISSING")
            return False
    
    # Test 2: VideoX-Fun imports
    print("\n2. Testing VideoX-Fun imports...")
    add_videox_fun_to_path()
    
    videox_modules = [
        'videox_fun.models',
        'videox_fun.pipeline', 
        'videox_fun.utils.lora_utils',
    ]
    
    for module in videox_modules:
        try:
            importlib.import_module(module)
            print(f"   {module}")
        except ImportError as e:
            print(f"   {module} - {e}")
            return False
    
    # Test 3: Check model path exists
    print("\n3. Testing model availability...")
    model_path = str(DEFAULT_MODEL_DIR)
    if os.path.exists(model_path):
        print(f"   Model found: {model_path}")
        
        # Check for key model components
        components = [
            "high_noise_model",
            "low_noise_model", 
            "google/umt5-xxl",
            "models_t5_umt5-xxl-enc-bf16.pth"
        ]
        
        for comp in components:
            comp_path = os.path.join(model_path, comp)
            if os.path.exists(comp_path):
                print(f"   Found: {comp}")
            else:
                print(f"   Missing: {comp}")
    else:
        print(f"   Model not found: {model_path}")
        return False
    
    # Test 4: Config file exists
    print("\n4. Testing config file...")
    config_path = str(DEFAULT_CONFIG_PATH)
    if os.path.exists(config_path):
        print(f"   Config found: {config_path}")
    else:
        print(f"   Config not found: {config_path}")
        return False
    
    # Test 5: Import our optimizer class (without initializing)
    print("\n5. Testing optimizer import...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        from ultimate_lora_t2v import UltimateFastVideoLoRAOptimizer
        print("   UltimateFastVideoLoRAOptimizer imported successfully!")
        
        # Test the quality presets are available
        presets = list(UltimateFastVideoLoRAOptimizer.QUALITY_PRESETS.keys())
        print(f"   Quality presets: {presets}")
        
    except Exception as e:
        print(f"   Failed to import optimizer: {e}")
        return False
    
    # Test 6: GPU availability
    print("\n6. Testing GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   CUDA available: {gpu_count} GPU(s)")
            print(f"   Primary GPU: {gpu_name}")
            print(f"   VRAM: {vram:.1f}GB")
            
            if vram < 8:
                print("   Low VRAM detected. Use 'lightning' or 'draft' quality.")
        else:
            print("   CUDA not available. CPU-only mode (very slow).")
    except:
        print("   Failed to check GPU status")
    
    print("\n" + "=" * 50)
    print("SETUP VALIDATION COMPLETE!")
    print("\nYour Ultimate LoRA Optimizer is ready to use!")
    print("\nNext steps:")
    print("1. To test without LoRA:")
    print("   python test_lora_inference.py --skip_lora_test")
    print("\n2. To test with a LoRA model:")
    print("   python test_lora_inference.py --lora_path /path/to/your/lora.safetensors")
    print("\n3. For production use:")
    print(f"   python ultimate_lora_t2v.py --model_path {DEFAULT_MODEL_DIR}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)