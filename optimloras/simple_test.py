#!/usr/bin/env python3
"""
Simple test to check dependencies and basic imports for LoRA optimization.
"""

import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO_ROOT, add_videox_fun_to_path, candidate_model_dirs

def test_imports():
    """Test essential imports."""
    print("Testing essential imports...")
    
    required_modules = [
        'torch',
        'numpy', 
        'PIL',
        'omegaconf',
        'diffusers',
        'transformers',
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  {module}")
        except ImportError:
            print(f"  {module} - MISSING")
            return False
    
    return True

def test_videox_fun_imports():
    """Test VideoX-Fun specific imports."""
    print("\n Testing VideoX-Fun imports...")
    
    # Add VideoX-Fun to path
    add_videox_fun_to_path()
    
    videox_modules = [
        'videox_fun.models',
        'videox_fun.pipeline', 
        'videox_fun.utils.lora_utils',
        'videox_fun.dist',
    ]
    
    for module in videox_modules:
        try:
            importlib.import_module(module)
            print(f"  {module}")
        except ImportError as e:
            print(f"  {module} - {e}")
            return False
    
    return True

def test_model_paths():
    """Test if model paths exist."""
    print("\n Testing model paths...")
    
    potential_models = [str(p) for p in candidate_model_dirs()]
    
    found_model = None
    for model_path in potential_models:
        if os.path.exists(model_path):
            print(f"  Found model: {model_path}")
            found_model = model_path
            break
        else:
            print(f"  Not found: {model_path}")
    
    if not found_model:
        print("  No WAN2.2 model found")
        # List available models
        wan_dir = str(REPO_ROOT)
        if os.path.exists(wan_dir):
            print("  Available in Wan2.2 directory:")
            for item in os.listdir(wan_dir):
                item_path = os.path.join(wan_dir, item)
                if os.path.isdir(item_path) and item.startswith("Wan2.2"):
                    print(f"      - {item}")
    
    return found_model

def main():
    print("Simple LoRA Optimizer Test")
    print("="*50)
    
    # Test 1: Basic imports
    if not test_imports():
        print("\n Basic imports failed. Install missing dependencies.")
        return 1
    
    # Test 2: VideoX-Fun imports  
    if not test_videox_fun_imports():
        print("\n VideoX-Fun imports failed. Check VideoX-Fun installation.")
        return 1
    
    # Test 3: Model paths
    model_path = test_model_paths()
    if not model_path:
        print("\n No model found. Download WAN2.2 model first.")
        return 1
    
    # Test 4: Try importing our optimizer
    print("\n Testing Ultimate LoRA Optimizer import...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from ultimate_lora_t2v import UltimateFastVideoLoRAOptimizer
        print("  UltimateFastVideoLoRAOptimizer imported successfully!")
        
        # Test basic initialization (without actually loading model)
        print("\n Testing optimizer initialization...")
        optimizer = UltimateFastVideoLoRAOptimizer(
            model_path=model_path,
            verbose=False  # Disable verbose for clean test output
        )
        print("  Optimizer created successfully!")
        
        return 0
        
    except Exception as e:
        print(f"  Failed to import/initialize optimizer: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())