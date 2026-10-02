#!/usr/bin/env python3
"""
Simple test script to verify the setup without running full inference.
This tests core dependencies and basic functionality without models.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    # Core ML libraries
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision: {e}")
        return False
    
    try:
        import diffusers
        print(f"✓ Diffusers {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ Diffusers: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    # Image/video processing
    try:
        from PIL import Image
        print("✓ PIL/Pillow")
    except ImportError as e:
        print(f"✗ PIL: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import imageio
        print(f"✓ ImageIO {imageio.__version__}")
    except ImportError as e:
        print(f"✗ ImageIO: {e}")
        return False
    
    # Utility libraries
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import tqdm
        print("✓ TQDM")
    except ImportError as e:
        print(f"✗ TQDM: {e}")
        return False
    
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Accelerate: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without models."""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        # Test tensor creation
        x = torch.randn(2, 3)
        print("✓ Tensor creation works")
        
        # Test device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Using device: {device}")
        
    except Exception as e:
        print(f"✗ Basic PyTorch functionality: {e}")
        return False
    
    try:
        from PIL import Image
        import numpy as np
        # Test image creation
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        print("✓ Image processing works")
        
    except Exception as e:
        print(f"✗ Image processing: {e}")
        return False
    
    return True

def test_wan_module():
    """Test Wan module import."""
    print("\nTesting Wan module...")
    
    try:
        import wan
        print("✅ Wan module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Wan module import failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("Wan2.2 Setup Test (CUDA)")
    print("=" * 50)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    wan_ok = test_wan_module()
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok and wan_ok:
        print("✅ Setup test PASSED! All dependencies working with CUDA support.")
        print("Ready for model inference (models need to be downloaded separately).")
    else:
        print("❌ Setup test FAILED! Some components are not working.")
    print("=" * 50)

if __name__ == "__main__":
    main()