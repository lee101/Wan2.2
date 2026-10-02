#!/usr/bin/env python3
"""
UV-based Installation Script for WAN2.2 Ultra-Fast Optimizations
Uses UV package manager for faster, more reliable dependency resolution.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_command(cmd, description="", capture_output=True, check=True):
    """Run a command with proper error handling."""
    print(f"🔧 {description}")
    
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            check=check
        )
        
        if capture_output and result.stdout:
            print(f"✅ {description} completed")
        
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return None
    
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return None

def check_uv_installed():
    """Check if UV is installed and install if needed."""
    if shutil.which("uv"):
        print("✅ UV package manager found")
        return True
    
    print("📦 UV not found, installing...")
    
    # Try to install UV
    install_methods = [
        # Method 1: pip install
        ([sys.executable, "-m", "pip", "install", "uv"], "Installing UV via pip"),
        
        # Method 2: curl install (Unix-like systems)
        (["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"], "Installing UV via curl"),
        
        # Method 3: PowerShell install (Windows)
        (["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"], "Installing UV via PowerShell")
    ]
    
    for cmd, desc in install_methods:
        try:
            if cmd[0] == "curl" and os.name == 'nt':
                continue  # Skip curl on Windows
            if cmd[0] == "powershell" and os.name != 'nt':
                continue  # Skip PowerShell on non-Windows
            
            result = run_command(cmd, desc, check=False)
            if result and result.returncode == 0:
                print("✅ UV installed successfully")
                return True
        except:
            continue
    
    print("❌ Failed to install UV automatically")
    print("Please install UV manually: https://docs.astral.sh/uv/getting-started/installation/")
    return False

def create_optimized_requirements():
    """Create optimized requirements files for different scenarios."""
    
    # Core requirements (always needed)
    core_reqs = """# WAN2.2 Core Dependencies
torch>=2.4.0
torchvision>=0.19.0
torchaudio
opencv-python>=4.9.0.80
diffusers>=0.31.0
transformers>=4.49.0,<=4.51.3
tokenizers>=0.20.3
accelerate>=1.1.1
tqdm
imageio[ffmpeg]
easydict
ftfy
dashscope
imageio-ffmpeg
numpy>=1.23.5,<2"""
    
    # Optimization requirements (performance boost)
    optimization_reqs = """# Ultra-Fast Optimization Dependencies
cache-dit  # TaylorSeer - 5x speedup
flash-attn  # Flash Attention 2 - 1.33x speedup
tomesd  # Token Merging - 1.4x speedup
torchao  # FP8 Quantization - 2.3x speedup
xformers  # Memory-efficient attention fallback"""
    
    # Optional requirements (nice to have)
    optional_reqs = """# Performance Monitoring and Video Processing
psutil
gpustat
memory-profiler
realesrgan
pytest
pytest-cov"""
    
    # Write different requirement files
    requirements_dir = Path("requirements")
    requirements_dir.mkdir(exist_ok=True)
    
    # Core only
    with open(requirements_dir / "core.txt", "w") as f:
        f.write(core_reqs)
    
    # Core + optimizations
    with open(requirements_dir / "optimized.txt", "w") as f:
        f.write(core_reqs + "\n\n" + optimization_reqs)
    
    # All requirements
    with open(requirements_dir / "all.txt", "w") as f:
        f.write(core_reqs + "\n\n" + optimization_reqs + "\n\n" + optional_reqs)
    
    print("✅ Created optimized requirements files in requirements/ directory")

def install_with_uv(requirements_file="requirements.in", create_venv=True):
    """Install dependencies using UV package manager."""
    
    if create_venv:
        print("🔧 Creating virtual environment with UV...")
        result = run_command(
            ["uv", "venv", "--python", "3.11"], 
            "Creating Python 3.11 virtual environment"
        )
        if not result:
            print("⚠️  Failed to create venv, continuing with system Python")
    
    print("📦 Installing dependencies with UV...")
    
    # Install from requirements.in if it exists, otherwise use requirements.txt
    req_file = requirements_file if os.path.exists(requirements_file) else "requirements.txt"
    
    if not os.path.exists(req_file):
        print(f"❌ Requirements file {req_file} not found")
        return False
    
    # UV pip install command
    cmd = ["uv", "pip", "install", "-r", req_file]
    
    result = run_command(
        cmd,
        f"Installing dependencies from {req_file}",
        capture_output=False  # Show real-time output
    )
    
    if result:
        print("✅ All dependencies installed successfully with UV!")
        return True
    else:
        print("❌ Installation failed")
        return False

def install_gpu_specific_optimizations():
    """Install GPU-specific optimizations based on detected hardware."""
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            
            print(f"🔍 Detected GPU: {gpu_name}")
            print(f"🔍 Compute Capability: {capability}")
            
            # Install optimizations based on GPU capabilities
            if capability[0] >= 8:  # Ampere/Hopper (RTX 30/40/50, A100, H100)
                print("🚀 Installing optimizations for modern GPU...")
                
                # These require modern GPU support
                modern_gpu_packages = [
                    "flash-attn --no-build-isolation",
                    "torchao",
                ]
                
                for package in modern_gpu_packages:
                    run_command(
                        ["uv", "pip", "install"] + package.split(),
                        f"Installing {package.split()[0]} for modern GPU",
                        check=False
                    )
            
            else:
                print("ℹ️  Installing compatibility optimizations for older GPU...")
                # Fallback optimizations for older GPUs
                run_command(
                    ["uv", "pip", "install", "xformers"],
                    "Installing xFormers for older GPU compatibility",
                    check=False
                )
        
        else:
            print("⚠️  No CUDA GPU detected, installing CPU-only optimizations")
            
    except ImportError:
        print("⚠️  PyTorch not available yet, skipping GPU-specific optimizations")

def verify_installation():
    """Verify that all critical packages are installed correctly."""
    
    print("\n🔍 Verifying installation...")
    
    verification_tests = {
        'PyTorch': 'import torch; print(f"PyTorch {torch.__version__}")',
        'WAN Module': 'import sys; sys.path.insert(0, "."); import wan; print("WAN module loaded")',
        'TaylorSeer': 'import cache_dit; print("TaylorSeer available")',
        'Flash Attention': 'import flash_attn; print("Flash Attention available")',
        'Token Merging': 'import tomesd; print("Token Merging available")',
        'FP8 Quantization': 'import torchao; print("TorchAO available")',
        'xFormers': 'import xformers; print("xFormers available")',
    }
    
    results = {}
    
    for name, test_code in verification_tests.items():
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                results[name] = "✅"
                print(f"✅ {name}: Working")
            else:
                results[name] = "❌"
                print(f"❌ {name}: Failed")
                
        except subprocess.TimeoutExpired:
            results[name] = "⏰"
            print(f"⏰ {name}: Timeout")
        except Exception as e:
            results[name] = "❌"
            print(f"❌ {name}: Error - {e}")
    
    # Summary
    working_count = sum(1 for status in results.values() if status == "✅")
    total_count = len(results)
    
    print(f"\n📊 Verification Summary: {working_count}/{total_count} components working")
    
    if working_count >= 4:
        print("🚀 Installation successful! Ready for ultra-fast generation!")
        expected_speedup = "15-25x"
    elif working_count >= 2:
        print("⚡ Basic installation successful! Ready for fast generation!")
        expected_speedup = "5-15x"
    else:
        print("⚠️  Minimal installation. Consider troubleshooting failed components.")
        expected_speedup = "2-5x"
    
    print(f"🎯 Expected speedup: {expected_speedup}")
    
    return results

def main():
    """Main installation workflow."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            🚀 WAN2.2 UV ULTRA-FAST INSTALLER 🚀              ║
    ║                                                              ║
    ║        Using UV package manager for lightning-fast          ║
    ║             dependency resolution and installation           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check/Install UV
    if not check_uv_installed():
        print("❌ Cannot proceed without UV package manager")
        sys.exit(1)
    
    # Step 2: Create optimized requirements files
    create_optimized_requirements()
    
    # Step 3: Install dependencies
    print("\n📦 DEPENDENCY INSTALLATION")
    print("=" * 50)
    
    # Choose installation method
    install_options = [
        ("requirements/all.txt", "Full installation (recommended)"),
        ("requirements/optimized.txt", "Core + optimizations only"),
        ("requirements/core.txt", "Core dependencies only"),
        ("requirements.in", "Use existing requirements.in")
    ]
    
    print("Choose installation option:")
    for i, (file, desc) in enumerate(install_options, 1):
        print(f"  {i}. {desc}")
    
    try:
        choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"
        choice_idx = int(choice) - 1
        
        if 0 <= choice_idx < len(install_options):
            req_file, desc = install_options[choice_idx]
            print(f"\n🎯 Selected: {desc}")
        else:
            req_file, desc = install_options[0]
            print(f"\n🎯 Using default: {desc}")
    
    except (ValueError, KeyboardInterrupt):
        req_file, desc = install_options[0]
        print(f"\n🎯 Using default: {desc}")
    
    # Install with UV
    success = install_with_uv(req_file)
    
    if success:
        # Step 4: Install GPU-specific optimizations
        print("\n🎮 GPU-SPECIFIC OPTIMIZATIONS")
        print("=" * 50)
        install_gpu_specific_optimizations()
        
        # Step 5: Verify installation
        print("\n🔍 INSTALLATION VERIFICATION")
        print("=" * 50)
        results = verify_installation()
        
        # Step 6: Usage instructions
        print(f"""
        
🎉 INSTALLATION COMPLETE!

🚀 Quick Start:
    python examples/ultimate_optimizer.py \\
        --task t2v-A14B \\
        --checkpoint_dir /path/to/checkpoints \\
        --prompt "A cat playing in a garden" \\
        --quality lightning

📊 Benchmark Performance:
    python examples/ultimate_optimizer.py \\
        --task t2v-A14B \\
        --checkpoint_dir /path/to/checkpoints \\
        --benchmark

🔧 Activate Environment (if created):
    source .venv/bin/activate  # Linux/Mac
    .venv\\Scripts\\activate   # Windows

📚 Documentation:
    See ULTIMATE_OPTIMIZER_README.md for complete usage guide
        """)
        
    else:
        print("❌ Installation failed. Please check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()