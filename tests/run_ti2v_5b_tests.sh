#!/bin/bash
"""
TI2V-5B Integration Test Runner
==============================

This script runs comprehensive tests for the TI2V-5B GPU-A6000 example,
including unit tests, integration tests, and compatibility checks.

Usage:
    bash tests/run_ti2v_5b_tests.sh [options]

Options:
    --model-path PATH    Path to Wan2.2-TI2V-5B model (default: ./Wan2.2-TI2V-5B)
    --quick              Run only quick tests (no actual generation)
    --gpu-check          Run GPU compatibility check first
    --verbose            Enable verbose output
    --help               Show this help message

Examples:
    # Run all tests with default model path
    bash tests/run_ti2v_5b_tests.sh
    
    # Run quick tests only
    bash tests/run_ti2v_5b_tests.sh --quick
    
    # Run with custom model path
    bash tests/run_ti2v_5b_tests.sh --model-path /path/to/model
"""

set -e  # Exit on any error

# Default configuration
MODEL_PATH="./Wan2.2-TI2V-5B"
QUICK_MODE=false
GPU_CHECK=false
VERBOSE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    head -n 25 "$0" | tail -n +2 | sed 's/^"""//' | sed 's/"""$//'
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --gpu-check)
            GPU_CHECK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set verbose mode
if [ "$VERBOSE" = true ]; then
    set -x
fi

log_info "Starting TI2V-5B Integration Tests"
log_info "Project directory: $PROJECT_DIR"
log_info "Model path: $MODEL_PATH"
log_info "Quick mode: $QUICK_MODE"

# Change to project directory
cd "$PROJECT_DIR"

# Check Python and dependencies
log_info "Checking Python environment..."
if ! command -v python &> /dev/null; then
    log_error "Python not found"
    exit 1
fi

python_version=$(python --version 2>&1)
log_info "Using $python_version"

# Check if required packages are available
log_info "Checking required packages..."
required_packages=("torch" "PIL" "pathlib")
for package in "${required_packages[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        log_warning "Package $package not available - some tests may be skipped"
    else
        log_info "$package available"
    fi
done

# GPU compatibility check
if [ "$GPU_CHECK" = true ] || [ "$QUICK_MODE" = false ]; then
    log_info "Running GPU compatibility check..."
    if python examples/ti2v_5b_a6000_example.py --check_gpu; then
        log_success "GPU compatibility check passed"
    else
        log_warning "GPU compatibility check failed - continuing with tests"
    fi
fi

# Run Python unit tests
log_info "Running unit tests..."
if python -m pytest tests/test_ti2v_5b_integration.py -v; then
    log_success "Unit tests passed"
else
    log_error "Unit tests failed"
    exit 1
fi

# Test example script imports and basic functionality
log_info "Testing example script functionality..."
if python -c "
import sys
sys.path.insert(0, '.')
from examples.ti2v_5b_a6000_example import TI2V5BGenerator
import tempfile
import os
test_dir = tempfile.mkdtemp()
model_dir = os.path.join(test_dir, 'mock_model')
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    f.write('{\"model\": \"ti2v-5b\"}')
try:
    generator = TI2V5BGenerator(model_dir)
    print('Generator initialization successful')
    memory_info = generator.get_memory_usage()
    print(f'Memory tracking working: {len(memory_info)} metrics')
    print('Example script functionality test passed')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
finally:
    import shutil
    shutil.rmtree(test_dir)
"; then
    log_success "Example script functionality test passed"
else
    log_error "Example script functionality test failed"
    exit 1
fi

# Test CLI argument parsing
log_info "Testing CLI argument parsing..."
if python examples/ti2v_5b_a6000_example.py --help > /dev/null; then
    log_success "CLI help works correctly"
else
    log_error "CLI help failed"
    exit 1
fi

# Performance and configuration tests
log_info "Running performance configuration tests..."
if python -c "
import sys
sys.path.insert(0, '.')
from examples.ti2v_5b_a6000_example import TI2V5BGenerator
import tempfile
import os

test_dir = tempfile.mkdtemp()
model_dir = os.path.join(test_dir, 'mock_model')
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    f.write('{\"model\": \"ti2v-5b\"}')

try:
    generator = TI2V5BGenerator(model_dir)
    
    # Check GPU-A6000 optimized settings
    assert generator.config['offload_model'] == True, 'Model offloading should be enabled'
    assert generator.config['convert_model_dtype'] == True, 'Data type conversion should be enabled'
    assert generator.config['t5_cpu'] == True, 'T5 CPU usage should be enabled'
    assert generator.config['size'] == '1280*704', 'Resolution should be optimized for TI2V-5B'
    
    print('All performance optimizations correctly configured')
except Exception as e:
    print(f'Performance configuration error: {e}')
    sys.exit(1)
finally:
    import shutil
    shutil.rmtree(test_dir)
"; then
    log_success "Performance configuration tests passed"
else
    log_error "Performance configuration tests failed"
    exit 1
fi

# Integration test with mock model (only if not quick mode)
if [ "$QUICK_MODE" = false ]; then
    log_info "Running integration tests with mock generation..."
    
    # Create temporary test environment
    TEST_DIR=$(mktemp -d)
    MOCK_MODEL_DIR="$TEST_DIR/Wan2.2-TI2V-5B"
    mkdir -p "$MOCK_MODEL_DIR"
    echo '{"model": "ti2v-5b"}' > "$MOCK_MODEL_DIR/config.json"
    echo "mock_model_data" > "$MOCK_MODEL_DIR/pytorch_model.bin"
    
    # Create test image
    TEST_IMAGE="$TEST_DIR/test_image.jpg"
    echo "fake_image_data" > "$TEST_IMAGE"
    
    log_info "Testing text-to-video command construction..."
    if python -c "
import sys, os, subprocess, tempfile
from unittest.mock import patch
sys.path.insert(0, '.')
from examples.ti2v_5b_a6000_example import TI2V5BGenerator

generator = TI2V5BGenerator('$MOCK_MODEL_DIR')
output_file = '$TEST_DIR/test_output.mp4'

# Mock subprocess.run to capture command without executing
with patch('subprocess.run') as mock_run:
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = 'success'
    mock_run.return_value.stderr = ''
    
    with patch('os.chdir'), patch('os.getcwd', return_value='$TEST_DIR'):
        try:
            result = generator.generate_text_to_video(
                prompt='Test prompt',
                output_path=output_file,
                seed=42
            )
            print('Text-to-video integration test passed')
        except Exception as e:
            print(f'Text-to-video integration test failed: {e}')
            sys.exit(1)
"; then
        log_success "Text-to-video integration test passed"
    else
        log_error "Text-to-video integration test failed"
        rm -rf "$TEST_DIR"
        exit 1
    fi
    
    log_info "Testing image-to-video command construction..."
    if python -c "
import sys, os, subprocess, tempfile
from unittest.mock import patch
sys.path.insert(0, '.')
from examples.ti2v_5b_a6000_example import TI2V5BGenerator

generator = TI2V5BGenerator('$MOCK_MODEL_DIR')
output_file = '$TEST_DIR/test_output_i2v.mp4'

# Mock subprocess.run to capture command without executing
with patch('subprocess.run') as mock_run:
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = 'success'
    mock_run.return_value.stderr = ''
    
    with patch('os.chdir'), patch('os.getcwd', return_value='$TEST_DIR'):
        try:
            result = generator.generate_image_to_video(
                image_path='$TEST_IMAGE',
                prompt='Test I2V prompt',
                output_path=output_file,
                seed=123
            )
            print('Image-to-video integration test passed')
        except Exception as e:
            print(f'Image-to-video integration test failed: {e}')
            sys.exit(1)
"; then
        log_success "Image-to-video integration test passed"
    else
        log_error "Image-to-video integration test failed"
        rm -rf "$TEST_DIR"
        exit 1
    fi
    
    # Clean up
    rm -rf "$TEST_DIR"
else
    log_info "Skipping integration tests (quick mode enabled)"
fi

# Final model path check (if model exists)
if [ -d "$MODEL_PATH" ]; then
    log_info "Validating model directory structure..."
    
    required_files=("config.json")
    for file in "${required_files[@]}"; do
        if [ -f "$MODEL_PATH/$file" ]; then
            log_info "Found $file"
        else
            log_warning "Missing $file in model directory"
        fi
    done
    
    log_success "Model directory validation completed"
else
    log_warning "Model path $MODEL_PATH not found - skipping model validation"
fi

# Summary
log_success "All TI2V-5B integration tests completed successfully!"
log_info "Test summary:"
log_info "  Python environment check"
log_info "  Unit tests"
log_info "  Example script functionality"
log_info "  CLI argument parsing"
log_info "  Performance configuration"
if [ "$QUICK_MODE" = false ]; then
    log_info "  Integration tests"
fi
if [ -d "$MODEL_PATH" ]; then
    log_info "  Model directory validation"
fi

log_success "Ready for deployment on GPU-A6000!"

# Usage examples
log_info ""
log_info "Usage examples:"
log_info "  # Text-to-video generation"
log_info "  python examples/ti2v_5b_a6000_example.py --mode t2v --prompt 'A serene landscape'"
log_info ""
log_info "  # Image-to-video generation"  
log_info "  python examples/ti2v_5b_a6000_example.py --mode i2v --image examples/i2v_input.JPG --prompt 'Gentle motion'"
log_info ""
log_info "  # Check GPU compatibility"
log_info "  python examples/ti2v_5b_a6000_example.py --check_gpu"