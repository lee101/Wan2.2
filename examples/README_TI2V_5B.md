# TI2V-5B GPU-A6000 Example

This directory contains an optimized example for running the Wan2.2 TI2V-5B model on GPU-A6000 hardware.

## Overview

The **TI2V-5B** model is specifically designed for efficient deployment and supports:
- **720P@24fps** video generation
- Both text-to-video and image-to-video generation
- Consumer-grade GPU compatibility
- High compression with Wan2.2-VAE (16×16×4 compression ratio)

## Hardware Requirements

### Recommended: GPU-A6000
- **CPU**: 10 cores
- **RAM**: 18GB 
- **VRAM**: 48GB (24GB minimum with optimizations)

### Also Compatible:
- **GPU-A100**: 12 cores, 60GB RAM, 40GB VRAM
- **RTX 4090**: Consumer card with 24GB VRAM

## Files

- `ti2v_5b_a6000_example.py` - Main example script optimized for GPU-A6000
- `README_TI2V_5B.md` - This documentation
- Integration tests in `../tests/test_ti2v_5b_integration.py`
- Test runner script `../tests/run_ti2v_5b_tests.sh`

## Quick Start

### 1. Check GPU Compatibility
```bash
python examples/ti2v_5b_a6000_example.py --check_gpu
```

### 2. Text-to-Video Generation
```bash
python examples/ti2v_5b_a6000_example.py \
  --mode t2v \
  --prompt "A serene mountain landscape with flowing water and gentle morning light" \
  --model_path ./Wan2.2-TI2V-5B \
  --output my_video.mp4
```

### 3. Image-to-Video Generation
```bash
python examples/ti2v_5b_a6000_example.py \
  --mode i2v \
  --image examples/i2v_input.JPG \
  --prompt "Gentle wind moving through the scene" \
  --model_path ./Wan2.2-TI2V-5B \
  --output my_i2v_video.mp4
```

## GPU-A6000 Optimizations

The example script includes several optimizations for GPU-A6000:

### Memory Optimizations
- **Model Offloading**: `--offload_model True` - Reduces GPU memory usage
- **Data Type Conversion**: `--convert_model_dtype` - Uses optimized precision
- **T5 CPU Usage**: `--t5_cpu` - Keeps T5 model on CPU to save VRAM

### Resolution Settings
- **720P Output**: `1280*704` resolution optimized for TI2V-5B
- **24fps**: Standard frame rate for smooth playback

### Performance Features
- Memory usage tracking and reporting
- Automatic optimization flag selection
- Progress logging and timing information

## Command Line Options

```bash
python examples/ti2v_5b_a6000_example.py --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Generation mode: `t2v` or `i2v` | Required |
| `--model_path` | Path to Wan2.2-TI2V-5B model | `./Wan2.2-TI2V-5B` |
| `--prompt` | Text prompt for generation | Auto-generated |
| `--image` | Input image path (required for i2v) | None |
| `--output` | Output video path | Auto-generated |
| `--seed` | Random seed | 42 |
| `--check_gpu` | Check GPU compatibility and exit | False |

## Expected Performance

Based on the original Wan2.2 documentation:

### TI2V-5B Model Performance
- **Generation Time**: ~9 minutes for 5-second 720P video (single GPU)
- **VRAM Usage**: ~24GB with optimizations
- **Output**: 720P@24fps video

### Comparison to Other Models
- **Faster than A14B models** - Requires less VRAM and computation
- **Consumer-friendly** - Can run on RTX 4090 and similar cards
- **Unified framework** - Supports both T2V and I2V in one model

## Testing

Run the comprehensive test suite:

```bash
# Quick tests (no actual model loading)
bash tests/run_ti2v_5b_tests.sh --quick

# Full integration tests
bash tests/run_ti2v_5b_tests.sh

# Unit tests only
python -m pytest tests/test_ti2v_5b_integration.py -v
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Use all memory optimizations
   python examples/ti2v_5b_a6000_example.py --mode t2v --prompt "test" \
     --model_path ./Wan2.2-TI2V-5B
   # The script automatically enables --offload_model, --convert_model_dtype, --t5_cpu
   ```

2. **CUDA Not Available**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Model Not Found**
   ```bash
   # Download the model first
   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
   ```

### Performance Tips

1. **For Maximum Speed**: Use multi-GPU setup with FSDP + DeepSpeed Ulysses
2. **For Memory Efficiency**: Keep all optimization flags enabled
3. **For Best Quality**: Use prompt extension with Qwen models

## Integration with Main Repository

This example integrates with the main Wan2.2 repository by:
- Using the standard `generate.py` script
- Following the same configuration patterns
- Supporting all standard Wan2.2 parameters
- Maintaining compatibility with existing workflows

## Model Downloads

| Model | HuggingFace | ModelScope | Description |
|-------|-------------|------------|-------------|
| TI2V-5B | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | [Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | High-compression VAE, T2V+I2V, 720P |

```bash
# Download with huggingface-cli
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

# Download with modelscope-cli  
pip install modelscope
modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B
```

## Contributing

When contributing to this example:

1. **Run Tests**: Always run the test suite before submitting changes
2. **Follow Conventions**: Use the same code style as the main repository
3. **Update Documentation**: Keep this README updated with any changes
4. **Test on Target Hardware**: Verify changes work on GPU-A6000 when possible

## License

This example follows the same license as the main Wan2.2 repository (Apache 2.0).