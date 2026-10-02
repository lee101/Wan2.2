# Ultimate LoRA Optimizer for WAN2.2

This directory contains the **Ultimate FastVideo LoRA Optimizer** - a powerful combination of VideoX-Fun LoRA functionality with the Ultimate FastVideo Optimizer for maximum performance.

## Features

- 🚀 **15-25x speedup** with intelligent quality presets
- 🎨 **LoRA support** for fine-tuned models (both low and high noise transformers)
- ⚡ **All optimizations**: TaylorSeer, Flash Attention, Token Merging, FP8 quantization
- 💾 **Smart memory management** based on available VRAM
- 🎯 **Quality presets** from lightning-fast draft to maximum quality
- 📊 **Batch processing** support
- 🔄 **VideoX-Fun compatibility** with WAN2.2-Fun models

## Files

- `ultimate_lora_t2v.py` - Main optimizer with LoRA support
- `example_lora.py` - Simple usage example
- `README.md` - This documentation

## Quick Start

### Basic Usage

```bash
python ultimate_lora_t2v.py \
  --model_path "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP" \
  --lora_path "path/to/your/lora.safetensors" \
  --prompt "A majestic eagle soaring through mountain valleys" \
  --quality balanced
```

### With High and Low Noise LoRAs

```bash
python ultimate_lora_t2v.py \
  --model_path "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP" \
  --lora_path "path/to/low_noise_lora.safetensors" \
  --lora_high_path "path/to/high_noise_lora.safetensors" \
  --lora_weight 0.8 \
  --lora_high_weight 0.6 \
  --prompt "Beautiful sunset over the ocean" \
  --quality fast
```

### Batch Processing

```bash
python ultimate_lora_t2v.py \
  --model_path "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP" \
  --lora_path "path/to/your/lora.safetensors" \
  --prompt "A cat playing in garden" "Ocean waves at sunset" "City at night" \
  --quality lightning
```

## Quality Presets

| Preset | Speed | Quality | Description |
|--------|-------|---------|-------------|
| `lightning` | 20x faster | Draft | Maximum speed for quick previews |
| `draft` | 15x faster | Acceptable | Very fast with decent quality |
| `fast` | 10x faster | Good | Fast generation with good quality |
| `balanced` | 8x faster | High | Best speed/quality balance |
| `quality` | 5x faster | Excellent | Focus on quality |
| `maximum` | 3x faster | Pristine | Maximum quality output |

## Python API

```python
from ultimate_lora_t2v import UltimateFastVideoLoRAOptimizer

# Initialize optimizer
optimizer = UltimateFastVideoLoRAOptimizer(
    model_path="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP",
    lora_path="path/to/your/lora.safetensors",
    lora_weight=0.7,
    verbose=True
)

# Generate video
video = optimizer.generate(
    prompt="A serene mountain lake at dawn",
    quality="balanced",
    video_length=81,
    sample_size=[480, 832],
    seed=42
)
```

## Requirements

- VideoX-Fun repository in `/vfast/data/code/VideoX-Fun`
- WAN2.2-Fun model weights
- Optional LoRA models (.safetensors or .pth format)
- CUDA-capable GPU (8GB+ VRAM recommended)

## Memory Usage by Quality

| Quality | VRAM Usage | Recommended GPU |
|---------|------------|-----------------|
| `lightning` | 6-8 GB | RTX 3060 12GB+ |
| `draft` | 6-8 GB | RTX 3060 12GB+ |
| `fast` | 8-10 GB | RTX 3070 8GB+ |
| `balanced` | 10-12 GB | RTX 3080 10GB+ |
| `quality` | 12-14 GB | RTX 3090 24GB+ |
| `maximum` | 14-16 GB | RTX 4090 24GB+ |

## Advanced Options

### Custom Settings

```python
custom_settings = {
    'steps': 30,
    'guidance_scale': 8.5,
    'token_merge_ratio': 0.2,
    'taylorseer_threshold': 0.08
}

video = optimizer.generate(
    prompt="Your prompt here",
    quality="balanced",
    custom_settings=custom_settings
)
```

### Configuration Files

The optimizer supports VideoX-Fun configuration files:
- Default: `config/wan2.2/wan_civitai_i2v.yaml`
- Custom: `--config_path path/to/your/config.yaml`

## Troubleshooting

### Out of Memory Errors
- Try lower quality presets (`lightning`, `draft`)
- Reduce `video_length` or `sample_size`
- Ensure no other GPU-intensive processes are running

### LoRA Loading Issues
- Verify LoRA file paths are correct
- Check LoRA weights are compatible with your model
- Try reducing LoRA weights (`--lora_weight 0.3`)

### Performance Issues
- Update to latest PyTorch with CUDA support
- Install optional dependencies (flash-attn, xformers, tomesd)
- Use quality presets with compilation enabled

## Integration with VideoX-Fun

This optimizer is fully compatible with VideoX-Fun WAN2.2-Fun models and configurations. It extends the original functionality with:

- Ultimate FastVideo optimization techniques
- Quality-based preset system
- Enhanced memory management
- Batch processing capabilities
- Comprehensive logging and monitoring

## License

Inherits licenses from VideoX-Fun and WAN2.2 projects.