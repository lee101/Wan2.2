# Memory Optimization Guide for Wan2.2

This guide covers memory optimization strategies for running Wan2.2 models efficiently on different GPU configurations.

## Quick Reference - Memory Optimization Flags

### For Limited VRAM (24GB or less)
```bash
# TI2V-5B (most memory efficient)
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "Your prompt here"
```

### For High VRAM (80GB+)
```bash
# Remove memory optimization flags for faster inference
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --prompt "Your prompt here"
```

## Memory Optimization Flags Explained

### `--offload_model True`
- **Purpose**: Moves model components to CPU after each forward pass
- **Memory Impact**: Significantly reduces GPU memory usage
- **Performance Impact**: Slower inference due to CPU↔GPU transfers
- **When to use**: Limited VRAM situations

### `--convert_model_dtype`
- **Purpose**: Converts model parameters to lower precision (config.param_dtype)
- **Memory Impact**: ~50% memory reduction (fp32 → fp16/bf16)
- **Performance Impact**: Minimal impact on quality
- **When to use**: Almost always recommended

### `--t5_cpu`
- **Purpose**: Places T5 text encoder on CPU instead of GPU
- **Memory Impact**: Saves ~8-12GB GPU memory
- **Performance Impact**: Slower text processing
- **When to use**: When GPU memory is limited

### `--t5_fsdp`
- **Purpose**: Uses Fully Sharded Data Parallel for T5 across multiple GPUs
- **Memory Impact**: Distributes T5 memory across GPUs
- **Performance Impact**: Better for multi-GPU setups
- **When to use**: Multi-GPU inference with limited per-GPU memory

### `--dit_fsdp`
- **Purpose**: Uses FSDP for the DiT (Diffusion Transformer) model
- **Memory Impact**: Distributes DiT memory across GPUs
- **Performance Impact**: Enables larger batch sizes
- **When to use**: Multi-GPU inference

### `--ulysses_size N`
- **Purpose**: Uses DeepSpeed Ulysses for sequence parallelism
- **Memory Impact**: Distributes sequence dimension across N GPUs
- **Performance Impact**: Faster inference with proper GPU count
- **When to use**: Multi-GPU setups (N = number of GPUs)

## Model-Specific Memory Requirements

### TI2V-5B (Most Efficient)
- **Base memory**: ~24GB
- **With optimizations**: ~12-16GB
- **Recommended GPU**: RTX 4090, RTX 3090, A6000+

### A14B Models (Higher Quality)
- **Base memory**: ~80GB
- **With optimizations**: ~40-50GB
- **Recommended GPU**: A100 80GB, H100+

## Example Configurations

### Single RTX 4090 (24GB)
```bash
# TI2V-5B with all optimizations
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "A cat playing with yarn"
```

### Multi-GPU (4x RTX 3090)
```bash
# A14B with FSDP and Ulysses
torchrun --nproc_per_node=4 generate.py \
  --task t2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp --t5_fsdp --ulysses_size 4 \
  --prompt "A beautiful sunset over mountains"
```

### Single A100 80GB
```bash
# A14B with minimal optimizations
python generate.py --task t2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --convert_model_dtype \
  --prompt "A bustling city street at night"
```

## Performance vs Memory Trade-offs

| Configuration | Memory Usage | Speed | Quality |
|---------------|-------------|--------|---------|
| No optimizations | Highest | Fastest | Best |
| `--convert_model_dtype` | -50% | ~Same | ~Same |
| `+ --t5_cpu` | -20% more | Slower | Same |
| `+ --offload_model` | -40% more | Much slower | Same |

## Troubleshooting

### Out of Memory Errors
1. Add `--offload_model True`
2. Add `--t5_cpu`
3. Use smaller resolution (e.g., 720*1280 → 480*832)
4. Reduce `--frame_num` (default 121 → 81 or 41)

### Slow Inference
1. Remove `--offload_model` if you have enough VRAM
2. Keep `--convert_model_dtype` (minimal impact)
3. Use GPU for T5 if memory allows

### Multi-GPU Issues
1. Ensure `--ulysses_size` matches GPU count
2. Use both `--dit_fsdp` and `--t5_fsdp`
3. Check CUDA visible devices: `export CUDA_VISIBLE_DEVICES=0,1,2,3`

## Model Download Commands

```bash
# TI2V-5B (most memory efficient)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

# T2V-A14B (highest quality text-to-video)
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

# I2V-A14B (image-to-video)
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
```