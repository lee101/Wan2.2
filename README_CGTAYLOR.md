# CG-Taylor Acceleration for Wan 2.2

This implementation provides training-free acceleration for Wan 2.2 video generation using CG-Taylor expansion to predict intermediate features and reduce computation time.

## Overview

The CG-Taylor acceleration method works by:
1. **Caching** intermediate features from previous timesteps
2. **Predicting** future features using Taylor expansion
3. **Validating** predictions to ensure quality
4. **Switching** between high-noise and low-noise expert configurations

## Key Features

- **MoE Integration**: Seamlessly works with Wan 2.2's existing high/low noise expert architecture
- **Adaptive Thresholds**: Automatically adjusts confidence thresholds based on prediction accuracy
- **Memory Efficient**: Optional CPU caching to save GPU memory
- **Quality Preservation**: Conservative validation ensures output quality matches original
- **Easy Integration**: Drop-in replacement for existing Wan T2V workflows

## Quick Start

### Basic Usage

```python
from wan.text2video_accelerated import WanT2VAccelerated
from wan.configs.wan_t2v_A14B import t2v_A14B

# Create accelerated model (replaces WanT2V)
model = WanT2VAccelerated(
    config=t2v_A14B,
    checkpoint_dir="path/to/wan22/models",
    enable_cgtaylor=True  # Enable acceleration (default)
)

# Generate video (same API as original)
video = model.generate(
    input_prompt="A majestic eagle soaring through mountains",
    size=(1280, 720),
    frame_num=81,
    sampling_steps=40
)

# Check acceleration statistics
stats = model.get_acceleration_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']}")
print(f"Estimated speedup: {stats['estimated_speedup']}")
```

### Configuration Tuning

```python
# Conservative settings for maximum quality
model.set_acceleration_config(
    high_noise_config={
        'confidence_threshold': 0.10,  # Very conservative
        'cache_window': 3,
        'order': 2
    },
    low_noise_config={
        'confidence_threshold': 0.08,
        'cache_window': 2, 
        'order': 2
    }
)

# Aggressive settings for maximum speed
model.set_acceleration_config(
    high_noise_config={
        'confidence_threshold': 0.35,  # More aggressive
        'cache_window': 6,
        'order': 3
    },
    low_noise_config={
        'confidence_threshold': 0.20,
        'cache_window': 4,
        'order': 3  
    }
)
```

### Temporary Disable Acceleration

```python
# Disable for this generation only
video = model.generate("prompt", enable_acceleration=False)

# Or use context manager
with model.acceleration_disabled():
    video = model.generate("prompt")
```

## Configuration Parameters

### TaylorConfig Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `order` | Taylor expansion order | 2 | 1-3 |
| `confidence_threshold` | Prediction confidence threshold | 0.15 | 0.05-0.5 |
| `cache_window` | Number of timesteps to cache | 5 | 2-10 |
| `adaptive_threshold` | Enable adaptive thresholding | True | True/False |
| `memory_efficient` | Cache features on CPU | True | True/False |

### Expert-Specific Defaults

**High Noise Expert** (t ≥ boundary):
- More aggressive acceleration for faster early denoising
- `confidence_threshold`: 0.25
- `cache_window`: 4
- `order`: 2

**Low Noise Expert** (t < boundary):
- Conservative settings to preserve fine details
- `confidence_threshold`: 0.12
- `cache_window`: 3
- `order`: 3

## Performance

Expected performance improvements:

| Resolution | Frames | Standard Time | Accelerated Time | Speedup |
|------------|--------|---------------|------------------|---------|
| 1280x720 | 81 | ~120s | ~75s | 1.6x |
| 960x540 | 49 | ~60s | ~38s | 1.6x |
| 1920x1080 | 81 | ~180s | ~110s | 1.6x |

*Times are approximate and depend on hardware, prompts, and settings*

## Memory Usage

The acceleration system is designed to be memory-efficient:

- **CPU Caching**: Features cached on CPU by default to save GPU memory
- **Adaptive Windows**: Cache size automatically adjusts based on available memory
- **Automatic Cleanup**: Caches cleared after generation when `offload_model=True`

For high-resolution generations, the system automatically:
1. Reduces cache window size to fit memory constraints
2. Adjusts confidence thresholds based on resolution
3. Enables CPU caching for large feature maps

## Testing

Run the quick test to verify installation:

```bash
cd examples
python quick_cgtaylor_test.py
```

Run comprehensive demos:

```bash
cd examples
python cgtaylor_acceleration_demo.py --checkpoint-dir /path/to/models --demo all
```

## Implementation Details

### Taylor Expansion

The system uses finite difference approximations for derivatives:

- **1st Order**: `f'(t) ≈ (f(t) - f(t-1)) / dt`
- **2nd Order**: `f''(t) ≈ (f(t) - 2f(t-1) + f(t-2)) / dt²`
- **3rd Order**: `f'''(t) ≈ (f(t) - 3f(t-1) + 3f(t-2) - f(t-3)) / dt³`

### Expert Switching

The system automatically switches between expert configurations based on the timestep boundary (default 0.875 * num_train_timesteps), matching Wan 2.2's built-in expert switching.

### Quality Validation

Each prediction is validated by comparing against a reference computation:
1. Compute prediction error: `|predicted - actual|`
2. Compare against confidence threshold
3. Update adaptive threshold based on recent prediction accuracy
4. Use actual computation if prediction fails validation

## Troubleshooting

### Common Issues

**Low Cache Hit Rate**:
- Increase `confidence_threshold` for more aggressive acceleration
- Increase `cache_window` to provide more history
- Enable `adaptive_threshold` for automatic tuning

**Quality Degradation**:
- Decrease `confidence_threshold` for more conservative predictions
- Use lower Taylor `order` (2 instead of 3)
- Disable acceleration for critical quality generations

**Memory Issues**:
- Enable `memory_efficient=True` (default)
- Reduce `cache_window` size
- Use smaller resolutions for testing

**Import Errors**:
- Ensure you're in the project root directory
- Check that all files are in the correct locations
- Run `python examples/quick_cgtaylor_test.py` to diagnose

### Debugging

Enable debug logging to see detailed acceleration behavior:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run generation - you'll see detailed logs about:
# - Expert switches
# - Cache hits/misses  
# - Prediction errors
# - Validation results
```

## Contributing

When modifying the acceleration system:

1. **Test Quality**: Always compare outputs with acceleration disabled
2. **Test Performance**: Benchmark on multiple resolutions and prompt types
3. **Test Memory**: Monitor GPU memory usage during generation
4. **Update Tests**: Add tests for new features in `quick_cgtaylor_test.py`

## License

Same as Wan 2.2 - refer to main project license.