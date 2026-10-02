# Copyright (c) 2025 Wan Team Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def cache_step_init_moe(num_steps=50, model_variant="A14B"):
    """
    Initialization for MoE-aware CG-Taylor cache system.
    
    :param num_steps: Number of denoising steps
    :param model_variant: Wan 2.2 model variant ("A14B", "TI2V-5B", etc.)
    """
    cache_dic = {}
    
    # MoE-specific cache structure with separate caches for each expert
    cache = {
        "high_noise": {
            "cond_hidden": {},
            "uncond_hidden": {},
            "firstblock_hidden": {},
        },
        "low_noise": {
            "cond_hidden": {},
            "uncond_hidden": {},
            "firstblock_hidden": {},
        }
    }
    
    # Legacy compatibility structure
    cache[-1] = {}
    cache[-1]["cond_stream"] = {}
    cache[-1]["uncond_stream"] = {}
    
    cache_dic["cache_counter"] = 0
    cache_dic["cache"] = cache
    
    # MoE-specific configurations
    cache_dic["moe_enabled"] = True
    cache_dic["expert_switch_snr"] = 0.5  # Configurable switching point
    cache_dic["current_expert"] = "high_noise"  # Start with high-noise expert
    cache_dic["pre_warm_enabled"] = True  # Enable cache pre-warming for expert switches
    
    # Expert-specific Taylor configuration
    # High-noise expert: More aggressive caching due to coarser features
    cache_dic["high_noise_taylor_threshold"] = 0.30
    cache_dic["high_noise_max_order"] = 2
    cache_dic["high_noise_firstblock_max_order"] = 3
    
    # Low-noise expert: Conservative thresholds to preserve fine details
    cache_dic["low_noise_taylor_threshold"] = 0.13
    cache_dic["low_noise_max_order"] = 3
    cache_dic["low_noise_firstblock_max_order"] = 4
    
    # Model variant specific configurations
    if model_variant == "TI2V-5B":
        cache_dic["flow_shift"] = 3.0
        cache_dic["latent_dims"] = (4, 16, 16)  # 4×16×16 compression
        cache_dic["output_resolution"] = (1280, 704, 24)  # 1280×704×24fps
        # TI2V-5B benefits from more conservative caching due to higher quality requirements
        cache_dic["high_noise_taylor_threshold"] = 0.25
        cache_dic["low_noise_taylor_threshold"] = 0.10
    elif "A14B" in model_variant:
        cache_dic["flow_shift"] = 5.0
        cache_dic["latent_dims"] = (8, 32, 32)  # Standard compression
        cache_dic["output_resolution"] = (1280, 720, 30)  # 1280×720×30fps
        # A14B models can handle more aggressive caching
        cache_dic["high_noise_taylor_threshold"] = 0.30
        cache_dic["low_noise_taylor_threshold"] = 0.13
    else:
        # Default configuration
        cache_dic["flow_shift"] = 5.0
        cache_dic["latent_dims"] = (8, 32, 32)
        cache_dic["output_resolution"] = (1280, 720, 30)
    
    # Common Taylor configuration
    cache_dic["taylor_cache"] = True
    cache_dic["Delta-DiT"] = False
    cache_dic["cache_type"] = "moe_taylor"
    cache_dic["fresh_ratio_schedule"] = "MoE-ToCa"
    cache_dic["fresh_ratio"] = 0.0
    cache_dic["fresh_threshold"] = 5
    cache_dic["force_fresh"] = "expert_aware"
    cache_dic["first_enhance"] = 1
    
    # Expert switching monitoring
    cache_dic["expert_switch_history"] = []
    cache_dic["cache_hit_rates"] = {
        "high_noise": {"cond": 0.0, "uncond": 0.0},
        "low_noise": {"cond": 0.0, "uncond": 0.0}
    }
    
    # Memory optimization settings
    cache_dic["enable_cpu_offload"] = True  # Offload inactive expert caches to CPU
    cache_dic["cache_eviction_policy"] = "expert_lru"  # LRU eviction per expert
    cache_dic["max_cache_memory"] = "auto"  # Auto-detect based on available GPU memory
    
    # Current step tracking
    current = {}
    current["activated_steps"] = [0]
    current["block_activated_steps"] = [0]
    current["step"] = 0
    current["num_steps"] = num_steps
    current["expert_type"] = "high_noise"
    current["previous_expert"] = None
    current["switch_pending"] = False
    current["stream"] = None  # Will be set to "cond_stream" or "uncond_stream"
    
    return cache_dic, current


def get_expert_cache(cache_dic, expert_type):
    """
    Get cache for specific expert type.
    
    :param cache_dic: Main cache dictionary
    :param expert_type: Expert type ('high_noise' or 'low_noise')
    """
    return cache_dic["cache"][expert_type]


def update_expert_metrics(cache_dic, expert_type, stream, cache_hit):
    """
    Update cache hit rate metrics for expert.
    
    :param cache_dic: Main cache dictionary
    :param expert_type: Expert type
    :param stream: Stream type ('cond' or 'uncond')
    :param cache_hit: Whether cache was hit (True/False)
    """
    if expert_type not in cache_dic["cache_hit_rates"]:
        cache_dic["cache_hit_rates"][expert_type] = {"cond": 0.0, "uncond": 0.0}
    
    # Simple moving average update
    current_rate = cache_dic["cache_hit_rates"][expert_type][stream]
    cache_dic["cache_hit_rates"][expert_type][stream] = (current_rate * 0.9) + (float(cache_hit) * 0.1)


def should_use_taylor_moe(cache_dic, current, expert_type):
    """
    Determine if Taylor prediction should be used for current expert and step.
    
    :param cache_dic: Main cache dictionary
    :param current: Current step information
    :param expert_type: Expert type
    """
    if not cache_dic.get("taylor_cache", False):
        return False
    
    # Check if we have enough cache history
    if len(current["activated_steps"]) < 2:
        return False
    
    # Get expert-specific threshold
    threshold_key = f"{expert_type}_taylor_threshold"
    threshold = cache_dic.get(threshold_key, 0.18)
    
    # Additional logic can be added here for expert-specific decision making
    # For now, use the threshold from the main forward pass logic
    return True  # Decision will be made in forward pass based on prediction loss


def near_expert_transition(cache_dic, current, snr, lookahead_steps=2):
    """
    Check if we're approaching an expert transition.
    
    :param cache_dic: Main cache dictionary  
    :param current: Current step information
    :param snr: Current signal-to-noise ratio
    :param lookahead_steps: How many steps ahead to check
    """
    switch_threshold = cache_dic.get("expert_switch_snr", 0.5)
    
    # Estimate SNR progression (simplified linear approximation)
    # In practice, this would use the scheduler's noise schedule
    steps_remaining = current["num_steps"] - current["step"]
    if steps_remaining <= lookahead_steps:
        return False
    
    # Simple heuristic: if SNR is within 10% of switch threshold
    threshold_margin = switch_threshold * 0.1
    return abs(snr - switch_threshold) < threshold_margin