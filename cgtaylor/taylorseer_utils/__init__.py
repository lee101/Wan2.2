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

import math
from typing import Dict, Optional

import torch


def firstblock_derivative_approximation_moe(
    cache_dic: Dict, current: Dict, feature: torch.Tensor, expert_type: str = "high_noise"
):
    """
    Compute derivative approximation for MoE first block.
    
    :param cache_dic: Cache dictionary with expert-specific caches
    :param current: Information of the current step
    :param feature: Feature tensor
    :param expert_type: Type of expert ('high_noise' or 'low_noise')
    """
    expert_cache = cache_dic["cache"][expert_type]
    difference_distance = current["block_activated_steps"][-1] - current["block_activated_steps"][-2]
    
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature
    
    max_order = cache_dic[f"{expert_type}_firstblock_max_order"]
    for i in range(max_order):
        if (expert_cache["firstblock_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - expert_cache["firstblock_hidden"][i]
            ) / difference_distance
        else:
            break
    
    expert_cache["firstblock_hidden"] = updated_taylor_factors


def firstblock_taylor_formula_moe(
    cache_dic: Dict, current: Dict, expert_type: str = "high_noise"
) -> torch.Tensor:
    """
    Compute Taylor expansion for MoE first block.
    
    :param cache_dic: Cache dictionary with expert-specific caches
    :param current: Information of the current step
    :param expert_type: Type of expert ('high_noise' or 'low_noise')
    """
    expert_cache = cache_dic["cache"][expert_type]
    x = current["step"] - current["block_activated_steps"][-1]
    
    output = 0
    for i in range(len(expert_cache["firstblock_hidden"])):
        output += (1 / math.factorial(i)) * expert_cache["firstblock_hidden"][i] * (x**i)
    
    return output


def step_uncond_derivative_approximation_moe(
    cache_dic: Dict, current: Dict, feature: torch.Tensor, expert_type: str = "high_noise"
):
    """
    Compute derivative approximation for unconditional path with MoE support.
    
    :param cache_dic: Cache dictionary with expert-specific caches
    :param current: Information of the current step
    :param feature: Feature tensor
    :param expert_type: Type of expert ('high_noise' or 'low_noise')
    """
    expert_cache = cache_dic["cache"][expert_type]
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]
    
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature
    
    max_order = cache_dic[f"{expert_type}_max_order"]
    for i in range(max_order):
        if (expert_cache["uncond_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - expert_cache["uncond_hidden"][i]
            ) / difference_distance
        else:
            break
    
    expert_cache["uncond_hidden"] = updated_taylor_factors


def step_cond_derivative_approximation_moe(
    cache_dic: Dict, current: Dict, feature: torch.Tensor, expert_type: str = "high_noise"
):
    """
    Compute derivative approximation for conditional path with MoE support.
    
    :param cache_dic: Cache dictionary with expert-specific caches
    :param current: Information of the current step
    :param feature: Feature tensor
    :param expert_type: Type of expert ('high_noise' or 'low_noise')
    """
    expert_cache = cache_dic["cache"][expert_type]
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]
    
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature
    
    max_order = cache_dic[f"{expert_type}_max_order"]
    for i in range(max_order):
        if (expert_cache["cond_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - expert_cache["cond_hidden"][i]
            ) / difference_distance
        else:
            break
    
    expert_cache["cond_hidden"] = updated_taylor_factors


def taylor_formula_moe(derivative_dict: Dict, distance: int) -> torch.Tensor:
    """
    Compute Taylor expansion for MoE experts.
    
    :param derivative_dict: Derivative dictionary for specific expert
    :param distance: Distance from last activated step
    """
    output = 0
    for i in range(len(derivative_dict)):
        output += (1 / math.factorial(i)) * derivative_dict[i] * (distance**i)
    
    return output


def compute_snr(timestep: torch.Tensor, scheduler) -> float:
    """
    Compute Signal-to-Noise Ratio for expert switching.
    
    :param timestep: Current timestep
    :param scheduler: Scheduler instance
    """
    # Convert timestep to alpha/sigma values
    if hasattr(scheduler, 'alphas_cumprod'):
        alpha_cumprod = scheduler.alphas_cumprod[timestep]
        snr = alpha_cumprod / (1 - alpha_cumprod)
    else:
        # Fallback for flow-based schedulers
        # Approximate SNR based on timestep
        normalized_t = timestep.float() / 1000.0
        snr = 1.0 - normalized_t
    
    return snr.item() if isinstance(snr, torch.Tensor) else snr


def should_switch_expert(snr: float, switch_threshold: float = 0.5) -> str:
    """
    Determine which expert to use based on SNR.
    
    :param snr: Current signal-to-noise ratio
    :param switch_threshold: Threshold for expert switching
    """
    return "high_noise" if snr > switch_threshold else "low_noise"


def prepare_expert_switch(cache_dic: Dict, current_expert: str, next_expert: str):
    """
    Prepare cache for expert switching by pre-warming the next expert's cache.
    
    :param cache_dic: Cache dictionary
    :param current_expert: Currently active expert
    :param next_expert: Expert to switch to
    """
    current_cache = cache_dic["cache"][current_expert]
    next_cache = cache_dic["cache"][next_expert]
    
    # Pre-warm next expert cache with interpolated values if current cache exists
    if current_cache.get("cond_hidden") and not next_cache.get("cond_hidden"):
        # Initialize next expert cache with current values as starting point
        next_cache["cond_hidden"] = {k: v.clone() for k, v in current_cache["cond_hidden"].items()}
    
    if current_cache.get("uncond_hidden") and not next_cache.get("uncond_hidden"):
        next_cache["uncond_hidden"] = {k: v.clone() for k, v in current_cache["uncond_hidden"].items()}


# Legacy compatibility functions (adapted from original Wan2.1 CG-Taylor)
def firstblock_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """Legacy compatibility wrapper"""
    return firstblock_derivative_approximation_moe(cache_dic, current, feature, "high_noise")


def firstblock_taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Legacy compatibility wrapper"""
    return firstblock_taylor_formula_moe(cache_dic, current, "high_noise")


def step_uncond_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """Legacy compatibility wrapper"""
    return step_uncond_derivative_approximation_moe(cache_dic, current, feature, "high_noise")


def step_cond_derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """Legacy compatibility wrapper"""
    return step_cond_derivative_approximation_moe(cache_dic, current, feature, "high_noise")


def taylor_formula(derivative_dict: Dict, distance: int) -> torch.Tensor:
    """Legacy compatibility wrapper"""
    return taylor_formula_moe(derivative_dict, distance)