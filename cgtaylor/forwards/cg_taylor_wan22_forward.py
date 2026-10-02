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

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from ..taylorseer_utils import (
    firstblock_derivative_approximation_moe,
    firstblock_taylor_formula_moe,
    step_cond_derivative_approximation_moe,
    step_uncond_derivative_approximation_moe,
    taylor_formula_moe,
    compute_snr,
    should_switch_expert,
    prepare_expert_switch,
)
from ..cache_functions.cache_step_init_moe import (
    update_expert_metrics,
    should_use_taylor_moe,
    near_expert_transition,
)


class MoECGTaylorPipeline(nn.Module):
    """
    MoE-aware CG-Taylor Pipeline for Wan 2.2.
    
    This class implements the conceptual MoE architecture by treating different
    noise levels as separate "experts" with independent caching systems.
    """
    
    def __init__(self, model, model_variant="A14B"):
        super().__init__()
        self.model = model
        self.model_variant = model_variant
        
        # MoE configuration based on requirements
        self.expert_switch_snr = 0.5  # Configurable switching point
        self.high_noise_threshold = 0.30  # More aggressive caching
        self.low_noise_threshold = 0.13   # Conservative caching
        
        # Dual caching system
        self.taylor_cache_high = None  # Will be set by cache_dic
        self.taylor_cache_low = None
        
        # Model-specific configurations
        if model_variant == "TI2V-5B":
            self.flow_shift = 3.0
            self.high_noise_threshold = 0.25  # More conservative for TI2V-5B
            self.low_noise_threshold = 0.10
        elif "A14B" in model_variant:
            self.flow_shift = 5.0
            self.high_noise_threshold = 0.30
            self.low_noise_threshold = 0.13
        else:
            self.flow_shift = 5.0
        
        # Performance tracking
        self.cache_hit_rates = {"high_noise": 0.0, "low_noise": 0.0}
        self.expert_switch_history = []


def CGTaylor_wan22_forward(
    self,
    x: List[torch.Tensor],
    t: torch.Tensor,
    context: List[torch.Tensor],
    seq_len: int,
    y: Optional[List[torch.Tensor]] = None,
    current: Optional[Dict] = None,
    cache_dic: Optional[Dict] = None,
    scheduler=None,
) -> List[torch.Tensor]:
    """
    MoE-aware CG-Taylor forward pass for Wan 2.2.
    
    This adapts the original Wan 2.2 forward pass to support MoE-like caching
    where different noise levels are treated as separate experts.
    
    Args:
        self: WanModel instance
        x: List of input video tensors
        t: Diffusion timesteps
        context: List of text embeddings
        seq_len: Maximum sequence length
        y: Conditional inputs for i2v mode
        current: Current step information for caching
        cache_dic: Cache dictionary with MoE support
        scheduler: Scheduler for SNR computation
    """
    
    if current is None or cache_dic is None:
        # Fallback to original forward pass if no caching
        return original_wan22_forward(self, x, t, context, seq_len, y)
    
    # Determine current expert based on SNR
    if scheduler is not None:
        snr = compute_snr(t, scheduler)
    else:
        # Fallback SNR estimation based on timestep
        normalized_t = t.float() / 1000.0
        snr = (1.0 - normalized_t).mean().item()
    
    current_expert = should_switch_expert(snr, cache_dic.get("expert_switch_snr", 0.5))
    previous_expert = current.get("expert_type", "high_noise")
    
    # Handle expert switching
    if current_expert != previous_expert:
        if cache_dic.get("pre_warm_enabled", True) and near_expert_transition(cache_dic, current, snr):
            prepare_expert_switch(cache_dic, previous_expert, current_expert)
        
        # Record expert switch
        cache_dic.setdefault("expert_switch_history", []).append({
            "step": current["step"], 
            "from": previous_expert, 
            "to": current_expert,
            "snr": snr
        })
        
        current["previous_expert"] = previous_expert
        current["switch_pending"] = True
    else:
        current["switch_pending"] = False
    
    current["expert_type"] = current_expert
    
    # Get expert-specific cache and threshold
    expert_cache = cache_dic["cache"][current_expert]
    threshold_key = f"{current_expert}_taylor_threshold"
    threshold = cache_dic.get(threshold_key, 0.18)
    
    # Original Wan 2.2 forward pass logic with MoE-aware modifications
    if self.model_type == 'i2v':
        assert y is not None
    
    # Device handling
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    
    # Embeddings - same as original
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])
    
    # Time embeddings - same as original
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t_flat = t.flatten()
        from wan.modules.model import sinusoidal_embedding_1d
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                  t_flat).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32
    
    # Context - same as original
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))
    
    # Arguments for blocks
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    # MoE-aware block processing with CG-Taylor
    if current["stream"] == "cond_stream":
        # First block with Taylor prediction for conditional stream
        pre_firstblock_hidden_states = firstblock_taylor_formula_moe(
            cache_dic=cache_dic, current=current, expert_type=current_expert)
        
        # Run first block
        x = self.blocks[0](x, **kwargs)
        
        # Compute prediction loss for current expert
        if hasattr(self, 'cnt') and self.cnt > 5:
            predict_loss = (
                pre_firstblock_hidden_states - x
            ).abs().mean() / x.abs().mean()
            can_use_cache = predict_loss < threshold
            
            # Update expert metrics
            update_expert_metrics(cache_dic, current_expert, "cond", can_use_cache)
            
            if not can_use_cache:
                current["block_activated_steps"].append(current["step"])
                firstblock_derivative_approximation_moe(
                    cache_dic=cache_dic, current=current, 
                    feature=x, expert_type=current_expert)
        else:
            # Cold start phase
            current["block_activated_steps"].append(current["step"])
            firstblock_derivative_approximation_moe(
                cache_dic=cache_dic, current=current, 
                feature=x, expert_type=current_expert)
            can_use_cache = False
        
        # Determine if we should calculate or use cache
        if hasattr(self, 'cnt'):
            if self.cnt == 0 or self.cnt == getattr(self, 'num_steps', 50) - 1:
                should_calc = True
            else:
                should_calc = not can_use_cache
        else:
            should_calc = True
    else:
        should_calc = True
    
    # Process remaining blocks
    if should_calc:
        # Full computation path
        if current["stream"] == "cond_stream":
            current["activated_steps"].append(current["step"])
        
        # Process blocks 1 to end
        for i, block in enumerate(self.blocks[1:], 1):
            x = block(x, **kwargs)
        
        # Update derivatives based on stream
        if current["stream"] == "cond_stream":
            step_cond_derivative_approximation_moe(
                cache_dic=cache_dic, current=current, 
                feature=x, expert_type=current_expert)
        else:
            step_uncond_derivative_approximation_moe(
                cache_dic=cache_dic, current=current, 
                feature=x, expert_type=current_expert)
    else:
        # Use Taylor prediction
        distance = current["step"] - current["activated_steps"][-1]
        if current["stream"] == "cond_stream":
            x = taylor_formula_moe(
                derivative_dict=expert_cache["cond_hidden"], 
                distance=distance)
        else:
            x = taylor_formula_moe(
                derivative_dict=expert_cache["uncond_hidden"], 
                distance=distance)
    
    # Head - same as original
    x = self.head(x, e)
    
    # Unpatchify - same as original
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def original_wan22_forward(self, x, t, context, seq_len, y=None):
    """
    Original Wan 2.2 forward pass without CG-Taylor acceleration.
    Used as fallback when caching is disabled.
    """
    return self.forward(x, t, context, seq_len, y)


def inject_cg_taylor_moe(model, cache_dic, current):
    """
    Inject MoE-aware CG-Taylor acceleration into a Wan 2.2 model.
    
    Args:
        model: WanModel instance
        cache_dic: MoE-aware cache dictionary
        current: Current step tracking dictionary
    """
    # Store original forward method
    if not hasattr(model, '_original_forward'):
        model._original_forward = model.forward
    
    # Create bound method with cache_dic and current
    def cg_taylor_forward(x, t, context, seq_len, y=None, scheduler=None):
        return CGTaylor_wan22_forward(
            model, x, t, context, seq_len, y, current, cache_dic, scheduler
        )
    
    # Replace forward method
    model.forward = cg_taylor_forward
    
    # Add CG-Taylor specific attributes
    model.cnt = getattr(model, 'cnt', 0)
    model.num_steps = getattr(model, 'num_steps', 50)
    model.predict_loss = None
    model.threshold = cache_dic.get(f"{current['expert_type']}_taylor_threshold", 0.18)
    model.should_calc = False
    
    return model


def restore_original_forward(model):
    """
    Restore the original forward method of a Wan 2.2 model.
    
    Args:
        model: WanModel instance with injected CG-Taylor
    """
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        delattr(model, '_original_forward')
    
    # Clean up CG-Taylor attributes
    for attr in ['cnt', 'num_steps', 'predict_loss', 'threshold', 'should_calc']:
        if hasattr(model, attr):
            delattr(model, attr)