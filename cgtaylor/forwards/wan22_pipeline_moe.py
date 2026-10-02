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

import logging
import torch
from tqdm import tqdm
from typing import Union, Tuple

from ..cache_functions.cache_step_init_moe import cache_step_init_moe
from .cg_taylor_wan22_forward import inject_cg_taylor_moe, restore_original_forward


def enable_cg_taylor_moe_acceleration(wan_t2v_instance, 
                                      model_variant="A14B",
                                      high_noise_threshold=0.30,
                                      low_noise_threshold=0.13,
                                      expert_switch_snr=0.5):
    """
    Enable MoE-aware CG-Taylor acceleration for a Wan T2V instance.
    
    Args:
        wan_t2v_instance: WanT2V instance
        model_variant: Model variant ("A14B", "TI2V-5B", etc.)
        high_noise_threshold: Taylor threshold for high noise expert
        low_noise_threshold: Taylor threshold for low noise expert
        expert_switch_snr: SNR threshold for expert switching
    """
    # Store original generate method if not already stored
    if not hasattr(wan_t2v_instance, '_original_generate'):
        wan_t2v_instance._original_generate = wan_t2v_instance.generate
    
    # Create bound method for CG-Taylor accelerated generation
    def cg_taylor_generate(
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        # CG-Taylor specific parameters
        enable_cg_taylor=True,
        cg_taylor_threshold_override=None
    ):
        """
        CG-Taylor accelerated generation method.
        """
        if not enable_cg_taylor:
            return wan_t2v_instance._original_generate(
                input_prompt, size, frame_num, shift, sample_solver,
                sampling_steps, guide_scale, n_prompt, seed, offload_model
            )
        
        return cg_taylor_moe_generate(
            wan_t2v_instance, input_prompt, size, frame_num, shift,
            sample_solver, sampling_steps, guide_scale, n_prompt, seed,
            offload_model, model_variant, high_noise_threshold,
            low_noise_threshold, expert_switch_snr, cg_taylor_threshold_override
        )
    
    # Replace generate method
    wan_t2v_instance.generate = cg_taylor_generate
    
    logging.info(f"CG-Taylor MoE acceleration enabled for {model_variant}")
    logging.info(f"High noise threshold: {high_noise_threshold}, Low noise threshold: {low_noise_threshold}")
    logging.info(f"Expert switch SNR: {expert_switch_snr}")


def disable_cg_taylor_moe_acceleration(wan_t2v_instance):
    """
    Disable CG-Taylor acceleration and restore original methods.
    
    Args:
        wan_t2v_instance: WanT2V instance
    """
    # Restore original generate method
    if hasattr(wan_t2v_instance, '_original_generate'):
        wan_t2v_instance.generate = wan_t2v_instance._original_generate
        delattr(wan_t2v_instance, '_original_generate')
    
    # Restore original forward methods for both models
    restore_original_forward(wan_t2v_instance.high_noise_model)
    restore_original_forward(wan_t2v_instance.low_noise_model)
    
    logging.info("CG-Taylor MoE acceleration disabled")


def cg_taylor_moe_generate(
    wan_t2v_instance,
    input_prompt: str,
    size: Tuple[int, int] = (1280, 720),
    frame_num: int = 81,
    shift: float = 5.0,
    sample_solver: str = 'unipc',
    sampling_steps: int = 50,
    guide_scale: Union[float, Tuple[float, float]] = 5.0,
    n_prompt: str = "",
    seed: int = -1,
    offload_model: bool = True,
    model_variant: str = "A14B",
    high_noise_threshold: float = 0.30,
    low_noise_threshold: float = 0.13,
    expert_switch_snr: float = 0.5,
    cg_taylor_threshold_override = None
):
    """
    MoE-aware CG-Taylor accelerated video generation.
    
    This function integrates CG-Taylor acceleration with Wan 2.2's existing
    MoE architecture (high_noise_model and low_noise_model).
    """
    
    # Initialize MoE-aware cache
    cache_dic, current = cache_step_init_moe(sampling_steps, model_variant)
    
    # Override thresholds if provided
    if cg_taylor_threshold_override is not None:
        if isinstance(cg_taylor_threshold_override, (list, tuple)) and len(cg_taylor_threshold_override) == 2:
            high_noise_threshold, low_noise_threshold = cg_taylor_threshold_override
        else:
            high_noise_threshold = low_noise_threshold = cg_taylor_threshold_override
    
    # Update cache with thresholds
    cache_dic["high_noise_taylor_threshold"] = high_noise_threshold
    cache_dic["low_noise_taylor_threshold"] = low_noise_threshold
    cache_dic["expert_switch_snr"] = expert_switch_snr
    
    # Inject CG-Taylor into both models
    inject_cg_taylor_moe(wan_t2v_instance.high_noise_model, cache_dic, current)
    inject_cg_taylor_moe(wan_t2v_instance.low_noise_model, cache_dic, current)
    
    # Configure model-specific parameters
    wan_t2v_instance.high_noise_model.num_steps = sampling_steps
    wan_t2v_instance.high_noise_model.threshold = high_noise_threshold
    wan_t2v_instance.low_noise_model.num_steps = sampling_steps
    wan_t2v_instance.low_noise_model.threshold = low_noise_threshold
    
    try:
        # Run generation with CG-Taylor acceleration
        result = _run_cg_taylor_generation(
            wan_t2v_instance, input_prompt, size, frame_num, shift,
            sample_solver, sampling_steps, guide_scale, n_prompt, seed,
            offload_model, cache_dic, current
        )
        
        # Log acceleration statistics
        _log_acceleration_stats(cache_dic, current, sampling_steps)
        
        return result
        
    finally:
        # Always restore original forward methods
        restore_original_forward(wan_t2v_instance.high_noise_model)
        restore_original_forward(wan_t2v_instance.low_noise_model)


def _run_cg_taylor_generation(
    wan_t2v_instance, input_prompt, size, frame_num, shift, sample_solver,
    sampling_steps, guide_scale, n_prompt, seed, offload_model, cache_dic, current
):
    """
    Run the actual generation with CG-Taylor acceleration.
    
    This closely follows the original Wan T2V generation logic but with
    CG-Taylor acceleration integrated into the denoising loop.
    """
    import gc
    import random
    import sys
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
    # Preprocessing (same as original)
    if seed == -1:
        seed = random.randint(0, sys.maxsize)
    torch.manual_seed(seed)
    
    if n_prompt == "":
        n_prompt = wan_t2v_instance.sample_neg_prompt
    
    if isinstance(guide_scale, (tuple, list)):
        guide_scale_low, guide_scale_high = guide_scale
    else:
        guide_scale_low = guide_scale_high = guide_scale
    
    w, h = size
    f = frame_num
    
    # Setup scheduler
    if sample_solver == 'dpm':
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=wan_t2v_instance.num_train_timesteps,
            shift=shift,
            solver_order=2
        )
        timesteps = scheduler.timesteps
    else:
        sigmas = get_sampling_sigmas(
            num_train_timesteps=wan_t2v_instance.num_train_timesteps,
            num_inference_steps=sampling_steps,
            shift=shift
        )
        timesteps, _ = retrieve_timesteps(scheduler=None, sigmas=sigmas, num_inference_steps=sampling_steps)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=wan_t2v_instance.num_train_timesteps,
            sigmas=sigmas,
            solver_order=2
        )
        scheduler.timesteps = timesteps
    
    timesteps = timesteps.long()
    
    # Initialize latents
    if wan_t2v_instance.sp_size == 1:
        sp_size = 1
    else:
        sp_size = wan_t2v_instance.sp_size
    
    assert f % sp_size == 0, f"f ({f}) should be divisible by sp_size ({sp_size})"
    f_sp = f // sp_size
    
    init_noise = torch.randn(16, f_sp, h // 8, w // 8, device=wan_t2v_instance.device)
    latents = [init_noise.clone()]
    
    # Text encoding
    current_device = next(wan_t2v_instance.text_encoder.parameters()).device
    if wan_t2v_instance.t5_cpu:
        if current_device.type == 'cpu':
            wan_t2v_instance.text_encoder.to(wan_t2v_instance.device)
        cond = wan_t2v_instance.text_encoder.encode([input_prompt])
        uncond = wan_t2v_instance.text_encoder.encode([n_prompt])
        if wan_t2v_instance.t5_cpu:
            wan_t2v_instance.text_encoder.to('cpu')
            torch.cuda.empty_cache()
    else:
        cond = wan_t2v_instance.text_encoder.encode([input_prompt])
        uncond = wan_t2v_instance.text_encoder.encode([n_prompt])
    
    # Denoising loop with CG-Taylor acceleration
    desc = f"CG-Taylor MoE T2V generation ({sampling_steps} steps)"
    with tqdm(total=sampling_steps, desc=desc, disable=(wan_t2v_instance.rank != 0)) as pbar:
        
        for i, t in enumerate(timesteps):
            current["step"] = i
            
            # Determine which model to use (existing Wan 2.2 logic)
            model = wan_t2v_instance._prepare_model_for_timestep(t, wan_t2v_instance.boundary, offload_model)
            
            # Set current expert type based on model
            if model is wan_t2v_instance.high_noise_model:
                current["expert_type"] = "high_noise"
                current_guide_scale = guide_scale_high
            else:
                current["expert_type"] = "low_noise"  
                current_guide_scale = guide_scale_low
            
            # Update model counter for CG-Taylor logic
            model.cnt = i
            
            # Conditional forward pass
            current["stream"] = "cond_stream"
            with amp.autocast(dtype=wan_t2v_instance.param_dtype):
                cond_pred = model.forward(
                    latents, t.unsqueeze(0), cond, wan_t2v_instance.config.seq_len,
                    scheduler=scheduler
                )[0]
            
            # Unconditional forward pass  
            current["stream"] = "uncond_stream"
            with amp.autocast(dtype=wan_t2v_instance.param_dtype):
                uncond_pred = model.forward(
                    latents, t.unsqueeze(0), uncond, wan_t2v_instance.config.seq_len,
                    scheduler=scheduler
                )[0]
            
            # Classifier-free guidance
            pred = uncond_pred + current_guide_scale * (cond_pred - uncond_pred)
            
            # Scheduler step
            latents = scheduler.step(pred, t, latents[0], return_dict=False)[0:1]
            
            pbar.update(1)
            
            # Optional memory cleanup
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    # VAE decode
    if offload_model or wan_t2v_instance.init_on_cpu:
        wan_t2v_instance.high_noise_model.to('cpu')
        wan_t2v_instance.low_noise_model.to('cpu')
    
    vae_output = wan_t2v_instance.vae.decode(latents[0])[0]
    vae_output = vae_output.float()
    
    return vae_output


def _log_acceleration_stats(cache_dic, current, total_steps):
    """
    Log CG-Taylor acceleration statistics.
    """
    # Calculate cache hit rates
    high_noise_hits = cache_dic["cache_hit_rates"]["high_noise"]["cond"] + \
                     cache_dic["cache_hit_rates"]["high_noise"]["uncond"]
    low_noise_hits = cache_dic["cache_hit_rates"]["low_noise"]["cond"] + \
                    cache_dic["cache_hit_rates"]["low_noise"]["uncond"]
    
    # Count expert switches
    num_switches = len(cache_dic.get("expert_switch_history", []))
    
    # Calculate acceleration ratio (estimated)
    activated_steps = len(current.get("activated_steps", [0]))
    if activated_steps > 0:
        acceleration_ratio = total_steps / activated_steps
    else:
        acceleration_ratio = 1.0
    
    logging.info("=== CG-Taylor MoE Acceleration Statistics ===")
    logging.info(f"Total denoising steps: {total_steps}")
    logging.info(f"Activated steps (full computation): {activated_steps}")
    logging.info(f"Estimated acceleration ratio: {acceleration_ratio:.2f}x")
    logging.info(f"Expert switches: {num_switches}")
    logging.info(f"High-noise expert cache hit rate: {high_noise_hits/2:.2%}")
    logging.info(f"Low-noise expert cache hit rate: {low_noise_hits/2:.2%}")
    
    if cache_dic.get("expert_switch_history"):
        logging.info("Expert switch timeline:")
        for switch in cache_dic["expert_switch_history"]:
            logging.info(f"  Step {switch['step']}: {switch['from']} -> {switch['to']} (SNR: {switch['snr']:.3f})")


# Utility functions for easy integration

def create_cg_taylor_wan_t2v(config, checkpoint_dir, **kwargs):
    """
    Create a Wan T2V instance with CG-Taylor acceleration pre-enabled.
    
    Args:
        config: Wan T2V configuration
        checkpoint_dir: Path to model checkpoints
        **kwargs: Additional arguments for WanT2V initialization and CG-Taylor configuration
    
    Returns:
        WanT2V instance with CG-Taylor acceleration enabled
    """
    from wan.text2video import WanT2V
    
    # Extract CG-Taylor specific kwargs
    cg_taylor_kwargs = {
        'model_variant': kwargs.pop('model_variant', 'A14B'),
        'high_noise_threshold': kwargs.pop('high_noise_threshold', 0.30),
        'low_noise_threshold': kwargs.pop('low_noise_threshold', 0.13),
        'expert_switch_snr': kwargs.pop('expert_switch_snr', 0.5),
    }
    
    # Create WanT2V instance
    wan_t2v = WanT2V(config, checkpoint_dir, **kwargs)
    
    # Enable CG-Taylor acceleration
    enable_cg_taylor_moe_acceleration(wan_t2v, **cg_taylor_kwargs)
    
    return wan_t2v