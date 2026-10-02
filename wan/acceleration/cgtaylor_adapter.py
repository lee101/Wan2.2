# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
CG-Taylor Acceleration Adapter for Wan 2.2 Video Generation
Adapts the training-free CG-Taylor method to work with Wan 2.2's MoE architecture
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from collections import deque


@dataclass
class TaylorConfig:
    """Configuration for Taylor expansion predictor"""
    order: int = 2  # Taylor expansion order
    confidence_threshold: float = 0.15  # Confidence threshold for using predictions
    cache_window: int = 5  # Number of past timesteps to cache
    first_block_weight: float = 0.8  # Weight for first block error in confidence
    adaptive_threshold: bool = True  # Enable adaptive confidence thresholding
    memory_efficient: bool = True  # Enable memory-efficient caching


class TaylorCache:
    """Expert-specific Taylor expansion cache with memory management"""
    
    def __init__(self, config: TaylorConfig):
        self.config = config
        self.feature_cache = deque(maxlen=config.cache_window)
        self.timestep_cache = deque(maxlen=config.cache_window)
        self.confidence_history = deque(maxlen=20)
        self.prediction_errors = deque(maxlen=10)
        
        # Adaptive threshold tracking
        self.successful_predictions = 0
        self.total_attempts = 0
        
    def add_features(self, timestep: int, features: torch.Tensor, prediction_error: float = None):
        """Add features to cache and update confidence metrics"""
        if self.config.memory_efficient:
            # Store on CPU to save GPU memory
            features = features.detach().cpu()
        else:
            features = features.detach().clone()
            
        self.feature_cache.append(features)
        self.timestep_cache.append(timestep)
        
        if prediction_error is not None:
            self.prediction_errors.append(prediction_error)
            
    def update_confidence(self, prediction_error: float, prediction_successful: bool):
        """Update confidence metrics based on prediction results"""
        self.confidence_history.append(prediction_error)
        self.total_attempts += 1
        if prediction_successful:
            self.successful_predictions += 1
            
    def get_adaptive_threshold(self) -> float:
        """Calculate adaptive confidence threshold"""
        if not self.config.adaptive_threshold or len(self.confidence_history) < 3:
            return self.config.confidence_threshold
            
        # Use recent error statistics to adapt threshold
        recent_errors = list(self.confidence_history)[-10:]
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        # Adaptive threshold based on error distribution
        adaptive_threshold = min(
            self.config.confidence_threshold * 2.0,
            max(self.config.confidence_threshold * 0.5, mean_error + 0.5 * std_error)
        )
        
        return adaptive_threshold
        
    def can_predict(self, timestep: int) -> bool:
        """Check if we have enough history to make predictions"""
        if len(self.feature_cache) < max(3, self.config.order + 1):
            return False
            
        # Check confidence based on recent prediction errors
        threshold = self.get_adaptive_threshold()
        if self.confidence_history:
            recent_error = np.mean(list(self.confidence_history)[-5:])
            return recent_error < threshold
            
        return True  # Allow first prediction attempt
        
    def taylor_predict(self, target_timestep: int) -> Optional[torch.Tensor]:
        """Predict features using Taylor expansion with numerical stability"""
        if not self.can_predict(target_timestep):
            return None
            
        try:
            # Get cached features and timesteps
            if self.config.memory_efficient:
                # Move back to GPU for computation
                features = [f.cuda() if f.device.type == 'cpu' else f for f in self.feature_cache]
            else:
                features = list(self.feature_cache)
                
            features = torch.stack(features)
            timesteps = torch.tensor(list(self.timestep_cache), dtype=torch.float32, device=features.device)
            
            # Ensure we have enough points for the requested order
            n_points = min(len(features), self.config.order + 1)
            features = features[-n_points:]
            timesteps = timesteps[-n_points:]
            
            # Compute derivatives using finite differences with numerical stability
            predicted = features[-1].clone()  # Start with the latest features
            
            if n_points >= 2:
                # First derivative
                dt = timesteps[-1] - timesteps[-2]
                if abs(dt) > 1e-8:  # Numerical stability check
                    first_derivative = (features[-1] - features[-2]) / dt
                    delta_t = target_timestep - timesteps[-1]
                    predicted += first_derivative * delta_t
                
            if n_points >= 3 and self.config.order >= 2:
                # Second derivative
                dt1 = timesteps[-1] - timesteps[-2]
                dt2 = timesteps[-2] - timesteps[-3]
                if abs(dt1) > 1e-8 and abs(dt2) > 1e-8:
                    # Use central difference for better numerical stability
                    second_derivative = 2 * (features[-1] / (dt1 * (dt1 + dt2)) -
                                           features[-2] / (dt1 * dt2) +
                                           features[-3] / (dt2 * (dt1 + dt2)))
                    delta_t = target_timestep - timesteps[-1]
                    predicted += 0.5 * second_derivative * (delta_t ** 2)
                    
            # Higher order terms if requested and available
            if n_points >= 4 and self.config.order >= 3:
                # Third derivative (simplified)
                dt_avg = (timesteps[-1] - timesteps[-4]) / 3
                if abs(dt_avg) > 1e-8:
                    third_derivative = (features[-1] - 3*features[-2] + 3*features[-3] - features[-4]) / (dt_avg ** 3)
                    delta_t = target_timestep - timesteps[-1]
                    predicted += (1/6) * third_derivative * (delta_t ** 3)
                    
            return predicted
            
        except Exception as e:
            logging.warning(f"Taylor prediction failed: {e}")
            return None


class WanCGTaylorAdapter:
    """CG-Taylor acceleration adapter for Wan 2.2 MoE architecture"""
    
    def __init__(self, config, boundary: float = 0.875):
        """
        Initialize the CG-Taylor adapter
        
        Args:
            config: Wan 2.2 configuration object
            boundary: SNR boundary for expert switching (0.875 matches Wan config)
        """
        self.config = config
        self.boundary = boundary
        self.num_train_timesteps = getattr(config, 'num_train_timesteps', 1000)
        
        # Expert-specific configurations optimized for Wan 2.2
        self.high_noise_config = TaylorConfig(
            order=2,
            confidence_threshold=0.25,  # More conservative for high noise
            cache_window=4,
            first_block_weight=0.7,
            adaptive_threshold=True,
            memory_efficient=True
        )
        
        self.low_noise_config = TaylorConfig(
            order=3,  # Higher order for fine details
            confidence_threshold=0.12,  # Strict threshold for quality
            cache_window=3,
            first_block_weight=0.9,
            adaptive_threshold=True,
            memory_efficient=True
        )
        
        # Initialize caches
        self.high_noise_cache = TaylorCache(self.high_noise_config)
        self.low_noise_cache = TaylorCache(self.low_noise_config)
        
        # Performance tracking
        self.reset_stats()
        
        # Current active cache
        self.current_cache = None
        self.current_config = None
        
    def reset_stats(self):
        """Reset performance statistics"""
        self.cache_hits = 0
        self.total_calls = 0
        self.expert_switches = 0
        self.last_expert = None
        self.total_speedup_time = 0.0
        
    def select_expert_cache(self, timestep: Union[int, torch.Tensor]) -> Tuple[str, TaylorCache, TaylorConfig]:
        """Select appropriate cache and config based on timestep"""
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.item()
        else:
            t_val = timestep
            
        # Use the same boundary logic as Wan 2.2
        boundary_timestep = self.boundary * self.num_train_timesteps
        
        if t_val >= boundary_timestep:
            expert_name = "high_noise"
            cache = self.high_noise_cache
            config = self.high_noise_config
        else:
            expert_name = "low_noise"
            cache = self.low_noise_cache
            config = self.low_noise_config
            
        # Track expert switches
        if self.last_expert is not None and self.last_expert != expert_name:
            self.expert_switches += 1
            logging.debug(f"Expert switch at t={t_val}: {self.last_expert} -> {expert_name}")
            
        self.last_expert = expert_name
        self.current_cache = cache
        self.current_config = config
        
        return expert_name, cache, config
        
    def can_accelerate(self, timestep: Union[int, torch.Tensor]) -> bool:
        """Check if acceleration can be applied for current timestep"""
        expert_name, cache, config = self.select_expert_cache(timestep)
        return cache.can_predict(timestep)
        
    def predict_features(self, timestep: Union[int, torch.Tensor]) -> Optional[torch.Tensor]:
        """Predict features using Taylor expansion"""
        self.total_calls += 1
        expert_name, cache, config = self.select_expert_cache(timestep)
        
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.item()
        else:
            t_val = timestep
            
        predicted = cache.taylor_predict(t_val)
        
        if predicted is not None:
            self.cache_hits += 1
            logging.debug(f"[t={t_val}] Using Taylor prediction from {expert_name} expert")
            
        return predicted
        
    def update_cache(self, timestep: Union[int, torch.Tensor], features: torch.Tensor, 
                    prediction_error: Optional[float] = None, prediction_successful: bool = False):
        """Update cache with new features and performance metrics"""
        expert_name, cache, config = self.select_expert_cache(timestep)
        
        if isinstance(timestep, torch.Tensor):
            t_val = timestep.item()
        else:
            t_val = timestep
            
        cache.add_features(t_val, features, prediction_error)
        
        if prediction_error is not None:
            cache.update_confidence(prediction_error, prediction_successful)
            
    def validate_prediction(self, predicted: torch.Tensor, actual: torch.Tensor) -> Tuple[float, bool]:
        """Validate Taylor prediction against actual computation"""
        if predicted.shape != actual.shape:
            # Handle shape mismatch by comparing compatible portions
            min_shape = tuple(min(p, a) for p, a in zip(predicted.shape, actual.shape))
            predicted_slice = predicted[:min_shape[0], :min_shape[1]] if len(min_shape) >= 2 else predicted[:min_shape[0]]
            actual_slice = actual[:min_shape[0], :min_shape[1]] if len(min_shape) >= 2 else actual[:min_shape[0]]
        else:
            predicted_slice = predicted
            actual_slice = actual
            
        error = torch.mean(torch.abs(predicted_slice - actual_slice)).item()
        threshold = self.current_config.confidence_threshold if self.current_config else 0.15
        is_successful = error < threshold
        
        return error, is_successful
        
    def get_acceleration_stats(self) -> Dict:
        """Get detailed acceleration statistics"""
        if self.total_calls == 0:
            return {"error": "No calls made yet"}
            
        cache_hit_rate = self.cache_hits / self.total_calls
        # Conservative speedup estimate - Taylor prediction saves ~60-75% computation
        estimated_speedup = 1.0 / (1.0 - cache_hit_rate * 0.7)
        
        stats = {
            "cache_hit_rate": f"{cache_hit_rate:.3f}",
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "expert_switches": self.expert_switches,
            "estimated_speedup": f"{estimated_speedup:.2f}x",
            "high_noise_stats": {
                "avg_confidence": np.mean(list(self.high_noise_cache.confidence_history)) if self.high_noise_cache.confidence_history else None,
                "success_rate": self.high_noise_cache.successful_predictions / max(1, self.high_noise_cache.total_attempts),
                "cache_size": len(self.high_noise_cache.feature_cache)
            },
            "low_noise_stats": {
                "avg_confidence": np.mean(list(self.low_noise_cache.confidence_history)) if self.low_noise_cache.confidence_history else None,
                "success_rate": self.low_noise_cache.successful_predictions / max(1, self.low_noise_cache.total_attempts),
                "cache_size": len(self.low_noise_cache.feature_cache)
            }
        }
        
        return stats
        
    def optimize_for_resolution(self, size: Tuple[int, int], frame_num: int, batch_size: int = 1):
        """Optimize cache settings for specific video parameters"""
        # Calculate expected feature map sizes
        vae_stride = getattr(self.config, 'vae_stride', (4, 8, 8))
        patch_size = getattr(self.config, 'patch_size', (1, 2, 2))
        
        latent_size = (
            (frame_num - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2]
        )
        
        # Estimate memory requirements
        hidden_dim = getattr(self.config, 'dim', 5120)
        feature_memory_mb = batch_size * hidden_dim * np.prod(latent_size) * 4 / (1024**2)
        
        # Adjust cache windows based on memory constraints
        max_memory_mb = 2048  # 2GB cache limit
        max_cache_entries = max(2, int(max_memory_mb / feature_memory_mb))
        
        self.high_noise_config.cache_window = min(4, max_cache_entries)
        self.low_noise_config.cache_window = min(3, max_cache_entries)
        
        # Adjust confidence thresholds for different resolutions
        if size[0] * size[1] > 1280 * 720:  # High resolution
            self.high_noise_config.confidence_threshold *= 1.2
            self.low_noise_config.confidence_threshold *= 1.1
        elif size[0] * size[1] < 640 * 360:  # Low resolution
            self.high_noise_config.confidence_threshold *= 0.8
            self.low_noise_config.confidence_threshold *= 0.9
            
        logging.info(f"Optimized for {size}x{frame_num} (est. {feature_memory_mb:.1f}MB per cache entry)")
        logging.info(f"Cache windows: high_noise={self.high_noise_config.cache_window}, low_noise={self.low_noise_config.cache_window}")
        
    def clear_caches(self):
        """Clear all caches to free memory"""
        self.high_noise_cache.feature_cache.clear()
        self.high_noise_cache.timestep_cache.clear()
        self.low_noise_cache.feature_cache.clear()
        self.low_noise_cache.timestep_cache.clear()
        torch.cuda.empty_cache()