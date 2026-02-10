"""
HunyuanImage-3.0 Simple VAE Manager - V2 Unified Node Support

Simple, hook-free VAE management using explicit .to() calls.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
import time
from enum import Enum
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class VAEPlacement(Enum):
    """VAE placement strategy."""
    ALWAYS_GPU = "always_gpu"  # VAE stays on GPU at all times
    MANAGED = "managed"        # Move to GPU for decode, back to CPU after


class SimpleVAEManager:
    """
    Simple VAE management without hooks.
    
    This manager uses explicit .to() calls instead of hooks to control
    VAE placement. This avoids conflicts with other ComfyUI extensions
    and provides predictable behavior.
    
    Usage:
        manager = SimpleVAEManager(model, VAEPlacement.MANAGED)
        manager.setup_initial_placement()
        
        # Before decode:
        manager.prepare_for_decode()
        images = vae.decode(latents)
        manager.cleanup_after_decode()
    """
    
    def __init__(
        self, 
        model: Any, 
        placement: VAEPlacement,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize VAE manager.
        
        Args:
            model: The Hunyuan model containing .vae attribute
            placement: VAE placement strategy
            device: Target GPU device
            dtype: Target dtype for VAE (default bfloat16)
        """
        self.model = model
        self.placement = placement
        self.device = torch.device(device)
        self.dtype = dtype
        self._vae_on_gpu = False
        self._original_device = None
        
    @property
    def vae(self):
        """Access the VAE from the model."""
        if not hasattr(self.model, 'vae'):
            raise AttributeError("Model does not have a 'vae' attribute")
        return self.model.vae
    
    @property
    def is_on_gpu(self) -> bool:
        """Check if VAE is currently on GPU."""
        return self._vae_on_gpu
    
    def setup_initial_placement(self) -> None:
        """
        Set up initial VAE placement based on configuration.
        
        Call this after model loading to establish VAE position.
        """
        if not hasattr(self.model, 'vae'):
            logger.warning("Model has no VAE attribute, skipping VAE setup")
            return
            
        if self.placement == VAEPlacement.ALWAYS_GPU:
            self._move_vae_to_gpu()
            logger.info(f"VAE placement: always_gpu ({self.device})")
        else:
            self._move_vae_to_cpu()
            logger.info("VAE placement: managed (parked on CPU)")
    
    def prepare_for_decode(self) -> float:
        """
        Prepare VAE for decoding by moving to GPU if needed.
        
        Returns:
            Time taken to move VAE (0 if already on GPU)
        """
        if self._vae_on_gpu:
            logger.debug("VAE already on GPU, no movement needed")
            return 0.0
        
        start_time = time.time()
        self._move_vae_to_gpu()
        elapsed = time.time() - start_time
        
        logger.info(f"VAE moved to GPU in {elapsed:.2f}s for decode")
        return elapsed
    
    def cleanup_after_decode(self) -> float:
        """
        Clean up VAE after decoding.
        
        For MANAGED placement, moves VAE back to CPU.
        For ALWAYS_GPU, does nothing.
        
        Returns:
            Time taken to move VAE (0 if not moved)
        """
        if self.placement == VAEPlacement.ALWAYS_GPU:
            logger.debug("VAE placement is always_gpu, keeping on GPU")
            return 0.0
        
        if not self._vae_on_gpu:
            logger.debug("VAE already on CPU, no movement needed")
            return 0.0
        
        start_time = time.time()
        self._move_vae_to_cpu()
        elapsed = time.time() - start_time
        
        # Clear CUDA cache after moving VAE off GPU
        torch.cuda.empty_cache()
        
        logger.info(f"VAE moved back to CPU in {elapsed:.2f}s")
        return elapsed
    
    def _move_vae_to_gpu(self) -> None:
        """Move VAE to GPU."""
        if not hasattr(self.model, 'vae'):
            return
            
        vae = self.model.vae
        
        # Move entire VAE module
        vae.to(device=self.device, dtype=self.dtype)
        
        # Ensure all parameters are moved (some may be lazy)
        for param in vae.parameters():
            if param.device != self.device:
                param.data = param.data.to(device=self.device, dtype=self.dtype)
        
        # Ensure buffers are moved too
        for buffer in vae.buffers():
            if buffer.device != self.device and buffer.device.type != "meta":
                if torch.is_floating_point(buffer):
                    buffer.data = buffer.data.to(device=self.device, dtype=self.dtype)
                else:
                    buffer.data = buffer.data.to(device=self.device)
        
        # Sync to ensure transfer is complete
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        self._vae_on_gpu = True
    
    def _move_vae_to_cpu(self) -> None:
        """Move VAE to CPU."""
        if not hasattr(self.model, 'vae'):
            return
            
        vae = self.model.vae
        cpu_device = torch.device("cpu")
        
        # Move entire VAE module
        vae.to(device=cpu_device)
        
        # Ensure all parameters are moved
        for param in vae.parameters():
            if param.device != cpu_device:
                param.data = param.data.to(device=cpu_device)
        
        # Ensure buffers are moved too
        for buffer in vae.buffers():
            if buffer.device != cpu_device and buffer.device.type != "meta":
                buffer.data = buffer.data.to(device=cpu_device)
        
        self._vae_on_gpu = False
    
    def get_vae_memory_gb(self) -> float:
        """Get estimated VAE memory usage in GB."""
        if not hasattr(self.model, 'vae'):
            return 0.0
        
        total_bytes = 0
        vae = self.model.vae
        
        for param in vae.parameters():
            total_bytes += param.numel() * param.element_size()
        
        for buffer in vae.buffers():
            if buffer.device.type != "meta":
                total_bytes += buffer.numel() * buffer.element_size()
        
        return total_bytes / (1024**3)
    
    def get_status(self) -> dict:
        """Get VAE manager status."""
        return {
            "placement": self.placement.value,
            "is_on_gpu": self._vae_on_gpu,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "memory_gb": self.get_vae_memory_gb(),
        }
    
    def __repr__(self) -> str:
        status = "GPU" if self._vae_on_gpu else "CPU"
        return f"SimpleVAEManager(placement={self.placement.value}, status={status})"


def enable_vae_tiling(model: Any, tile_size: int = 256) -> bool:
    """
    Enable tiled VAE decoding for memory efficiency.
    
    This is separate from VAE placement - tiling happens during decode
    regardless of where VAE is located.
    
    Args:
        model: Model with VAE
        tile_size: Tile size for spatial tiling
        
    Returns:
        True if tiling was enabled
    """
    if not hasattr(model, 'vae'):
        logger.warning("Model has no VAE, cannot enable tiling")
        return False
    
    vae = model.vae
    
    # Check for enable_tiling method (diffusers-style VAE)
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
        logger.info(f"VAE tiling enabled via enable_tiling()")
        return True
    
    # Check for tile_latent_input attribute
    if hasattr(vae, 'tile_latent_input'):
        vae.tile_latent_input = True
        if hasattr(vae, 'tile_sample_min_size'):
            vae.tile_sample_min_size = tile_size
        logger.info(f"VAE tiling enabled via tile_latent_input (size={tile_size})")
        return True
    
    logger.debug("VAE does not support tiling")
    return False


def disable_vae_tiling(model: Any) -> bool:
    """
    Disable tiled VAE decoding.
    
    Args:
        model: Model with VAE
        
    Returns:
        True if tiling was disabled
    """
    if not hasattr(model, 'vae'):
        return False
    
    vae = model.vae
    
    if hasattr(vae, 'disable_tiling'):
        vae.disable_tiling()
        logger.info("VAE tiling disabled via disable_tiling()")
        return True
    
    if hasattr(vae, 'tile_latent_input'):
        vae.tile_latent_input = False
        logger.info("VAE tiling disabled via tile_latent_input")
        return True
    
    return False
