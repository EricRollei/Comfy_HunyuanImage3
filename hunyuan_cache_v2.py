"""
HunyuanImage-3.0 V2 Cache System

Simple cache for V2 unified node. Stores loaded models and their
associated managers (BlockSwapManager, SimpleVAEManager).

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """A cached model with its associated managers."""
    model: Any
    quant_type: str
    is_moveable: bool
    device: torch.device
    dtype: torch.dtype
    model_path: str
    
    # Associated managers
    block_swap_manager: Optional[Any] = None
    vae_manager: Optional[Any] = None
    
    # State tracking
    is_on_gpu: bool = True
    load_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    
    # Config at load time
    blocks_to_swap: int = 0
    vae_placement: str = "always_gpu"
    
    # BF16-specific: track the VRAM reserve model was loaded with
    # If a new generation needs more reserve, cache must be invalidated
    loaded_with_reserve_gb: float = 0.0
    
    def touch(self):
        """Update last used time and increment use count."""
        self.last_used = time.time()
        self.use_count += 1
    
    def __repr__(self) -> str:
        return (
            f"CachedModel({self.quant_type}, "
            f"{'GPU' if self.is_on_gpu else 'CPU'}, "
            f"uses={self.use_count})"
        )


class ModelCacheV2:
    """
    Simple model cache for V2 unified node.
    
    Features:
    - Single model cache (one model at a time for simplicity)
    - Stores model + managers together
    - Soft unload support (move to CPU, keep in cache)
    - Full unload (remove from cache)
    - Thread-safe operations
    
    Usage:
        cache = ModelCacheV2()
        
        # Check if model is cached
        cached = cache.get(model_path, quant_type)
        if cached:
            return cached.model
        
        # Load and cache
        model = load_model(...)
        cache.put(model_path, quant_type, model, ...)
    """
    
    _instance: Optional['ModelCacheV2'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global cache."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._cache: Dict[str, CachedModel] = {}
        self._cache_lock = threading.Lock()
        self._initialized = True
        
        logger.info("ModelCacheV2 initialized")
    
    def _make_key(self, model_path: str, quant_type: str) -> str:
        """Create cache key from path and quant type."""
        # Normalize path
        path = Path(model_path).resolve()
        return f"{path}::{quant_type}"
    
    def get(
        self,
        model_path: str,
        quant_type: str
    ) -> Optional[CachedModel]:
        """
        Get a cached model if it exists and is valid.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type (bf16, int8, nf4)
            
        Returns:
            CachedModel if found and valid, None otherwise
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached:
                # CRITICAL: Validate the model is actually in memory
                # After a restart, cache entry may exist but model is gone
                try:
                    model = cached.model
                    if model is None:
                        logger.warning(f"Cache entry exists but model is None: {key}")
                        del self._cache[key]
                        return None
                    
                    # Check if model has any parameters and they're on a real device
                    try:
                        first_param = next(model.parameters())
                        device = first_param.device
                        
                        # If on meta device, model didn't load properly
                        if device.type == 'meta':
                            logger.warning(f"Cached model has meta tensors (not loaded): {key}")
                            del self._cache[key]
                            return None
                        
                        # Check if tensor data is accessible (not deallocated)
                        # This will throw if the storage is gone
                        _ = first_param.data_ptr()
                        
                    except (StopIteration, RuntimeError) as e:
                        logger.warning(f"Cached model appears invalid (no valid params): {key}, {e}")
                        del self._cache[key]
                        return None
                    
                    cached.touch()
                    logger.debug(f"Cache hit (validated): {key}")
                    return cached
                    
                except Exception as e:
                    logger.warning(f"Cache validation failed: {key}, {e}")
                    try:
                        del self._cache[key]
                    except:
                        pass
                    return None
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def put(
        self,
        model_path: str,
        quant_type: str,
        model: Any,
        is_moveable: bool,
        device: torch.device,
        dtype: torch.dtype,
        block_swap_manager: Optional[Any] = None,
        vae_manager: Optional[Any] = None,
        load_time: float = 0.0,
        blocks_to_swap: int = 0,
        vae_placement: str = "always_gpu",
        loaded_with_reserve_gb: float = 0.0
    ) -> CachedModel:
        """
        Add a model to the cache.
        
        If a different model is already cached, it will be unloaded first.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type
            model: The loaded model
            is_moveable: Whether model can be moved between devices
            device: Current device
            dtype: Model dtype
            block_swap_manager: Optional BlockSwapManager
            vae_manager: Optional SimpleVAEManager
            load_time: Time taken to load
            blocks_to_swap: Number of blocks configured for swap
            vae_placement: VAE placement strategy
            loaded_with_reserve_gb: VRAM reserve the model was loaded with (for BF16)
            
        Returns:
            The CachedModel entry
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            # Check if same model already cached
            if key in self._cache:
                logger.info(f"Model already cached: {key}")
                cached = self._cache[key]
                cached.touch()
                return cached
            
            # Clear any existing cache entries (single model cache)
            if self._cache:
                self._clear_cache_internal()
            
            # Create new cache entry
            cached = CachedModel(
                model=model,
                quant_type=quant_type,
                is_moveable=is_moveable,
                device=device,
                dtype=dtype,
                model_path=model_path,
                block_swap_manager=block_swap_manager,
                vae_manager=vae_manager,
                is_on_gpu=True,
                load_time=load_time,
                blocks_to_swap=blocks_to_swap,
                vae_placement=vae_placement,
                loaded_with_reserve_gb=loaded_with_reserve_gb
            )
            
            self._cache[key] = cached
            logger.info(f"Cached model: {key}")
            
            return cached
    
    def soft_unload(self, model_path: str, quant_type: str) -> bool:
        """
        Soft unload a model (move to CPU, keep in cache).
        
        Only works for moveable models (NF4, BF16).
        INT8 models cannot be soft unloaded.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type
            
        Returns:
            True if unloaded, False if not possible
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            cached = self._cache.get(key)
            if not cached:
                logger.warning(f"Cannot soft unload: model not in cache: {key}")
                return False
            
            if not cached.is_moveable:
                logger.warning(f"Cannot soft unload: model not moveable (INT8): {key}")
                return False
            
            if not cached.is_on_gpu:
                logger.info(f"Model already on CPU: {key}")
                return True
            
            # Use block swap manager if available
            if cached.block_swap_manager:
                cached.block_swap_manager.move_all_to_cpu()
            else:
                # Direct move
                cached.model.to("cpu")
            
            # Move VAE to CPU
            if cached.vae_manager:
                cached.vae_manager.cleanup_after_decode()
            
            cached.is_on_gpu = False
            torch.cuda.empty_cache()
            
            logger.info(f"Soft unloaded model: {key}")
            return True
    
    def restore(self, model_path: str, quant_type: str, device: str = "cuda:0") -> bool:
        """
        Restore a soft-unloaded model to GPU.
        
        Args:
            model_path: Path to model
            quant_type: Quantization type
            device: Target device
            
        Returns:
            True if restored, False if not possible
        """
        key = self._make_key(model_path, quant_type)
        
        with self._cache_lock:
            cached = self._cache.get(key)
            if not cached:
                logger.warning(f"Cannot restore: model not in cache: {key}")
                return False
            
            if cached.is_on_gpu:
                logger.info(f"Model already on GPU: {key}")
                return True
            
            # Use block swap manager if available
            if cached.block_swap_manager:
                cached.block_swap_manager.move_all_to_gpu()
            else:
                # Direct move
                cached.model.to(device)
            
            cached.is_on_gpu = True
            cached.device = torch.device(device)
            
            logger.info(f"Restored model to {device}: {key}")
            return True
    
    def full_unload(self, model_path: str = None, quant_type: str = None) -> bool:
        """
        Fully unload and remove a model from cache.
        
        If no arguments provided, clears entire cache.
        
        Args:
            model_path: Path to model (optional)
            quant_type: Quantization type (optional)
            
        Returns:
            True if unloaded
        """
        with self._cache_lock:
            if model_path and quant_type:
                key = self._make_key(model_path, quant_type)
                if key in self._cache:
                    self._unload_entry(key)
                    return True
                return False
            else:
                self._clear_cache_internal()
                return True
    
    def _unload_entry(self, key: str):
        """Unload a single cache entry (internal, must hold lock)."""
        if key not in self._cache:
            return
        
        cached = self._cache[key]
        
        # Clean up managers
        if cached.block_swap_manager:
            del cached.block_swap_manager
        if cached.vae_manager:
            del cached.vae_manager
        
        # Delete model
        del cached.model
        del self._cache[key]
        
        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"Fully unloaded: {key}")
    
    def _clear_cache_internal(self):
        """Clear entire cache (internal, must hold lock)."""
        for key in list(self._cache.keys()):
            self._unload_entry(key)
        self._cache.clear()
    
    def get_status(self) -> Dict:
        """Get cache status information."""
        with self._cache_lock:
            if not self._cache:
                return {"cached": False}
            
            # Get first (only) entry
            key, cached = next(iter(self._cache.items()))
            
            return {
                "cached": True,
                "model_path": cached.model_path,
                "quant_type": cached.quant_type,
                "is_on_gpu": cached.is_on_gpu,
                "is_moveable": cached.is_moveable,
                "use_count": cached.use_count,
                "blocks_to_swap": cached.blocks_to_swap,
                "vae_placement": cached.vae_placement,
                "load_time": cached.load_time,
            }
    
    def clear(self):
        """Clear entire cache."""
        self.full_unload()


# Global cache instance
_cache: Optional[ModelCacheV2] = None


def get_cache() -> ModelCacheV2:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = ModelCacheV2()
    return _cache
