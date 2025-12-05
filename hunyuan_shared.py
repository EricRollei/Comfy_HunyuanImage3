"""
HunyuanImage-3.0 ComfyUI Custom Nodes - Shared Utilities
Shared utilities, VRAM cache management, and dtype compatibility patches

Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at [eric@historic.camera, eric@rollei.us] for licensing options.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT.

Note: HunyuanImage-3.0 model is subject to Tencent's Apache 2.0 license.
"""

import logging
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)

_SDPA_PATCHED = False
_ORIGINAL_SDPA = None
_GELU_PATCHED = False
_ORIGINAL_GELU = None
_SILU_PATCHED = False
_ORIGINAL_SILU = None
_SCATTER_PATCHED = False
_ORIGINAL_SCATTER = None
_ORIGINAL_SCATTER_INPLACE = None
_INDEX_COPY_PATCHED = False
_ORIGINAL_INDEX_COPY = None
_ORIGINAL_INDEX_COPY_INPLACE = None
_DYNAMIC_CACHE_PATCHED = False
_ORIGINAL_DYNAMIC_CACHE_UPDATE = None
_DTYPE_HOOKS_INSTALLED = False


def install_dtype_harmonization_hooks(model) -> None:
    """Install forward hooks to harmonize dtypes across quantized and non-quantized layers."""
    global _DTYPE_HOOKS_INSTALLED
    if _DTYPE_HOOKS_INSTALLED:
        return
    
    def _harmonize_dtype_hook(module, args, kwargs, output):
        """Ensure output tensors are in float32/bfloat16, not int8."""
        if isinstance(output, torch.Tensor):
            # If output is int8 from a quantized layer, convert to bfloat16
            if output.dtype == torch.int8:
                output = output.to(dtype=torch.bfloat16)
            # If output is uint8, convert to float32
            elif output.dtype == torch.uint8:
                output = output.to(dtype=torch.float32)
        elif isinstance(output, (tuple, list)):
            # Handle tuple/list outputs
            output = type(output)(
                _harmonize_dtype_hook(module, args, kwargs, item) if isinstance(item, torch.Tensor) else item
                for item in output
            )
        return output
    
    # Register hooks on all modules
    hook_count = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module.register_forward_hook(_harmonize_dtype_hook, with_kwargs=True)
            hook_count += 1
    
    _DTYPE_HOOKS_INSTALLED = True
    logger.info(f"Installed dtype harmonization hooks on {hook_count} modules")


def patch_scaled_dot_product_attention() -> None:
    """Patch PyTorch SDPA so attn_bias tensors follow the query device."""
    global _SDPA_PATCHED, _ORIGINAL_SDPA
    if _SDPA_PATCHED:
        return
    _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention

    def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if isinstance(attn_mask, torch.Tensor) and attn_mask.device != query.device:
            attn_mask = attn_mask.to(device=query.device)
        return _ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa  # type: ignore[assignment]
    _SDPA_PATCHED = True
    logger.info("Patched scaled_dot_product_attention to enforce GPU attn_bias")


def patch_gelu_uint8() -> None:
    """Promote stray uint8/int8 activations to float before GELU."""
    global _GELU_PATCHED, _ORIGINAL_GELU
    if _GELU_PATCHED:
        return
    _ORIGINAL_GELU = torch.nn.functional.gelu

    def _patched_gelu(input, approximate='none'):
        if isinstance(input, torch.Tensor):
            # Handle quantized types (uint8, int8, etc.)
            if input.dtype in (torch.uint8, torch.int8):
                input = input.to(dtype=torch.float32)
            # Handle bitsandbytes quantized tensors with quant_state
            elif hasattr(input, 'quant_state'):
                # This is a quantized tensor - it should have been dequantized before GELU
                # Force dequantization by converting to float
                input = input.to(dtype=torch.float32)
        return _ORIGINAL_GELU(input, approximate=approximate)

    torch.nn.functional.gelu = _patched_gelu  # type: ignore[assignment]
    _GELU_PATCHED = True
    logger.info("Patched GELU to promote uint8/int8/quantized activations to float32")


def patch_silu_uint8() -> None:
    """Promote stray uint8/int8 activations to float before SiLU."""
    global _SILU_PATCHED, _ORIGINAL_SILU
    if _SILU_PATCHED:
        return
    _ORIGINAL_SILU = torch.nn.functional.silu

    def _patched_silu(input, inplace=False):
        if isinstance(input, torch.Tensor):
            # Handle quantized types (uint8, int8, etc.)
            if input.dtype in (torch.uint8, torch.int8):
                input = input.to(dtype=torch.float32)
            # Handle bitsandbytes quantized tensors with quant_state
            elif hasattr(input, 'quant_state'):
                input = input.to(dtype=torch.float32)
        return _ORIGINAL_SILU(input, inplace=inplace)

    torch.nn.functional.silu = _patched_silu  # type: ignore[assignment]
    _SILU_PATCHED = True
    logger.info("Patched SiLU to promote uint8/int8/quantized activations to float32")


def patch_scatter_dtype() -> None:
    """Ensure scatter operations upgrade src tensors to match destination dtype."""
    global _SCATTER_PATCHED, _ORIGINAL_SCATTER, _ORIGINAL_SCATTER_INPLACE
    if _SCATTER_PATCHED:
        return

    _ORIGINAL_SCATTER = torch.Tensor.scatter
    _ORIGINAL_SCATTER_INPLACE = torch.Tensor.scatter_

    def _align_src(dtype_tensor: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if isinstance(src, torch.Tensor) and src.dtype != dtype_tensor.dtype:
            return src.to(dtype=dtype_tensor.dtype)
        return src

    def _patched_scatter(self, dim, index, src, reduce=None):
        src = _align_src(self, src)
        if reduce is None:
            return _ORIGINAL_SCATTER(self, dim, index, src)
        try:
            return _ORIGINAL_SCATTER(self, dim, index, src, reduce=reduce)
        except TypeError:
            return _ORIGINAL_SCATTER(self, dim, index, src)

    def _patched_scatter_(self, dim, index, src, reduce=None):
        src = _align_src(self, src)
        if reduce is None:
            return _ORIGINAL_SCATTER_INPLACE(self, dim, index, src)
        try:
            return _ORIGINAL_SCATTER_INPLACE(self, dim, index, src, reduce=reduce)
        except TypeError:
            return _ORIGINAL_SCATTER_INPLACE(self, dim, index, src)

    torch.Tensor.scatter = _patched_scatter  # type: ignore[assignment]
    torch.Tensor.scatter_ = _patched_scatter_  # type: ignore[assignment]
    _SCATTER_PATCHED = True
    logger.info("Patched Tensor.scatter[ _] to align src dtype with destination")


def patch_index_copy_dtype() -> None:
    """Ensure index_copy behaves like scatter dtype-wise."""
    global _INDEX_COPY_PATCHED, _ORIGINAL_INDEX_COPY, _ORIGINAL_INDEX_COPY_INPLACE
    if _INDEX_COPY_PATCHED:
        return

    _ORIGINAL_INDEX_COPY = torch.Tensor.index_copy
    _ORIGINAL_INDEX_COPY_INPLACE = torch.Tensor.index_copy_

    def _align_src(dtype_tensor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if isinstance(source, torch.Tensor) and source.dtype != dtype_tensor.dtype:
            return source.to(dtype=dtype_tensor.dtype)
        return source

    def _patched_index_copy(self, dim, index, source):
        source = _align_src(self, source)
        return _ORIGINAL_INDEX_COPY(self, dim, index, source)

    def _patched_index_copy_(self, dim, index, source):
        source = _align_src(self, source)
        return _ORIGINAL_INDEX_COPY_INPLACE(self, dim, index, source)

    torch.Tensor.index_copy = _patched_index_copy  # type: ignore[assignment]
    torch.Tensor.index_copy_ = _patched_index_copy_  # type: ignore[assignment]
    _INDEX_COPY_PATCHED = True
    logger.info("Patched Tensor.index_copy[ _] to align source dtype with destination")


def patch_dynamic_cache_dtype() -> None:
    """Ensure KV cache promotes incoming tensors to the cache dtype."""
    global _DYNAMIC_CACHE_PATCHED, _ORIGINAL_DYNAMIC_CACHE_UPDATE
    if _DYNAMIC_CACHE_PATCHED:
        return

    try:
        from transformers.cache_utils import DynamicCache  # type: ignore
    except ImportError:
        logger.warning("transformers.cache_utils.DynamicCache unavailable; dtype patch skipped")
        return

    _ORIGINAL_DYNAMIC_CACHE_UPDATE = DynamicCache.update

    def _cast(obj, target_dtype: torch.dtype):
        if target_dtype is None:
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype=target_dtype) if obj.dtype != target_dtype else obj
        if isinstance(obj, (list, tuple)):
            casted = [
                _cast(item, target_dtype) for item in obj
            ]
            return type(obj)(casted)
        return obj

    def _get_cache_dtype(cache, idx):
        if not hasattr(cache, "__len__"):
            return None
        if idx >= len(cache):
            return None
        entry = cache[idx]
        if isinstance(entry, torch.Tensor):
            return entry.dtype
        if isinstance(entry, (list, tuple)) and entry:
            first = entry[0]
            if isinstance(first, torch.Tensor):
                return first.dtype
        return None

    def _patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_dtype = _get_cache_dtype(self.key_cache, layer_idx) if hasattr(self, "key_cache") else None
        value_dtype = _get_cache_dtype(self.value_cache, layer_idx) if hasattr(self, "value_cache") else key_dtype

        if key_dtype is not None:
            key_states = _cast(key_states, key_dtype)
        if value_dtype is not None:
            value_states = _cast(value_states, value_dtype)

        return _ORIGINAL_DYNAMIC_CACHE_UPDATE(
            self,
            key_states,
            value_states,
            layer_idx,
            cache_kwargs=cache_kwargs,
        )

    DynamicCache.update = _patched_update  # type: ignore[assignment]
    _DYNAMIC_CACHE_PATCHED = True
    logger.info("Patched DynamicCache.update to harmonize KV dtype")


def _ensure_bias_device(module, target_device: torch.device) -> None:
    bias = getattr(module, "attn_bias", None)
    if isinstance(bias, torch.Tensor) and bias.device != target_device:
        module.attn_bias = bias.to(target_device)


def ensure_model_on_device(model, device: torch.device, skip_quantized_params: bool) -> None:
    """Move buffers/params plus attach hooks so future biases stay on the right device."""
    if not torch.cuda.is_available():
        return

    patch_scaled_dot_product_attention()
    patch_gelu_uint8()
    patch_silu_uint8()
    patch_scatter_dtype()
    patch_index_copy_dtype()
    patch_dynamic_cache_dtype()

    moved_total = 0
    moved_buffers = 0
    moved_params = 0
    meta_buffer_count = 0
    meta_param_count = 0
    quant_skip_count = 0
    sampled_meta_buffers = []
    sampled_meta_params = []

    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            meta_buffer_count += 1
            if len(sampled_meta_buffers) < 3:
                sampled_meta_buffers.append(name)
            continue
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
            moved_total += 1
            moved_buffers += 1

    for name, param in model.named_parameters():
        if skip_quantized_params and hasattr(param, "quant_state"):
            quant_skip_count += 1
            continue
        if param.device.type == "meta":
            meta_param_count += 1
            if len(sampled_meta_params) < 3:
                sampled_meta_params.append(name)
            continue
        if param.device != device:
            param.data = param.data.to(device)
            moved_total += 1
            moved_params += 1

    if meta_buffer_count:
        suffix = f"; examples: {', '.join(sampled_meta_buffers)}" if sampled_meta_buffers else ""
        logger.info(
            "Skipped %d buffers on meta device (managed by HF device_map)%s",
            meta_buffer_count,
            suffix,
        )
    if meta_param_count:
        suffix = f"; examples: {', '.join(sampled_meta_params)}" if sampled_meta_params else ""
        logger.info(
            "Skipped %d parameters on meta device (managed by HF device_map)%s",
            meta_param_count,
            suffix,
        )
    if quant_skip_count:
        logger.info("Deferred %d quantized parameters to backend hooks", quant_skip_count)

    if moved_total:
        logger.info(
            "Moved %d tensors to %s (%d buffers, %d parameters)",
            moved_total,
            device,
            moved_buffers,
            moved_params,
        )
    else:
        logger.info("All tensors already on %s", device)

    for module in model.modules():
        if hasattr(module, "attn_bias") and not getattr(module, "_hunyuan_bias_hooked", False):
            _ensure_bias_device(module, device)

            def _make_hook():
                def _hook(mod, inputs):
                    first_tensor: Optional[torch.Tensor] = next(
                        (inp for inp in inputs if isinstance(inp, torch.Tensor)),
                        None,
                    )
                    target = first_tensor.device if first_tensor is not None else device
                    _ensure_bias_device(mod, target)

                return _hook

            module.register_forward_pre_hook(_make_hook())
            module._hunyuan_bias_hooked = True  # type: ignore[attr-defined]


class HunyuanModelCache:
    """Simple singleton cache so loaders can share state and we can drop it on demand."""

    _cached_model = None
    _cached_path: Optional[str] = None
    _model_on_cpu = False  # Track if model is parked on CPU for fast reload

    @classmethod
    def _normalize_path(cls, path: str) -> str:
        """Normalize path for consistent comparison."""
        import os
        # Resolve to absolute, normalize slashes, remove trailing slashes
        try:
            return os.path.normpath(os.path.abspath(path)).lower()
        except Exception:
            return path.lower().rstrip('/\\')

    @classmethod
    def get(cls, model_path: str):
        if cls._cached_model is not None:
            # Normalize both paths for comparison
            requested = cls._normalize_path(model_path)
            cached = cls._normalize_path(cls._cached_path) if cls._cached_path else ""
            
            if requested == cached:
                if cls._model_on_cpu:
                    logger.info("Found cached Hunyuan model on CPU for %s - will move to GPU", model_path)
                else:
                    logger.info("Using cached Hunyuan model for %s", model_path)
                return cls._cached_model
            else:
                logger.info("Cache path mismatch: requested=%s, cached=%s", requested, cached)
        return None

    @classmethod
    def is_on_cpu(cls) -> bool:
        """Check if cached model is currently parked on CPU."""
        return cls._model_on_cpu and cls._cached_model is not None

    @classmethod
    def store(cls, model_path: str, model) -> None:
        # Normalize path for consistent cache key
        normalized_path = os.path.normpath(os.path.abspath(model_path))
        logger.info("Caching Hunyuan model for reuse: %s", normalized_path)
        logger.info(f"  Model type: {type(model).__name__}, id: {id(model)}")
        cls._cached_model = model
        cls._cached_path = normalized_path
        cls._model_on_cpu = False
        logger.info(f"  Cache now has model: {cls._cached_model is not None}")

    @classmethod
    def debug_status(cls) -> dict:
        """Return current cache status for debugging."""
        status = {
            "has_model": cls._cached_model is not None,
            "cached_path": cls._cached_path,
            "on_cpu": cls._model_on_cpu,
        }
        if cls._cached_model is not None:
            # Sample device placement
            cuda_count = 0
            cpu_count = 0
            meta_count = 0
            try:
                for i, param in enumerate(cls._cached_model.parameters()):
                    if i >= 50:
                        break
                    if param.device.type == 'cuda':
                        cuda_count += 1
                    elif param.device.type == 'cpu':
                        cpu_count += 1
                    elif param.device.type == 'meta':
                        meta_count += 1
                status["cuda_params"] = cuda_count
                status["cpu_params"] = cpu_count
                status["meta_params"] = meta_count
            except Exception as e:
                status["error"] = str(e)
        logger.info(f"Cache status: {status}")
        return status

    @classmethod
    def soft_unload(cls) -> bool:
        """
        Move model to CPU RAM but keep it cached for fast reload.
        
        This frees GPU VRAM while avoiding the slow disk reload.
        CPU->GPU reload is ~10-20 seconds vs 1-2 minutes from disk.
        
        Returns:
            True if model was moved to CPU, False if no model or already on CPU
        """
        import gc
        
        if cls._cached_model is None:
            logger.info("No model cached, nothing to soft unload")
            return False
        
        if cls._model_on_cpu:
            logger.info("Model already on CPU")
            # Still clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        
        # Check if model is quantized (INT8/NF4)
        # With bitsandbytes >= 0.48.2, quantized models CAN be moved with .to()
        is_quantized = cls._is_model_quantized(cls._cached_model)
        if is_quantized:
            # Verify bitsandbytes version supports .to()
            try:
                import bitsandbytes
                bnb_version = tuple(int(x) for x in bitsandbytes.__version__.split('.')[:3])
                if bnb_version < (0, 48, 2):
                    logger.warning("=" * 60)
                    logger.warning("SOFT UNLOAD REQUIRES bitsandbytes >= 0.48.2")
                    logger.warning(f"Your version: {bitsandbytes.__version__}")
                    logger.warning("Upgrade with: pip install bitsandbytes>=0.48.2")
                    logger.warning("=" * 60)
                    return False
                logger.info(f"Quantized model detected, using bitsandbytes {bitsandbytes.__version__} .to() support")
            except Exception as e:
                logger.warning(f"Could not verify bitsandbytes version: {e}")
                # Continue anyway - let it fail naturally if unsupported
        
        # Check if model has meta tensors (from HF device_map offloading)
        has_meta = cls._has_meta_tensors(cls._cached_model)
        if has_meta:
            logger.warning("=" * 60)
            logger.warning("SOFT UNLOAD NOT SUPPORTED FOR OFFLOADED MODELS")
            logger.warning("Model was loaded with device_map offloading (smart mode)")
            logger.warning("which uses meta tensors that cannot be moved.")
            logger.warning("To use soft unload, load with offload_mode='disabled'.")
            logger.warning("=" * 60)
            return False
        
        logger.info("Soft unloading: Moving model to CPU RAM (fast reload later)...")
        import time
        start_time = time.time()
        
        try:
            model = cls._cached_model
            
            # Remove any GPU-specific hooks before moving
            hook = getattr(model, '_cpu_offload_hook', None)
            if hook is not None:
                try:
                    hook.remove()
                except Exception:
                    pass
            
            # Move all parameters and buffers to CPU
            model.cpu()
            
            # Mark as on CPU
            cls._model_on_cpu = True
            
            # Clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"✓ Model moved to CPU in {elapsed:.1f}s - VRAM now {free/1024**3:.1f}GB free")
            else:
                logger.info(f"✓ Model moved to CPU in {elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to soft unload: {e}")
            return False

    @classmethod
    def _has_meta_tensors(cls, model) -> bool:
        """Check if model has meta tensors (from HuggingFace device_map offloading)."""
        if model is None:
            return False
        
        try:
            for param in model.parameters():
                if param.device.type == 'meta':
                    return True
            for buffer in model.buffers():
                if buffer.device.type == 'meta':
                    return True
        except Exception:
            pass
        
        # Also check for hf_device_map attribute which indicates offloading was used
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # If device_map has 'cpu' or 'disk' values, meta tensors are likely
            device_map = model.hf_device_map
            if isinstance(device_map, dict):
                for device in device_map.values():
                    if device in ('cpu', 'disk', 'meta'):
                        return True
        
        return False

    @classmethod
    def _is_model_quantized(cls, model) -> bool:
        """Check if model uses bitsandbytes quantization (INT8 or NF4)."""
        if model is None:
            return False
        
        try:
            # Check for bitsandbytes quantized layers
            from bitsandbytes.nn import Linear8bitLt, Linear4bit
            for module in model.modules():
                if isinstance(module, (Linear8bitLt, Linear4bit)):
                    return True
        except ImportError:
            pass
        
        # Check for quantization config
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            return True
        
        # Check for quant_state on any parameter
        for param in model.parameters():
            if hasattr(param, 'quant_state'):
                return True
        
        return False

    @classmethod 
    def restore_to_gpu(cls, device: torch.device = None) -> bool:
        """
        Move CPU-cached model back to GPU.
        
        This is MUCH faster than reloading from disk (~10-20s vs 1-2min).
        NOTE: Does NOT work for INT8/NF4 quantized models (bitsandbytes limitation).
        
        Returns:
            True if model was moved to GPU, False otherwise
        """
        if cls._cached_model is None:
            logger.info("No model cached to restore")
            return False
        
        if not cls._model_on_cpu:
            logger.info("Model is not on CPU, no restore needed")
            return True
        
        # Check if model is quantized
        # With bitsandbytes >= 0.48.2, quantized models CAN be restored with .to()
        if cls._is_model_quantized(cls._cached_model):
            try:
                import bitsandbytes
                bnb_version = tuple(int(x) for x in bitsandbytes.__version__.split('.')[:3])
                if bnb_version < (0, 48, 2):
                    logger.error("Cannot restore quantized model - bitsandbytes < 0.48.2")
                    logger.error("Upgrade with: pip install bitsandbytes>=0.48.2")
                    logger.error("Model must be reloaded from disk. Clearing cache...")
                    cls._cached_model = None
                    cls._cached_path = None
                    cls._model_on_cpu = False
                    return False
                logger.info(f"Restoring quantized model using bitsandbytes {bitsandbytes.__version__}")
            except Exception as e:
                logger.warning(f"Could not verify bitsandbytes version: {e}")
        
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        if device.type != "cuda":
            logger.warning("Cannot restore to non-CUDA device")
            return False
        
        logger.info(f"Restoring model from CPU to {device}...")
        import time
        start_time = time.time()
        
        try:
            model = cls._cached_model
            
            # Move back to GPU
            model.to(device)
            
            cls._model_on_cpu = False
            
            elapsed = time.time() - start_time
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"✓ Model restored to GPU in {elapsed:.1f}s - VRAM: {allocated:.1f}GB")
            else:
                logger.info(f"✓ Model restored to GPU in {elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore model to GPU: {e}")
            # Model may be in inconsistent state, mark for full reload
            cls._model_on_cpu = False
            cls._cached_model = None
            cls._cached_path = None
            return False

    @classmethod
    def clear(cls) -> bool:
        import gc
        import traceback
        had_model = cls._cached_model is not None
        
        # Log who is calling clear and the call stack
        if had_model:
            logger.info("=" * 40)
            logger.info("HunyuanModelCache.clear() called!")
            logger.info("Call stack (most recent call last):")
            for line in traceback.format_stack()[-5:-1]:
                logger.info(line.strip())
            logger.info("=" * 40)
        
        if had_model:
            logger.info("Clearing cached Hunyuan model from VRAM...")
            try:
                # Store reference for cleanup
                model_to_clear = cls._cached_model
                
                # Clear cache variables FIRST to prevent reuse during cleanup
                cls._cached_model = None
                cls._cached_path = None
                cls._model_on_cpu = False
                
                # Remove accelerate hooks if present
                hook = getattr(model_to_clear, '_cpu_offload_hook', None)
                if hook is not None:
                    try:
                        hook.remove()
                    except Exception as hook_err:
                        logger.warning("Failed to remove cpu_offload hook: %s", hook_err)
                if hasattr(model_to_clear, '_forced_device'):
                    try:
                        delattr(model_to_clear, '_forced_device')
                    except Exception:
                        pass

                # Try to move to CPU first (helps release CUDA memory)
                # But catch errors for meta tensors or quantized models
                try:
                    if hasattr(model_to_clear, 'cpu'):
                        model_to_clear.cpu()
                except Exception as move_err:
                    # Meta tensors or quantized models can't be moved
                    # Just log and continue with deletion
                    logger.debug("Could not move model to CPU (expected for meta/quantized): %s", move_err)
                
                # Aggressively clear all parameter data
                try:
                    for param in model_to_clear.parameters():
                        param.data = torch.empty(0)
                        if param.grad is not None:
                            param.grad = None
                except Exception:
                    pass  # Some params may be meta or locked
                
                # Clear all buffers too
                try:
                    for buffer in model_to_clear.buffers():
                        buffer.data = torch.empty(0)
                except Exception:
                    pass
                
                # Remove all hooks that might hold references
                try:
                    for module in model_to_clear.modules():
                        module._forward_hooks.clear()
                        module._forward_pre_hooks.clear()
                        module._backward_hooks.clear()
                        if hasattr(module, '_state_dict_hooks'):
                            module._state_dict_hooks.clear()
                except Exception:
                    pass
                
                # Delete all references
                del model_to_clear
                
            except Exception as e:
                logger.warning("Error during model cleanup: %s", e)
                # Ensure cache is cleared even if cleanup fails
                cls._cached_model = None
                cls._cached_path = None
                cls._model_on_cpu = False
            
            # Force garbage collection - multiple passes
            for _ in range(3):
                gc.collect()
            
            # Clear CUDA cache on all GPUs
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            logger.info("✓ Cleared cached model and freed VRAM")
        else:
            # Ensure flags are reset even if no model
            cls._model_on_cpu = False
            cls._cached_path = None
            
            # Still clear CUDA cache even if no model was cached
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache (no model was cached)")
        
        return had_model


class HunyuanImage3Unload:
    """
    Utility node that clears cached Hunyuan models from GPU VRAM.
    
    For successive runs with downstream models (Flux detailer, SAM2, etc.):
    - Place this node AFTER Hunyuan generation but BEFORE downstream nodes
    - Enable 'clear_for_downstream' to free Hunyuan VRAM for other models
    - The unload_signal output can trigger downstream nodes after clearing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_clear": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear the cached Hunyuan model from VRAM"
                }),
            },
            "optional": {
                "clear_for_downstream": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Aggressively clear VRAM for downstream models (Flux, SAM2, etc). Use for successive runs where Hunyuan needs to reload each time anyway."
                }),
                "trigger": ("*", {"default": None}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "*")
    RETURN_NAMES = ("cleared", "vram_status", "unload_signal")
    FUNCTION = "unload"
    CATEGORY = "HunyuanImage3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run this node to ensure VRAM is cleared when requested
        return float("nan")

    def unload(self, force_clear, clear_for_downstream=False, trigger=None):
        import gc
        import time
        
        cleared = False
        vram_status = ""
        
        # Get initial VRAM state
        vram_before = 0
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram_before = (total - free) / 1024**3
            except Exception:
                pass
        
        if force_clear:
            cleared = HunyuanModelCache.clear()
        
        # Additional aggressive clearing for downstream models
        if clear_for_downstream:
            logger.info("=" * 60)
            logger.info("CLEARING VRAM FOR DOWNSTREAM MODELS")
            logger.info("Freeing space for Flux detailer, SAM2, etc.")
            logger.info("=" * 60)
            
            # Multiple GC passes
            for _ in range(3):
                gc.collect()
            
            # Clear CUDA cache aggressively
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Try to release cached blocks
                try:
                    torch.cuda.memory._set_allocator_settings("")
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        
        # Get final VRAM state
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram_after = (total - free) / 1024**3
                vram_freed = vram_before - vram_after
                vram_status = f"{free/1024**3:.1f}GB free ({vram_freed:.1f}GB freed)"
                logger.info(f"VRAM after unload: {vram_status}")
            except Exception:
                vram_status = "Could not query VRAM"
        
        # Return a signal that changes to force downstream nodes to re-evaluate if needed
        return (cleared, vram_status, float(time.time()))


class HunyuanImage3SoftUnload:
    """
    Fast VRAM release - moves model to CPU RAM instead of deleting.
    
    ✅ NOW WORKS WITH: (requires bitsandbytes >= 0.48.2)
    - INT8 quantized models
    - NF4 quantized models  
    - BF16 models loaded entirely on GPU
    
    ⚠️ Does NOT work with:
    - Models loaded with device_map offloading (has meta tensors)
    - Use offload_mode='disabled' in loader to enable soft unload
    
    BENEFITS:
    - Soft unload + restore: ~10-30 seconds (scales with model size)
    - Full unload + reload from disk: ~2+ minutes
    
    USE CASE:
    Run Hunyuan → Soft Unload → Run downstream models (Flux, SAM2, etc.)
    → Restore Hunyuan → Generate again (no disk reload needed!)
    
    ACTIONS:
    - soft_unload: Move model from GPU to CPU RAM, free VRAM
    - restore_to_gpu: Move model back to GPU for inference
    - check_status: Report current model location
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["soft_unload", "restore_to_gpu", "check_status"], {
                    "default": "soft_unload",
                    "tooltip": "soft_unload: Move to CPU. restore_to_gpu: Move back. check_status: Just report."
                }),
            },
            "optional": {
                "trigger": ("*", {"default": None}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "*")
    RETURN_NAMES = ("success", "status", "signal")
    FUNCTION = "execute"
    CATEGORY = "HunyuanImage3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, action="soft_unload", trigger=None):
        import time
        
        # Debug: Log cache status at start
        cache_status = HunyuanModelCache.debug_status()
        logger.info(f"SoftUnload [{action}]: Cache status = {cache_status}")
        
        if action == "soft_unload":
            # Check for unsupported model types BEFORE attempting soft unload
            is_quantized = False
            has_meta = False
            if HunyuanModelCache._cached_model is not None:
                is_quantized = HunyuanModelCache._is_model_quantized(HunyuanModelCache._cached_model)
                has_meta = HunyuanModelCache._has_meta_tensors(HunyuanModelCache._cached_model)
            
            success = HunyuanModelCache.soft_unload()
            if success:
                status = "Model moved to CPU RAM - VRAM freed for other tasks"
            elif HunyuanModelCache.is_on_cpu():
                status = "Model already on CPU"
                success = True
            elif is_quantized:
                # Quantized model detected - check bitsandbytes version
                try:
                    import bitsandbytes
                    status = f"⚠️ Soft unload failed. bitsandbytes {bitsandbytes.__version__} - upgrade to >=0.48.2 for quantized model support."
                except ImportError:
                    status = "⚠️ bitsandbytes not found. Install with: pip install bitsandbytes>=0.48.2"
                success = False
            elif has_meta:
                # Model loaded with device_map offloading
                status = "⚠️ Model loaded with offloading (smart mode) - has meta tensors. Use offload_mode='disabled' for soft unload, or use Force Unload."
                success = False
            elif HunyuanModelCache._cached_model is None:
                status = "No model cached to unload"
            else:
                status = "Soft unload failed - try Force Unload"
        
        elif action == "restore_to_gpu":
            if HunyuanModelCache._cached_model is None:
                status = "No model cached - use loader node"
                success = False
            elif not HunyuanModelCache.is_on_cpu():
                status = "Model already on GPU"
                success = True
            else:
                success = HunyuanModelCache.restore_to_gpu()
                if success:
                    status = "Model restored to GPU from CPU RAM"
                else:
                    status = "Failed to restore - model may need full reload"
        
        elif action == "check_status":
            if HunyuanModelCache._cached_model is None:
                status = "No model cached"
            elif HunyuanModelCache.is_on_cpu():
                status = f"Model cached on CPU: {HunyuanModelCache._cached_path}"
            else:
                status = f"Model cached on GPU: {HunyuanModelCache._cached_path}"
            success = HunyuanModelCache._cached_model is not None
        
        else:
            status = f"Unknown action: {action}"
            success = False
        
        logger.info(f"SoftUnload [{action}]: {status}")
        return (success, status, float(time.time()))


class HunyuanImage3ForceUnload:
    """
    Aggressive VRAM clearing node - nuclear option for stubborn memory leaks.
    
    This node performs multiple aggressive cleanup steps:
    1. Clears model cache (same as regular Unload)
    2. Clears all ComfyUI model management caches
    3. Forces Python garbage collection multiple times
    4. Resets CUDA memory allocator
    5. Optionally hunts and destroys ALL orphaned CUDA tensors
    
    The "nuke_orphaned_tensors" option is the most aggressive - it scans
    all Python objects and destroys any CUDA tensors it finds. Use this
    after OOM errors when memory is stuck and nothing else works.
    
    NEW: "first_run_only" option - when enabled, the node will perform
    the nuclear clear on the first run, then automatically skip itself
    on subsequent runs (preserving the loaded Hunyuan model).
    Use "Reset First Run Flag" button to re-enable for next queue.
    
    WARNING: This may affect other loaded models in ComfyUI!
    Use only when regular Unload doesn't work or after OOM errors.
    """
    
    # Class variable to track first-run state per node instance
    _first_run_completed = {}  # node_id -> bool

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_run_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only run on FIRST execution, then auto-skip. Perfect for cleaning up cross-tab pollution while keeping Hunyuan loaded for successive runs."
                }),
                "clear_all_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear Hunyuan model cache"
                }),
                "aggressive_gc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run aggressive garbage collection (3 passes)"
                }),
                "reset_cuda_allocator": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reset CUDA memory allocator (may help after OOM)"
                }),
                "clear_comfy_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear ComfyUI's internal model cache (affects all models!)"
                }),
                "nuke_orphaned_tensors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "DANGEROUS: Hunt and destroy ALL CUDA tensors in memory. Use after OOM when VRAM is stuck."
                }),
                "nuke_ram_tensors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "DANGEROUS: Also hunt and destroy orphaned CPU tensors in RAM. Use when system RAM is full of leaked model weights."
                }),
            },
            "optional": {
                "trigger": ("*", {"default": None}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "*")
    RETURN_NAMES = ("cleared", "memory_report", "unload_signal")
    FUNCTION = "force_unload"
    CATEGORY = "HunyuanImage3"
    OUTPUT_NODE = True
    
    @classmethod
    def reset_first_run_flags(cls):
        """Reset all first-run flags - call this to re-enable first_run_only nodes."""
        cls._first_run_completed.clear()
        logger.info("Force Unload: All first-run flags reset")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def force_unload(self, first_run_only=False, clear_all_models=True, aggressive_gc=True, 
                     reset_cuda_allocator=True, clear_comfy_cache=True,
                     nuke_orphaned_tensors=True, nuke_ram_tensors=True, 
                     trigger=None, unique_id=None):
        import gc
        import time
        
        # Check first_run_only logic
        node_id = str(unique_id) if unique_id else "default"
        
        if first_run_only:
            if node_id in HunyuanImage3ForceUnload._first_run_completed:
                # Already ran once - SKIP this execution
                skip_report = [
                    "=== FORCE UNLOAD SKIPPED (first_run_only) ===",
                    f"Node {node_id} already executed once this session.",
                    "Nuclear clear was done on first run, now preserving loaded models.",
                    "To re-enable: Restart ComfyUI or reload the workflow.",
                    "=== END SKIP ==="
                ]
                logger.info("Force Unload: Skipping (first_run_only mode, already executed)")
                return (False, "\n".join(skip_report), trigger)
            else:
                # First run - mark as completed and proceed
                HunyuanImage3ForceUnload._first_run_completed[node_id] = True
                logger.info(f"Force Unload: First run for node {node_id}, will skip on subsequent runs")
        
        report_lines = ["=== FORCE UNLOAD REPORT ==="]
        if first_run_only:
            report_lines.append("Mode: first_run_only (will skip on subsequent runs)")
        cleared = False
        
        # Get initial VRAM state
        vram_before = 0
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram_before = (total - free) / 1024**3
                report_lines.append(f"VRAM before: {vram_before:.2f}GB used")
            except Exception as e:
                report_lines.append(f"Could not get initial VRAM: {e}")
        
        # Step 1: Clear Hunyuan model cache
        if clear_all_models:
            try:
                cleared = HunyuanModelCache.clear()
                report_lines.append(f"Hunyuan cache cleared: {cleared}")
            except Exception as e:
                report_lines.append(f"Failed to clear Hunyuan cache: {e}")
        
        # Step 2: Clear ComfyUI's internal caches (optional, more aggressive)
        if clear_comfy_cache:
            try:
                import comfy.model_management as mm
                if hasattr(mm, 'unload_all_models'):
                    mm.unload_all_models()
                    report_lines.append("ComfyUI unload_all_models called")
                if hasattr(mm, 'soft_empty_cache'):
                    mm.soft_empty_cache()
                    report_lines.append("ComfyUI soft_empty_cache called")
                if hasattr(mm, 'cleanup_models'):
                    mm.cleanup_models()
                    report_lines.append("ComfyUI cleanup_models called")
            except ImportError:
                report_lines.append("ComfyUI model_management not available")
            except Exception as e:
                report_lines.append(f"ComfyUI cache clear error: {e}")
        
        # Step 3: Aggressive garbage collection
        if aggressive_gc:
            try:
                # Multiple GC passes to catch circular references
                for i in range(3):
                    unreachable = gc.collect()
                    if i == 0:
                        report_lines.append(f"GC pass {i+1}: {unreachable} objects collected")
                
                # Try to clear __del__ finalizers that might hold GPU tensors
                gc.collect()
                
                report_lines.append("Aggressive GC complete (3 passes)")
            except Exception as e:
                report_lines.append(f"GC error: {e}")
        
        # Step 4: CUDA cleanup
        if torch.cuda.is_available():
            try:
                # Synchronize all streams first
                torch.cuda.synchronize()
                
                # Empty cache on all devices
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                report_lines.append("CUDA cache cleared on all devices")
                
                # Reset peak memory stats for accurate tracking
                if reset_cuda_allocator:
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)
                        torch.cuda.reset_accumulated_memory_stats(i)
                    report_lines.append("CUDA memory stats reset")
                    
                    # Try to release cached memory back to CUDA driver
                    try:
                        # This is more aggressive - releases cached blocks
                        torch.cuda.memory._set_allocator_settings("")
                        torch.cuda.empty_cache()
                        report_lines.append("CUDA allocator reset")
                    except Exception:
                        # Not all PyTorch versions support this
                        pass
                
            except Exception as e:
                report_lines.append(f"CUDA cleanup error: {e}")
        
        # Step 5: NUCLEAR OPTION - Hunt and destroy orphaned tensors
        if nuke_orphaned_tensors or nuke_ram_tensors:
            report_lines.append("--- NUCLEAR TENSOR HUNT ---")
            if nuke_orphaned_tensors:
                report_lines.append("Targeting: CUDA (VRAM) tensors")
            if nuke_ram_tensors:
                report_lines.append("Targeting: CPU (RAM) tensors")
            
            try:
                cuda_tensors_found = 0
                cpu_tensors_found = 0
                tensors_cleared = 0
                modules_found = 0
                ram_freed_estimate = 0
                
                # Get all objects in memory
                all_objects = gc.get_objects()
                report_lines.append(f"Scanning {len(all_objects)} Python objects...")
                
                # First pass: Find and clear nn.Module instances
                for obj in all_objects:
                    try:
                        if isinstance(obj, torch.nn.Module):
                            # Skip our cached model - we handle that separately
                            if obj is HunyuanModelCache._cached_model:
                                continue
                            modules_found += 1
                            # Clear all parameters
                            for param in obj.parameters():
                                if param.device.type == 'cuda' and nuke_orphaned_tensors:
                                    ram_freed_estimate += param.numel() * param.element_size()
                                    param.data = torch.empty(0, device='cpu')
                                    tensors_cleared += 1
                                elif param.device.type == 'cpu' and nuke_ram_tensors:
                                    ram_freed_estimate += param.numel() * param.element_size()
                                    param.data = torch.empty(0)
                                    tensors_cleared += 1
                            # Clear all buffers
                            for buf in obj.buffers():
                                if buf.device.type == 'cuda' and nuke_orphaned_tensors:
                                    ram_freed_estimate += buf.numel() * buf.element_size()
                                    buf.data = torch.empty(0, device='cpu')
                                    tensors_cleared += 1
                                elif buf.device.type == 'cpu' and nuke_ram_tensors:
                                    ram_freed_estimate += buf.numel() * buf.element_size()
                                    buf.data = torch.empty(0)
                                    tensors_cleared += 1
                    except Exception:
                        pass
                
                # Second pass: Find raw tensors
                for obj in all_objects:
                    try:
                        if isinstance(obj, torch.Tensor):
                            if obj.device.type == 'cuda':
                                cuda_tensors_found += 1
                                if nuke_orphaned_tensors:
                                    ram_freed_estimate += obj.numel() * obj.element_size()
                                    obj.data = torch.empty(0, device='cpu')
                                    tensors_cleared += 1
                            elif obj.device.type == 'cpu':
                                cpu_tensors_found += 1
                                if nuke_ram_tensors:
                                    ram_freed_estimate += obj.numel() * obj.element_size()
                                    obj.data = torch.empty(0)
                                    tensors_cleared += 1
                    except Exception:
                        pass
                
                report_lines.append(f"Found {modules_found} nn.Module instances")
                report_lines.append(f"Found {cuda_tensors_found} CUDA tensors, {cpu_tensors_found} CPU tensors")
                report_lines.append(f"Cleared {tensors_cleared} tensors (~{ram_freed_estimate/1024**3:.2f}GB)")
                
                # Force cleanup after tensor hunting
                del all_objects
                for _ in range(5):
                    gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                report_lines.append("Nuclear cleanup complete")
                
            except Exception as e:
                report_lines.append(f"Nuclear cleanup error: {e}")
        
        # Final GC pass after CUDA cleanup
        if aggressive_gc:
            gc.collect()
        
        # Get final VRAM state
        vram_after = 0
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram_after = (total - free) / 1024**3
                vram_freed = vram_before - vram_after
                report_lines.append(f"VRAM after: {vram_after:.2f}GB used")
                report_lines.append(f"VRAM freed: {vram_freed:.2f}GB")
            except Exception as e:
                report_lines.append(f"Could not get final VRAM: {e}")
        
        report_lines.append("=== END FORCE UNLOAD ===")
        
        # Log the report
        for line in report_lines:
            logger.info(line)
        
        memory_report = "\n".join(report_lines)
        return (cleared or clear_comfy_cache, memory_report, float(time.time()))


class HunyuanImage3ClearDownstream:
    """
    Clear downstream models (Flux, SAM2, Florence, etc.) while KEEPING Hunyuan loaded.
    
    Use this node at the END of your workflow (after all downstream processing)
    to free VRAM from other models before the next Hunyuan generation.
    
    Workflow placement:
    [Hunyuan] → [Generate] → [Flux Detailer] → [SAM2] → [Output] → [THIS NODE]
                                                                        ↓
                                                              (triggers on next run)
    
    This allows successive Hunyuan runs without reloading the large model each time,
    while still freeing VRAM used by downstream models between runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_comfy_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use ComfyUI's model management to unload non-Hunyuan models"
                }),
                "aggressive_gc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run aggressive garbage collection"
                }),
            },
            "optional": {
                # Accept common output types - IMAGE is most common at end of workflow
                "trigger_image": ("IMAGE", {
                    "tooltip": "Connect an IMAGE output to trigger after that node completes"
                }),
                "trigger_string": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect a STRING output (like filepath) to trigger after that node"
                }),
                "trigger_latent": ("LATENT", {
                    "tooltip": "Connect a LATENT output to trigger after that node completes"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("memory_report",)
    FUNCTION = "clear_downstream"
    CATEGORY = "HunyuanImage3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def clear_downstream(self, clear_comfy_models=True, aggressive_gc=True, 
                         trigger_image=None, trigger_string=None, trigger_latent=None,
                         unique_id=None):
        import gc
        import time
        
        report_lines = ["=== CLEARING DOWNSTREAM MODELS ==="]
        report_lines.append("Keeping Hunyuan model loaded")
        
        # Get initial VRAM state
        vram_before = 0
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram_before = (total - free) / 1024**3
                report_lines.append(f"VRAM before: {vram_before:.2f}GB used")
            except Exception:
                pass
        
        # Use ComfyUI's model management to unload models
        if clear_comfy_models:
            try:
                import comfy.model_management as mm
                
                # Get current loaded models info before clearing
                if hasattr(mm, 'current_loaded_models'):
                    loaded = mm.current_loaded_models
                    if loaded:
                        report_lines.append(f"ComfyUI has {len(loaded)} models loaded")
                
                # Unload all models that ComfyUI is managing
                # Note: This won't touch our Hunyuan cache since we manage it separately
                if hasattr(mm, 'unload_all_models'):
                    mm.unload_all_models()
                    report_lines.append("Called ComfyUI unload_all_models()")
                
                # Soft empty cache to release memory
                if hasattr(mm, 'soft_empty_cache'):
                    mm.soft_empty_cache()
                    report_lines.append("Called ComfyUI soft_empty_cache()")
                
                # Cleanup models - more aggressive
                if hasattr(mm, 'cleanup_models'):
                    mm.cleanup_models()
                    report_lines.append("Called ComfyUI cleanup_models()")
                    
            except ImportError:
                report_lines.append("ComfyUI model_management not available")
            except Exception as e:
                report_lines.append(f"ComfyUI cleanup error: {e}")
        
        # Aggressive garbage collection
        if aggressive_gc:
            for i in range(3):
                collected = gc.collect()
                if i == 0:
                    report_lines.append(f"GC collected {collected} objects")
        
        # Clear CUDA cache (but NOT our Hunyuan model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            report_lines.append("Cleared CUDA cache")
        
        # Verify Hunyuan is still cached
        if HunyuanModelCache._cached_model is not None:
            report_lines.append(f"✓ Hunyuan model still cached: {HunyuanModelCache._cached_path}")
            if HunyuanModelCache._model_on_cpu:
                report_lines.append("  (model is on CPU)")
            else:
                report_lines.append("  (model is on GPU)")
        else:
            report_lines.append("⚠ No Hunyuan model in cache")
        
        # Get final VRAM state
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(0)
                vram_after = (total - free) / 1024**3
                vram_freed = vram_before - vram_after
                report_lines.append(f"VRAM after: {vram_after:.2f}GB used")
                report_lines.append(f"VRAM freed: {vram_freed:.2f}GB")
            except Exception:
                pass
        
        report_lines.append("=== END CLEAR DOWNSTREAM ===")
        
        # Log the report
        for line in report_lines:
            logger.info(line)
        
        memory_report = "\n".join(report_lines)
        return (memory_report,)


class MemoryTracker:
    """Track system RAM and GPU VRAM usage during a generation run."""

    def __init__(self, device_index: int = 0):
        self.process = psutil.Process()
        self.device_index = device_index if torch.cuda.is_available() else None
        self.ram_start_bytes = self.process.memory_info().rss
        self._finished_stats = None

        if self.device_index is not None:
            torch.cuda.reset_peak_memory_stats(self.device_index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_index)
            self.vram_start_bytes = total_bytes - free_bytes
            self.vram_total_bytes = total_bytes
        else:
            self.vram_start_bytes = None
            self.vram_total_bytes = None

    @staticmethod
    def _bytes_to_gb(value: Optional[int]) -> Optional[float]:
        if value is None:
            return None
        return value / 1024**3

    @staticmethod
    def _extract_peak_ram(mem_info) -> int:
        candidates = []
        for attr in ("peak_wset", "peak_rss", "peak_pagefile", "peak_pagefaults", "rss"):
            value = getattr(mem_info, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                candidates.append(int(value))
        return max(candidates) if candidates else mem_info.rss

    def finish(self) -> dict:
        if self._finished_stats is not None:
            return self._finished_stats

        mem_end = self.process.memory_info()
        ram_end_bytes = mem_end.rss
        ram_peak_bytes = max(self._extract_peak_ram(mem_end), self.ram_start_bytes, ram_end_bytes)

        vram_end_bytes = None
        vram_peak_bytes = None
        if self.device_index is not None:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_index)
            vram_end_bytes = total_bytes - free_bytes
            vram_peak_bytes = torch.cuda.max_memory_reserved(self.device_index)

        stats = {
            "ram_start_gb": self._bytes_to_gb(self.ram_start_bytes),
            "ram_end_gb": self._bytes_to_gb(ram_end_bytes),
            "ram_peak_gb": self._bytes_to_gb(ram_peak_bytes),
            "vram_start_gb": self._bytes_to_gb(self.vram_start_bytes),
            "vram_end_gb": self._bytes_to_gb(vram_end_bytes),
            "vram_peak_gb": self._bytes_to_gb(vram_peak_bytes),
            "vram_total_gb": self._bytes_to_gb(self.vram_total_bytes),
        }

        self._finished_stats = stats
        return stats

    def start_snapshot(self) -> dict:
        return {
            "ram_start_gb": self._bytes_to_gb(self.ram_start_bytes),
            "vram_start_gb": self._bytes_to_gb(self.vram_start_bytes),
            "vram_total_gb": self._bytes_to_gb(self.vram_total_bytes),
        }


def format_memory_stats(stats: dict) -> str:
    """Create a short human-readable summary of memory usage."""
    parts = []
    peak_vram = stats.get("vram_peak_gb")
    if peak_vram is not None:
        parts.append(f"Peak VRAM {peak_vram:.1f}GB")

    vram_start = stats.get("vram_start_gb")
    vram_end = stats.get("vram_end_gb")
    if vram_start is not None and vram_end is not None:
        parts.append(f"VRAM Start {vram_start:.1f}GB → End {vram_end:.1f}GB")

    peak_ram = stats.get("ram_peak_gb")
    if peak_ram is not None:
        parts.append(f"Peak RAM {peak_ram:.1f}GB")

    if not parts:
        return ""
    return " | ".join(parts)


def patch_hunyuan_generate_image(model):
    """
    Monkey-patch the generate_image method on the model instance to correctly handle
    callback_on_step_end by only passing it to the final image generation step.
    This fixes issues where the callback is passed to text generation steps (CoT, ratio)
    which causes a ValueError in transformers.
    """
    import sys
    import types
    
    # Get the module where the model is defined to access its globals
    model_module = sys.modules[model.__class__.__module__]
    get_system_prompt = getattr(model_module, "get_system_prompt", None)
    t2i_system_prompts = getattr(model_module, "t2i_system_prompts", None)
    default = getattr(model_module, "default", lambda val, d: val if val is not None else d)
    
    def new_generate_image(
            self,
            prompt,
            seed=None,
            image_size="auto",
            use_system_prompt=None,
            system_prompt=None,
            bot_task=None,
            stream=False,
            **kwargs,
    ):
        max_new_tokens = kwargs.pop("max_new_tokens", 8192)
        verbose = kwargs.pop("verbose", 0)

        # Extract callback_on_step_end so we can pass it ONLY to the final image gen step
        callback_on_step_end = kwargs.pop("callback_on_step_end", None)

        if stream:
            from transformers import TextStreamer
            streamer = TextStreamer(self._tkwrapper.tokenizer, skip_prompt=True, skip_special_tokens=False)
            kwargs["streamer"] = streamer

        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)
        system_prompt = get_system_prompt(use_system_prompt, bot_task, system_prompt)

        if bot_task in ["think", "recaption"]:
            # Cot
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, bot_task=bot_task, system_prompt=system_prompt, max_new_tokens=max_new_tokens)
            print(f"<{bot_task}>", end="", flush=True)
            # Do NOT pass callback_on_step_end here
            outputs = self._generate(**model_inputs, **kwargs, verbose=verbose)
            cot_text = self.get_cot_text(outputs[0])
            # Switch system_prompt to `en_recaption` if drop_think is enabled.
            if self.generation_config.drop_think and system_prompt:
                system_prompt = t2i_system_prompts["en_recaption"][0]
        else:
            cot_text = None

        # Image ratio
        if image_size == "auto":
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, cot_text=cot_text, bot_task="img_ratio", system_prompt=system_prompt, seed=seed)
            # Do NOT pass callback_on_step_end here
            outputs = self._generate(**model_inputs, **kwargs, verbose=verbose)
            ratio_index = outputs[0, -1].item() - self._tkwrapper.ratio_token_offset
            # In some cases, the generated ratio_index is out of range. A valid ratio_index should be in [0, 32].
            # If ratio_index is out of range, we set it to 16 (i.e., 1:1).
            if ratio_index < 0 or ratio_index >= len(self.image_processor.reso_group):
                ratio_index = 16
            reso = self.image_processor.reso_group[ratio_index]
            image_size = reso.height, reso.width

        # Generate image
        model_inputs = self.prepare_model_inputs(
            prompt=prompt, cot_text=cot_text, system_prompt=system_prompt, mode="gen_image", seed=seed,
            image_size=image_size,
        )
        
        # Ensure we don't pass callback_on_step_end if it's None, just to be safe
        gen_kwargs = kwargs.copy()
        if callback_on_step_end is not None:
            gen_kwargs["callback_on_step_end"] = callback_on_step_end
            
        # PASS callback_on_step_end here
        outputs = self._generate(**model_inputs, **gen_kwargs, verbose=verbose)
        return outputs[0]

    logger.info("Patching model.generate_image to support progress bars...")
    model.generate_image = types.MethodType(new_generate_image, model)

    # Also patch the pipeline class to extract callback_on_step_end from model_kwargs
    # This is needed because the cached _generate method puts everything into model_kwargs
    if hasattr(model, 'pipeline'):
        pipeline_class = model.pipeline.__class__
        if not getattr(pipeline_class, '_is_patched_for_comfy', False):
            logger.info("Patching HunyuanImage3Text2ImagePipeline to handle callback_on_step_end...")
            original_call = pipeline_class.__call__
            
            def new_pipeline_call(self, *args, **kwargs):
                model_kwargs = kwargs.get('model_kwargs')
                if model_kwargs and isinstance(model_kwargs, dict):
                    if 'callback_on_step_end' in model_kwargs:
                        cb = model_kwargs.pop('callback_on_step_end')
                        # Only set it if not already provided explicitly
                        if kwargs.get('callback_on_step_end') is None:
                            kwargs['callback_on_step_end'] = cb
                return original_call(self, *args, **kwargs)
            
            pipeline_class.__call__ = new_pipeline_call
            pipeline_class._is_patched_for_comfy = True
