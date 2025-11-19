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

import torch

logger = logging.getLogger(__name__)

_SDPA_PATCHED = False
_ORIGINAL_SDPA = None
_GELU_PATCHED = False
_ORIGINAL_GELU = None
_SCATTER_PATCHED = False
_ORIGINAL_SCATTER = None
_ORIGINAL_SCATTER_INPLACE = None
_INDEX_COPY_PATCHED = False
_ORIGINAL_INDEX_COPY = None
_ORIGINAL_INDEX_COPY_INPLACE = None
_DYNAMIC_CACHE_PATCHED = False
_ORIGINAL_DYNAMIC_CACHE_UPDATE = None


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
    """Promote stray uint8 activations to float before GELU."""
    global _GELU_PATCHED, _ORIGINAL_GELU
    if _GELU_PATCHED:
        return
    _ORIGINAL_GELU = torch.nn.functional.gelu

    def _patched_gelu(input, approximate='none'):
        if isinstance(input, torch.Tensor) and input.dtype == torch.uint8:
            input = input.to(dtype=torch.float32)
        return _ORIGINAL_GELU(input, approximate=approximate)

    torch.nn.functional.gelu = _patched_gelu  # type: ignore[assignment]
    _GELU_PATCHED = True
    logger.info("Patched GELU to promote uint8 activations to float32")


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

    @classmethod
    def get(cls, model_path: str):
        if cls._cached_model is not None and cls._cached_path == model_path:
            logger.info("Using cached Hunyuan model for %s", model_path)
            return cls._cached_model
        return None

    @classmethod
    def store(cls, model_path: str, model) -> None:
        logger.info("Caching Hunyuan model for reuse: %s", model_path)
        cls._cached_model = model
        cls._cached_path = model_path

    @classmethod
    def clear(cls) -> bool:
        import gc
        had_model = cls._cached_model is not None
        
        if had_model:
            logger.info("Clearing cached Hunyuan model from VRAM...")
            try:
                # Move model to CPU first to release VRAM
                if hasattr(cls._cached_model, 'cpu'):
                    cls._cached_model.cpu()
                # Delete all references
                del cls._cached_model
            except Exception as e:
                logger.warning("Error during model cleanup: %s", e)
            
            cls._cached_model = None
            cls._cached_path = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache on all GPUs
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            logger.info("✓ Cleared cached model and freed VRAM")
        else:
            # Still clear CUDA cache even if no model was cached
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache (no model was cached)")
        
        return had_model


class HunyuanImage3Unload:
    """Utility node that clears cached Hunyuan models from GPU VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_clear": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "trigger": ("*", {"default": None}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("cleared",)
    FUNCTION = "unload"
    CATEGORY = "HunyuanImage3"
    OUTPUT_NODE = True

    def unload(self, force_clear, trigger=None):
        if torch.cuda.is_available():
            before_mem = {}
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                before_mem[i] = (total - free) / 1024**3
        
        released = HunyuanModelCache.clear()
        
        if torch.cuda.is_available():
            logger.info("="*60)
            logger.info("VRAM USAGE AFTER UNLOAD")
            logger.info("="*60)
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                after_used = (total - free) / 1024**3
                before_used = before_mem.get(i, 0)
                freed = before_used - after_used
                logger.info(
                    "GPU %d: %.2f GiB used (freed %.2f GiB) / %.2f GiB total",
                    i, after_used, freed, total / 1024**3
                )
            logger.info("="*60)
        
        if released:
            logger.info("✓ Hunyuan cache cleared via unload node")
        else:
            logger.info("Cache already empty, cleared CUDA cache only")
        
        return (released,)
