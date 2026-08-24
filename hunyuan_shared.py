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

import inspect
import logging
import os
from pathlib import Path
from typing import List, Optional

import psutil
import torch
import transformers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transformers version detection
# ---------------------------------------------------------------------------
_TRANSFORMERS_VERSION: tuple = tuple(
    int(x) for x in transformers.__version__.split(".")[:2]
)
"""Parsed (major, minor) of the installed transformers package."""

_TRANSFORMERS_GTE_5 = _TRANSFORMERS_VERSION >= (5, 0)
"""True when running transformers >= 5.0 (lazy_initialization signature change)."""

logger.info(
    "Transformers %s detected (parsed %s, >=5.0: %s)",
    transformers.__version__,
    _TRANSFORMERS_VERSION,
    _TRANSFORMERS_GTE_5,
)

# ---------------------------------------------------------------------------
# ComfyUI folder_paths integration
# ---------------------------------------------------------------------------
try:
    import folder_paths as _folder_paths
    _COMFYUI_FOLDER_PATHS = True
except ImportError:
    _folder_paths = None
    _COMFYUI_FOLDER_PATHS = False

HUNYUAN_FOLDER_NAME = "hunyuan"
"""folder_paths category used by all base-model loaders."""

HUNYUAN_INSTRUCT_FOLDER_NAME = "hunyuan_instruct"
"""folder_paths category used by Instruct-model loaders."""


def _register_hunyuan_model_paths() -> None:
    """Register ``hunyuan`` and ``hunyuan_instruct`` as ComfyUI model folders.

    Both default to ``ComfyUI/models/`` so that models placed there are found
    automatically.  Users can add extra search paths through
    ``extra_model_paths.yaml``::

        comfyui:
            hunyuan: |
                models/
                H:/MyModels/
            hunyuan_instruct: |
                models/
                H:/MyModels/
    """
    if not _COMFYUI_FOLDER_PATHS:
        return
    try:
        _folder_paths.add_model_folder_path(
            HUNYUAN_FOLDER_NAME, _folder_paths.models_dir, is_default=True
        )
        _folder_paths.add_model_folder_path(
            HUNYUAN_INSTRUCT_FOLDER_NAME, _folder_paths.models_dir, is_default=True
        )
    except Exception as exc:
        logger.warning("Could not register Hunyuan model folders: %s", exc)


_register_hunyuan_model_paths()


def get_hunyuan_search_paths(category: str = HUNYUAN_FOLDER_NAME) -> List[str]:
    """Return the list of directories registered for *category*.

    Falls back to ``folder_paths.models_dir`` when the category has not been
    registered (or ComfyUI is not available).
    """
    if not _COMFYUI_FOLDER_PATHS:
        return []
    try:
        return list(_folder_paths.get_folder_paths(category))
    except Exception:
        return [_folder_paths.models_dir]


def resolve_hunyuan_model_path(
    model_name: str,
    category: str = HUNYUAN_FOLDER_NAME,
) -> str:
    """Resolve a display name (or bare folder name) to an absolute path.

    Search order:
    1. Exact match in the ``_path_map`` stashed by the scanner that built the
       dropdown list (handles the ``"Name [location]"`` display names).
    2. Walk every directory registered under *category* looking for a matching
       subfolder.
    3. Treat *model_name* as an absolute path if it already exists on disk.
    4. Return *model_name* unchanged and let the caller error.
    """
    # 1. Scanner path map (populated by get_available_hunyuan_models)
    path_map = getattr(get_available_hunyuan_models, "_path_map", {})
    if model_name in path_map:
        return path_map[model_name]

    # 2. Walk registered search dirs
    for search_dir in get_hunyuan_search_paths(category):
        candidate = os.path.join(search_dir, model_name)
        if os.path.isdir(candidate):
            return candidate

    # 3. Already an absolute path?
    if os.path.isdir(model_name):
        return model_name

    # 4. Last resort
    return model_name


def get_available_hunyuan_models(
    category: str = HUNYUAN_FOLDER_NAME,
    *,
    name_filter=None,
    require_file: Optional[str] = None,
    sort_key=None,
    fallback: Optional[List[str]] = None,
) -> List[str]:
    """Scan registered model directories and return display names.

    Parameters
    ----------
    category:
        The ``folder_paths`` category to scan (``"hunyuan"`` or
        ``"hunyuan_instruct"``).
    name_filter:
        Optional callable ``(folder_name: str) -> bool``.  Only directories
        for which this returns ``True`` are included.
    require_file:
        If given, only directories containing a file with this name are
        included (e.g. ``"quantization_metadata.json"``).
    sort_key:
        Optional key function passed to ``list.sort()``.
    fallback:
        Returned when no matching directories are found.
    """
    found: dict[str, str] = {}  # display_name -> full_path
    models_dir_norm = ""
    if _COMFYUI_FOLDER_PATHS:
        models_dir_norm = os.path.normpath(_folder_paths.models_dir)

    for base_path in get_hunyuan_search_paths(category):
        if not os.path.isdir(base_path):
            continue
        try:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if not os.path.isdir(item_path):
                    continue

                # Apply name filter
                if name_filter and not name_filter(item):
                    continue

                # Apply file-existence filter
                if require_file and not os.path.isfile(
                    os.path.join(item_path, require_file)
                ):
                    continue

                # Build display name
                if os.path.normpath(base_path) == models_dir_norm:
                    display = item
                else:
                    display = f"{item} [{os.path.basename(base_path)}]"

                if display not in found:
                    found[display] = item_path
        except Exception as exc:
            logger.warning("Error scanning %s: %s", base_path, exc)

    # Stash for resolve_hunyuan_model_path()
    get_available_hunyuan_models._path_map = found

    result = sorted(found.keys(), key=sort_key) if sort_key else sorted(found.keys())
    return result if result else (fallback or [])

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
_STATIC_CACHE_LAZY_INIT_PATCHED = False
_DTYPE_HOOKS_INSTALLED = False
_MOE_EFFICIENT_PATCHED = False
_MOE_ORIGINAL_FORWARDS = {}  # module id -> original forward


def _efficient_moe_forward(self, hidden_states):
    """
    Memory-efficient MoE forward that avoids the giant dispatch_mask.
    
    The default 'eager' MoE implementation materializes a dispatch_mask of shape
    [N_tokens, num_experts, expert_capacity] which is O(N² * topk / experts).
    At 3MP with CFG (batch=2, ~25K tokens), this requires 10-37GB just for
    the dispatch_mask and related einsum intermediates.
    
    This efficient version uses a simple loop-based approach:
    1. Use easy_topk to get top-k expert indices and weights (tiny: [N, topk])
    2. For each expert, gather its assigned tokens and run the expert MLP
    3. Scatter-add the weighted outputs back
    
    Memory: O(N * hidden_size) instead of O(N * experts * capacity)
    Speed: Similar — same expert MLPs run on same data, just dispatched differently.
    """
    # NOTE: removed torch.cuda.set_device(hidden_states.device.index) — it's
    # unnecessary (CUDA kernels launch on the tensor's device, not the global
    # default) and mutates thread-local state that could confuse other code.
    bsz, seq_len, hidden_size = hidden_states.shape

    # Shared MLP (if used)
    if self.config.use_mixed_mlp_moe:
        hidden_states_mlp = self.shared_mlp(hidden_states)

    reshaped_input = hidden_states.reshape(-1, hidden_size)  # [N, hidden_size]
    N = reshaped_input.shape[0]

    # Get top-k expert assignments using the lightweight path
    topk_weight, topk_index = self.gate(hidden_states, topk_impl='easy')
    # topk_weight: [N, topk] float32, topk_index: [N, topk] int64

    # Single-token fast path (autoregressive decode, bsz=1, seq_len=1).
    # Inspired by Tencent PR #93: when N == 1 only `topk` experts are needed,
    # so we skip iterating all 64 experts and the per-expert mask test.
    if N == 1:
        token_input = reshaped_input  # [1, hidden_size]
        weights_row = topk_weight[0]  # [topk]
        index_row = topk_index[0]     # [topk]
        out = torch.zeros_like(reshaped_input)
        for slot in range(index_row.shape[0]):
            expert_idx = int(index_row[slot].item())
            w = weights_row[slot]
            expert_out = self.experts[expert_idx](token_input.unsqueeze(0)).squeeze(0)
            out = out + expert_out * w.to(expert_out.dtype)
        combined_output = out.reshape(bsz, seq_len, hidden_size)
        if self.config.use_mixed_mlp_moe:
            return hidden_states_mlp + combined_output
        return combined_output

    # Prepare output buffer
    combined_output = torch.zeros_like(reshaped_input)  # [N, hidden_size]

    # Process each expert: gather assigned tokens, run MLP, scatter-add back
    for expert_idx in range(self.num_experts):
        # Find which (token, topk_slot) pairs route to this expert
        # expert_mask: [N, topk] bool
        expert_mask = (topk_index == expert_idx)

        if not expert_mask.any():
            continue

        # Get token indices that route to this expert (any topk slot)
        # token_mask: [N] bool — True if any topk slot routes to this expert
        token_mask = expert_mask.any(dim=1)
        token_indices = token_mask.nonzero(as_tuple=True)[0]

        if token_indices.numel() == 0:
            continue

        # Gather the input tokens assigned to this expert
        expert_input = reshaped_input[token_indices]  # [n_tokens, hidden_size]

        # Run the expert MLP
        # Expert MLPs expect 3D input [batch, seq, hidden] (they use .chunk(2, dim=2))
        # so we unsqueeze to [1, n_tokens, hidden] and squeeze back after
        expert_output = self.experts[expert_idx](expert_input.unsqueeze(0)).squeeze(0)  # [n_tokens, hidden_size]

        # Compute combined weight for this expert across all topk slots
        # For each selected token, sum the weights from all slots that chose this expert
        expert_weights_for_tokens = (topk_weight[token_indices] * expert_mask[token_indices].float()).sum(dim=1)
        # expert_weights_for_tokens: [n_tokens]

        # Weighted scatter-add back to combined output
        combined_output[token_indices] += expert_output * expert_weights_for_tokens.unsqueeze(1).to(expert_output.dtype)

    combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

    if self.config.use_mixed_mlp_moe:
        output = hidden_states_mlp + combined_output
    else:
        output = combined_output

    return output


def patch_moe_efficient_forward(model) -> None:
    """
    Monkey-patch all HunyuanMoE modules to use memory-efficient forward.
    
    This replaces the dispatch_mask-based MoE routing with a loop-based approach
    that uses ~75x less VRAM at high resolutions (3MP+).
    
    Safe to call multiple times (idempotent). Can be reverted with unpatch_moe_efficient_forward.
    """
    global _MOE_EFFICIENT_PATCHED, _MOE_ORIGINAL_FORWARDS
    
    patched_count = 0
    for name, module in model.named_modules():
        # Match by class name since we can't import the class directly without side effects
        if type(module).__name__ == 'HunyuanMoE':
            module_id = id(module)
            if module_id not in _MOE_ORIGINAL_FORWARDS:
                _MOE_ORIGINAL_FORWARDS[module_id] = module.forward
            # Bind the efficient forward as a bound method
            import types
            module.forward = types.MethodType(_efficient_moe_forward, module)
            patched_count += 1
    
    if patched_count > 0:
        _MOE_EFFICIENT_PATCHED = True
        logger.info(f"Patched {patched_count} HunyuanMoE layers with memory-efficient forward")
    else:
        logger.warning("No HunyuanMoE layers found to patch")


def unpatch_moe_efficient_forward(model) -> None:
    """Restore original MoE forward methods."""
    global _MOE_EFFICIENT_PATCHED, _MOE_ORIGINAL_FORWARDS
    
    restored_count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'HunyuanMoE':
            module_id = id(module)
            if module_id in _MOE_ORIGINAL_FORWARDS:
                module.forward = _MOE_ORIGINAL_FORWARDS[module_id]
                del _MOE_ORIGINAL_FORWARDS[module_id]
                restored_count += 1
    
    if restored_count > 0:
        logger.info(f"Restored {restored_count} HunyuanMoE layers to original forward")
    
    # Safety net: force-clear any orphaned entries (e.g. if model was
    # partially cleaned up and named_modules() missed some).
    # These entries hold bound methods whose __self__ keeps MoE modules
    # and all their expert weight tensors alive in RAM.
    orphaned = len(_MOE_ORIGINAL_FORWARDS)
    if orphaned > 0:
        logger.warning(f"Force-clearing {orphaned} orphaned MoE forward entries "
                       f"from global _MOE_ORIGINAL_FORWARDS dict")
        _MOE_ORIGINAL_FORWARDS.clear()
    
    _MOE_EFFICIENT_PATCHED = False


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

    def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
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
        **kwargs,
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


# =============================================================================
# Patch: Fix upstream to_device() to handle dict inputs
# =============================================================================
# The upstream instruct model's to_device() helper (used in generate_image)
# only handles Tensor and list types. When cond_vit_image_kwargs (a dict
# containing attention_mask tensors) is passed through to_device(), the dict
# falls through to the else branch and is returned unchanged, leaving
# attention_mask tensors on CPU while the model runs on CUDA.
# This causes: "Expected all tensors to be on the same device, but found
# at least two devices, cuda:0 and cpu!" in the SigLIP2 vision encoder.

_TO_DEVICE_PATCHED = False

def patch_to_device_for_instruct(model) -> bool:
    """
    Patch the upstream to_device() function in the instruct model's module
    to handle dict inputs recursively.
    
    The upstream to_device() only handles Tensor and list, but
    cond_vit_image_kwargs is a dict containing tensors. Without this patch,
    the dict passes through unchanged and attention_mask stays on CPU.
    
    Args:
        model: The loaded instruct model (AutoModelForCausalLM)
        
    Returns:
        True if patched successfully, False otherwise
    """
    global _TO_DEVICE_PATCHED
    
    # Find the module that contains to_device
    # It's defined in the model's modeling_hunyuan_image_3.py
    model_module = type(model).__module__
    
    try:
        import sys
        mod = sys.modules.get(model_module)
        if mod is None:
            # Try the parent module
            parent_module = '.'.join(model_module.split('.')[:-1])
            mod = sys.modules.get(parent_module)
        
        if mod is None:
            logger.warning(f"Could not find module {model_module} for to_device patch")
            return False
        
        original_to_device = getattr(mod, 'to_device', None)
        if original_to_device is None:
            logger.warning("to_device function not found in model module")
            return False
        
        # Check if already patched (has dict handling)
        import inspect
        source = inspect.getsource(original_to_device)
        if 'isinstance(data, dict)' in source:
            logger.info("to_device already handles dicts, skipping patch")
            return True
        
        # Create fixed version
        def fixed_to_device(data, device):
            """Fixed to_device that handles dict, Tensor, list, and tuple."""
            if device is None:
                return data
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, dict):
                return {k: fixed_to_device(v, device) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(fixed_to_device(x, device) for x in data)
            else:
                return data
        
        # Apply patch
        mod.to_device = fixed_to_device
        _TO_DEVICE_PATCHED = True
        logger.info("Patched to_device() to handle dict inputs (fixes vision encoder device mismatch)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to patch to_device: {e}")
        return False


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

    def _get_cache_info(cache, idx):
        """Return (dtype, device) of the cache entry at *idx*, or (None, None)."""
        if not hasattr(cache, "__len__"):
            return None, None
        if idx >= len(cache):
            return None, None
        entry = cache[idx]
        if isinstance(entry, torch.Tensor):
            return entry.dtype, entry.device
        if isinstance(entry, (list, tuple)) and entry:
            first = entry[0]
            if isinstance(first, torch.Tensor):
                return first.dtype, first.device
        return None, None

    def _harmonize(obj, target_dtype, target_device):
        """Cast *obj* to *target_dtype* and/or move to *target_device*."""
        if target_dtype is None and target_device is None:
            return obj
        if isinstance(obj, torch.Tensor):
            needs_dtype = target_dtype is not None and obj.dtype != target_dtype
            needs_device = target_device is not None and obj.device != target_device
            if needs_dtype or needs_device:
                kw = {}
                if needs_dtype:
                    kw["dtype"] = target_dtype
                if needs_device:
                    kw["device"] = target_device
                return obj.to(**kw)
            return obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(_harmonize(item, target_dtype, target_device) for item in obj)
        return obj

    def _patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_dtype, key_device = _get_cache_info(self.key_cache, layer_idx) if hasattr(self, "key_cache") else (None, None)
        value_dtype, value_device = _get_cache_info(self.value_cache, layer_idx) if hasattr(self, "value_cache") else (key_dtype, key_device)

        if key_dtype is not None or key_device is not None:
            key_states = _harmonize(key_states, key_dtype, key_device)
        if value_dtype is not None or value_device is not None:
            value_states = _harmonize(value_states, value_dtype, value_device)

        return _ORIGINAL_DYNAMIC_CACHE_UPDATE(
            self,
            key_states,
            value_states,
            layer_idx,
            cache_kwargs=cache_kwargs,
        )

    DynamicCache.update = _patched_update  # type: ignore[assignment]
    _DYNAMIC_CACHE_PATCHED = True
    logger.info("Patched DynamicCache.update to harmonize KV dtype and device")


# ---------------------------------------------------------------------------
# HunyuanStaticCache device harmonization (fixes multi-GPU mismatch)
# ---------------------------------------------------------------------------

def patch_hunyuan_static_cache_device(model) -> bool:
    """Patch ``HunyuanStaticCache.update`` to harmonize device on Path B.

    The model's ``HunyuanStaticCache`` (defined in the checkpoint's
    ``hunyuan.py``) has **two code paths** in its ``update()`` method:

    - **Path A** (old transformers — ``self.key_cache`` list):  Already has a
      device migration guard that moves the cached tensor to ``key_states.device``.
    - **Path B** (new transformers ≥5.0 — ``self.layers[idx].keys``):  **No**
      device migration.  If the lazily-initialised cache tensor ends up on a
      different device than subsequent ``key_states`` (common with
      ``CUDA_VISIBLE_DEVICES`` remapping, accelerate hooks, or multi-GPU
      device maps), the ``index_copy_`` call crashes with:
      ``RuntimeError: Expected all tensors to be on the same device``.

    This function monkey-patches the loaded model's ``HunyuanStaticCache.update``
    to add the missing device migration on Path B, exactly mirroring Path A.

    Must be called **after** the model is loaded (the class is dynamically
    imported via ``trust_remote_code=True``).

    Args:
        model: The loaded ``HunyuanImage3ForCausalMM`` model instance.

    Returns:
        ``True`` if the patch was applied, ``False`` otherwise.
    """
    # Find the HunyuanStaticCache class from the model's module
    cache_cls = None
    try:
        # The model's module is registered under its config's auto_map
        model_module = type(model).__module__
        import importlib
        mod = importlib.import_module(model_module)
        cache_cls = getattr(mod, "HunyuanStaticCache", None)
    except Exception:
        pass

    if cache_cls is None:
        # Fallback: walk the model's class hierarchy
        try:
            import sys
            for mod_name, mod in sys.modules.items():
                if mod is not None and hasattr(mod, "HunyuanStaticCache"):
                    cache_cls = getattr(mod, "HunyuanStaticCache")
                    break
        except Exception:
            pass

    if cache_cls is None:
        logger.warning("HunyuanStaticCache not found; device harmonization patch skipped")
        return False

    # Check if already patched
    if getattr(cache_cls, "_device_patch_applied", False):
        return True

    _original_update = cache_cls.update

    def _device_patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Wrapper that adds device migration for the StaticLayer (Path B) code path."""
        # Path B: self.layers[layer_idx].keys (transformers >=5.0 with StaticLayer)
        if not (hasattr(self, "key_cache") and hasattr(self, "value_cache")):
            # This is Path B — check if cache is initialised and on wrong device
            if hasattr(self, "layers") and layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                if getattr(layer, "keys", None) is not None:
                    if layer.keys.device != key_states.device:
                        layer.keys = layer.keys.to(key_states.device)
                        layer.values = layer.values.to(value_states.device)

        return _original_update(self, key_states, value_states, layer_idx, cache_kwargs)

    cache_cls.update = _device_patched_update
    cache_cls._device_patch_applied = True
    logger.info("Patched HunyuanStaticCache.update with device harmonization (multi-GPU fix)")
    return True


def patch_model_device_meta_safe(model) -> bool:
    """Override ``model.device`` to skip parameters/buffers stuck on ``meta``.

    When loading with ``device_map="cpu"`` + ``low_cpu_mem_usage=True``,
    accelerate uses ``init_empty_weights`` to materialize the model on the
    ``meta`` device and then loads the checkpoint over it. Buffers that are
    not present in the checkpoint (e.g. registered ``inv_freq`` / RoPE caches
    that the model lazily fills) can remain on ``meta``. The default
    ``PreTrainedModel.device`` property returns ``next(self.parameters()).device``,
    which in that case can return ``meta`` and crash upstream code such as::

        File "modeling_hunyuan_image_3.py", line 2762, in prepare_model_inputs
            generator = [torch.Generator(self.device).manual_seed(seed) ...]
        RuntimeError: META device type not an accelerator.

    This patch installs a ``device`` property on the model's class that walks
    parameters/buffers and returns the first non-``meta`` device, falling
    back to ``cuda:0`` (or ``cpu``) if every tensor is on ``meta``. Safe for
    block-swap setups where parts of the model live on different devices.
    """
    cls = type(model)
    if getattr(cls, "_eric_meta_safe_device", False):
        return True

    @property
    def _meta_safe_device(self):
        for p in self.parameters(recurse=True):
            if p.device.type != "meta":
                return p.device
        for b in self.buffers(recurse=True):
            if b.device.type != "meta":
                return b.device
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    try:
        cls.device = _meta_safe_device
        cls._eric_meta_safe_device = True
        logger.info("Patched %s.device to skip meta-device tensors", cls.__name__)
        return True
    except (TypeError, AttributeError) as exc:
        logger.warning("Could not install meta-safe device property on %s: %s", cls.__name__, exc)
        return False


def strip_accelerate_hooks_from_vae(model) -> int:
    """Remove all accelerate hooks from the VAE submodule tree.

    The VAE is moved to a single GPU by ``_move_non_block_components_to_gpu``
    and never spans devices, so accelerate's dispatch hooks add no value.
    They DO add a fragility: when hooks are installed/removed across
    runs (e.g. between image-edit invocations) the underlying
    ``module._old_forward`` chain can get corrupted, surfacing as::

        RuntimeError: super(): __class__ is not a type (NoneType)

    inside the custom ``Conv3d`` subclass in ``autoencoder_kl_3d.py``
    (its ``forward`` uses zero-arg ``super()`` which needs an intact
    ``__class__`` closure cell).

    This function walks the VAE recursively and:
      1. Calls ``accelerate.hooks.remove_hook_from_module`` on every module
         carrying an ``_hf_hook``.
      2. Deletes any leftover instance-level ``forward`` attribute so the
         class method is used (preserves ``__class__`` for ``super()``).

    Returns the number of modules cleaned. Safe to call repeatedly
    (idempotent: re-running has no further effect).
    """
    vae = getattr(model, "vae", None)
    if vae is None:
        return 0

    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        remove_hook_from_module = None

    cleaned = 0
    for _name, module in vae.named_modules():
        had_hook = hasattr(module, "_hf_hook")
        if had_hook and remove_hook_from_module is not None:
            try:
                remove_hook_from_module(module, recurse=False)
                cleaned += 1
            except (AttributeError, RuntimeError):
                pass
        # Drop any instance-level forward override left behind so the class
        # method (with its intact __class__ closure) is used.
        if "forward" in vars(module):
            try:
                delattr(module, "forward")
            except (AttributeError, TypeError):
                pass
        if "_old_forward" in vars(module):
            try:
                delattr(module, "_old_forward")
            except (AttributeError, TypeError):
                pass

    if cleaned:
        logger.info("Stripped accelerate hooks from %d VAE submodule(s)", cleaned)
    return cleaned


def patch_static_cache_lazy_init() -> None:
    """Patch ``StaticLayer.lazy_initialization`` for transformers >=5.0 compat.

    **The problem**:  The HunyuanImage-3 model's ``HunyuanStaticCache.update()``
    calls ``self.layers[layer_idx].lazy_initialization(key_states)`` with a
    single positional argument.  In transformers <=4.57 the ``StaticLayer``
    signature was ``lazy_initialization(self, key_states)`` — one positional arg
    was correct.  Starting with transformers 5.0 the signature changed to
    ``lazy_initialization(self, key_states, value_states)`` (two positional
    args), so the model's single-arg call crashes with a ``TypeError``.

    **The fix**:  We monkey-patch ``StaticLayer.lazy_initialization`` to accept
    ``value_states`` as an **optional** second argument.  When it is omitted
    (the model's call path) we derive ``v_head_dim`` from ``key_states``,
    which is correct for HunyuanImage-3 because its key and value heads have
    the same dimension.

    The patch is only applied when transformers >=5.0 is detected and is
    completely harmless for <=4.57 (it will simply not be invoked).
    """
    global _STATIC_CACHE_LAZY_INIT_PATCHED

    if _STATIC_CACHE_LAZY_INIT_PATCHED:
        return

    if not _TRANSFORMERS_GTE_5:
        # On transformers <=4.57 the old 1-arg signature is still in place —
        # nothing to patch.
        return

    try:
        from transformers.cache_utils import StaticLayer
    except ImportError:
        logger.warning("transformers.cache_utils.StaticLayer unavailable; "
                       "lazy_init compat patch skipped")
        return

    _original_lazy_init = StaticLayer.lazy_initialization

    # Check how many positional params the current implementation expects
    # (excluding ``self``).  If it already accepts 1, no patch needed.
    sig = inspect.signature(_original_lazy_init)
    # Parameters excluding 'self'
    params = [
        p for name, p in sig.parameters.items()
        if name != "self"
    ]
    positional_params = [
        p for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if len(positional_params) < 2:
        # Already accepts 1 required positional arg — nothing to do.
        _STATIC_CACHE_LAZY_INIT_PATCHED = True
        return

    # transformers >=5.0: lazy_initialization(self, key_states, value_states)
    # Make value_states optional so the model's 1-arg call works.
    def _compat_lazy_initialization(self, key_states, value_states=None):
        if value_states is None:
            # The model only passes key_states.  For HunyuanImage-3 the
            # key and value head dims are identical (head_dim = 128), so
            # we simply duplicate key_states for shape inference.
            value_states = key_states
        return _original_lazy_init(self, key_states, value_states)

    StaticLayer.lazy_initialization = _compat_lazy_initialization  # type: ignore[assignment]
    _STATIC_CACHE_LAZY_INIT_PATCHED = True
    logger.info(
        "Patched StaticLayer.lazy_initialization for transformers >=5.0 compat "
        "(value_states now optional, derived from key_states when omitted)"
    )


def apply_nf4_transformers_compat(model) -> None:
    """Apply runtime compatibility shims for NF4 models on transformers >=5.0.

    Two known issues require workarounds (issues #24, #27, #34):

    1. ``bitsandbytes`` ``Params4bit`` quantization state is not auto-initialized
       by transformers >=5.0's pre-quantized loader, leading to::

           AssertionError: assert module.weight.shape[1] == 1

       inside ``fix_4bit_weight_quant_state_from_module``.  Fix: walk every
       ``Linear4bit`` module and call its ``.cuda(...)`` to materialize the
       quant state from the on-disk packed storage.

    2. The bundled ``image_processor.vit_process_image`` does
       ``inputs["pixel_values"].squeeze(0)`` but in newer transformers
       ``BaseImageProcessor.__call__`` may return ``pixel_values`` as a list.
       Fix: monkey-patch the bound method to coerce list -> stacked tensor
       before the squeeze.

    Both shims are no-ops on transformers <5.0 or when the issue is not
    present.  Safe to call multiple times.
    """
    if not _TRANSFORMERS_GTE_5:
        return

    # ---- 1. Materialize NF4 quant_state on Params4bit weights ----
    try:
        from bitsandbytes.nn import Linear4bit, Params4bit
    except ImportError:
        Linear4bit = None  # type: ignore[assignment]
        Params4bit = None  # type: ignore[assignment]

    if Linear4bit is not None and Params4bit is not None and torch.cuda.is_available():
        fixed = 0
        for module in model.modules():
            if not isinstance(module, Linear4bit):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, Params4bit):
                continue
            quant_state = getattr(weight, "quant_state", None)
            needs_init = (
                quant_state is None
                or weight.data.dim() != 2
                or weight.data.shape[1] != 1
            )
            if not needs_init:
                continue
            try:
                target = weight.data.device if weight.data.is_cuda else torch.device("cuda")
                module.cuda(target)
                fixed += 1
            except Exception as exc:
                logger.debug("apply_nf4_transformers_compat: %s init failed: %s",
                             type(module).__name__, exc)
        if fixed:
            logger.info(
                "Initialized quant_state on %d NF4 layers (transformers >=5.0 compat)",
                fixed,
            )

    # ---- 2. Patch image_processor.vit_process_image (issue #34) ----
    image_processor = getattr(model, "image_processor", None)
    vit_process_image = getattr(image_processor, "vit_process_image", None)
    if image_processor is not None and callable(vit_process_image) and \
            not getattr(vit_process_image, "_hunyuan_t5_compat_patched", False):
        import types

        original_vit = vit_process_image

        def _compat_vit_process_image(self, *args, **kwargs):
            # Run the original first; if it succeeds, we're done.
            try:
                return original_vit(*args, **kwargs)
            except AttributeError as exc:
                if "'list' object has no attribute 'squeeze'" not in str(exc):
                    raise

            # Replay the call by patching the underlying processor temporarily
            # to coerce list -> stacked tensor.
            inner = getattr(self, "vit_processor", None) or \
                getattr(self, "processor", None) or \
                getattr(self, "image_processor", None)
            if inner is None or not hasattr(inner, "__call__"):
                raise

            inner_call = inner.__call__

            def _coerced_call(*a, **kw):
                kw.setdefault("return_tensors", "pt")
                out = inner_call(*a, **kw)
                pv = out.get("pixel_values") if hasattr(out, "get") else \
                    getattr(out, "pixel_values", None)
                if isinstance(pv, list):
                    pv = torch.stack([
                        t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                        for t in pv
                    ], dim=0)
                    if hasattr(out, "__setitem__"):
                        out["pixel_values"] = pv
                    else:
                        out.pixel_values = pv
                return out

            try:
                inner.__call__ = _coerced_call  # type: ignore[assignment]
                return original_vit(*args, **kwargs)
            finally:
                try:
                    del inner.__call__
                except Exception:
                    inner.__call__ = inner_call  # type: ignore[assignment]

        # Set the marker on the function, not the bound method: MethodType objects
        # reject attribute assignment, so marking `bound` raises AttributeError and
        # kills every Instruct load on transformers >=5.0 (issue #48). getattr on
        # the bound method falls through to the function, so the guard still sees it.
        _compat_vit_process_image._hunyuan_t5_compat_patched = True  # type: ignore[attr-defined]
        bound = types.MethodType(_compat_vit_process_image, image_processor)
        image_processor.vit_process_image = bound
        logger.info(
            "Patched image_processor.vit_process_image for transformers >=5.0 "
            "(coerces pixel_values list -> tensor, issue #34)"
        )


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
    patch_static_cache_lazy_init()

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

                # Skip model.cpu() — the tensor gutting below handles everything.
                # model.cpu() would create ~50GB of new CRT heap allocations
                # (via block.to('cpu')) that the CRT never returns to the OS,
                # causing permanent RSS bloat. The tensor gutting that follows
                # replaces every param.data with empty(0), which frees both
                # GPU and CPU storage directly.
                logger.info("  Skipping model.cpu() — tensor gutting handles memory release")
                
                # ---- Remove external monkey-patches on submodules ----
                # Other custom nodes (e.g. seedvr2_videoupscaler) may patch
                # methods on our model's submodules with closures that
                # capture the model, keeping the tree alive after unload.
                import types as _types
                ext_broken = 0
                try:
                    for _, submod in model_to_clear.named_modules():
                        for attr_name in list(vars(submod).keys()):
                            attr = vars(submod).get(attr_name)
                            if attr is None:
                                continue
                            closure = None
                            if isinstance(attr, _types.FunctionType):
                                closure = attr.__closure__
                            elif isinstance(attr, _types.MethodType):
                                closure = getattr(
                                    attr.__func__, '__closure__', None)
                            if closure:
                                import ctypes as _ct
                                for cell in closure:
                                    try:
                                        _val = cell.cell_contents
                                    except ValueError:
                                        continue
                                    # CRITICAL: never nuke __class__ cells.
                                    # Python stores the enclosing class in a
                                    # closure cell for zero-arg super(); the
                                    # cell lives on the class-level function
                                    # object and is shared by ALL instances.
                                    # Nuking it permanently breaks super()
                                    # for the entire class (including future
                                    # instances loaded via sys.modules).
                                    if isinstance(_val, type):
                                        continue
                                    try:
                                        _ct.pythonapi.PyCell_Set(
                                            _ct.py_object(cell),
                                            _ct.py_object(None))
                                        ext_broken += 1
                                    except Exception:
                                        pass
                                try:
                                    delattr(submod, attr_name)
                                except Exception:
                                    pass
                except Exception:
                    pass
                if ext_broken:
                    logger.info("  Broke %d external monkey-patch closure "
                                "cells on submodules", ext_broken)
                
                # ---- Break closure references that keep the model alive ----
                # patch_hunyuan_generate_image() binds generate_image via
                # MethodType and patches pipeline_class.__call__.  These
                # closures hold strong refs to model, preventing gc.
                # RESTORE originals instead of breaking cells to avoid
                # corrupting the class for future loads.
                unpatch_hunyuan_generate_image(model_to_clear)
                
                # Unpatch MoE efficient forward — releases the global
                # _MOE_ORIGINAL_FORWARDS dict which holds bound methods
                # keeping all MoE expert weights alive in RAM
                unpatch_moe_efficient_forward(model_to_clear)
                
                # Reset dtype hooks flag so they get reinstalled on next load
                global _DTYPE_HOOKS_INSTALLED
                _DTYPE_HOOKS_INSTALLED = False
                
                # Unpatch VAE decode closure
                unpatch_pipeline_pre_vae_cleanup(model_to_clear)
                # ---- End closure unpatching ----
                
                # ---- Nuclear gc closure scan ----
                # Walk ALL gc objects; break any closure cell that still
                # references this model or one of its submodules.
                _cell_sentinel = None
                cell_type = type((lambda: _cell_sentinel).__closure__[0])
                gc.collect()
                # Pre-build set of module ids for O(1) lookup
                model_module_ids = set()
                try:
                    for _, _sub in model_to_clear.named_modules():
                        model_module_ids.add(id(_sub))
                except Exception:
                    pass
                model_module_ids.add(id(model_to_clear))
                nuked = 0
                for _obj in gc.get_objects():
                    if type(_obj) is not cell_type:
                        continue
                    try:
                        _val = _obj.cell_contents
                    except ValueError:
                        continue
                    # CRITICAL: skip class objects (types) — Python stores
                    # __class__ in a closure cell for super(); breaking it
                    # destroys super() for the entire class permanently.
                    if isinstance(_val, torch.nn.Module) and not isinstance(_val, type) and id(_val) in model_module_ids:
                        import ctypes as _ct
                        _ct.pythonapi.PyCell_Set(
                            _ct.py_object(_obj), _ct.py_object(None))
                        nuked += 1
                del model_module_ids
                if nuked:
                    logger.info(f"  Nuclear closure scan broke {nuked} cells")
                    gc.collect()
                # ---- End nuclear gc closure scan ----
                
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
            
            # Also clear the Instruct model cache if it exists
            try:
                try:
                    from .hunyuan_instruct_nodes import _instruct_cache
                except ImportError:
                    from hunyuan_instruct_nodes import _instruct_cache
                if _instruct_cache.model is not None:
                    _instruct_cache.clear()
                    logger.info("Also cleared Instruct model cache")
            except ImportError:
                pass
            except Exception:
                pass
        
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
        
        # Force Windows to return freed memory to OS
        if force_clear or clear_for_downstream:
            force_windows_memory_release()
        
        # Return a signal that changes to force downstream nodes to re-evaluate if needed
        return (cleared, vram_status, float(time.time()))


class HunyuanImage3SoftUnload:
    """
    Fast VRAM release - moves model to CPU RAM instead of deleting.
    
    NOTE: With the new `post_action` dropdown in Generate nodes, you may not
    need this node at all. The Generate nodes can now soft_unload automatically
    after generation. This standalone node is kept for advanced workflows.
    
    ✅ WORKS WITH: (requires bitsandbytes >= 0.48.2)
    - INT8 quantized models
    - NF4 quantized models  
    - BF16 models loaded entirely on GPU
    
    ⚠️ Does NOT work with:
    - Models loaded with device_map offloading (has meta tensors)
    - Use offload_mode='disabled' in loader to enable soft unload
    
    BENEFITS:
    - Soft unload + restore: ~10-30 seconds (scales with model size)
    - Full unload + reload from disk: ~2+ minutes
    
    ACTIONS:
    - soft_unload: Move model from GPU to CPU RAM, free VRAM
    - check_status: Report current model location
    - restore_to_gpu: (Deprecated) Loader now handles this automatically
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["soft_unload", "check_status", "restore_to_gpu"], {
                    "default": "soft_unload",
                    "tooltip": "soft_unload: Move to CPU and free VRAM. check_status: Report location. restore_to_gpu: (deprecated, loader handles this)"
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
            # Deprecated: Loader now handles this automatically
            logger.warning("restore_to_gpu is deprecated - the Loader node now automatically restores CPU-cached models")
            if HunyuanModelCache._cached_model is None:
                status = "No model cached - use loader node (it will restore automatically)"
                success = False
            elif not HunyuanModelCache.is_on_cpu():
                status = "Model already on GPU"
                success = True
            else:
                success = HunyuanModelCache.restore_to_gpu()
                if success:
                    status = "Model restored to GPU (note: loader does this automatically now)"
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
                    "default": False,
                    "tooltip": "Hunt and destroy orphaned Hunyuan CUDA tensors. "
                               "Safe for downstream models (scoped to Hunyuan only). "
                               "Use after OOM when VRAM is stuck."
                }),
                "nuke_ram_tensors": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hunt and destroy orphaned Hunyuan CPU tensors in RAM. "
                               "Safe for downstream models (scoped to Hunyuan only). "
                               "Use when system RAM is full of leaked model weights."
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
        
        # Step 1: Clear Hunyuan model cache (base AND Instruct)
        if clear_all_models:
            try:
                cleared = HunyuanModelCache.clear()
                report_lines.append(f"Hunyuan base cache cleared: {cleared}")
            except Exception as e:
                report_lines.append(f"Failed to clear Hunyuan base cache: {e}")
            
            # Also clear the Instruct model cache if it exists
            try:
                try:
                    from .hunyuan_instruct_nodes import _instruct_cache
                except ImportError:
                    from hunyuan_instruct_nodes import _instruct_cache
                if _instruct_cache.model is not None:
                    _instruct_cache.clear()
                    report_lines.append("Hunyuan Instruct cache cleared")
            except ImportError:
                pass  # Instruct nodes not installed
            except Exception as e:
                report_lines.append(f"Failed to clear Instruct cache: {e}")
        
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
                # Build a set of Hunyuan model IDs — ONLY these get gutted.
                # Everything else (Marigold, segmentation, ComfyUI internals)
                # must be left untouched.
                _hunyuan_ids = set()
                if HunyuanModelCache._cached_model is not None:
                    try:
                        for _, sub in HunyuanModelCache._cached_model.named_modules():
                            _hunyuan_ids.add(id(sub))
                    except Exception:
                        pass
                    _hunyuan_ids.add(id(HunyuanModelCache._cached_model))
                try:
                    try:
                        from .hunyuan_instruct_nodes import _instruct_cache
                    except ImportError:
                        from hunyuan_instruct_nodes import _instruct_cache
                    if _instruct_cache.model is not None:
                        try:
                            for _, sub in _instruct_cache.model.named_modules():
                                _hunyuan_ids.add(id(sub))
                        except Exception:
                            pass
                        _hunyuan_ids.add(id(_instruct_cache.model))
                except ImportError:
                    pass
                
                for obj in all_objects:
                    try:
                        if isinstance(obj, torch.nn.Module):
                            if id(obj) not in _hunyuan_ids:
                                continue  # NOT a Hunyuan module — leave it alone
                            modules_found += 1
                            for param in obj.parameters():
                                if param.device.type == 'cuda' and nuke_orphaned_tensors:
                                    ram_freed_estimate += param.numel() * param.element_size()
                                    param.data = torch.empty(0, device='cpu')
                                    tensors_cleared += 1
                                elif param.device.type == 'cpu' and nuke_ram_tensors:
                                    ram_freed_estimate += param.numel() * param.element_size()
                                    param.data = torch.empty(0)
                                    tensors_cleared += 1
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
                
                # Second pass: Find raw tensors (only if explicitly enabled)
                # NOTE: Raw tensor scan cannot be scoped to Hunyuan-only,
                # so only run it when the user knowingly enables it.
                if nuke_orphaned_tensors or nuke_ram_tensors:
                    for obj in all_objects:
                        try:
                            if isinstance(obj, torch.Tensor) and not isinstance(obj, torch.nn.Parameter):
                                if obj.device.type == 'cuda':
                                    cuda_tensors_found += 1
                                elif obj.device.type == 'cpu':
                                    cpu_tensors_found += 1
                                # Don't gut raw tensors — we can't tell who owns them
                                # and gutting them destroys downstream models
                        except Exception:
                            pass
                
                del _hunyuan_ids
                
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
        
        # Force Windows to return freed memory to OS
        force_windows_memory_release()
        
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

        # Some model variants (newer Instruct/Distil v2) only expose ``generate``
        # while older ones expose ``_generate`` for the cached/streaming path.
        # Pick whichever is available so the patch works across versions.
        _gen = getattr(self, "_generate", None) or self.generate

        # Extract callback_on_step_end so we can pass it ONLY to the final image gen step
        callback_on_step_end = kwargs.pop("callback_on_step_end", None)
        latents = kwargs.pop("latents", None)  # LATENT CONTROL: extract custom latents

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
            outputs = _gen(**model_inputs, **kwargs, verbose=verbose)
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
            outputs = _gen(**model_inputs, **kwargs, verbose=verbose)
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
        if latents is not None:  # LATENT CONTROL: pass through custom latents
            gen_kwargs["latents"] = latents
            
        # PASS callback_on_step_end here
        outputs = _gen(**model_inputs, **gen_kwargs, verbose=verbose)
        return outputs[0]

    logger.info("Patching model.generate_image to support progress bars...")
    # Save original generate_image so we can restore it on unload
    if not hasattr(model, '_comfy_original_generate_image'):
        model._comfy_original_generate_image = getattr(model, 'generate_image', None)
    model.generate_image = types.MethodType(new_generate_image, model)

    # Also patch the pipeline class to extract callback_on_step_end from model_kwargs
    # This is needed because the cached _generate method puts everything into model_kwargs
    if hasattr(model, 'pipeline'):
        pipeline_class = model.pipeline.__class__
        if not getattr(pipeline_class, '_is_patched_for_comfy', False):
            logger.info("Patching HunyuanImage3Text2ImagePipeline to handle callback_on_step_end...")
            original_call = pipeline_class.__call__
            # Save the TRUE original on the class so we can restore it on unload.
            # Only save once — if _comfy_original_call already exists, a previous
            # unload failed to restore, so keep the earlier (real) original.
            if not hasattr(pipeline_class, '_comfy_original_call'):
                pipeline_class._comfy_original_call = original_call
            
            def new_pipeline_call(self, *args, **kwargs):
                model_kwargs = kwargs.get('model_kwargs')
                if model_kwargs and isinstance(model_kwargs, dict):
                    if 'callback_on_step_end' in model_kwargs:
                        cb = model_kwargs.pop('callback_on_step_end')
                        # Only set it if not already provided explicitly
                        if kwargs.get('callback_on_step_end') is None:
                            kwargs['callback_on_step_end'] = cb
                    # LATENT CONTROL: extract latents from model_kwargs → top-level pipeline kwarg
                    if 'latents' in model_kwargs:
                        lat = model_kwargs.pop('latents')
                        if kwargs.get('latents') is None:
                            kwargs['latents'] = lat
                return original_call(self, *args, **kwargs)
            
            pipeline_class.__call__ = new_pipeline_call
            pipeline_class._is_patched_for_comfy = True

    # Apply HunyuanStaticCache device harmonization (multi-GPU fix)
    patch_hunyuan_static_cache_device(model)


def unpatch_hunyuan_generate_image(model):
    """
    Cleanly reverse all patches applied by patch_hunyuan_generate_image().
    
    This RESTORES original methods instead of just breaking closure cells,
    which is critical for allowing re-patching on the next model load.
    Breaking closure cells leaves broken wrappers in place that corrupt
    subsequent loads.
    """
    # 1. Restore original generate_image on the model instance
    if hasattr(model, '_comfy_original_generate_image'):
        original = model._comfy_original_generate_image
        if original is not None:
            model.generate_image = original
            logger.info("Restored original generate_image method")
        else:
            # Original was None (model didn't have generate_image before our patch)
            try:
                del model.generate_image
            except (AttributeError, Exception):
                pass
            logger.info("Removed patched generate_image (no original existed)")
        try:
            del model._comfy_original_generate_image
        except (AttributeError, Exception):
            pass
    elif hasattr(model, 'generate_image'):
        # No saved original, but we have a patched generate_image.
        # Just delete the instance-level override to fall back to class method.
        try:
            del model.generate_image
            logger.info("Removed generate_image instance override (no saved original)")
        except (AttributeError, Exception):
            pass
    
    # 2. Restore original pipeline class __call__
    pipeline = getattr(model, 'pipeline', None)
    if pipeline is not None:
        pipeline_class = pipeline.__class__
        if getattr(pipeline_class, '_is_patched_for_comfy', False):
            if hasattr(pipeline_class, '_comfy_original_call'):
                pipeline_class.__call__ = pipeline_class._comfy_original_call
                del pipeline_class._comfy_original_call
                logger.info("Restored original pipeline __call__")
            else:
                # No saved original — just log warning, don't break cells
                logger.warning("Pipeline patched but no _comfy_original_call saved")
            pipeline_class._is_patched_for_comfy = False


def clear_generation_cache(model) -> None:
    """
    Clear KV cache and intermediate state after generation to free VRAM.

    This addresses the issue where subsequent generations use more VRAM than the first,
    because transformers keeps the KV cache (past_key_values) between calls.

    Call this after each generation to ensure consistent VRAM usage.
    """
    import gc

    cleared_items = []

    # Clear past_key_values / KV cache
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
        cleared_items.append("past_key_values")

    # Clear any cached hidden states
    if hasattr(model, '_cache'):
        model._cache = None
        cleared_items.append("_cache")

    # Clear model's internal cache if it has one
    if hasattr(model, 'model'):
        inner_model = model.model
        if hasattr(inner_model, '_cache'):
            inner_model._cache = None
            cleared_items.append("model._cache")
        if hasattr(inner_model, 'past_key_values'):
            inner_model.past_key_values = None
            cleared_items.append("model.past_key_values")

    # Clear any generation_config cache
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, '_cache'):
        model.generation_config._cache = None
        cleared_items.append("generation_config._cache")

    # For transformer-based models, clear any layer-wise caches
    for name, module in model.named_modules():
        # Clear attention caches
        if hasattr(module, 'key_value_cache'):
            module.key_value_cache = None
            cleared_items.append(f"{name}.key_value_cache")
        if hasattr(module, '_past_key_value'):
            module._past_key_value = None
            cleared_items.append(f"{name}._past_key_value")

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Log memory state
        free, total = torch.cuda.mem_get_info(0)
        logger.info(
            f"Post-generation cleanup: Cleared {len(cleared_items)} cache items. "
            f"VRAM: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free"
        )
        if cleared_items:
            logger.debug(
                f"  Cleared: {', '.join(cleared_items[:5])}"
                f"{'...' if len(cleared_items) > 5 else ''}"
            )
    else:
        if cleared_items:
            logger.info(f"Cleared {len(cleared_items)} cache items")


def patch_pipeline_pre_vae_cleanup(model, enabled: bool = True):
    """
    Patch the VAE decode method to clear KV cache BEFORE decoding.

    This addresses the issue where INT8/quantized models run out of memory during
    VAE decode because the transformer's KV cache is still consuming VRAM.

    The patch wraps the VAE's decode method to:
    1. Clear KV cache, gc collect, empty CUDA cache
    2. Then proceed with VAE decode

    Args:
        model: The HunyuanImage3 model
        enabled: Whether to enable the patch
    """
    if not enabled:
        return

    # Try to get pipeline, but some models don't have one
    pipeline = getattr(model, 'pipeline', None)

    # Find the VAE - some models have it directly on model, others on pipeline
    vae = None
    if pipeline is not None:
        vae = getattr(pipeline, 'vae', None)
    if vae is None:
        vae = getattr(model, 'vae', None)

    if vae is None:
        logger.warning("No VAE found on model or pipeline, cannot patch pre-VAE cleanup")
        return

    # Check if already patched - use VAE itself as marker
    if getattr(vae, '_prevae_cleanup_patched', False):
        logger.info("✓ VAE already patched for pre-decode cleanup")
        return

    # Store original decode
    original_decode = vae.decode

    def decode_with_cleanup(*args, **kwargs):
        """Wrapper that clears KV cache before VAE decode to free VRAM.

        The transformer's KV cache can consume 1-4 GB depending on resolution.
        Clearing it before VAE decode ensures enough VRAM for the decode step.

        IMPORTANT: We do NOT move the VAE between devices by default. Moving
        modules with .to(cpu)/.to(cuda) breaks accelerate's dispatch hooks
        that were set up by device_map='auto' during model loading, causing
        'tensors on different devices' errors on subsequent runs.  When the
        user explicitly opts into ``vae_offload="on"`` or block-swap is
        active (no accelerate hooks), offload is safe.
        """
        import gc

        tiling_mode = getattr(model, "_vae_tiling_mode", "auto")
        offload_mode = getattr(model, "_vae_offload_mode", "auto")

        logger.info(
            f"PRE-VAE DECODE: Clearing KV cache before decode "
            f"(tiling={tiling_mode}, offload={offload_mode})"
        )

        # Clear transformer caches to free VRAM for decode
        clear_generation_cache(model)

        # Release any swapped transformer blocks back to CPU BEFORE we measure
        # VRAM and decide on tiling.  Without this, block-swap retains all
        # blocks on GPU after generation, masking how much VRAM is available
        # for the VAE decode (issue #22).
        block_swap = getattr(model, "_block_swap_manager", None) or \
            getattr(model, "block_swap_manager", None)
        if block_swap is not None and hasattr(block_swap, "release_all_blocks"):
            try:
                block_swap.release_all_blocks()
                logger.info("  Released all swap blocks back to CPU before VAE decode")
            except Exception as exc:
                logger.warning(f"  Could not release swap blocks: {exc}")

        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            free, total = torch.cuda.mem_get_info(0)
            logger.info(f"  VRAM after cleanup: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

        # Decide on offload
        offloaded = False
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            free_gb = free / 1024**3
            should_offload = (
                offload_mode == "on"
                or (offload_mode == "auto" and free_gb < 8.0
                    and block_swap is not None)  # only auto-offload when block-swap is active
            )
            if should_offload and hasattr(vae, "to"):
                try:
                    target_device = next(vae.parameters()).device
                    if target_device.type == "cuda":
                        vae.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                        vae.to(target_device)
                    offloaded = True
                except Exception as exc:
                    logger.debug(f"  VAE offload skipped: {exc}")

        # Decide on tiling
        if torch.cuda.is_available() and hasattr(vae, "enable_tiling"):
            free, _ = torch.cuda.mem_get_info(0)
            free_gb = free / 1024**3
            should_tile = (
                tiling_mode == "on"
                or (tiling_mode == "auto" and free_gb < 15.0)
            )
            if should_tile:
                logger.info(
                    f"  Enabling VAE tiling (mode={tiling_mode}, free={free_gb:.1f}GB)"
                )
                try:
                    vae.enable_tiling()
                except Exception as exc:
                    logger.debug(f"  enable_tiling failed: {exc}")
            elif tiling_mode == "off" and hasattr(vae, "disable_tiling"):
                try:
                    vae.disable_tiling()
                except Exception:
                    pass

        # Call original decode — VAE stays on its original device
        result = original_decode(*args, **kwargs)

        return result

    # Replace decode method
    vae.decode = decode_with_cleanup
    vae._prevae_cleanup_patched = True
    vae._original_decode = original_decode

    logger.info("✓ Patched VAE decode: will clear KV cache before decode to free VRAM")


def unpatch_pipeline_pre_vae_cleanup(model):
    """Remove the pre-VAE cleanup patch and restore original decode method."""
    # Find the VAE
    pipeline = getattr(model, 'pipeline', None)
    vae = None
    if pipeline is not None:
        vae = getattr(pipeline, 'vae', None)
    if vae is None:
        vae = getattr(model, 'vae', None)

    if vae is None:
        return

    if not getattr(vae, '_prevae_cleanup_patched', False):
        return

    if hasattr(vae, '_original_decode'):
        vae.decode = vae._original_decode
        del vae._original_decode

    vae._prevae_cleanup_patched = False
    logger.info("Removed pre-VAE cleanup patch")


# ---------------------------------------------------------------------------
# Windows memory release utility
# ---------------------------------------------------------------------------

def force_windows_memory_release():
    """Force Windows to return freed memory from the process working set.
    
    On Windows, Python's allocator and PyTorch's _aligned_malloc/_aligned_free
    do NOT return freed memory pages to the OS.  They stay committed in the
    process working set, making RSS/Task-Manager look inflated even though
    the memory is logically free inside the CRT heap.
    
    This function aggressively tries every available mechanism:
    1. gc.collect() — break Python reference cycles
    2. msvcrt._heapmin() — return CRT-heap free blocks to the OS
    3. GetProcessHeaps + HeapCompact — compact ALL heaps (not just default)
    4. EmptyWorkingSet — nuclear: evict ALL pages from working set
    5. SetProcessWorkingSetSizeEx(-1, -1) — trim working set to minimum
    
    Call this AFTER all model cleanup (cache.clear, del model, gc.collect,
    torch.cuda.empty_cache) to reduce the reported RSS.
    """
    import gc
    import sys
    
    # Triple gc to break nested cycles
    gc.collect()
    gc.collect()
    gc.collect()
    
    if sys.platform != 'win32':
        return
    
    import ctypes
    import ctypes.wintypes as wintypes
    
    ram_before = None
    try:
        ram_before = psutil.Process().memory_info().rss / 1024**3
    except Exception:
        pass
    
    kernel32 = ctypes.windll.kernel32
    
    # Flush PyTorch's pinned memory cache FIRST — these pages are allocated
    # via cudaHostAlloc, not the CRT heap, so _heapmin/HeapCompact can't touch them.
    if torch.cuda.is_available():
        try:
            torch._C._cuda_CachingHostAllocator_emptyCache()
            logger.info("  Flushed PyTorch CachingHostAllocator (pinned memory)")
        except (AttributeError, RuntimeError):
            pass
        
    # 1. Ask CRT heap to return free blocks to the OS
    try:
        ctypes.cdll.msvcrt._heapmin()
    except Exception:
        pass
    
    # 2. Compact ALL process heaps (not just default)
    #    PyTorch and other C libraries may create private heaps.
    try:
        GetProcessHeaps = kernel32.GetProcessHeaps
        GetProcessHeaps.restype = wintypes.DWORD
        HeapCompact = kernel32.HeapCompact
        HeapCompact.restype = ctypes.c_size_t
        
        # First call to get count
        num_heaps = GetProcessHeaps(0, None)
        if num_heaps > 0:
            heap_array = (wintypes.HANDLE * num_heaps)()
            got = GetProcessHeaps(num_heaps, heap_array)
            compacted_total = 0
            for i in range(got):
                try:
                    result = HeapCompact(heap_array[i], 0)
                    compacted_total += result
                except Exception:
                    pass
            if compacted_total > 0:
                logger.debug(f"  HeapCompact across {got} heaps: "
                             f"largest free block sum = "
                             f"{compacted_total / 1024**2:.1f} MB")
    except Exception:
        # Fallback: just compact default heap
        try:
            heap = kernel32.GetProcessHeap()
            kernel32.HeapCompact(heap, 0)
        except Exception:
            pass
    
    # 3. EmptyWorkingSet — aggressively evicts ALL pages from working set.
    #    Pages that are truly unused will go to the standby list and the
    #    OS can reclaim them.  Pages that are still accessed will be
    #    soft-faulted back in quickly.
    try:
        psapi = ctypes.windll.psapi
        handle = kernel32.GetCurrentProcess()
        psapi.EmptyWorkingSet(handle)
    except Exception:
        try:
            # EmptyWorkingSet might be in kernel32 on newer Windows
            handle = kernel32.GetCurrentProcess()
            kernel32.EmptyWorkingSet(handle)
        except Exception:
            pass
    
    # 4. SetProcessWorkingSetSizeEx with (-1, -1) — trim to minimum
    try:
        handle = kernel32.GetCurrentProcess()
        SIZE_T = ctypes.c_size_t
        kernel32.SetProcessWorkingSetSizeEx(
            handle,
            SIZE_T(ctypes.c_size_t(-1).value),
            SIZE_T(ctypes.c_size_t(-1).value),
            ctypes.c_uint32(0),
        )
    except Exception:
        pass
    
    if ram_before is not None:
        try:
            ram_after = psutil.Process().memory_info().rss / 1024**3
            freed = ram_before - ram_after
            if freed > 0.1:
                logger.info(f"Windows memory release: freed {freed:.1f}GB "
                           f"(RSS {ram_before:.1f}GB -> {ram_after:.1f}GB)")
            else:
                logger.debug(f"Windows memory release: RSS {ram_after:.1f}GB "
                            f"(no significant change)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RAM Diagnostic node
# ---------------------------------------------------------------------------

class HunyuanRAMDiagnostic:
    """
    Diagnostic node that reports detailed memory state.
    
    Use this to troubleshoot RAM leaks.  It scans all Python objects to find
    orphaned nn.Module instances and their parameter footprints.  With
    trace_referrers enabled, it also shows WHO is holding each large module
    alive (the reference chain), which is the key to finding the actual leak.
    
    Run this node BEFORE and AFTER running a cleanup/unload node to see
    what changed.
    """
    
    CATEGORY = "Hunyuan/Debug"
    FUNCTION = "diagnose"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trace_referrers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use gc.get_referrers() to trace WHY the largest "
                               "modules are still alive. SLOW but reveals the leak source."}),
                "compact_windows_heap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After reporting, call Windows APIs to return freed "
                               "memory to the OS. Only affects reported RSS, not "
                               "actual leaks."}),
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-run so the report is fresh."""
        return float("NaN")
    
    def diagnose(self, trace_referrers: bool = False,
                 compact_windows_heap: bool = True):
        import gc
        import sys
        import torch.nn as nn
        from collections import defaultdict
        
        lines = []
        proc = psutil.Process()
        rss = proc.memory_info().rss / 1024**3
        vm = psutil.virtual_memory()
        lines.append("=" * 60)
        lines.append("          HUNYUAN RAM DIAGNOSTIC REPORT")
        lines.append("=" * 60)
        lines.append(f"Process RSS:   {rss:.2f} GB")
        lines.append(f"System RAM:    {vm.used / 1024**3:.1f}GB used / "
                     f"{vm.total / 1024**3:.1f}GB total ({vm.percent}%)")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(i)
                alloc_b = torch.cuda.memory_allocated(i)
                lines.append(f"GPU {i}:         {alloc_b/1024**3:.1f}GB alloc, "
                             f"{free_b/1024**3:.1f}GB free / "
                             f"{total_b/1024**3:.1f}GB total")
        lines.append("")
        
        # ------------------------------------------------------------------
        # Scan all gc objects for nn.Module and raw Tensor instances
        # ------------------------------------------------------------------
        gc.collect()
        all_objects = gc.get_objects()
        
        module_types = defaultdict(lambda: {"count": 0, "cpu_bytes": 0,
                                            "cuda_bytes": 0})
        total_cpu_bytes = 0
        total_cuda_bytes = 0
        module_count = 0
        tensor_count = 0
        tensor_cpu_bytes = 0
        tensor_cuda_bytes = 0
        
        # For referrer tracing: keep references to the biggest modules
        largest_modules = []  # (bytes, type_name, obj_or_None)
        
        for obj in all_objects:
            try:
                if isinstance(obj, nn.Module):
                    module_count += 1
                    type_name = type(obj).__qualname__
                    mod_cpu = 0
                    mod_cuda = 0
                    # recurse=False avoids double-counting child modules
                    for p in obj.parameters(recurse=False):
                        nb = p.numel() * p.element_size()
                        if p.device.type == "cpu":
                            mod_cpu += nb
                        else:
                            mod_cuda += nb
                    for b in obj.buffers(recurse=False):
                        nb = b.numel() * b.element_size()
                        if b.device.type == "cpu":
                            mod_cpu += nb
                        else:
                            mod_cuda += nb
                    module_types[type_name]["count"] += 1
                    module_types[type_name]["cpu_bytes"] += mod_cpu
                    module_types[type_name]["cuda_bytes"] += mod_cuda
                    total_cpu_bytes += mod_cpu
                    total_cuda_bytes += mod_cuda
                    mod_total = mod_cpu + mod_cuda
                    if mod_total > 0:
                        largest_modules.append(
                            (mod_total, type_name,
                             obj if trace_referrers else None))
                elif isinstance(obj, torch.Tensor) and not isinstance(obj, nn.Parameter):
                    tensor_count += 1
                    nb = obj.numel() * obj.element_size()
                    if obj.device.type == "cpu":
                        tensor_cpu_bytes += nb
                    else:
                        tensor_cuda_bytes += nb
            except Exception:
                continue
        
        del all_objects  # release the huge list ASAP
        
        lines.append(f"nn.Module instances alive:  {module_count}")
        lines.append(f"  CPU parameter bytes:      {total_cpu_bytes / 1024**3:.2f} GB")
        lines.append(f"  CUDA parameter bytes:     {total_cuda_bytes / 1024**3:.2f} GB")
        lines.append(f"Raw Tensors (non-Param):    {tensor_count}")
        lines.append(f"  CPU tensor bytes:         {tensor_cpu_bytes / 1024**3:.2f} GB")
        lines.append(f"  CUDA tensor bytes:        {tensor_cuda_bytes / 1024**3:.2f} GB")
        lines.append("")
        
        # Top-10 module types by parameter size
        sorted_types = sorted(
            module_types.items(),
            key=lambda x: x[1]["cpu_bytes"] + x[1]["cuda_bytes"],
            reverse=True,
        )[:15]
        lines.append("Top module types by parameter size:")
        lines.append(f"  {'Type':<45} {'Count':>5}  {'CPU GB':>8}  {'CUDA GB':>8}")
        lines.append("  " + "-" * 70)
        for type_name, info in sorted_types:
            cpu_gb = info["cpu_bytes"] / 1024**3
            cuda_gb = info["cuda_bytes"] / 1024**3
            if cpu_gb + cuda_gb < 0.001:
                continue
            # Truncate long names
            short = type_name[-44:] if len(type_name) > 44 else type_name
            lines.append(f"  {short:<45} {info['count']:>5}  "
                         f"{cpu_gb:>8.3f}  {cuda_gb:>8.3f}")
        lines.append("")
        
        # ------------------------------------------------------------------
        # Comprehensive tensor STORAGE scan
        #
        # The module-parameter scan above only counts parameters owned by
        # live nn.Module objects.  But RSS can be much higher if there are:
        #   1. Tensor storages shared between multiple tensors (views)
        #   2. Oversized storages (tensor is a slice of a big storage)
        #   3. Tensors held by dicts/lists/closures not attached to modules
        #   4. Memory-mapped file pages still mapped
        #
        # This section deduplicates by storage data_ptr() to give the
        # TRUE memory footprint, and categorises by owner type.
        # ------------------------------------------------------------------
        lines.append("--- Tensor Storage Analysis (unique storages) ---")
        gc.collect()
        seen_ptrs = {}  # data_ptr -> (nbytes, device_type, owner_desc)
        all_objs = gc.get_objects()
        
        for obj in all_objs:
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
                storage = obj.untyped_storage()
                dptr = storage.data_ptr()
                nbytes = storage.nbytes()
                if nbytes == 0:
                    continue
                dev = str(obj.device)
                if dptr not in seen_ptrs or nbytes > seen_ptrs[dptr][0]:
                    # Classify the tensor's owner
                    if isinstance(obj, nn.Parameter):
                        owner = "nn.Parameter"
                    else:
                        owner = "Tensor"
                    seen_ptrs[dptr] = (nbytes, dev, owner)
            except Exception:
                continue
        
        # Also scan for mmap-backed file handles
        mmap_count = 0
        mmap_bytes = 0
        try:
            import mmap as _mmap_mod
            for obj in all_objs:
                if isinstance(obj, _mmap_mod.mmap):
                    mmap_count += 1
                    try:
                        mmap_bytes += obj.size()
                    except Exception:
                        pass
        except Exception:
            pass
        
        del all_objs
        
        # Aggregate by device
        storage_summary = {}  # device -> (count, total_bytes)
        for dptr, (nbytes, dev, owner) in seen_ptrs.items():
            if dev not in storage_summary:
                storage_summary[dev] = [0, 0]
            storage_summary[dev][0] += 1
            storage_summary[dev][1] += nbytes
        
        total_storage_gb = sum(v[1] for v in storage_summary.values()) / 1024**3
        lines.append(f"  Unique tensor storages: {len(seen_ptrs)} "
                     f"({total_storage_gb:.2f} GB total)")
        for dev in sorted(storage_summary.keys()):
            cnt, nbytes = storage_summary[dev]
            lines.append(f"    {dev}: {cnt} storages, "
                         f"{nbytes / 1024**3:.2f} GB")
        
        if mmap_count > 0:
            lines.append(f"  Memory-mapped files: {mmap_count} "
                         f"({mmap_bytes / 1024**3:.2f} GB)")
        
        # Show top-10 largest individual storages
        top_storages = sorted(seen_ptrs.values(),
                              key=lambda x: x[0], reverse=True)[:10]
        if top_storages:
            lines.append("  Top-10 largest tensor storages:")
            for i, (nbytes, dev, owner) in enumerate(top_storages):
                lines.append(f"    [{i}] {nbytes / 1024**3:.3f} GB on "
                             f"{dev} ({owner})")
        
        # Gap analysis: RSS vs tracked memory
        tracked_gb = total_storage_gb + (mmap_bytes / 1024**3)
        gap_gb = rss - tracked_gb
        lines.append(f"\n  RSS: {rss:.2f} GB | Tracked: {tracked_gb:.2f} GB | "
                     f"Gap: {gap_gb:.2f} GB")
        if gap_gb > 5.0:
            lines.append(f"  ⚠ {gap_gb:.1f}GB unaccounted: likely Python heap, "
                         f"mmap pages, or C-level allocations")
        
        del seen_ptrs
        lines.append("")
        
        # ------------------------------------------------------------------
        # Reference tracing for the largest individual modules
        # ------------------------------------------------------------------
        if trace_referrers and largest_modules:
            largest_modules.sort(reverse=True, key=lambda x: x[0])
            
            # Extract just the objects into a separate list to avoid our
            # own tuples showing up as referrers
            trace_targets = [(m[0], m[1], m[2]) for m in largest_modules[:5]
                             if m[2] is not None]
            # Now clear our tuples so they don't pollute get_referrers()
            largest_modules.clear()
            
            lines.append("=" * 60)
            lines.append("  REFERRER TRACE (top-5 largest modules)")
            lines.append("=" * 60)
            
            for mod_bytes, type_name, obj in trace_targets:
                lines.append(f"\n  {type_name} ({mod_bytes / 1024**3:.2f} GB):")
                if obj is None:
                    lines.append("    (skipped — trace_referrers was False)")
                    continue
                try:
                    referrers = gc.get_referrers(obj)
                    lines.append(f"    {len(referrers)} referrers:")
                    for i, ref in enumerate(referrers[:10]):
                        ref_type = type(ref).__name__
                        if isinstance(ref, dict):
                            keys = [k for k, v in ref.items()
                                    if v is obj][:3]
                            # Who owns this dict?
                            owners = gc.get_referrers(ref)
                            owner_types = []
                            for o in owners[:4]:
                                if o is not ref and not isinstance(o, list):
                                    oname = type(o).__qualname__
                                    owner_types.append(oname)
                            lines.append(
                                f"    [{i}] dict(keys={keys}) "
                                f"owned by: {owner_types}")
                            del owners
                        elif ref_type == 'cell':
                            # A closure cell — find which function owns it
                            cell_id = id(ref)
                            owner_func = "(unknown)"
                            cell_val_type = "(unreadable)"
                            try:
                                cell_val = ref.cell_contents
                                cell_val_type = type(cell_val).__qualname__
                            except ValueError:
                                cell_val_type = "(empty cell)"
                            except Exception:
                                pass
                            try:
                                # Search gc objects for function owning this cell
                                # Check plain functions AND MethodType.__func__
                                import types as _types
                                for scan_obj in gc.get_objects():
                                    closure = None
                                    if isinstance(scan_obj, _types.FunctionType):
                                        closure = scan_obj.__closure__
                                    elif isinstance(scan_obj, _types.MethodType):
                                        func = scan_obj.__func__
                                        if hasattr(func, '__closure__'):
                                            closure = func.__closure__
                                    elif hasattr(scan_obj, '__wrapped__'):
                                        # functools.wraps etc
                                        w = scan_obj.__wrapped__
                                        if hasattr(w, '__closure__'):
                                            closure = w.__closure__
                                    if closure and any(
                                            id(c) == cell_id for c in closure):
                                        qn = getattr(scan_obj, '__qualname__',
                                                      getattr(scan_obj, '__name__', '?'))
                                        mod = getattr(scan_obj, '__module__', '?')
                                        owner_func = f"{mod}.{qn}"
                                        break
                            except Exception:
                                pass
                            lines.append(
                                f"    [{i}] CLOSURE CELL id={cell_id:#x} "
                                f"contains={cell_val_type} "
                                f"owner: {owner_func}")
                        elif isinstance(ref, (list, tuple)):
                            idx = next(
                                (j for j, item in enumerate(ref)
                                 if item is obj), "?")
                            lines.append(
                                f"    [{i}] {ref_type}[{idx}] "
                                f"(len={len(ref)})")
                        elif isinstance(ref, nn.Module):
                            lines.append(
                                f"    [{i}] {type(ref).__qualname__} "
                                f"(parent module)")
                        else:
                            r = repr(ref)[:120]
                            lines.append(
                                f"    [{i}] {ref_type}: {r}")
                    del referrers
                except Exception as e:
                    lines.append(f"    Error tracing: {e}")
            # Clean up trace targets
            del trace_targets
            lines.append("")
        
        # Clear our own references to the objects
        largest_modules.clear()
        del largest_modules
        
        # ------------------------------------------------------------------
        # Closure cell scan: find ALL cells holding nn.Module references
        # This directly identifies the leak source regardless of cell owner
        # ------------------------------------------------------------------
        if trace_referrers:
            lines.append("=" * 60)
            lines.append("  ALL CLOSURE CELLS HOLDING nn.Module OBJECTS")
            lines.append("=" * 60)
            import types as _types
            gc.collect()
            _cell_sentinel_diag = None
            cell_type = type((lambda: _cell_sentinel_diag).__closure__[0])
            cell_count = 0
            for obj in gc.get_objects():
                if type(obj) is not cell_type:
                    continue
                try:
                    val = obj.cell_contents
                except ValueError:
                    continue
                if not isinstance(val, nn.Module):
                    continue
                cell_count += 1
                cell_id = id(obj)
                val_type = type(val).__qualname__
                # Count parameters in this module
                try:
                    param_bytes = sum(
                        p.numel() * p.element_size()
                        for p in val.parameters())
                    param_gb = param_bytes / 1024**3
                except Exception:
                    param_gb = 0
                
                # Find owner function
                owner = "(orphaned cell — no function owns it)"
                for scan_obj in gc.get_objects():
                    closure = None
                    if isinstance(scan_obj, _types.FunctionType):
                        closure = scan_obj.__closure__
                    elif isinstance(scan_obj, _types.MethodType):
                        func = scan_obj.__func__
                        if hasattr(func, '__closure__'):
                            closure = func.__closure__
                    if closure and any(id(c) == cell_id for c in closure):
                        qn = getattr(scan_obj, '__qualname__',
                                     getattr(scan_obj, '__name__', '?'))
                        mod = getattr(scan_obj, '__module__', '?')
                        owner = f"{mod}.{qn}"
                        break
                
                lines.append(f"  Cell {cell_id:#x}: contains {val_type} "
                             f"({param_gb:.2f}GB), owner: {owner}")
                
                # If owner is orphaned, find what keeps the CELL alive
                if "orphaned" in owner:
                    cell_referrers = gc.get_referrers(obj)
                    for ci, cref in enumerate(cell_referrers[:5]):
                        ct = type(cref).__name__
                        if isinstance(cref, tuple):
                            # A tuple of cells = a __closure__ tuple
                            # Find who owns THAT tuple
                            tuple_refs = gc.get_referrers(cref)
                            tuple_owners = []
                            for tr in tuple_refs[:5]:
                                if tr is not cref:
                                    tn = type(tr).__qualname__
                                    if hasattr(tr, '__qualname__'):
                                        tn = f"{tr.__module__}.{tr.__qualname__}"
                                    tuple_owners.append(tn)
                            del tuple_refs
                            lines.append(f"    cell-ref[{ci}]: tuple "
                                         f"(closure tuple) owners: {tuple_owners}")
                        else:
                            lines.append(f"    cell-ref[{ci}]: {ct}")
                    del cell_referrers
            
            if cell_count == 0:
                lines.append("  No closure cells holding nn.Module objects found")
            lines.append("")
        
        # ------------------------------------------------------------------
        # ComfyUI model management inspection
        # ------------------------------------------------------------------
        lines.append("--- ComfyUI Model Management ---")
        try:
            import comfy.model_management as mm
            models = mm.current_loaded_models
            lines.append(f"ModelPatcher loaded models: {len(models)}")
            for m in models:
                try:
                    mem_gb = m.model_memory() / 1024**3
                    lines.append(f"  {type(m.model).__name__}: {mem_gb:.2f}GB")
                except Exception:
                    lines.append(f"  {type(m).__name__}: (size unknown)")
        except Exception as e:
            lines.append(f"  Could not inspect: {e}")
        
        # Our custom caches
        lines.append("")
        lines.append("--- Hunyuan Custom Caches ---")
        try:
            if HunyuanModelCache._cached_model is not None:
                m = HunyuanModelCache._cached_model
                lines.append(f"Base cache: {type(m).__name__} "
                             f"(refcount={sys.getrefcount(m)})")
            else:
                lines.append("Base cache: empty")
        except Exception:
            lines.append("Base cache: (error)")
        
        try:
            from .hunyuan_instruct_nodes import _instruct_cache
            if _instruct_cache.model is not None:
                m = _instruct_cache.model
                lines.append(f"Instruct cache: {type(m).__name__} "
                             f"(refcount={sys.getrefcount(m)})")
            else:
                lines.append("Instruct cache: empty")
        except Exception:
            try:
                from hunyuan_instruct_nodes import _instruct_cache
                if _instruct_cache.model is not None:
                    m = _instruct_cache.model
                    lines.append(f"Instruct cache: {type(m).__name__} "
                                 f"(refcount={sys.getrefcount(m)})")
                else:
                    lines.append("Instruct cache: empty")
            except Exception:
                lines.append("Instruct cache: (not loaded)")
        
        try:
            from .hunyuan_cache_v2 import get_cache
            v2_status = get_cache().get_status()
            if v2_status.get("cached"):
                lines.append(f"V2 cache: {v2_status['quant_type']} model "
                             f"(on_gpu={v2_status['is_on_gpu']})")
            else:
                lines.append("V2 cache: empty")
        except Exception:
            try:
                from hunyuan_cache_v2 import get_cache
                v2_status = get_cache().get_status()
                if v2_status.get("cached"):
                    lines.append(f"V2 cache: {v2_status['quant_type']} model "
                                 f"(on_gpu={v2_status['is_on_gpu']})")
                else:
                    lines.append("V2 cache: empty")
            except Exception:
                lines.append("V2 cache: (not loaded)")
        
        lines.append("")
        
        # ------------------------------------------------------------------
        # Open file handles — mmap'd safetensors files keep pages in RSS
        # ------------------------------------------------------------------
        lines.append("--- Open File Handles (model-related) ---")
        try:
            open_files = proc.open_files()
            model_files = [f for f in open_files
                           if any(ext in f.path.lower()
                                  for ext in ('.safetensors', '.bin', '.pt',
                                              '.pth', '.gguf', '.onnx'))]
            if model_files:
                lines.append(f"  {len(model_files)} model file(s) still open:")
                for f in model_files[:20]:
                    lines.append(f"    {f.path}")
            else:
                lines.append("  No model files open")
            lines.append(f"  Total open files: {len(open_files)}")
        except Exception as e:
            lines.append(f"  Could not inspect: {e}")
        lines.append("")
        
        # ------------------------------------------------------------------
        # Optional: Compact Windows heap
        # ------------------------------------------------------------------
        if compact_windows_heap:
            rss_before = proc.memory_info().rss / 1024**3
            force_windows_memory_release()
            rss_after = proc.memory_info().rss / 1024**3
            freed = rss_before - rss_after
            lines.append(f"Windows heap compact + EmptyWorkingSet: "
                         f"RSS {rss_before:.2f}GB -> {rss_after:.2f}GB "
                         f"(freed {freed:.2f}GB)")
            if freed < 0.5:
                lines.append("  -> Minimal change. The committed virtual memory "
                             "is held by the C runtime heap (CRT) due to")
                lines.append("     heap fragmentation from large PyTorch "
                             "allocations during generation.")
                lines.append("     This memory CANNOT be returned to the OS "
                             "without restarting the process.")
                lines.append("     It IS reusable by the process for the next "
                             "model load (no new allocation needed).")
            else:
                lines.append("  -> Pages evicted from working set. RSS dropped "
                             "but commit charge may still be high.")
                lines.append("     The OS can repurpose the physical RAM for "
                             "other processes.")
        
        lines.append("")
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        # Print to ComfyUI console as well
        for line in lines:
            logger.info(line)
        
        return (report,)
