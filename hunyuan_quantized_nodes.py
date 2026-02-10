"""
HunyuanImage-3.0 ComfyUI Custom Nodes - NF4 Quantized Loaders & Generation
NF4 quantized loading and advanced image generation with prompt enhancement

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

import json
import math
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import folder_paths
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map
from comfy.utils import ProgressBar

from .hunyuan_shared import (
    HUNYUAN_FOLDER_NAME,
    HunyuanImage3Unload,
    HunyuanImage3ForceUnload,
    HunyuanImage3SoftUnload,
    HunyuanImage3ClearDownstream,
    HunyuanModelCache,
    MemoryTracker,
    ensure_model_on_device,
    format_memory_stats,
    get_available_hunyuan_models,
    get_hunyuan_search_paths,
    patch_dynamic_cache_dtype,
    patch_hunyuan_generate_image,
    resolve_hunyuan_model_path,
)
from .hunyuan_api_config import get_api_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _force_model_device_property(model, device: torch.device) -> None:
    """Ensure model.device returns a usable accelerator even when accelerate parks params on meta."""
    cls = model.__class__
    original_prop = getattr(cls, "device", None)
    if isinstance(original_prop, property) and not getattr(cls, "_hunyuan_device_prop_patched", False):
        def _device_getter(self):  # type: ignore[override]
            forced = getattr(self, "_forced_device", None)
            if forced is not None:
                return forced
            return original_prop.fget(self) if original_prop.fget else torch.device("cpu")

        cls.device = property(_device_getter)  # type: ignore[assignment]
        cls._hunyuan_device_prop_patched = True  # type: ignore[attr-defined]

    model._forced_device = device  # type: ignore[attr-defined]


def _is_model_int8_quantized(model) -> bool:
    """Check if a model was loaded with INT8 quantization."""
    # Check for INT8 Linear layers from bitsandbytes
    try:
        from bitsandbytes.nn import Linear8bitLt
        for module in model.modules():
            if isinstance(module, Linear8bitLt):
                return True
    except ImportError:
        pass
    
    # Check model metadata/attributes
    if hasattr(model, '_hunyuan_quantization_type'):
        return 'int8' in str(model._hunyuan_quantization_type).lower()
    
    # Check config for quantization hints
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'quantization_config'):
            qconfig = config.quantization_config
            if hasattr(qconfig, 'load_in_8bit') and qconfig.load_in_8bit:
                return True
    
    return False


def _get_gpu_info() -> dict:
    """Get GPU name and VRAM info."""
    info = {
        'name': 'Unknown',
        'total_vram_gb': 0.0,
        'free_vram_gb': 0.0,
        'is_blackwell': False,
        'is_rtx_6000': False,
    }
    
    if not torch.cuda.is_available():
        return info
    
    try:
        info['name'] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        info['total_vram_gb'] = total / 1024**3
        info['free_vram_gb'] = free / 1024**3
        
        # Detect specific GPUs
        name_lower = info['name'].lower()
        info['is_blackwell'] = 'blackwell' in name_lower or 'b200' in name_lower or '6000' in name_lower
        info['is_rtx_6000'] = '6000' in name_lower or 'rtx 6000' in name_lower
    except Exception as e:
        logger.debug(f"Could not get GPU info: {e}")
    
    return info


def _calculate_resolution_megapixels(resolution_str: str) -> float:
    """Calculate megapixels from a resolution string like '1920x1080' or 'Auto (model default)'."""
    if 'auto' in resolution_str.lower():
        # Auto mode can choose up to ~2.5MP, assume worst case
        return 2.5
    
    try:
        # Extract WxH from strings like "1920x1080 - Landscape 1080p (2.1MP)"
        parts = resolution_str.split(' - ')[0].strip()
        if 'x' in parts:
            w, h = map(int, parts.split('x'))
            return (w * h) / 1_000_000
    except Exception:
        pass
    
    return 1.0  # Default fallback


def _estimate_inference_vram_gb(megapixels: float) -> float:
    """
    Estimate VRAM required for inference based on resolution.
    
    Based on real-world measurements and MoE memory analysis:
    - 1MP: ~12GB (attention + KV cache + small MoE intermediates)
    - 2MP: ~25GB (MoE dispatch_mask starts growing significantly)
    - 3MP: ~65GB (MoE dispatch_mask with CFG doubles tokens: 25K tokens,
                   dispatch_mask [25K, 64, 3136] = 10GB bf16, plus combine_weights
                   and einsum intermediates = ~30-37GB peak per layer)
    
    The MoE "eager" implementation materializes a dispatch_mask of shape
    [N_tokens, num_experts, expert_capacity] which grows as O(N² * topk / experts).
    With CFG (guidance_scale > 1), batch=2 doubles N.
    """
    # Below 2MP, the original estimate works well
    if megapixels <= 2.0:
        base_vram = 12.0
        return base_vram * (megapixels ** 1.4)
    
    # Above 2MP, MoE dispatch_mask dominates memory
    # Empirical: 3MP with CFG needs ~65GB free (model weights 30GB + 65GB = 95GB total)
    # Formula: accounts for quadratic growth of MoE intermediates
    base_vram = 12.0
    attention_vram = base_vram * (megapixels ** 1.4)
    
    # MoE dispatch_mask memory (dominant at high res):
    # N_tokens ≈ 2 * (MP * 1e6 / 256) for CFG batch=2, VAE downsample 16x16
    # dispatch_mask bf16: N * 64 * (8*N/64) * 2 bytes = N² * 16 bytes  
    # Plus combine_weights, einsum intermediates: ~3x dispatch_mask
    n_tokens = 2 * megapixels * 1_000_000 / 256  # CFG doubles batch
    moe_dispatch_gb = (n_tokens ** 2 * 16 * 3) / (1024 ** 3)
    
    return max(attention_vram, moe_dispatch_gb)


@lru_cache(maxsize=1)
def _resolution_metadata():
    base_size = 1024
    min_multiple = 0.5
    max_multiple = 2.0
    step = None
    align = 1
    presets = None

    candidate_dirs = []
    default_path = resolve_hunyuan_model_path("HunyuanImage-3")
    default_dir = Path(default_path)
    if default_dir.exists():
        candidate_dirs.append(default_dir)
    # Also scan registered search paths so externally-stored models are found
    for search_root in get_hunyuan_search_paths():
        search_root_path = Path(search_root)
        if not search_root_path.is_dir():
            continue
        for item in search_root_path.iterdir():
            if item.is_dir() and item not in candidate_dirs:
                candidate_dirs.append(item)

    for directory in candidate_dirs:
        config_path = directory / "config.json"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
                base_size = int(config_data.get("image_base_size", base_size))
                min_multiple = float(config_data.get("image_min_multiple", min_multiple))
                max_multiple = float(config_data.get("image_max_multiple", max_multiple))
                step_val = config_data.get("image_resolution_step", step)
                align_val = config_data.get("image_resolution_align", align)
                step = int(step_val) if step_val is not None else step
                align = int(align_val) if align_val else align
                config_presets = config_data.get("image_resolution_presets")
                if config_presets:
                    presets = config_presets
            except Exception:
                logger.debug("Failed to parse %s", config_path, exc_info=True)

        gen_path = directory / "generation_config.json"
        if presets is None and gen_path.exists():
            try:
                with gen_path.open("r", encoding="utf-8") as handle:
                    gen_data = json.load(handle)
                presets = gen_data.get("image_size_presets")
            except Exception:
                logger.debug("Failed to parse %s", gen_path, exc_info=True)

        if presets is not None and config_path.exists():
            break

    count = 33
    if 'config_data' in locals():
        count = int(config_data.get("image_resolution_count", count))

    return {
        "base_size": base_size,
        "min_multiple": min_multiple,
        "max_multiple": max_multiple,
        "step": step,
        "align": align,
        "presets": presets,
        "count": count,
    }


class HunyuanImage3QuantizedLoader:
    """Loads NF4-quantized HunyuanImage-3.0 model onto GPU."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls._get_available_models(),),
                "force_reload": ("BOOLEAN", {"default": False}),
                "reserve_memory_gb": ("FLOAT", {
                    "default": 6.0, 
                    "min": 2.0, 
                    "max": 80.0, 
                    "step": 0.5,
                    "tooltip": "VRAM to leave free. Standard: 6GB. Large images (>2MP) can eat ~15GB/MP (use Large Generate node to offload)."
                }),
            },
            "optional": {
                "unload_signal": ("*", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    @classmethod
    def IS_CHANGED(cls, model_name, force_reload=False, unload_signal=None, **kwargs):
        if force_reload:
            return float("nan")
        return model_name
    
    @classmethod
    def _get_available_models(cls):
        available = get_available_hunyuan_models(
            require_file="quantization_metadata.json",
            sort_key=lambda x: 0 if "nf4" in x.lower() else 1,
            fallback=["HunyuanImage-3-NF4"],
        )
        return available
    
    def load_model(self, model_name, force_reload=False, unload_signal=None, reserve_memory_gb=6.0):
        # force_reload: if True, always reload model even if cached
        # unload_signal: forces re-execution if model was cleared (changes on each unload)
        model_path = Path(resolve_hunyuan_model_path(model_name))
        model_path_str = str(model_path)

        # If force_reload is True, skip cache and reload fresh
        if force_reload:
            logger.info("force_reload=True, clearing cache and reloading model...")
            HunyuanModelCache.clear()
            cached = None
        else:
            cached = HunyuanModelCache.get(model_path_str)
        
        if cached is not None:
            try:
                # Validate cached model is in a usable state
                if not hasattr(cached, 'generate_image'):
                    logger.warning("Cached model is invalid (missing generate_image), clearing cache and reloading...")
                    HunyuanModelCache.clear()
                    cached = None
                else:
                    # Check if model is soft-unloaded to CPU (fast restore path)
                    if HunyuanModelCache.is_on_cpu():
                        logger.info("Cached model is on CPU, restoring to GPU (fast path)...")
                        if HunyuanModelCache.restore_to_gpu():
                            logger.info("✓ Model restored from CPU to GPU")
                            self._apply_dtype_patches()
                            patch_hunyuan_generate_image(cached)
                            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
                            return (cached,)
                        else:
                            # Restore failed, need to reload from disk
                            logger.warning("Failed to restore model from CPU, reloading from disk...")
                            HunyuanModelCache.clear()
                            cached = None
                    else:
                        # Model should be on GPU - validate device placement
                        has_valid_device = False
                        try:
                            # Check a model parameter to see if it has a valid device
                            for param in cached.parameters():
                                if param.device.type == 'cuda' and param.device.index is not None:
                                    has_valid_device = True
                                    break
                                elif param.device.type == 'cpu':
                                    # Model on CPU but not tracked as soft-unloaded - inconsistent state
                                    logger.warning("Cached model on CPU but not tracked as soft-unloaded, clearing...")
                                    HunyuanModelCache.clear()
                                    cached = None
                                    break
                        except Exception:
                            has_valid_device = False
                        
                        if not has_valid_device and cached is not None:
                            logger.warning("Cached model has invalid device placement, clearing cache and reloading...")
                            HunyuanModelCache.clear()
                            cached = None
                        elif cached is not None:
                            logger.info("Using cached model from previous load")
                            self._apply_dtype_patches()
                            patch_hunyuan_generate_image(cached)
                            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
                            return (cached,)
            except Exception as e:
                logger.warning(f"Failed to validate cached model: {e}. Clearing cache and reloading...")
                HunyuanModelCache.clear()
                cached = None

        # Clear CUDA cache to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Report VRAM before loading
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            logger.info(f"VRAM before load: {(total-free)/1024**3:.2f}GB used / {total/1024**3:.2f}GB total ({free/1024**3:.2f}GB free)")
        
        logger.info(f"Loading {model_name}")
        logger.info("Creating NF4 quantization config (attention layers in full precision)...")

        skip_modules = [
            "vae",
            "model.vae",
            "vae.decoder",
            "vae.encoder",
            "autoencoder",
            "model.autoencoder",
            "autoencoder.decoder",
            "autoencoder.encoder",
            "patch_embed",
            "model.patch_embed",
            "final_layer",
            "model.final_layer",
            "time_embed",
            "model.time_embed",
            "time_embed_2",
            "model.time_embed_2",
            "timestep_emb",
            "model.timestep_emb",
            # Critical: exclude attention projections to prevent corruption
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "attn.qkv_proj",
            "attn.out_proj",
            "self_attn",
            "cross_attn",
        ]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Critical: use bfloat16 for stable attention
            modules_to_not_convert=skip_modules,
            llm_int8_skip_modules=skip_modules,
        )

        logger.info("Loading model with selective quantization (VAE kept in full precision)...")

        model = None
        try:
            # Calculate max memory to use (leave some headroom for generation)
            max_memory_dict = None
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                # Reserve specified amount for generation overhead and leave 10% headroom
                reserve_gb = float(reserve_memory_gb)
                headroom = 0.10
                max_gpu_memory = int((total / 1024**3 - reserve_gb) * (1 - headroom) * 1024**3)
                max_memory_dict = {0: max_gpu_memory, "cpu": "100GiB"}
                logger.info(f"Setting max GPU memory to {max_gpu_memory/1024**3:.1f}GB (reserving {reserve_gb}GB)")
            
            load_kwargs = dict(
                quantization_config=quant_config,
                device_map="auto" if max_memory_dict else "cuda:0",
                max_memory=max_memory_dict,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    **load_kwargs,
                )
            except TypeError as exc:
                logger.warning("Falling back to legacy load args due to: %s", exc)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    quantization_config=quant_config,
                    device_map={'': 0},
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()
            patch_hunyuan_generate_image(model)

            if hasattr(model, "vae"):
                target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                logger.info("Ensuring VAE stays in bfloat16 on %s", target_device)
                model.vae = model.vae.to(device=target_device, dtype=torch.bfloat16)
                for param in model.vae.parameters():
                    if hasattr(param, "quant_state"):
                        continue
                    move_kwargs = {"device": target_device}
                    if torch.is_floating_point(param):
                        move_kwargs["dtype"] = torch.bfloat16
                    param.data = param.data.to(**move_kwargs)
                for buffer in model.vae.buffers():
                    if buffer.device.type != "meta":
                        move_kwargs = {"device": target_device}
                        if torch.is_floating_point(buffer):
                            move_kwargs["dtype"] = torch.bfloat16
                        buffer.data = buffer.data.to(**move_kwargs)
                
                try:
                    from bitsandbytes.nn import Linear4bit  # type: ignore

                    quantized_layers = [
                        name for name, module in model.vae.named_modules()
                        if isinstance(module, Linear4bit)
                    ]
                    if quantized_layers:
                        logger.warning("VAE still has quantized linear layers: %s", quantized_layers[:5])
                    else:
                        logger.info("✓ VAE contains no quantized linear layers")
                except Exception:
                    logger.debug("bitsandbytes Linear4bit check unavailable; skipping VAE inspection")
                logger.info("✓ VAE configured in full precision")

            logger.info("Verifying device placement...")
            ensure_model_on_device(model, torch.device("cuda:0"), skip_quantized_params=True)

            HunyuanModelCache.store(model_path_str, model)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"✓ Model loaded - VRAM Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                logger.info("  (Expected ~45GB with attention in full precision)")

            logger.info("✓ Model ready for generation")
            return (model,)
        
        except Exception as e:
            logger.error("Failed to load quantized model: %s", e)
            logger.info("Cleaning up VRAM after failed load...")
            if model is not None:
                try:
                    del model
                except:
                    pass
            HunyuanModelCache.clear()
            raise

    @classmethod
    def _apply_dtype_patches(cls):
        logger.info("Applying dtype compatibility patches...")
        patch_dynamic_cache_dtype()


class HunyuanImage3Int8Loader:
    """Loads INT8-quantized HunyuanImage-3.0 model onto GPU."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls._get_available_models(),),
                "force_reload": ("BOOLEAN", {"default": False}),
                "reserve_memory_gb": ("FLOAT", {
                    "default": 6.0, 
                    "min": 2.0, 
                    "max": 80.0, 
                    "step": 0.5,
                    "tooltip": "VRAM to leave free. Standard: 6GB. Large images (>2MP) can eat ~15GB/MP (use Large Generate node to offload)."
                }),
            },
            "optional": {
                "unload_signal": ("*", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    @classmethod
    def IS_CHANGED(cls, model_name, force_reload=False, unload_signal=None, **kwargs):
        if force_reload:
            return float("nan")
        return model_name
    
    @classmethod
    def _get_available_models(cls):
        available = get_available_hunyuan_models(
            require_file="quantization_metadata.json",
            sort_key=lambda x: 0 if "int8" in x.lower() else 1,
            fallback=["HunyuanImage-3-INT8"],
        )
        return available
    
    def load_model(self, model_name, force_reload=False, unload_signal=None, reserve_memory_gb=6.0):
        # force_reload: if True, always reload model even if cached
        # unload_signal: forces re-execution if model was cleared (changes on each unload)
        # reserve_memory_gb: kept for API compatibility but not used (INT8 needs all available memory)
        
        model_path = Path(resolve_hunyuan_model_path(model_name))
        model_path_str = str(model_path)

        # If force_reload is True, skip cache and reload fresh
        if force_reload:
            logger.info("force_reload=True, clearing cache and reloading model...")
            HunyuanModelCache.clear()
            cached = None
        else:
            cached = HunyuanModelCache.get(model_path_str)
        
        if cached is not None:
            try:
                # Validate cached model is in a usable state
                if not hasattr(cached, 'generate_image'):
                    logger.warning("Cached model is invalid (missing generate_image), clearing cache and reloading...")
                    HunyuanModelCache.clear()
                    cached = None
                else:
                    # Check if model is soft-unloaded to CPU (fast restore path)
                    if HunyuanModelCache.is_on_cpu():
                        logger.info("Cached model is on CPU, restoring to GPU (fast path)...")
                        if HunyuanModelCache.restore_to_gpu():
                            logger.info("✓ Model restored from CPU to GPU")
                            self._apply_dtype_patches()
                            patch_hunyuan_generate_image(cached)
                            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
                            return (cached,)
                        else:
                            # Restore failed, need to reload from disk
                            logger.warning("Failed to restore model from CPU, reloading from disk...")
                            HunyuanModelCache.clear()
                            cached = None
                    else:
                        # Model should be on GPU - validate device placement
                        has_cuda_device = False
                        cuda_count = 0
                        cpu_count = 0
                        try:
                            for i, param in enumerate(cached.parameters()):
                                if i >= 100:  # Sample first 100 params
                                    break
                                if param.device.type == 'cuda':
                                    has_cuda_device = True
                                    cuda_count += 1
                                elif param.device.type == 'cpu':
                                    cpu_count += 1
                            logger.info(f"Cache validation: {cuda_count} CUDA params, {cpu_count} CPU params (sampled 100)")
                        except Exception as e:
                            logger.warning(f"Error during cache validation: {e}")
                            has_cuda_device = False
                        
                        if not has_cuda_device and cached is not None:
                            logger.warning("Cached model has NO CUDA params, clearing cache and reloading...")
                            HunyuanModelCache.clear()
                            cached = None
                        elif cached is not None:
                            logger.info("✓ Using cached INT8 model (validated CUDA params present)")
                            self._apply_dtype_patches()
                            patch_hunyuan_generate_image(cached)
                            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
                            return (cached,)
            except Exception as e:
                logger.warning(f"Failed to validate cached model: {e}. Clearing cache and reloading...")
                HunyuanModelCache.clear()
                cached = None

        # Clear CUDA cache to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Report VRAM before loading
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            logger.info(f"VRAM before load: {(total-free)/1024**3:.2f}GB used / {total/1024**3:.2f}GB total ({free/1024**3:.2f}GB free)")
        
        # Check metadata to warn about potential mismatch
        meta_path = model_path / "quantization_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get("quantization_method") == "bitsandbytes_nf4" or meta.get("bnb_4bit_quant_type") == "nf4":
                    logger.warning(f"⚠️ WARNING: Model '{model_name}' appears to be NF4 quantized, but you are using the INT8 loader.")
            except Exception:
                pass

        logger.info(f"Loading {model_name} (INT8)")
        logger.info("Creating INT8 quantization config (attention layers in full precision)...")

        skip_modules = [
            "vae",
            "model.vae",
            "vae.decoder",
            "vae.encoder",
            "autoencoder",
            "model.autoencoder",
            "autoencoder.decoder",
            "autoencoder.encoder",
            "patch_embed",
            "model.patch_embed",
            "final_layer",
            "model.final_layer",
            "time_embed",
            "model.time_embed",
            "time_embed_2",
            "model.time_embed_2",
            "timestep_emb",
            "model.timestep_emb",
            # Critical: exclude attention projections to prevent corruption
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "attn.qkv_proj",
            "attn.out_proj",
            "self_attn",
            "cross_attn",
        ]
        
        # Loading a PRE-QUANTIZED INT8 model (weights already quantized on disk)
        # We don't need BitsAndBytesConfig at all - just load the quantized weights directly
        # The model metadata already contains the quantization info
        
        logger.info("Loading PRE-QUANTIZED INT8 model with GPU/CPU RAM distribution...")
        logger.info("  Model was already quantized with skip_modules (VAE/attention in full precision)")
        logger.info("  Expected: ~80GB in RAM, works like BF16 loader")

        model = None
        try:
            # Calculate max memory - same approach as BF16 loader
            max_memory = None
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                # Reserve specified amount for generation overhead
                reserve_bytes = int(reserve_memory_gb * 1024**3)
                max_gpu_memory = free - reserve_bytes
                
                # Ensure we don't set a negative or too small limit
                if max_gpu_memory < 4 * 1024**3:
                    logger.warning(f"Reserved memory ({reserve_memory_gb}GB) leaves too little for model. Using minimum 4GB.")
                    max_gpu_memory = 4 * 1024**3
                
                max_memory = {0: max_gpu_memory, "cpu": "100GiB"}
                logger.info(f"Setting max GPU memory to {max_gpu_memory/1024**3:.1f}GB (reserving {reserve_memory_gb}GB)")
            
            # The model has quantization metadata embedded that triggers validation
            # We need to explicitly disable quantization detection OR use a custom loader
            
            # Try using load_quantized.py if it exists (custom loader that bypasses validation)
            load_quantized_path = model_path / "load_quantized.py"
            if load_quantized_path.exists():
                logger.info("Found load_quantized.py - using custom loader to bypass validation")
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("load_quantized", load_quantized_path)
                    load_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(load_module)
                    
                    # Use custom loader if available
                    if hasattr(load_module, 'load_quantized_model'):
                        model = load_module.load_quantized_model(
                            model_path_str,
                            device_map="auto",
                            max_memory=max_memory,
                        )
                        logger.info("✓ Loaded using custom load_quantized.py")
                    else:
                        raise AttributeError("load_quantized_model not found in custom loader")
                except Exception as e:
                    logger.warning(f"Custom loader failed: {e}, falling back to standard loading")
                    model = None
            else:
                model = None
            
            # Fallback to standard loading
            if model is None:
                load_kwargs = dict(
                    device_map="auto",
                    max_memory=max_memory,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    moe_impl="eager",
                    moe_drop_tokens=True,
                    low_cpu_mem_usage=True,
                    # Explicitly set quantization_config to None to bypass embedded config
                    quantization_config=None,
                )

                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path_str,
                        **load_kwargs,
                    )
                except TypeError as exc:
                    logger.warning("Falling back to legacy load args due to: %s", exc)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path_str,
                        device_map="auto",
                        max_memory=max_memory,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )

            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()
            patch_hunyuan_generate_image(model)

            if hasattr(model, "vae"):
                target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                logger.info("Ensuring VAE stays in bfloat16 on %s", target_device)
                model.vae = model.vae.to(device=target_device, dtype=torch.bfloat16)
                for param in model.vae.parameters():
                    if hasattr(param, "quant_state"):
                        continue
                    move_kwargs = {"device": target_device}
                    if torch.is_floating_point(param):
                        move_kwargs["dtype"] = torch.bfloat16
                    param.data = param.data.to(**move_kwargs)
                for buffer in model.vae.buffers():
                    if buffer.device.type != "meta":
                        move_kwargs = {"device": target_device}
                        if torch.is_floating_point(buffer):
                            move_kwargs["dtype"] = torch.bfloat16
                        buffer.data = buffer.data.to(**move_kwargs)
                
                logger.info("✓ VAE configured in full precision")

            logger.info("Verifying device placement...")
            logger.info(f"  Device map: {model.hf_device_map}")
            ensure_model_on_device(model, torch.device("cuda:0"), skip_quantized_params=True)

            HunyuanModelCache.store(model_path_str, model)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"✓ INT8 model loaded - VRAM Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                logger.info("  Model distributed across GPU + CPU as needed")

            logger.info("✓ INT8 model ready for generation (expect 2-3x faster than full BF16)")
            return (model,)
        
        except Exception as e:
            logger.error("Failed to load INT8 model: %s", e)
            logger.info("Cleaning up VRAM after failed load...")
            if model is not None:
                try:
                    del model
                except:
                    pass
            HunyuanModelCache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache (no model was cached)")
            raise

    @classmethod
    def _apply_dtype_patches(cls):
        logger.info("Applying dtype compatibility patches...")
        patch_dynamic_cache_dtype()


class HunyuanImage3NF4LoaderLowVRAMBudget:
    """NF4 loader variant with explicit GPU budget controls for 24-32GB cards."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls._get_available_models(),),
                "force_reload": ("BOOLEAN", {"default": False}),
                "reserve_memory_gb": ("FLOAT", {
                    "default": 6.0,
                    "min": 2.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Fallback headroom if GPU budget <= 0. Leave ≥6GB so auto mode keeps inference breathing room."
                }),
                "gpu_memory_target_gb": ("FLOAT", {
                    "default": 18.0,
                    "min": 4.0,
                    "max": 128.0,
                    "step": 0.5,
                    "tooltip": "Approximate GPU budget for NF4 weights (GB). 18‑20GB for 24GB GPUs, 26‑28GB for 32GB GPUs."
                }),
            },
            "optional": {
                "unload_signal": ("*", {"default": None}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    @classmethod
    def IS_CHANGED(cls, model_name, force_reload=False, unload_signal=None, **kwargs):
        if force_reload:
            return float("nan")
        return model_name

    @classmethod
    def _get_available_models(cls):
        available = get_available_hunyuan_models(
            require_file="quantization_metadata.json",
            sort_key=lambda x: 0 if "nf4" in x.lower() else 1,
            fallback=["HunyuanImage-3-NF4"],
        )
        return available

    def load_model(self, model_name, force_reload=False, unload_signal=None,
                  reserve_memory_gb=6.0, gpu_memory_target_gb=18.0):
        model_path = Path(resolve_hunyuan_model_path(model_name))
        model_path_str = str(model_path)

        if force_reload:
            logger.info("force_reload=True, clearing cache and reloading model...")
            HunyuanModelCache.clear()
            cached = None
        else:
            cached = HunyuanModelCache.get(model_path_str)

        if cached is not None:
            try:
                if not hasattr(cached, 'generate_image'):
                    logger.warning("Cached model invalid, clearing cache and reloading...")
                    HunyuanModelCache.clear()
                    cached = None
                else:
                    # For NF4/device_map models, we may have BOTH cuda and cpu params
                    # Just need to verify at least SOME params are on cuda
                    has_cuda_device = False
                    cuda_count = 0
                    cpu_count = 0
                    try:
                        for i, param in enumerate(cached.parameters()):
                            if i >= 100:  # Sample first 100 params
                                break
                            if param.device.type == 'cuda':
                                has_cuda_device = True
                                cuda_count += 1
                            elif param.device.type == 'cpu':
                                cpu_count += 1
                        logger.info(f"Cache validation: {cuda_count} CUDA params, {cpu_count} CPU params (sampled 100)")
                    except Exception as e:
                        logger.warning(f"Error during cache validation: {e}")
                        has_cuda_device = False

                    if not has_cuda_device and cached is not None:
                        logger.warning("Cached model has NO CUDA params, clearing cache and reloading...")
                        HunyuanModelCache.clear()
                        cached = None
                    elif cached is not None:
                        logger.info("✓ Using cached NF4 Low VRAM model (validated CUDA params present)")
                        self._apply_dtype_patches()
                        patch_hunyuan_generate_image(cached)
                        ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
                        return (cached,)
            except Exception as e:
                logger.warning(f"Failed to validate cached model: {e}. Clearing cache and reloading...")
                HunyuanModelCache.clear()
                cached = None

        max_memory = None
        target_gpu_gb = None
        total_vram_gb = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_vram, total_vram = torch.cuda.mem_get_info(0)
            total_vram_gb = total_vram / 1024**3
            logger.info(
                f"VRAM before load: {(total_vram-free_vram)/1024**3:.2f}GB used / {total_vram/1024**3:.2f}GB total"
                f" ({free_vram/1024**3:.2f}GB free)"
            )

            if gpu_memory_target_gb and gpu_memory_target_gb > 0:
                desired_budget = min(gpu_memory_target_gb, total_vram_gb)
                reserve_cap = max(reserve_memory_gb, 0.0)
                target_gpu_gb = max(min(desired_budget, total_vram_gb - reserve_cap), 8.0)
                if target_gpu_gb < desired_budget:
                    logger.info(
                        "  Clamped GPU budget from %.1fGB to %.1fGB to leave %.1fGB headroom",
                        desired_budget,
                        target_gpu_gb,
                        reserve_cap,
                    )
            else:
                reserve_bytes = int(max(reserve_memory_gb, 0.0) * 1024**3)
                target_gpu_gb = max((free_vram - reserve_bytes) / 1024**3, 8.0)

            max_vram_for_model = max(int(target_gpu_gb * 1024**3), int(8 * 1024**3))
            max_memory = {0: max_vram_for_model, "cpu": "100GiB"}

            headroom_gb = max(total_vram_gb - target_gpu_gb, 0.0)
            logger.info(
                "LOW VRAM MODE: Targeting %.1fGB GPU budget (total %.1fGB, headroom %.1fGB)",
                target_gpu_gb,
                total_vram_gb,
                headroom_gb,
            )
            if gpu_memory_target_gb and gpu_memory_target_gb > 0:
                logger.info("  Reserve slider only kicks in if the GPU target would leave <%.1fGB free", max(reserve_memory_gb, 0.0))
            else:
                logger.info("  GPU budget derived from reserve slider (%.1fGB)", reserve_memory_gb)
            logger.info("  Excess weights will be kept in system RAM via accelerate")
        else:
            max_memory = None
            logger.warning("CUDA not available, loading model to CPU")

        logger.info(f"Loading {model_name} (NF4 Low VRAM Budget)")

        skip_modules = [
            "vae", "model.vae", "vae.decoder", "vae.encoder",
            "autoencoder", "model.autoencoder", "autoencoder.decoder", "autoencoder.encoder",
            "patch_embed", "model.patch_embed",
            "final_layer", "model.final_layer",
            "time_embed", "model.time_embed", "time_embed_2", "model.time_embed_2",
            "timestep_emb", "model.timestep_emb",
            "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
            "attn.qkv_proj", "attn.out_proj",
            "self_attn", "cross_attn",
            "norm", "model.norm",
            "lm_head", "model.lm_head",
            "embed_tokens", "model.embed_tokens",
        ]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            modules_to_not_convert=skip_modules,
            llm_int8_skip_modules=skip_modules,
            llm_int8_enable_fp32_cpu_offload=True,
            keep_in_fp32_modules=skip_modules,
        )

        model = None
        try:
            # Strategy: Calculate custom device map to force NF4 layers to GPU
            # This prevents bitsandbytes from seeing 4-bit layers on CPU (which causes validation error)
            logger.info("Calculating custom device map to force NF4 layers to GPU...")
            config = AutoConfig.from_pretrained(model_path_str, trust_remote_code=True)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            
            device_map = infer_auto_device_map(
                empty_model,
                max_memory=max_memory,
                no_split_module_classes=["HunyuanImage3DecoderLayer"],
                dtype=torch.bfloat16,
            )
            
            # Force transformer blocks to GPU
            forced_count = 0
            for name in list(device_map.keys()):
                if "model.layers" in name:
                    device_map[name] = 0
                    forced_count += 1

            # Critical output modules must stay on GPU or bitsandbytes rejects the map
            critical_gpu_modules = [
                "model.ln_f",
                "ln_f",
                "lm_head",
                "model.lm_head",
                "model.norm",
            ]
            critical_forced = 0
            for module_name in critical_gpu_modules:
                if module_name in device_map and device_map[module_name] != 0:
                    device_map[module_name] = 0
                    critical_forced += 1

            if critical_forced:
                logger.info("Forced %d critical output modules to GPU to satisfy NF4 requirements", critical_forced)
            
            logger.info(f"Forced {forced_count} transformer blocks to GPU to satisfy NF4 requirements")

            safe_device_map = device_map
            try:
                non_gpu_entries = [
                    (name, target)
                    for name, target in safe_device_map.items()
                    if target not in (0, "cuda", "cuda:0")
                ]
                if non_gpu_entries:
                    logger.info("Device map contains non-GPU placements:")
                    for name, target in non_gpu_entries:
                        logger.info("  %s -> %s", name, target)
                else:
                    logger.info("Device map summary: all modules currently mapped to GPU 0")
            except Exception as exc:
                logger.warning("Could not summarize device map: %s", exc)

            quant_config_path = Path(model_path_str) / "quantization_config.json"
            if quant_config_path.exists():
                try:
                    with quant_config_path.open("r", encoding="utf-8") as handle:
                        saved_quant_config = json.load(handle)
                    logger.info(
                        "Checkpoint quantization config preview: %s",
                        {k: saved_quant_config.get(k) for k in sorted(saved_quant_config) if k in {"bnb_4bit_quant_type", "llm_int8_enable_fp32_cpu_offload", "modules_to_not_convert", "llm_int8_skip_modules"}}
                    )
                except Exception as exc:
                    logger.warning("Failed to read checkpoint quantization_config.json: %s", exc)

            load_kwargs = dict(
                device_map=device_map,
                quantization_config=quant_config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
                low_cpu_mem_usage=True,
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(model_path_str, **load_kwargs)
            except TypeError as exc:
                logger.warning("Falling back to legacy load args due to: %s", exc)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    device_map=device_map,
                    quantization_config=quant_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()
            patch_hunyuan_generate_image(model)
            model._needs_meta_patch = True

            if hasattr(model, "vae"):
                target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                logger.info("Ensuring VAE stays in bfloat16 on %s", target_device)
                model.vae = model.vae.to(device=target_device, dtype=torch.bfloat16)
                for param in model.vae.parameters():
                    if hasattr(param, "quant_state"):
                        continue
                    move_kwargs = {"device": target_device}
                    if torch.is_floating_point(param):
                        move_kwargs["dtype"] = torch.bfloat16
                    param.data = param.data.to(**move_kwargs)
                for buffer in model.vae.buffers():
                    if buffer.device.type != "meta":
                        move_kwargs = {"device": target_device}
                        if torch.is_floating_point(buffer):
                            move_kwargs["dtype"] = torch.bfloat16
                        buffer.data = buffer.data.to(**move_kwargs)

                logger.info("✓ VAE configured in full precision")

            logger.info("Verifying device placement...")
            if torch.cuda.is_available():
                model._loader_gpu_budget_gb = target_gpu_gb
                model._loader_reserve_gb = max(total_vram_gb - target_gpu_gb, 0.0)
            ensure_model_on_device(model, torch.device("cuda:0"), skip_quantized_params=True)

            HunyuanModelCache.store(model_path_str, model)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                free_after, _ = torch.cuda.mem_get_info(0)
                logger.info("✓ Low VRAM NF4 model loaded")
                logger.info(f"  VRAM: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB, Free {free_after/1024**3:.2f}GB")

            logger.info("✓ Model ready for generation with Low VRAM Budget node")
            return (model,)

        except Exception as e:
            logger.error("Failed to load Low VRAM NF4 model: %s", e)
            if model is not None:
                try:
                    del model
                except Exception:
                    pass
            HunyuanModelCache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    @classmethod
    def _apply_dtype_patches(cls):
        logger.info("Applying dtype compatibility patches...")
        patch_dynamic_cache_dtype()


class HunyuanImage3Int8LoaderBudget:
    """INT8 loader variant with GPU budget and telemetry-friendly metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls._get_available_models(),),
                "force_reload": ("BOOLEAN", {"default": False}),
                "reserve_memory_gb": ("FLOAT", {
                    "default": 20.0,
                    "min": 5.0,
                    "max": 80.0,
                    "step": 0.5,
                    "tooltip": "VRAM to reserve for inference headroom. INT8 needs ~20GB minimum."
                }),
                "gpu_memory_target_gb": ("FLOAT", {
                    "default": 80.0,
                    "min": 10.0,
                    "max": 128.0,
                    "step": 0.5,
                    "tooltip": "Approximate GPU budget for INT8 weights (GB). 80GB hits the sweet spot on 96GB cards."
                }),
            },
            "optional": {
                "unload_signal": ("*", {"default": None}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    @classmethod
    def IS_CHANGED(cls, model_name, force_reload=False, unload_signal=None, **kwargs):
        if force_reload:
            return float("nan")
        return model_name

    @classmethod
    def _get_available_models(cls):
        available = get_available_hunyuan_models(
            require_file="quantization_metadata.json",
            sort_key=lambda x: 0 if "int8" in x.lower() else 1,
            fallback=["HunyuanImage-3-INT8"],
        )
        return available

    def load_model(self, model_name, force_reload=False, unload_signal=None,
                  reserve_memory_gb=20.0, gpu_memory_target_gb=80.0):
        model_path = Path(resolve_hunyuan_model_path(model_name))
        model_path_str = str(model_path)

        # If force_reload is True, skip cache and reload fresh
        if force_reload:
            logger.info("force_reload=True, clearing cache and reloading model...")
            HunyuanModelCache.clear()
            cached = None
        else:
            cached = HunyuanModelCache.get(model_path_str)
        
        if cached is not None:
            try:
                # Validate cached model is in a usable state
                if not hasattr(cached, 'generate_image'):
                    logger.warning("Cached model is invalid (missing generate_image), clearing cache and reloading...")
                    HunyuanModelCache.clear()
                    cached = None
                else:
                    # Check if model is on CPU (soft unloaded) - restore it fast!
                    if HunyuanModelCache.is_on_cpu():
                        logger.info("Found model cached on CPU - restoring to GPU (fast reload)...")
                        import time
                        start_time = time.time()
                        if HunyuanModelCache.restore_to_gpu():
                            elapsed = time.time() - start_time
                            logger.info(f"✓ Model restored from CPU to GPU in {elapsed:.1f}s (vs ~100s from disk)")
                            self._apply_dtype_patches()
                            patch_hunyuan_generate_image(cached)
                            return (cached,)
                        else:
                            logger.warning("Failed to restore from CPU, will reload from disk...")
                            HunyuanModelCache.clear()
                            cached = None
                    else:
                        # Model should be on GPU, validate it
                        # For INT8/device_map models, we may have BOTH cuda and cpu params
                        # Just need to verify at least SOME params are on cuda
                        has_cuda_device = False
                        cuda_count = 0
                        cpu_count = 0
                        try:
                            for i, param in enumerate(cached.parameters()):
                                if i >= 100:  # Sample first 100 params
                                    break
                                if param.device.type == 'cuda':
                                    has_cuda_device = True
                                    cuda_count += 1
                                elif param.device.type == 'cpu':
                                    cpu_count += 1
                            logger.info(f"Cache validation: {cuda_count} CUDA params, {cpu_count} CPU params (sampled 100)")
                        except Exception as e:
                            logger.warning(f"Error during cache validation: {e}")
                            has_cuda_device = False
                        
                        if not has_cuda_device and cached is not None:
                            logger.warning("Cached model has NO CUDA params, clearing cache and reloading...")
                            HunyuanModelCache.clear()
                            cached = None
                        elif cached is not None:
                            logger.info("✓ Using cached INT8 model (validated CUDA params present)")
                            self._apply_dtype_patches()
                            patch_hunyuan_generate_image(cached)
                            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
                            return (cached,)
                            return (cached,)
            except Exception as e:
                logger.warning(f"Failed to validate cached model: {e}. Clearing cache and reloading...")
                HunyuanModelCache.clear()
                cached = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            logger.info(
                f"VRAM before load: {(total-free)/1024**3:.2f}GB used / {total/1024**3:.2f}GB total"
                f" ({free/1024**3:.2f}GB free)"
            )

        logger.info(f"Loading {model_name} (INT8 Budget)")

        meta_path = model_path / "quantization_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get("quantization_method") == "bitsandbytes_nf4":
                    logger.warning(f"⚠️ Model '{model_name}' appears to be NF4 quantized, not INT8")
            except Exception:
                pass

        skip_modules = [
            "vae",
            "model.vae",
            "vae.decoder",
            "vae.encoder",
            "autoencoder",
            "model.autoencoder",
            "autoencoder.decoder",
            "autoencoder.encoder",
            "patch_embed",
            "model.patch_embed",
            "final_layer",
            "model.final_layer",
            "time_embed",
            "model.time_embed",
            "time_embed_2",
            "model.time_embed_2",
            "timestep_emb",
            "model.timestep_emb",
            # Critical: exclude attention projections to prevent corruption
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "attn.qkv_proj",
            "attn.out_proj",
            "self_attn",
            "cross_attn",
        ]

        logger.info("Detected INT8 selective quantization")
        logger.info("  Critical layers already bfloat16 on disk")
        logger.info("  Skipping quantization_config since model is pre-quantized")

        # NOTE: For PRE-QUANTIZED models, we should NOT pass quantization_config
        # The weights are already INT8 on disk, passing config causes redundant processing
        # and can slow down loading significantly

        model = None
        target_gpu_gb = None
        total_vram_gb = None
        if torch.cuda.is_available():
            _, total_bytes = torch.cuda.mem_get_info(0)
            total_vram_gb = total_bytes / 1024**3
            if gpu_memory_target_gb and gpu_memory_target_gb > 0:
                target_gpu_gb = min(gpu_memory_target_gb, total_vram_gb)

        try:
            logger.info("Selective INT8: Loading pre-quantized model to GPU")
            logger.info("  TIP: Pre-quantized models don't need quantization_config")
            
            # For pre-quantized INT8, load directly without quantization_config
            # This is faster because bitsandbytes doesn't need to validate/process
            load_kwargs = dict(
                device_map="cuda:0",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
                low_cpu_mem_usage=True,
                # Don't pass quantization_config for pre-quantized models!
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(model_path_str, **load_kwargs)
            except TypeError as exc:
                logger.warning("Falling back to legacy load args due to: %s", exc)
                fallback_kwargs = {
                    k: v for k, v in load_kwargs.items()
                    if k in ["device_map", "max_memory", "quantization_config", "trust_remote_code", "torch_dtype", "low_cpu_mem_usage"]
                }
                model = AutoModelForCausalLM.from_pretrained(model_path_str, **fallback_kwargs)

            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()
            patch_hunyuan_generate_image(model)

            from .hunyuan_shared import install_dtype_harmonization_hooks
            install_dtype_harmonization_hooks(model)

            if hasattr(model, "vae"):
                target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                logger.info("Ensuring VAE stays in bfloat16 on %s", target_device)
                model.vae = model.vae.to(device=target_device, dtype=torch.bfloat16)
                for param in model.vae.parameters():
                    if hasattr(param, "quant_state"):
                        continue
                    move_kwargs = {"device": target_device}
                    if torch.is_floating_point(param):
                        move_kwargs["dtype"] = torch.bfloat16
                    param.data = param.data.to(**move_kwargs)
                for buffer in model.vae.buffers():
                    if buffer.device.type != "meta":
                        move_kwargs = {"device": target_device}
                        if torch.is_floating_point(buffer):
                            move_kwargs["dtype"] = torch.bfloat16
                        buffer.data = buffer.data.to(**move_kwargs)

            logger.info("Verifying device placement...")
            logger.info(f"  Device map: {model.hf_device_map}")
            ensure_model_on_device(model, torch.device("cuda:0"), skip_quantized_params=True)

            HunyuanModelCache.store(model_path_str, model)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"✓ INT8 model loaded - VRAM Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                logger.info("  Model distributed across GPU + CPU as needed")

            logger.info("✓ INT8 model ready for generation with budget-aware nodes")
            return (model,)

        except Exception as e:
            logger.error("Failed to load INT8 model: %s", e)
            if model is not None:
                try:
                    del model
                except Exception:
                    pass
            HunyuanModelCache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    @classmethod
    def _apply_dtype_patches(cls):
        logger.info("Applying dtype compatibility patches...")
        patch_dynamic_cache_dtype()


class HunyuanImage3Generate:
    """
    Generate images using HunyuanImage-3.0 (works with both quantized and full models).
    
    Note: HunyuanImage-3.0 uses autoregressive architecture (like GPT), not diffusion.
    It does not support negative prompts - be explicit about what you want in your prompt.
    
    Optional Prompt Rewriting:
    - Uses official HunyuanImage-3.0 system prompts for professional prompt expansion
    - Supports any OpenAI-compatible API (DeepSeek, OpenAI, Claude via proxy, etc.)
    - Rewrite styles:
      * none: Use your original prompt without modification
      * en_recaption: Structured, detail-rich professional expansion (recommended)
      * en_think_recaption: Advanced with thinking phase + detailed expansion
    
    API Setup:
    - Configure API key in api_config.ini or environment variables (HUNYUAN_API_KEY)
    - See api_config.ini.example for details
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "resolution": (cls._resolution_choices(),),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "post_action": (["keep_loaded", "soft_unload_to_cpu", "full_unload"], {
                    "default": "keep_loaded",
                    "tooltip": "After generation: keep_loaded (fastest reruns), soft_unload_to_cpu (free VRAM, ~10s restore), full_unload (free VRAM+RAM, ~35s reload)"
                }),
            },
            "optional": {
                "enable_prompt_rewrite": ("BOOLEAN", {"default": False}),
                "rewrite_style": (["none", "en_recaption", "en_think_recaption"], {"default": "none"}),
                # api_key removed for security, use api_config.ini or env vars
                "api_url": ("STRING", {"default": "https://api.deepseek.com/v1/chat/completions"}),
                "model_name": ("STRING", {"default": "deepseek-chat"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "*")
    RETURN_NAMES = ("image", "rewritten_prompt", "status", "trigger")
    FUNCTION = "generate"
    CATEGORY = "HunyuanImage3"
    
    def generate(self, model, prompt, seed, steps, resolution, guidance_scale, post_action="keep_loaded",
                 enable_prompt_rewrite=False, rewrite_style="none", 
                 api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat", 
                 skip_device_check=False, keep_model_loaded=None):
        # Backward compatibility: if old keep_model_loaded param is passed, convert to post_action
        if keep_model_loaded is not None:
            post_action = "keep_loaded" if keep_model_loaded else "full_unload"
        
        # Validate model has valid device placement before generation
        if not skip_device_check:
            try:
                # For device_map models, we may have BOTH cuda and cpu params
                # Just need to verify at least SOME params are on cuda
                has_cuda_device = False
                for i, param in enumerate(model.parameters()):
                    if i >= 100:  # Sample first 100 params
                        break
                    if param.device.type == 'cuda':
                        has_cuda_device = True
                        break
                
                if not has_cuda_device:
                    raise RuntimeError(
                        "Model has no CUDA params (likely cleared by Unload node). "
                        "Please ensure the model loader runs before generation in your workflow. "
                        "The model may have been unloaded by a previous Unload node."
                    )
            except Exception as e:
                if "no cuda params" in str(e).lower() or "cleared by unload" in str(e).lower():
                    raise
                logger.warning(f"Could not validate model device: {e}")
        
        logger.info(f"Generating image with {steps} steps")
        
        # Handle prompt rewriting if enabled
        original_prompt = prompt
        rewritten_prompt = prompt
        status_message = "Generation complete"
        
        if enable_prompt_rewrite and rewrite_style != "none":
            # Get API config
            config = get_api_config()
            api_key = config.get("api_key")
            
            # Use provided URL/model if they differ from default, otherwise prefer config
            # Actually, the inputs have defaults. If user didn't change them, we might want to use config.
            # But here we just use what's passed, or fallback to config if passed is default?
            # Simpler: Use passed values for URL/Model, but Key comes from config.
            
            if api_key:
                try:
                    logger.info(f"Rewriting prompt using LLM API (style: {rewrite_style})...")
                    api_config = {
                        "url": api_url,
                        "key": api_key,
                        "model": model_name
                    }
                    prompt = self._rewrite_prompt_with_llm(prompt, rewrite_style, api_config)
                    rewritten_prompt = prompt
                    logger.info(f"Original prompt: {original_prompt[:80]}...")
                    logger.info(f"Rewritten prompt: {prompt[:80]}...")
                    status_message = f"Prompt rewritten using {rewrite_style}"
                except Exception as e:
                    logger.warning(f"Prompt rewriting failed: {e}, using original prompt")
                    prompt = original_prompt
                    rewritten_prompt = original_prompt
                    status_message = f"Prompt rewriting failed: {str(e)[:100]}"
            else:
                logger.warning("Prompt rewrite enabled but no API key found in config/env.")
                status_message = "Prompt rewrite skipped (missing API key)"
        else:
            status_message = "Using original prompt (no rewriting)"
        
        logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "diff_infer_steps": steps,
            "stream": True,
            "diff_guidance_scale": guidance_scale,
        }

        # Progress bar callback
        pbar = ProgressBar(steps)
        def callback(pipe, step, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs
        
        # Only add callback if model supports it (patched models do)
        # But to be safe against unpatched models, we can wrap it or check
        # The patch_hunyuan_generate_image ensures it's handled, but if that failed...
        gen_kwargs["callback_on_step_end"] = callback
        
        if resolution == self._default_resolution_label():
            logger.info("Using model default resolution")
        else:
            height, width = self._parse_resolution(resolution)
            # The model expects "height x width" string for image_size
            # But wait, the model's internal logic might be parsing it differently or snapping it.
            # Let's pass explicit width/height integers if possible, or ensure the string is correct.
            # The pipeline usually takes width/height or image_size.
            # Let's try passing width and height directly to be safe if the pipeline supports it,
            # otherwise rely on the string.
            # Based on Hunyuan code, it often snaps to buckets.
            
            # CRITICAL FIX: We must patch the resolution group to force our exact resolution
            # just like we did in the Large node, otherwise it snaps to the nearest bucket.
            # 1536x1920 (3MP) is snapping to 768x1024 (0.8MP) because it's the "closest" standard bucket
            # if the 3MP buckets aren't in its default list.
            
            if hasattr(model, "image_processor") and hasattr(model.image_processor, "reso_group"):
                reso_group = model.image_processor.reso_group
                original_get_target_size = reso_group.get_target_size
                
                def new_get_target_size(w, h):
                    # Force exact requested size
                    return int(w), int(h)
                
                reso_group.get_target_size = new_get_target_size
                logger.info("Applied resolution patch to bypass bucket snapping")
                
                # We need to restore this after generation
                self._original_get_target_size = original_get_target_size
                self._model_to_restore = model

            gen_kwargs["image_size"] = f"{height}x{width}"
            logger.info(f"Using custom resolution: {width}x{height}")
        
        logger.info(f"Guidance scale: {guidance_scale}")
        
        try:
            image = model.generate_image(**gen_kwargs)
        finally:
            # Restore original method if we patched it
            if hasattr(self, "_model_to_restore"):
                self._model_to_restore.image_processor.reso_group.get_target_size = self._original_get_target_size
                del self._model_to_restore
                del self._original_get_target_size
                logger.info("Restored original resolution logic")

        # Some configurations return an iterator of intermediate frames when streaming is enabled.
        if isinstance(image, (list, tuple)) and image:
            image = image[-1]
        elif not isinstance(image, Image.Image) and hasattr(image, "__iter__"):
            last_frame = None
            for frame in image:
                last_frame = frame
            if last_frame is None:
                raise RuntimeError("Streaming generator returned no frames")
            image = last_frame
        elif not isinstance(image, Image.Image):
            raise RuntimeError("Unexpected return type from generate_image")

        logger.info(f"Converting image (mode: {image.mode}, size: {image.size})")

        if image.mode == "I":
            image = image.point(lambda i: i * (1 / 255))

        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert PIL to ComfyUI format
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Handle post-generation action
        if post_action == "soft_unload_to_cpu":
            logger.info("Post-action: Soft unloading model to CPU RAM...")
            success = HunyuanModelCache.soft_unload()
            if success:
                logger.info("✓ Model moved to CPU RAM - VRAM freed for downstream tasks")
            else:
                logger.warning("Soft unload failed or model already on CPU")
        elif post_action == "full_unload":
            logger.info("Post-action: Full unload - clearing model from memory...")
            HunyuanModelCache.clear()
            logger.info("✓ Model cleared from memory")
        else:
            logger.info("Post-action: Keeping model loaded on GPU")

        logger.info(f"✓ Image generated successfully - tensor shape: {image_tensor.shape}")
        return (image_tensor, rewritten_prompt, status_message, True)

    @classmethod
    def _default_resolution_label(cls) -> str:
        return "Auto (model default)"

    @classmethod
    def _resolution_choices(cls):
        if not hasattr(cls, "_RESOLUTION_CACHE"):
            # Standard resolutions for 1MP, 2MP, 3MP
            # Sorted by Aspect Ratio: Tall -> Square -> Wide
            # Aligned to 64 pixels (based on config.json image_resolution_align)
            resolutions = [
                # 1MP Class
                (768, 1344, "9:16 (1.0MP)"),
                (768, 1280, "5:8 (1.0MP)"),
                (832, 1216, "2:3 (1.0MP)"),
                (896, 1152, "3:4 (1.0MP)"),
                (896, 1088, "4:5 (1.0MP)"),
                (1024, 1024, "1:1 (1.0MP)"),
                (1088, 896, "5:4 (1.0MP)"),
                (1152, 896, "4:3 (1.0MP)"),
                (1216, 832, "3:2 (1.0MP)"),
                (1280, 768, "8:5 (1.0MP)"),
                (1344, 768, "16:9 (1.0MP)"),

                # 2MP Class
                (1088, 1856, "9:16 (2.0MP)"),
                (1088, 1792, "5:8 (1.9MP)"),
                (1152, 1728, "2:3 (2.0MP)"),
                (1216, 1664, "3:4 (2.0MP)"),
                (1280, 1600, "4:5 (2.0MP)"),
                (1408, 1408, "1:1 (2.0MP)"),
                (1600, 1280, "5:4 (2.0MP)"),
                (1664, 1216, "4:3 (2.0MP)"),
                (1728, 1152, "3:2 (2.0MP)"),
                (1792, 1088, "8:5 (1.9MP)"),
                (1856, 1088, "16:9 (2.0MP)"),

                # 3MP Class
                (1280, 2304, "9:16 (2.9MP)"),
                (1344, 2176, "5:8 (2.9MP)"),
                (1408, 2112, "2:3 (3.0MP)"),
                (1472, 1984, "3:4 (2.9MP)"),
                (1536, 1920, "4:5 (2.9MP)"),
                (1728, 1728, "1:1 (3.0MP)"),
                (1920, 1536, "5:4 (2.9MP)"),
                (1984, 1472, "4:3 (2.9MP)"),
                (2112, 1408, "3:2 (3.0MP)"),
                (2176, 1344, "8:5 (2.9MP)"),
                (2304, 1280, "16:9 (2.9MP)"),
            ]
            
            options = [cls._default_resolution_label()]
            for w, h, label in resolutions:
                options.append(f"{w}x{h} - {label}")
                
            cls._RESOLUTION_CACHE = tuple(options)
        return cls._RESOLUTION_CACHE

    @staticmethod
    def _compute_resolutions(
            base_size: int,
            *,
            step: Optional[int] = None,
            align: int = 1,
            min_multiple: float = 0.5,
            max_multiple: float = 2.0,
            max_entries: int = 33,
        ):
        if base_size <= 0:
            raise ValueError("base_size must be positive")
        if not (0 < min_multiple < max_multiple):
            raise ValueError("min_multiple must be positive and smaller than max_multiple")

        if step is None:
            step = max(align, base_size // 16)
        else:
            if step <= 0:
                raise ValueError("step must be positive")
            step = max(align, int(math.ceil(step / align)) * align)

        min_height = max(align, int(math.ceil(base_size * min_multiple / align)) * align)
        min_width = min_height
        max_height = int(math.floor(base_size * max_multiple / align)) * align
        max_width = max_height

        resolutions = {(base_size, base_size)}

        cur_height, cur_width = base_size, base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break
            cur_height = min(cur_height + step, max_height)
            cur_width = max(cur_width - step, min_width)
            resolutions.add((cur_height, cur_width))

        cur_height, cur_width = base_size, base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break
            cur_height = max(cur_height - step, min_height)
            cur_width = min(cur_width + step, max_width)
            resolutions.add((cur_height, cur_width))

        sorted_resolutions = sorted(resolutions, key=lambda hw: hw[0] / hw[1])

        if max_entries and len(sorted_resolutions) > max_entries:
            sorted_resolutions = HunyuanImage3Generate._downsample_resolutions(sorted_resolutions, max_entries)

        return sorted_resolutions

    @classmethod
    def _parse_resolution(cls, resolution: str):
        try:
            # Handle new labeled format: "1024x768 - Landscape (0.8MP) [<2MP]"
            # Extract just the WxH part
            if " - " in resolution:
                resolution = resolution.split(" - ")[0]
            width_str, height_str = resolution.split("x")
            return int(height_str), int(width_str)
        except Exception as exc:
            raise ValueError(f"Invalid resolution selection: {resolution}") from exc

    @staticmethod
    def _downsample_resolutions(resolutions, max_entries):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if len(resolutions) <= max_entries:
            return list(resolutions)

        step = (len(resolutions) - 1) / (max_entries - 1) if max_entries > 1 else 1
        indices = []
        for i in range(max_entries):
            idx = int(round(i * step))
            idx = max(0, min(idx, len(resolutions) - 1))
            if indices and idx <= indices[-1]:
                idx = indices[-1] + 1
            if idx >= len(resolutions):
                idx = len(resolutions) - 1
            indices.append(idx)

        indices = sorted(set(indices))
        for idx in range(len(resolutions)):
            if len(indices) >= max_entries:
                break
            if idx not in indices:
                indices.append(idx)
        indices.sort()

        mid = len(indices) // 2
        base_idx = min(range(len(resolutions)), key=lambda i: abs(resolutions[i][0] - resolutions[i][1]))
        if base_idx not in indices:
            indices[mid] = base_idx
            indices.sort()

        return [resolutions[i] for i in indices[:max_entries]]
    
    def _rewrite_prompt_with_llm(self, prompt: str, style: str, api_config: dict) -> str:
        """
        Rewrite prompt using LLM API with official HunyuanImage-3.0 system prompts.
        
        Official system prompts from:
        https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/hyvideo_prompt/prompt_rewrite.py
        
        Supports any OpenAI-compatible API (DeepSeek, OpenAI, Claude via proxy, etc.)
        """
        import requests
        import re
        
        # Official HunyuanImage-3.0 system prompts
        system_prompts = {
            "en_recaption": """You are a world-class image generation prompt expert. Your task is to rewrite a user's simple description into a **structured, objective, and detail-rich** professional-level prompt.

The final output must be wrapped in `<recaption>` tags.

### **Universal Core Principles**

When rewriting the prompt (inside the `<recaption>` tags), you must adhere to the following principles:

1.  **Absolute Objectivity**: Describe only what is visually present. Avoid subjective words like "beautiful" or "sad". Convey aesthetic qualities through specific descriptions of color, light, shadow, and composition.
2.  **Physical and Logical Consistency**: All scene elements (e.g., gravity, light, shadows, reflections, spatial relationships, object proportions) must strictly adhere to real-world physics and common sense.
3.  **Structured Description**: Strictly follow a logical order: from general to specific, background to foreground, and primary to secondary elements.
4.  **Use Present Tense**: Describe the scene from an observer's perspective using the present tense.
5.  **Use Rich and Specific Descriptive Language**: Use precise adjectives to describe quantity, size, shape, color, and other attributes.

### **Final Output Requirements**

1.  **Output the Final Prompt Only**: Do not show any thought process or formatting.
2.  **Adhere to the Input**: Retain the core concepts and attributes from the user's input.
3.  **Style Reinforcement**: Mention the core style 3-5 times within the prompt.
4.  **Avoid Self-Reference**: Describe the image content directly.
5.  **The final output must be wrapped in `<recaption>xxxx</recaption>` tags.**""",
            
            "en_think_recaption": """You will act as a top-tier Text-to-Image AI. Your task is to deeply analyze the user's text input and transform it into a detailed, artistic image description.

Your workflow is divided into two phases:

1. **Thinking Phase (<think>)**: Break down the image elements:
   - Subject: Core character(s) or object(s), appearance, posture, expression
   - Composition: Camera angle, layout (close-up, long shot, etc.)
   - Environment/Background: Scene location, time, weather
   - Lighting: Type, direction, quality of light source
   - Color Palette: Main color tone and scheme
   - Quality/Style: Artistic style and technical details
   - Details: Minute elements that enhance realism

2. **Recaption Phase (<recaption>)**: Merge all details into a coherent, precise description.

**Key Principles:**
- Absolutely Objective: Describe only what is visually present
- Physical and Logical Consistency: Follow real-world physics
- Structured Description: Whole to part, background to foreground
- Use Present Tense: "A man stands," "light shines on..."
- Rich and Specific Language: Precise adjectives, no vague expressions

**Output Format:**
<think>Thinking process</think><recaption>Refined image description</recaption>"""
        }
        
        # Map old style names to official prompt types
        style_mapping = {
            "none": None,
            "universal": "en_recaption",
            "text_rendering": "en_recaption",  # Use recaption for detailed descriptions
            "en_recaption": "en_recaption",
            "en_think_recaption": "en_think_recaption"
        }
        
        prompt_type = style_mapping.get(style, "en_recaption")
        if not prompt_type:
            return prompt
            
        system_prompt = system_prompts[prompt_type]
        
        try:
            # Extract API configuration
            api_url = api_config.get("url", "https://api.deepseek.com/v1/chat/completions")
            api_key = api_config.get("key", "")
            model_name = api_config.get("model", "deepseek-chat")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            logger.info(f"Calling LLM API: {api_url} (model: {model_name})")
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            rewritten = result["choices"][0]["message"]["content"].strip()
            
            # Extract content from tags if present
            if "<recaption>" in rewritten:
                match = re.search(r'<recaption>(.*?)</recaption>', rewritten, re.DOTALL)
                if match:
                    rewritten = match.group(1).strip()
            elif "<think>" in rewritten and "<recaption>" in rewritten:
                # For think_recaption, extract only the recaption part
                match = re.search(r'<recaption>(.*?)</recaption>', rewritten, re.DOTALL)
                if match:
                    rewritten = match.group(1).strip()
            
            return rewritten
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 402:
                error_msg = (
                    "LLM API requires payment. For DeepSeek, add credits at:\n"
                    "https://platform.deepseek.com/top_up\n"
                    "Or disable prompt rewriting to use your original prompt."
                )
                logger.warning(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.error(f"LLM API HTTP error: {e}")
                raise RuntimeError(f"LLM API error: {e}")
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise RuntimeError(f"Failed to rewrite prompt: {e}")


class HunyuanImage3GenerateLarge:
    """
    Generate large high-resolution images with CPU offload support.
    Slower but handles >2MP images without OOM errors.
    
    Note: HunyuanImage-3.0 uses autoregressive architecture (like GPT), not diffusion.
    It does not support negative prompts - be explicit about what you want in your prompt.
    
    Optional Prompt Rewriting:
    - Uses official HunyuanImage-3.0 system prompts
    - Supports any OpenAI-compatible API
    - See HunyuanImage3Generate for full documentation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "resolution": (cls._get_large_resolutions(),),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "offload_mode": (["smart", "always", "disabled"], {"default": "smart"}),
                "post_action": (["keep_loaded", "soft_unload_to_cpu", "full_unload"], {
                    "default": "keep_loaded",
                    "tooltip": "After generation: keep_loaded (fastest reruns), soft_unload_to_cpu (free VRAM, ~10s restore), full_unload (free VRAM+RAM, ~35s reload)"
                }),
            },
            "optional": {
                "enable_prompt_rewrite": ("BOOLEAN", {"default": False}),
                "rewrite_style": (["none", "en_recaption", "en_think_recaption"], {"default": "none"}),
                # api_key removed for security, use api_config.ini or env vars
                "api_url": ("STRING", {"default": "https://api.deepseek.com/v1/chat/completions"}),
                "model_name": ("STRING", {"default": "deepseek-chat"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "*")
    RETURN_NAMES = ("image", "rewritten_prompt", "status", "trigger")
    FUNCTION = "generate_large"
    CATEGORY = "HunyuanImage3"
    
    @classmethod
    def _get_large_resolutions(cls):
        """Generate resolution options sorted by size then aspect ratio."""
        resolutions = []
        
        # Define standard sizes to include
        # Format: (width, height, label)
        base_sizes = [
            # 1MP Class (Standard)
            (720, 1280, "Portrait 720p"),
            (864, 1152, "Portrait 3:4"),
            (1024, 1024, "Square 1K"),
            (1152, 864, "Landscape 4:3"),
            (1280, 720, "Landscape 720p"),
            
            # 2MP Class
            (1080, 1920, "Portrait 1080p"),
            (1200, 1600, "Portrait 3:4"),
            (1440, 1440, "Square 1.5K"),
            (1600, 1200, "Landscape 4:3"),
            (1920, 1080, "Landscape 1080p"),
            
            # 4MP Class (2K)
            (1440, 2560, "Portrait 2K"),
            (1536, 2048, "Portrait 3:4"),
            (2048, 2048, "Square 2K"),
            (2048, 1536, "Landscape 4:3"),
            (2560, 1440, "Landscape 2K"),
            
            # 9MP Class (3K/4K)
            (2160, 3840, "Portrait 4K"),
            (3072, 3072, "Square 3K"),
            (3840, 2160, "Landscape 4K"),
            
            # Extreme
            (4096, 4096, "Square 4K"),
        ]
        
        options = ["Auto (model default)"]
        for w, h, desc in base_sizes:
            mp = (w * h) / 1_000_000
            options.append(f"{w}x{h} - {desc} ({mp:.1f}MP)")
            
        return options
    
    def generate_large(self, model, prompt, seed, steps, resolution, guidance_scale, offload_mode="smart", 
                      post_action="keep_loaded",
                      enable_prompt_rewrite=False, rewrite_style="none",
                      api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat", 
                      cpu_offload=None, keep_model_loaded=None):
        
        # Backward compatibility for old keep_model_loaded param
        if keep_model_loaded is not None:
            post_action = "keep_loaded" if keep_model_loaded else "full_unload"
        
        # Backward compatibility for cpu_offload boolean
        if cpu_offload is not None:
            offload_mode = "always" if cpu_offload else "disabled"

        # === AUTO OFFLOAD OVERRIDE FOR INT8 MODELS ===
        # Automatically switch to 'smart' offload if:
        # 1. Model is INT8 quantized
        # 2. offload_mode is currently 'disabled'
        # 3. Resolution is 'Auto' or > 1.2MP
        # This prevents OOM errors on GPUs like RTX 6000 Pro Blackwell
        original_offload_mode = offload_mode
        if offload_mode == "disabled":
            is_int8 = _is_model_int8_quantized(model)
            megapixels = _calculate_resolution_megapixels(resolution)
            gpu_info = _get_gpu_info()
            
            if is_int8 and (megapixels > 1.2 or 'auto' in resolution.lower()):
                logger.warning("="*60)
                logger.warning("AUTO OFFLOAD OVERRIDE TRIGGERED")
                logger.warning(f"  Model: INT8 quantized")
                logger.warning(f"  Resolution: {resolution} ({megapixels:.1f}MP)")
                logger.warning(f"  GPU: {gpu_info['name']} ({gpu_info['total_vram_gb']:.0f}GB)")
                logger.warning(f"  Switching offload_mode from 'disabled' to 'smart' to prevent OOM")
                logger.warning(f"  TIP: For 1MP resolutions, 'disabled' mode is faster and safe")
                logger.warning("="*60)
                offload_mode = "smart"

        # Validate model has valid device placement before generation
        try:
            # For device_map models, we may have BOTH cuda and cpu params
            # Just need to verify at least SOME params are on cuda
            has_cuda_device = False
            for i, param in enumerate(model.parameters()):
                if i >= 100:  # Sample first 100 params
                    break
                if param.device.type == 'cuda':
                    has_cuda_device = True
                    break
            
            if not has_cuda_device:
                raise RuntimeError(
                    "Model has no CUDA params (likely cleared by Unload node). "
                    "Please ensure the model loader runs before generation in your workflow. "
                    "The model may have been unloaded by a previous Unload node."
                )
        except Exception as e:
            if "no cuda params" in str(e).lower() or "cleared by unload" in str(e).lower():
                raise
            logger.warning(f"Could not validate model device: {e}")
        
        # Parse resolution
        generator = HunyuanImage3Generate()
        if resolution == "Auto (model default)":
            # Auto mode can be unpredictable and choose large resolutions (up to 2.5MP+)
            # To be safe for Smart Offload, we assume a "worst case" standard resolution
            # of ~2.5MP (e.g. 1600x1600) to ensure we offload if VRAM is tight.
            width, height = 1600, 1600 
            parsed_resolution = generator._default_resolution_label()
            logger.info("Auto resolution selected: Assuming ~2.5MP for memory calculation safety.")
        else:
            parsed_resolution = resolution.split(" - ")[0]  # Extract WxH
            width, height = map(int, parsed_resolution.split("x"))

        # Smart Offload Logic
        should_offload = False
        if offload_mode == "always":
            should_offload = True
        elif offload_mode == "smart" and torch.cuda.is_available():
            # Estimate memory requirements using empirical formula
            # Based on real measurements: 1MP=12GB, 2MP=22GB, 3MP=45GB
            megapixels = (width * height) / 1_000_000
            required_free_gb = _estimate_inference_vram_gb(megapixels)
            
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1024**3
            
            if free_gb < required_free_gb:
                logger.info(f"Smart Offload: Free VRAM ({free_gb:.1f}GB) < Required ({required_free_gb:.1f}GB) for {megapixels:.1f}MP. Enabling offload.")
                should_offload = True
            else:
                logger.info(f"Smart Offload: Sufficient VRAM ({free_gb:.1f}GB >= {required_free_gb:.1f}GB). Skipping offload for speed.")
                should_offload = False
        
        logger.info("=" * 60)
        logger.info("LARGE IMAGE GENERATION MODE")
        logger.info(f"Resolution: {width}x{height}")
        logger.info(f"Offload Mode: {offload_mode} -> {'Enabled' if should_offload else 'Disabled'}")
        logger.info("=" * 60)
        
        # Temporarily enable CPU offload for this generation if requested
        if should_offload:
            # Check if this is a device_map model — those already have accelerate
            # hooks managing device placement. Adding cpu_offload on top breaks them.
            has_device_map = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
            has_meta = any(p.device.type == 'meta' for i, p in zip(range(50), model.parameters()))
            is_device_map_model = has_device_map or has_meta
            
            if is_device_map_model:
                logger.info("Model was loaded with device_map — accelerate hooks already manage offloading.")
                logger.info("Skipping manual CPU offload to avoid 'Cannot copy out of meta tensor' errors.")
            else:
                try:
                    from accelerate import cpu_offload as accelerate_cpu_offload
                    logger.info("Enabling CPU offload for large image generation...")
                    
                    # FORCE MOVE TO CPU FIRST
                    # This ensures VRAM is actually freed before we start offloading hooks
                    logger.info("Moving model to CPU to ensure VRAM is free...")
                    model.cpu()
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        free, total = torch.cuda.mem_get_info(0)
                        logger.info(f"GPU memory before clearing: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")

                    # Apply cpu_offload
                    accelerate_cpu_offload(model, execution_device="cuda:0")
                    logger.info("✓ CPU offload enabled via accelerate")
                    
                except ImportError:
                    logger.warning("accelerate library not found, cannot enable CPU offload")
                except Exception as e:
                    logger.warning(f"Could not configure CPU offload: {e}")
        
        # Patch ResolutionGroup to allow arbitrary resolutions
        # The model's image_processor snaps to the nearest bucket by default.
        # We need to bypass this for large/custom resolutions.
        original_get_target_size = None
        if hasattr(model, "image_processor") and hasattr(model.image_processor, "reso_group"):
            reso_group = model.image_processor.reso_group
            original_get_target_size = reso_group.get_target_size
            
            def new_get_target_size(w, h):
                # If exact match exists in buckets, use it (preserves standard behavior)
                key = (int(round(h)), int(round(w)))
                if key in reso_group._lookup:
                    return reso_group._lookup[key].w, reso_group._lookup[key].h
                # Otherwise return exact requested size
                # This allows 4K etc. to pass through without being snapped to 1K
                return int(w), int(h)
            
            reso_group.get_target_size = new_get_target_size
            logger.info("Applied resolution patch to bypass bucket snapping")

        try:
            image_tensor, rewritten_prompt, status, trigger = generator.generate(
                model=model,
                prompt=prompt,
                seed=seed,
                steps=steps,
                resolution=parsed_resolution,
                guidance_scale=guidance_scale,
                post_action=post_action,
                enable_prompt_rewrite=enable_prompt_rewrite,
                rewrite_style=rewrite_style,
                api_url=api_url,
                model_name=model_name,
                skip_device_check=should_offload  # Skip check if offloading, as model will be on CPU
            )
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"GPU memory after generation: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")
            
            logger.info("=" * 60)
            logger.info("✓ Large image generated successfully")
            logger.info("=" * 60)
            
            # Update status to indicate large image mode
            status = f"Large image mode (Offload: {offload_mode}) - {status}"
            
            return (image_tensor, rewritten_prompt, status, True)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU Out of Memory! Try:")
                logger.error("  1. Set offload_mode to 'always'")
                logger.error("  2. Use a smaller resolution")
                logger.error("  3. Reduce guidance_scale or steps")
                logger.error("  4. Clear GPU memory with Unload node first")
            raise
        finally:
            # Restore original method
            if original_get_target_size is not None:
                model.image_processor.reso_group.get_target_size = original_get_target_size
                logger.info("Restored original resolution logic")


class HunyuanImage3GenerateLowVRAM(HunyuanImage3GenerateLarge):
    """
    Specialized generation node for Low VRAM (e.g. 24GB) users running Quantized models (NF4/INT8).

    Differences from Standard/Large nodes:
    1. Optimized for models loaded with device_map="auto" (NF4/INT8)
    2. Skips conflicting 'cpu_offload' calls that break quantized models
    3. Aggressively clears cache before/after generation
    4. Supports the same resolution/prompt features as Large node
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Inherit inputs but set defaults for low VRAM
        inputs = super().INPUT_TYPES()

        # Update defaults and tooltips for Low VRAM context
        inputs["required"]["offload_mode"] = (["smart", "always", "disabled"], {
            "default": "smart",
            "tooltip": "For NF4/INT8 models, offloading is handled automatically by the loader (device_map). This setting is ignored for quantized models."
        })

        # Low VRAM users should default to soft_unload to free VRAM for downstream
        inputs["required"]["post_action"] = (["keep_loaded", "soft_unload_to_cpu", "full_unload"], {
            "default": "soft_unload_to_cpu",
            "tooltip": "After generation: soft_unload_to_cpu recommended for Low VRAM to free GPU for downstream tasks"
        })

        inputs["required"]["steps"] = ("INT", {
            "default": 40, "min": 1, "max": 100,
            "tooltip": "Number of sampling steps. 30-40 is recommended."
        })

        inputs["required"]["guidance_scale"] = ("FLOAT", {
            "default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1,
            "tooltip": "Classifier-free guidance scale. Higher = closer to prompt, Lower = more creative."
        })

        return inputs

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "*")
    RETURN_NAMES = ("image", "rewritten_prompt", "status", "trigger")
    FUNCTION = "generate_low_vram"
    CATEGORY = "HunyuanImage3"

    def generate_low_vram(self, model, prompt, seed, steps, resolution, guidance_scale, offload_mode="smart", 
                          post_action="soft_unload_to_cpu",
                          enable_prompt_rewrite=False, rewrite_style="none",
                          api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat", 
                          cpu_offload=None, keep_model_loaded=None):
        
        # Backward compatibility
        if keep_model_loaded is not None:
            post_action = "keep_loaded" if keep_model_loaded else "full_unload"

        logger.info("=" * 60)
        logger.info("LOW VRAM GENERATION MODE (NF4/INT8 Optimized)")
        logger.info("=" * 60)

        # Check if model is managed by accelerate (common for NF4/INT8)
        is_accelerate_mapped = hasattr(model, "hf_device_map")
        if is_accelerate_mapped:
            logger.info("✓ Detected Accelerate/Quantized model (hf_device_map present)")
            logger.info("  Skipping manual CPU offload to avoid conflicts with device_map.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("! Model does not appear to be quantized/mapped. Using standard Large logic.")

        original_offload_mode = offload_mode
        if is_accelerate_mapped:
            offload_mode = "disabled"
            logger.info("  Internal offload_mode set to 'disabled' for parent call (handled by device_map)")

        try:
            return super().generate_large(
                model, prompt, seed, steps, resolution, guidance_scale,
                offload_mode=offload_mode,
                post_action=post_action,
                enable_prompt_rewrite=enable_prompt_rewrite,
                rewrite_style=rewrite_style,
                api_url=api_url,
                model_name=model_name,
                cpu_offload=cpu_offload,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"Post-generation cleanup: {(total-free)/1024**3:.2f}GB used")


class HunyuanImage3GenerateTelemetry(HunyuanImage3Generate):
    """Standard generator with RAM/VRAM telemetry output in node status."""

    @classmethod
    def INPUT_TYPES(cls):
        return HunyuanImage3Generate.INPUT_TYPES()

    def generate(self, model, prompt, seed, steps, resolution, guidance_scale, post_action="keep_loaded",
                 enable_prompt_rewrite=False, rewrite_style="none",
                 api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat",
                 skip_device_check=False, keep_model_loaded=None):
        # Backward compatibility
        if keep_model_loaded is not None:
            post_action = "keep_loaded" if keep_model_loaded else "full_unload"
            
        tracker = MemoryTracker()
        image, rewritten_prompt, status, trigger = super().generate(
            model=model,
            prompt=prompt,
            seed=seed,
            steps=steps,
            resolution=resolution,
            guidance_scale=guidance_scale,
            post_action=post_action,
            enable_prompt_rewrite=enable_prompt_rewrite,
            rewrite_style=rewrite_style,
            api_url=api_url,
            model_name=model_name,
            skip_device_check=skip_device_check,
        )

        stats_text = format_memory_stats(tracker.finish())
        if stats_text:
            logger.info(f"MEMORY: {stats_text}")
            status = f"{status} | {stats_text}"
        return (image, rewritten_prompt, status, trigger)


class HunyuanImage3GenerateLargeBudget(HunyuanImage3GenerateLarge):
    """Large/offload generator with GPU budget awareness and telemetry."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        optional = dict(inputs.get("optional", {}))
        optional["gpu_budget_gb"] = ("FLOAT", {
            "default": -1.0,
            "min": -1.0,
            "max": 128.0,
            "step": 0.5,
            "tooltip": "Override loader GPU budget for this run (GB). -1 uses loader default."
        })
        inputs["optional"] = optional
        return inputs

    def _resolve_loader_reserve(self, model, total_vram_gb: float, gpu_budget_gb: float) -> float:
        """
        Determine how much VRAM was reserved by the loader for inference headroom.
        
        Args:
            model: The loaded model
            total_vram_gb: Total GPU VRAM in GB
            gpu_budget_gb: User-specified GPU budget override (-1 means use default)
            
        Returns:
            Estimated reserved VRAM in GB for inference
        """
        # If user specified a budget, calculate reserve from that
        if gpu_budget_gb > 0:
            return max(total_vram_gb - gpu_budget_gb, 0.0)
        
        # Try to get reserve from model metadata if available
        if hasattr(model, '_loader_reserve_gb'):
            return model._loader_reserve_gb
        
        # Default reserve based on model type
        if _is_model_int8_quantized(model):
            # INT8 models typically reserve ~15-20GB for inference
            return 15.0
        else:
            # NF4/other models reserve less
            return 6.0

    def generate_large(self, model, prompt, seed, steps, resolution, guidance_scale, offload_mode="smart", 
                       post_action="keep_loaded",
                       enable_prompt_rewrite=False, rewrite_style="none",
                       api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat",
                       cpu_offload=None, gpu_budget_gb=-1.0, keep_model_loaded=None):
        
        # Backward compatibility
        if keep_model_loaded is not None:
            post_action = "keep_loaded" if keep_model_loaded else "full_unload"

        if cpu_offload is not None:
            offload_mode = "always" if cpu_offload else "disabled"

        # === AUTO OFFLOAD OVERRIDE FOR INT8 MODELS ===
        # Automatically switch to 'smart' offload if:
        # 1. Model is INT8 quantized
        # 2. offload_mode is currently 'disabled'
        # 3. Resolution is 'Auto' or > 1.2MP
        # This prevents OOM errors on GPUs like RTX 6000 Pro Blackwell
        original_offload_mode = offload_mode
        if offload_mode == "disabled":
            is_int8 = _is_model_int8_quantized(model)
            megapixels = _calculate_resolution_megapixels(resolution)
            gpu_info = _get_gpu_info()
            
            if is_int8 and (megapixels > 1.2 or 'auto' in resolution.lower()):
                logger.warning("="*60)
                logger.warning("AUTO OFFLOAD OVERRIDE TRIGGERED")
                logger.warning(f"  Model: INT8 quantized")
                logger.warning(f"  Resolution: {resolution} ({megapixels:.1f}MP)")
                logger.warning(f"  GPU: {gpu_info['name']} ({gpu_info['total_vram_gb']:.0f}GB)")
                logger.warning(f"  Switching offload_mode from 'disabled' to 'smart' to prevent OOM")
                logger.warning(f"  TIP: For 1MP resolutions, 'disabled' mode is faster and safe")
                logger.warning("="*60)
                offload_mode = "smart"

        try:
            # For device_map models, we may have BOTH cuda and cpu params
            # Just need to verify at least SOME params are on cuda
            has_cuda_device = False
            for i, param in enumerate(model.parameters()):
                if i >= 100:  # Sample first 100 params
                    break
                if param.device.type == 'cuda':
                    has_cuda_device = True
                    break

            if not has_cuda_device:
                raise RuntimeError(
                    "Model has no CUDA params (likely cleared by Unload node). "
                    "Please ensure the model loader runs before generation in your workflow. "
                    "The model may have been unloaded by a previous Unload node."
                )
        except Exception as e:
            if "no cuda params" in str(e).lower() or "cleared by unload" in str(e).lower():
                raise
            logger.warning(f"Could not validate model device: {e}")

        tracker = MemoryTracker()
        tracker_starts = tracker.start_snapshot()
        if gpu_budget_gb is not None and gpu_budget_gb > 0:
            logger.info(f"GPU budget override set to {gpu_budget_gb:.1f}GB for this run")

        generator = HunyuanImage3Generate()
        if resolution == "Auto (model default)":
            width, height = 1600, 1600
            parsed_resolution = generator._default_resolution_label()
            logger.info("Auto resolution selected: assuming 1600x1600 for safety")
        else:
            parsed_resolution = resolution.split(" - ")[0]
            width, height = map(int, parsed_resolution.split("x"))

        should_offload = False
        if offload_mode == "always":
            should_offload = True
        elif offload_mode == "smart" and torch.cuda.is_available():
            megapixels = (width * height) / 1_000_000
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3

            loader_reserve_gb = self._resolve_loader_reserve(model, total_gb, gpu_budget_gb)
            # Use empirical formula: 1MP=12GB, 2MP=22GB, 3MP=45GB
            inference_requirement = _estimate_inference_vram_gb(megapixels) + loader_reserve_gb
            percent_used = ((total_gb - free_gb) / total_gb) * 100 if total_gb else 0

            if free_gb < 15.0:
                logger.info(f"Smart Offload: Low free VRAM ({free_gb:.1f}GB < 15GB). Enabling offload.")
                should_offload = True
            elif percent_used > 85.0:
                logger.info(f"Smart Offload: High VRAM usage ({percent_used:.1f}% > 85%). Enabling offload.")
                should_offload = True
            elif free_gb < inference_requirement:
                logger.info(f"Smart Offload: Need {inference_requirement:.1f}GB but only {free_gb:.1f}GB free. Enabling offload.")
                should_offload = True
            else:
                logger.info(f"Smart Offload: Sufficient VRAM ({free_gb:.1f}GB >= {inference_requirement:.1f}GB). Keeping on GPU.")

        logger.info("=" * 60)
        logger.info("LARGE IMAGE GENERATION MODE (Budget)")
        logger.info(f"Resolution: {width}x{height}")
        logger.info(f"Offload Mode: {offload_mode} -> {'Enabled' if should_offload else 'Disabled'}")
        start_ram = tracker_starts.get("ram_start_gb")
        start_vram = tracker_starts.get("vram_start_gb")
        total_vram = tracker_starts.get("vram_total_gb")
        if start_ram is not None and start_vram is not None and total_vram is not None:
            logger.info(f"Starting Memory: RAM={start_ram:.2f}GB, VRAM={start_vram:.2f}GB / {total_vram:.2f}GB")
        elif start_ram is not None:
            logger.info(f"Starting Memory: RAM={start_ram:.2f}GB")
        logger.info("=" * 60)

        if should_offload:
            try:
                from accelerate import cpu_offload as accelerate_cpu_offload
                logger.info("Enabling CPU offload for large image generation (budget node)...")

                has_meta = False
                for param in model.parameters():
                    if param.device.type == 'meta':
                        has_meta = True
                        break

                if has_meta:
                    logger.info("Model already managed by accelerate device_map; skipping manual cpu_offload call")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        free, total = torch.cuda.mem_get_info(0)
                        logger.info(f"GPU memory after clearing: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")
                else:
                    model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

                        free, total = torch.cuda.mem_get_info(0)
                        logger.info(f"GPU memory after clearing: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")

                    execution_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    offload_hook = accelerate_cpu_offload(model, execution_device=execution_device)
                    model._cpu_offload_hook = offload_hook
                    try:
                        model._offload_execution_device = torch.device(execution_device)
                    except Exception:
                        model._offload_execution_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                    _force_model_device_property(model, model._offload_execution_device)
                    logger.info("✓ CPU offload enabled via accelerate")

                    if hasattr(model, '_needs_meta_patch') and model._needs_meta_patch and hasattr(model, 'prepare_model_inputs'):
                        original_prepare = model.prepare_model_inputs

                        def patched_prepare_model_inputs(self, *args, **kwargs):
                            needs_patch = hasattr(self, 'device') and getattr(self.device, 'type', None) == 'meta'
                            target_device = getattr(self, '_offload_execution_device', None)
                            if target_device is None and torch.cuda.is_available():
                                target_device = torch.device("cuda:0")
                            if target_device is None:
                                target_device = torch.device("cpu")

                            if needs_patch:
                                original_device = self.device
                                self.__dict__['device'] = target_device
                            else:
                                original_device = None

                            try:
                                return original_prepare(*args, **kwargs)
                            finally:
                                if needs_patch:
                                    if original_device is None:
                                        self.__dict__.pop('device', None)
                                    else:
                                        self.__dict__['device'] = original_device

                        model.prepare_model_inputs = patched_prepare_model_inputs.__get__(model, type(model))
                        logger.info("✓ Applied meta device patch after CPU offload")
                        model._needs_meta_patch = False

            except ImportError:
                logger.warning("accelerate library not found, cannot enable CPU offload")
            except Exception as e:
                logger.warning(f"Could not configure CPU offload: {e}")

        original_get_target_size = None
        if hasattr(model, "image_processor") and hasattr(model.image_processor, "reso_group"):
            reso_group = model.image_processor.reso_group
            original_get_target_size = reso_group.get_target_size

            def new_get_target_size(w, h):
                key = (int(round(h)), int(round(w)))
                if key in reso_group._lookup:
                    return reso_group._lookup[key].w, reso_group._lookup[key].h
                return int(w), int(h)

            reso_group.get_target_size = new_get_target_size
            logger.info("Applied resolution patch to bypass bucket snapping")

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            image_tensor, rewritten_prompt, status, trigger = generator.generate(
                model=model,
                prompt=prompt,
                seed=seed,
                steps=steps,
                resolution=parsed_resolution,
                guidance_scale=guidance_scale,
                post_action=post_action,
                enable_prompt_rewrite=enable_prompt_rewrite,
                rewrite_style=rewrite_style,
                api_url=api_url,
                model_name=model_name,
                skip_device_check=should_offload
            )

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"GPU memory after generation: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")

            stats_text = format_memory_stats(tracker.finish())
            logger.info("=" * 60)
            logger.info("✓ Large image generated successfully (budget node)")
            if stats_text:
                logger.info(f"MEMORY: {stats_text}")
            logger.info("=" * 60)

            if stats_text:
                status = f"Large image mode (Offload: {offload_mode}) - {status} | {stats_text}"
            else:
                status = f"Large image mode (Offload: {offload_mode}) - {status}"

            return (image_tensor, rewritten_prompt, status, True)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU Out of Memory! Try reducing resolution or enabling offload.")
            raise
        finally:
            if original_get_target_size is not None:
                model.image_processor.reso_group.get_target_size = original_get_target_size
                logger.info("Restored original resolution logic")


class HunyuanImage3GenerateLowVRAMBudget(HunyuanImage3GenerateLargeBudget):
    """Low VRAM optimized generator with GPU budget override support."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = HunyuanImage3GenerateLowVRAM.INPUT_TYPES()
        optional = dict(inputs.get("optional", {}))
        optional["gpu_budget_gb"] = ("FLOAT", {
            "default": -1.0,
            "min": -1.0,
            "max": 128.0,
            "step": 0.5,
            "tooltip": "Override loader GPU budget for this run (GB). -1 uses loader default."
        })
        inputs["optional"] = optional
        return inputs

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "*")
    RETURN_NAMES = ("image", "rewritten_prompt", "status", "trigger")
    FUNCTION = "generate_low_vram"
    CATEGORY = "HunyuanImage3"

    def generate_low_vram(self, model, prompt, seed, steps, resolution, guidance_scale, offload_mode="smart", 
                          post_action="soft_unload_to_cpu",
                          enable_prompt_rewrite=False, rewrite_style="none",
                          api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat",
                          cpu_offload=None, gpu_budget_gb=-1.0, keep_model_loaded=None):
        
        # Backward compatibility
        if keep_model_loaded is not None:
            post_action = "keep_loaded" if keep_model_loaded else "full_unload"

        logger.info("=" * 60)
        logger.info("LOW VRAM GENERATION MODE (Budget)")
        logger.info("=" * 60)

        is_accelerate_mapped = hasattr(model, "hf_device_map")
        if is_accelerate_mapped:
            logger.info("✓ Detected Accelerate/Quantized model (hf_device_map present)")
            logger.info("  Skipping manual CPU offload to avoid conflicts with device_map.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            offload_mode = "disabled"
        else:
            logger.info("! Model does not appear to be quantized/mapped. Using standard Large logic.")

        try:
            return super().generate_large(
                model, prompt, seed, steps, resolution, guidance_scale,
                offload_mode=offload_mode,
                post_action=post_action,
                enable_prompt_rewrite=enable_prompt_rewrite,
                rewrite_style=rewrite_style,
                api_url=api_url,
                model_name=model_name,
                cpu_offload=cpu_offload,
                gpu_budget_gb=gpu_budget_gb,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"Post-generation cleanup: {(total-free)/1024**3:.2f}GB used")


NODE_CLASS_MAPPINGS = {
    "HunyuanImage3QuantizedLoader": HunyuanImage3QuantizedLoader,
    "HunyuanImage3Int8Loader": HunyuanImage3Int8Loader,
    "HunyuanImage3NF4LoaderLowVRAMBudget": HunyuanImage3NF4LoaderLowVRAMBudget,
    "HunyuanImage3Int8LoaderBudget": HunyuanImage3Int8LoaderBudget,
    "HunyuanImage3Generate": HunyuanImage3Generate,
    "HunyuanImage3GenerateTelemetry": HunyuanImage3GenerateTelemetry,
    "HunyuanImage3GenerateLarge": HunyuanImage3GenerateLarge,
    "HunyuanImage3GenerateLargeBudget": HunyuanImage3GenerateLargeBudget,
    "HunyuanImage3GenerateLowVRAM": HunyuanImage3GenerateLowVRAM,
    "HunyuanImage3GenerateLowVRAMBudget": HunyuanImage3GenerateLowVRAMBudget,
    "HunyuanImage3Unload": HunyuanImage3Unload,
    "HunyuanImage3SoftUnload": HunyuanImage3SoftUnload,
    "HunyuanImage3ForceUnload": HunyuanImage3ForceUnload,
    "HunyuanImage3ClearDownstream": HunyuanImage3ClearDownstream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage3QuantizedLoader": "Hunyuan 3 Loader (NF4)",
    "HunyuanImage3Int8Loader": "Hunyuan 3 Loader (INT8)",
    "HunyuanImage3NF4LoaderLowVRAMBudget": "Hunyuan 3 Loader (NF4 Low VRAM+)",
    "HunyuanImage3Int8LoaderBudget": "Hunyuan 3 Loader (INT8 Budget)",
    "HunyuanImage3Generate": "Hunyuan 3 Generate",
    "HunyuanImage3GenerateTelemetry": "Hunyuan 3 Generate (Telemetry)",
    "HunyuanImage3GenerateLarge": "Hunyuan 3 Generate (Large/Offload)",
    "HunyuanImage3GenerateLargeBudget": "Hunyuan 3 Generate (Large Budget)",
    "HunyuanImage3GenerateLowVRAM": "Hunyuan 3 Generate (Low VRAM)",
    "HunyuanImage3GenerateLowVRAMBudget": "Hunyuan 3 Generate (Low VRAM Budget)",
    "HunyuanImage3Unload": "Hunyuan 3 Unload",
    "HunyuanImage3SoftUnload": "Hunyuan 3 Soft Unload (Fast)",
    "HunyuanImage3ForceUnload": "Hunyuan 3 Force Unload (Nuclear)",
    "HunyuanImage3ClearDownstream": "Hunyuan 3 Clear Downstream Models",
}