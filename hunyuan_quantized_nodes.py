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
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from comfy.utils import ProgressBar

from .hunyuan_shared import (
    HunyuanImage3Unload,
    HunyuanModelCache,
    ensure_model_on_device,
    patch_dynamic_cache_dtype,
    patch_hunyuan_generate_image,
)
from .hunyuan_api_config import get_api_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _resolution_metadata():
    base_size = 1024
    min_multiple = 0.5
    max_multiple = 2.0
    step = None
    align = 1
    presets = None

    models_dir = Path(folder_paths.models_dir)
    candidate_dirs = []
    default_dir = models_dir / "HunyuanImage-3"
    if default_dir.exists():
        candidate_dirs.append(default_dir)
    for item in models_dir.iterdir():
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
                    "tooltip": "VRAM to leave free. Standard: 6GB. Large images (>2MP) need ~12GB/MP free (use Large Generate node to offload)."
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
        models_dir = folder_paths.models_dir
        hunyuan_dir = Path(models_dir)
        available = []
        
        for item in hunyuan_dir.iterdir():
            if item.is_dir() and (item / "quantization_metadata.json").exists():
                available.append(item.name)
        
        return available if available else ["HunyuanImage-3-NF4"]
    
    def load_model(self, model_name, force_reload=False, unload_signal=None, reserve_memory_gb=6.0):
        # force_reload: if True, always reload model even if cached
        # unload_signal: forces re-execution if model was cleared (changes on each unload)
        model_path = Path(folder_paths.models_dir) / model_name
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
                    # Check if model has valid device placement
                    # After unload, models may have None device indices which cause errors
                    has_valid_device = False
                    try:
                        # Check a model parameter to see if it has a valid device
                        for param in cached.parameters():
                            if param.device.type == 'cuda' and param.device.index is not None:
                                has_valid_device = True
                                break
                            elif param.device.type == 'cpu':
                                # Model was moved to CPU, needs to be reloaded to GPU
                                logger.info("Cached model is on CPU, clearing and reloading to GPU...")
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
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
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
    
    def generate(self, model, prompt, seed, steps, resolution, guidance_scale, keep_model_loaded=True,
                 enable_prompt_rewrite=False, rewrite_style="none", 
                 api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat", skip_device_check=False):
        # Validate model has valid device placement before generation
        if not skip_device_check:
            try:
                has_valid_device = False
                for param in model.parameters():
                    if param.device.type == 'cuda' and param.device.index is not None:
                        has_valid_device = True
                        break
                    elif param.device.type == 'cpu' or param.device.index is None:
                        break
                
                if not has_valid_device:
                    raise RuntimeError(
                        "Model has invalid device placement (likely cleared by Unload node). "
                        "Please ensure the model loader runs before generation in your workflow. "
                        "The model may have been unloaded by a previous Unload node."
                    )
            except Exception as e:
                if "invalid device" in str(e).lower() or "cleared by unload" in str(e).lower():
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
        
        if not keep_model_loaded:
            HunyuanModelCache.clear()

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
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "resolution": (cls._get_large_resolutions(),),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "offload_mode": (["smart", "always", "disabled"], {"default": "smart"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
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
    
    def generate_large(self, model, prompt, seed, steps, resolution, guidance_scale, offload_mode="smart", keep_model_loaded=True,
                      enable_prompt_rewrite=False, rewrite_style="none",
                      api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat", cpu_offload=None):
        
        # Backward compatibility for cpu_offload boolean
        if cpu_offload is not None:
            offload_mode = "always" if cpu_offload else "disabled"

        # Validate model has valid device placement before generation
        try:
            has_valid_device = False
            for param in model.parameters():
                if param.device.type == 'cuda' and param.device.index is not None:
                    has_valid_device = True
                    break
                elif param.device.type == 'cpu' or param.device.index is None:
                    break
            
            if not has_valid_device:
                raise RuntimeError(
                    "Model has invalid device placement (likely cleared by Unload node). "
                    "Please ensure the model loader runs before generation in your workflow. "
                    "The model may have been unloaded by a previous Unload node."
                )
        except Exception as e:
            if "invalid device" in str(e).lower() or "cleared by unload" in str(e).lower():
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
            # Estimate memory requirements
            megapixels = (width * height) / 1_000_000
            # Empirical testing shows very high VRAM usage for activations
            # 2.1MP -> ~24GB
            # 3.1MP -> ~38GB
            # Formula adjusted to: 2GB base + 12GB per MP to be safe
            required_free_gb = 2.0 + (megapixels * 12.0)
            
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
            try:
                from accelerate import cpu_offload as accelerate_cpu_offload
                logger.info("Enabling CPU offload for large image generation...")
                
                # FORCE MOVE TO CPU FIRST
                # This ensures VRAM is actually freed before we start offloading hooks
                # If the model is currently on GPU (from FullLoader), this is critical.
                logger.info("Moving model to CPU to ensure VRAM is free...")
                
                # Check for meta tensors first to avoid crash
                has_meta = False
                for param in model.parameters():
                    if param.device.type == 'meta':
                        has_meta = True
                        break
                
                if has_meta:
                    logger.info("Model has meta tensors (managed by accelerate), skipping manual .cpu() move.")
                else:
                    model.cpu()
                    
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info(0)
                    logger.info(f"GPU memory before generation: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")

                # Apply cpu_offload
                # This moves the model to CPU (redundant but safe) and adds hooks to move layers to GPU during forward
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
                keep_model_loaded=keep_model_loaded,
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


NODE_CLASS_MAPPINGS = {
    "HunyuanImage3QuantizedLoader": HunyuanImage3QuantizedLoader,
    "HunyuanImage3Generate": HunyuanImage3Generate,
    "HunyuanImage3GenerateLarge": HunyuanImage3GenerateLarge,
    "HunyuanImage3Unload": HunyuanImage3Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage3QuantizedLoader": "Hunyuan 3 Loader (NF4)",
    "HunyuanImage3Generate": "Hunyuan 3 Generate",
    "HunyuanImage3GenerateLarge": "Hunyuan 3 Generate (Large/Offload)",
    "HunyuanImage3Unload": "Hunyuan 3 Unload",
}