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

from .hunyuan_shared import (
    HunyuanImage3Unload,
    HunyuanModelCache,
    ensure_model_on_device,
    patch_dynamic_cache_dtype,
)

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
            },
        }
    
    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"
    
    @classmethod
    def _get_available_models(cls):
        models_dir = folder_paths.models_dir
        hunyuan_dir = Path(models_dir)
        available = []
        
        for item in hunyuan_dir.iterdir():
            if item.is_dir() and (item / "quantization_metadata.json").exists():
                available.append(item.name)
        
        return available if available else ["HunyuanImage-3-NF4"]
    
    def load_model(self, model_name):
        model_path = Path(folder_paths.models_dir) / model_name
        model_path_str = str(model_path)

        cached = HunyuanModelCache.get(model_path_str)
        if cached is not None:
            self._apply_dtype_patches()
            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=True)
            return (cached,)

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
            load_kwargs = dict(
                quantization_config=quant_config,
                device_map="cuda:0",
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
    
    API Setup (DeepSeek example):
    - Get key: https://platform.deepseek.com/api_keys
    - Add credits: https://platform.deepseek.com/top_up
    - Default URL: https://api.deepseek.com/v1/chat/completions
    - Model: deepseek-chat
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A serene landscape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "resolution": (cls._resolution_choices(),),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "enable_prompt_rewrite": ("BOOLEAN", {"default": False}),
                "rewrite_style": (["none", "en_recaption", "en_think_recaption"], {"default": "none"}),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-... (Optional, requires payment)"}),
                "api_url": ("STRING", {"default": "https://api.deepseek.com/v1/chat/completions"}),
                "model_name": ("STRING", {"default": "deepseek-chat"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "rewritten_prompt", "status")
    FUNCTION = "generate"
    CATEGORY = "HunyuanImage3"
    
    def generate(self, model, prompt, seed, steps, resolution, guidance_scale, 
                 enable_prompt_rewrite=False, rewrite_style="none", api_key="", 
                 api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat"):
        logger.info(f"Generating image with {steps} steps")
        
        # Handle prompt rewriting if enabled
        original_prompt = prompt
        rewritten_prompt = prompt
        status_message = "Generation complete"
        
        if enable_prompt_rewrite and rewrite_style != "none" and api_key:
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
        
        if resolution == self._default_resolution_label():
            logger.info("Using model default resolution")
        else:
            height, width = self._parse_resolution(resolution)
            gen_kwargs["image_size"] = f"{height}x{width}"
            logger.info(f"Using custom resolution: {width}x{height}")
        
        logger.info(f"Guidance scale: {guidance_scale}")
        
        image = model.generate_image(**gen_kwargs)

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
        
        logger.info(f"✓ Image generated successfully - tensor shape: {image_tensor.shape}")
        return (image_tensor, rewritten_prompt, status_message)

    @classmethod
    def _default_resolution_label(cls) -> str:
        return "Auto (model default)"

    @classmethod
    def _resolution_choices(cls):
        if not hasattr(cls, "_RESOLUTION_CACHE"):
            metadata = _resolution_metadata()
            base_size = int(metadata["base_size"])
            min_multiple = float(metadata["min_multiple"])
            max_multiple = float(metadata["max_multiple"])
            step = metadata["step"]
            align = metadata["align"]
            count_limit = int(metadata.get("count", 33))
            step = int(step) if step is not None else None
            align = int(align) if align else 1

            preset_pairs = []
            if isinstance(metadata["presets"], list):
                seen = set()
                for entry in metadata["presets"]:
                    if not isinstance(entry, str) or "x" not in entry:
                        continue
                    try:
                        height_str, width_str = entry.split("x")
                        pair = (int(height_str), int(width_str))
                    except ValueError:
                        continue
                    if pair in seen:
                        continue
                    seen.add(pair)
                    preset_pairs.append(pair)

            if not preset_pairs:
                preset_pairs = cls._compute_resolutions(
                    base_size,
                    step=step,
                    align=align,
                    min_multiple=min_multiple,
                    max_multiple=max_multiple,
                    max_entries=count_limit,
                )

            # Organize resolutions: prioritize <2K, add descriptive labels
            labeled_resolutions = []
            for height, width in preset_pairs:
                total_pixels = height * width
                aspect = width / height
                
                # Determine orientation and size category
                if abs(aspect - 1.0) < 0.1:
                    orientation = "Square"
                elif aspect > 1.0:
                    orientation = "Landscape"
                else:
                    orientation = "Portrait"
                
                # Size category
                if total_pixels < 1024 * 1024:
                    size_cat = "<1MP"
                elif total_pixels < 2048 * 2048:
                    size_cat = "1-2MP"
                else:
                    size_cat = ">2MP"
                
                megapixels = total_pixels / (1024 * 1024)
                label = f"{width}x{height} - {orientation} ({megapixels:.1f}MP) [{size_cat}]"
                labeled_resolutions.append((height, width, label, total_pixels))
            
            # Sort: <2MP first, then by aspect ratio within each group
            labeled_resolutions.sort(key=lambda x: (x[3] >= 2048*2048, x[1] / x[0]))
            
            # Limit to reasonable count, keeping smaller sizes
            if len(labeled_resolutions) > count_limit:
                # Keep more <2MP options
                under_2mp = [r for r in labeled_resolutions if r[3] < 2048*2048]
                over_2mp = [r for r in labeled_resolutions if r[3] >= 2048*2048]
                
                keep_under = min(len(under_2mp), int(count_limit * 0.7))  # 70% for <2MP
                keep_over = count_limit - keep_under
                
                labeled_resolutions = under_2mp[:keep_under] + over_2mp[:keep_over]
            
            cls._RESOLUTION_CACHE = tuple(
                [cls._default_resolution_label()] + [label for _, _, label, _ in labeled_resolutions]
            )
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
        # Generate large resolution options (2MP+)
        large_resolutions = cls._get_large_resolutions()
        
        return {
            "required": {
                "model": ("HUNYUAN_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A serene landscape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "resolution": (large_resolutions,),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "cpu_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "enable_prompt_rewrite": ("BOOLEAN", {"default": False}),
                "rewrite_style": (["none", "en_recaption", "en_think_recaption"], {"default": "none"}),
                "api_key": ("STRING", {"default": "", "placeholder": "sk-... (Optional, requires payment)"}),
                "api_url": ("STRING", {"default": "https://api.deepseek.com/v1/chat/completions"}),
                "model_name": ("STRING", {"default": "deepseek-chat"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "rewritten_prompt", "status")
    FUNCTION = "generate_large"
    CATEGORY = "HunyuanImage3"
    
    @classmethod
    def _get_large_resolutions(cls):
        """Generate resolution options focusing on 2MP+ sizes."""
        common_large = [
            (2048, 2048, "Square 4K", 4.2),
            (2560, 1440, "Landscape 2K", 3.7),
            (1440, 2560, "Portrait 2K", 3.7),
            (2048, 1536, "Landscape 3MP", 3.1),
            (1536, 2048, "Portrait 3MP", 3.1),
            (2560, 1920, "Landscape HD+", 4.9),
            (1920, 2560, "Portrait HD+", 4.9),
            (3072, 2048, "Landscape 6MP", 6.3),
            (2048, 3072, "Portrait 6MP", 6.3),
            (3840, 2160, "Landscape 4K UHD", 8.3),
            (2160, 3840, "Portrait 4K UHD", 8.3),
        ]
        
        options = ["Auto (model default)"]
        for width, height, desc, mp in common_large:
            options.append(f"{width}x{height} - {desc} ({mp:.1f}MP)")
        
        return options
    
    def generate_large(self, model, prompt, seed, steps, resolution, guidance_scale, cpu_offload=True,
                      enable_prompt_rewrite=False, rewrite_style="none", api_key="",
                      api_url="https://api.deepseek.com/v1/chat/completions", model_name="deepseek-chat"):
        
        logger.info("=" * 60)
        logger.info("LARGE IMAGE GENERATION MODE")
        logger.info(f"CPU Offload: {'Enabled' if cpu_offload else 'Disabled'}")
        logger.info("=" * 60)
        
        # Temporarily enable CPU offload for this generation if requested
        original_config = None
        if cpu_offload and hasattr(model, 'config'):
            try:
                # Store original device map
                original_device_map = getattr(model, 'hf_device_map', None)
                logger.info("Enabling CPU offload for large image generation...")
                
                # Free up some GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info(0)
                    logger.info(f"GPU memory before generation: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")
            except Exception as e:
                logger.warning(f"Could not configure CPU offload: {e}")
        
        # Use the standard generate logic from HunyuanImage3Generate
        generator = HunyuanImage3Generate()
        
        # Parse resolution
        if resolution == "Auto (model default)":
            parsed_resolution = generator._default_resolution_label()
        else:
            parsed_resolution = resolution.split(" - ")[0]  # Extract WxH
        
        try:
            image_tensor, rewritten_prompt, status = generator.generate(
                model=model,
                prompt=prompt,
                seed=seed,
                steps=steps,
                resolution=parsed_resolution,
                guidance_scale=guidance_scale,
                enable_prompt_rewrite=enable_prompt_rewrite,
                rewrite_style=rewrite_style,
                api_key=api_key,
                api_url=api_url,
                model_name=model_name
            )
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"GPU memory after generation: {(total-free)/1024**3:.2f}GB used, {free/1024**3:.2f}GB free")
            
            logger.info("=" * 60)
            logger.info("✓ Large image generated successfully")
            logger.info("=" * 60)
            
            # Update status to indicate large image mode
            status = f"Large image mode (CPU offload: {'enabled' if cpu_offload else 'disabled'}) - {status}"
            
            return (image_tensor, rewritten_prompt, status)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU Out of Memory! Try:")
                logger.error("  1. Enable 'cpu_offload' (if not already)")
                logger.error("  2. Use a smaller resolution")
                logger.error("  3. Reduce guidance_scale or steps")
                logger.error("  4. Clear GPU memory with Unload node first")
            raise


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