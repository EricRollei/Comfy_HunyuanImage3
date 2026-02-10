"""
HunyuanImage-3.0 ComfyUI Custom Nodes - High Resolution Generation
Memory-efficient generation nodes for 2MP+ resolutions using optimized MoE dispatch.

The standard generate nodes use the upstream MoE "eager" implementation which
materializes a dispatch_mask of shape [N_tokens, num_experts, expert_capacity].
At 3MP with CFG (guidance_scale > 1), this dispatch_mask requires 10-37GB of VRAM
just for MoE routing intermediates, causing OOM on even 96GB GPUs.

This module provides a HighRes generate node that monkey-patches the MoE forward
pass to use a loop-based expert dispatch. This reduces MoE intermediate memory from
O(N² * topk / experts) to O(N * hidden_size) — roughly 75x less VRAM at 3MP.

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

import gc
import logging

import numpy as np
import torch
from PIL import Image

from .hunyuan_shared import (
    HunyuanModelCache,
    patch_moe_efficient_forward,
    unpatch_moe_efficient_forward,
)

logger = logging.getLogger(__name__)


class HunyuanImage3GenerateHighRes:
    """
    Generate high-resolution images (2MP–4MP+) using memory-efficient MoE dispatch.
    
    This node is specifically designed for large BF16 model generations at resolutions
    above 2MP where the standard generate nodes run out of VRAM.
    
    How it works:
    The standard MoE "eager" implementation creates a dispatch_mask tensor of shape
    [N_tokens, 64_experts, expert_capacity] which requires 10-37GB+ at 3MP with CFG.
    This node patches the MoE forward to use a simple loop-based dispatch that needs
    only O(N * hidden_size) — reducing peak intermediate memory by ~75x.
    
    VRAM Requirements (BF16 model, ~30GB weights on GPU):
    - 2MP (1920x1080): ~50GB total (works on standard generate too)
    - 3MP (2048x1536): ~55GB total (OOMs on standard generate, works here)
    - 4MP (2560x1920): ~70GB total
    - 9MP (3840x2160): ~90GB+ total (tight on 96GB GPU)
    
    Speed: Similar to standard generate — same expert MLPs run on same data.
    Quality: Identical — same routing decisions, just dispatched differently.
    
    Connect to any Hunyuan 3 BF16 loader node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "resolution": (cls._get_highres_resolutions(),),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "offload_mode": (["smart", "always", "disabled"], {
                    "default": "smart",
                    "tooltip": "smart: auto-offload if VRAM tight. always: always offload (slow but safe). disabled: keep everything on GPU."
                }),
                "post_action": (["keep_loaded", "soft_unload_to_cpu", "full_unload"], {
                    "default": "keep_loaded",
                    "tooltip": "After generation: keep_loaded (fastest reruns), soft_unload_to_cpu (free VRAM, ~10s restore), full_unload (free VRAM+RAM, ~35s reload)"
                }),
            },
            "optional": {
                "enable_prompt_rewrite": ("BOOLEAN", {"default": False}),
                "rewrite_style": (["none", "en_recaption", "en_think_recaption"], {"default": "none"}),
                "api_url": ("STRING", {"default": "https://api.deepseek.com/v1/chat/completions"}),
                "model_name": ("STRING", {"default": "deepseek-chat"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "*")
    RETURN_NAMES = ("image", "rewritten_prompt", "status", "trigger")
    FUNCTION = "generate_highres"
    CATEGORY = "HunyuanImage3"
    
    @classmethod
    def _get_highres_resolutions(cls):
        """Resolution options focused on high-res (2MP+) but including standard sizes."""
        base_sizes = [
            # 1MP Class (Standard) — included for versatility
            (1024, 1024, "Square 1K"),
            (1152, 864, "Landscape 4:3"),
            (1280, 720, "Landscape 720p"),
            (864, 1152, "Portrait 3:4"),
            (720, 1280, "Portrait 720p"),
            
            # 2MP Class
            (1440, 1440, "Square 1.5K"),
            (1600, 1200, "Landscape 4:3"),
            (1920, 1080, "Landscape 1080p"),
            (1200, 1600, "Portrait 3:4"),
            (1080, 1920, "Portrait 1080p"),
            
            # 3MP Class
            (2048, 1536, "Landscape 4:3"),
            (2048, 2048, "Square 2K"),
            (1536, 2048, "Portrait 3:4"),
            (2560, 1440, "Landscape 2K"),
            (1440, 2560, "Portrait 2K"),
            
            # 4MP+ Class
            (2560, 1920, "Landscape 5:4"),
            (1920, 2560, "Portrait 4:5"),
            (2560, 1536, "Landscape 5:3"),
            (1536, 2560, "Portrait 3:5"),
            
            # 6MP+ (4K-class)
            (3072, 2048, "Landscape 3:2"),
            (2048, 3072, "Portrait 2:3"),
            (3072, 1728, "Landscape 16:9"),
            (1728, 3072, "Portrait 9:16"),
            
            # 8MP+ (Ultra)
            (3840, 2160, "Landscape 4K"),
            (2160, 3840, "Portrait 4K"),
            (3584, 2048, "Landscape 7:4"),
            (2048, 3584, "Portrait 4:7"),
            (4096, 4096, "Square 4K"),
        ]
        
        options = ["Auto (model default)"]
        for w, h, desc in base_sizes:
            mp = (w * h) / 1_000_000
            options.append(f"{w}x{h} - {desc} ({mp:.1f}MP)")
            
        return options

    def generate_highres(self, model, prompt, seed, steps, resolution, guidance_scale,
                         offload_mode="smart", post_action="keep_loaded",
                         enable_prompt_rewrite=False, rewrite_style="none",
                         api_url="https://api.deepseek.com/v1/chat/completions",
                         model_name="deepseek-chat"):
        
        from comfy.utils import ProgressBar
        
        # ── Validate model has CUDA params ──
        try:
            has_cuda = any(p.device.type == 'cuda' for i, p in zip(range(100), model.parameters()))
            if not has_cuda:
                raise RuntimeError(
                    "Model has no CUDA params (likely cleared by Unload node). "
                    "Please ensure the loader runs before generation in your workflow."
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not validate model device: {e}")
        
        # ── Parse resolution ──
        if resolution == "Auto (model default)":
            width, height = 1600, 1600
            resolution_str = None  # Let model choose
            logger.info("Auto resolution selected")
        else:
            parts = resolution.split(" - ")[0]
            width, height = map(int, parts.split("x"))
            resolution_str = f"{height}x{width}"
        
        megapixels = (width * height) / 1_000_000
        
        # ── Detect device_map model ──
        # Models loaded with device_map="auto" already have accelerate hooks that
        # manage device placement automatically (layers move to GPU during forward,
        # back to CPU after). We must NOT apply cpu_offload on top of device_map.
        has_device_map = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
        has_meta = any(p.device.type == 'meta' for i, p in zip(range(50), model.parameters()))
        is_device_map_model = has_device_map or has_meta
        
        if is_device_map_model:
            logger.info("Model was loaded with device_map (accelerate manages device placement)")
            logger.info("Skipping manual CPU offload — accelerate hooks handle layer movement automatically")
        
        # ── Smart Offload Logic ──
        # With efficient MoE, we need MUCH less VRAM than the standard estimate.
        # Attention + KV cache dominate: ~12 * MP^1.4 GB, plus ~2-4GB MoE overhead.
        # (The standard nodes need 30-40GB extra for dispatch_mask, but we don't.)
        should_offload = False
        if is_device_map_model:
            # Device-map models ALREADY offload — don't add another layer.
            # The accelerate hooks move each layer GPU⟷CPU automatically.
            should_offload = False
            offload_status = "Handled by device_map"
        elif offload_mode == "always":
            should_offload = True
            offload_status = "Forced ON"
        elif offload_mode == "smart" and torch.cuda.is_available():
            # Efficient MoE overhead is ~2-4GB (vs 30-40GB for standard dispatch_mask)
            required_free_gb = 12.0 * (megapixels ** 1.4) + 4.0
            
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1024**3
            
            if free_gb < required_free_gb:
                logger.info(f"Smart Offload: Free VRAM ({free_gb:.1f}GB) < Required ({required_free_gb:.1f}GB) for {megapixels:.1f}MP. Enabling offload.")
                should_offload = True
                offload_status = "Enabled (low VRAM)"
            else:
                logger.info(f"Smart Offload: Sufficient VRAM ({free_gb:.1f}GB >= {required_free_gb:.1f}GB). Skipping offload.")
                offload_status = "Skipped (enough VRAM)"
        else:
            offload_status = "Disabled"
        
        logger.info("=" * 60)
        logger.info("HIGH-RESOLUTION GENERATION (Efficient MoE)")
        logger.info(f"Resolution: {width}x{height} ({megapixels:.1f}MP)")
        logger.info(f"Offload: {offload_status}")
        logger.info("=" * 60)
        
        # ── Apply CPU offload if needed (only for non-device_map models) ──
        if should_offload and not is_device_map_model:
            try:
                from accelerate import cpu_offload as accelerate_cpu_offload
                logger.info("Enabling CPU offload for high-res generation...")
                
                model.cpu()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info(0)
                    logger.info(f"After CPU move: {free/1024**3:.1f}GB free of {total/1024**3:.1f}GB")
                
                accelerate_cpu_offload(model, execution_device="cuda:0")
                logger.info("✓ CPU offload enabled via accelerate")
            except ImportError:
                logger.warning("accelerate not available, cannot enable CPU offload")
            except Exception as e:
                logger.warning(f"Could not configure CPU offload: {e}")
        
        # ── Prompt rewriting ──
        original_prompt = prompt
        rewritten_prompt = prompt
        status_message = "Using original prompt"
        
        if enable_prompt_rewrite and rewrite_style != "none":
            try:
                from .hunyuan_api_config import HunyuanAPIConfig
                config = HunyuanAPIConfig.load_config()
                api_key = config.get("api_key")
                
                if api_key:
                    from .hunyuan_quantized_nodes import HunyuanImage3Generate
                    gen = HunyuanImage3Generate()
                    prompt = gen._rewrite_prompt_with_llm(prompt, rewrite_style, {
                        "url": api_url, "key": api_key, "model": model_name
                    })
                    rewritten_prompt = prompt
                    status_message = f"Prompt rewritten ({rewrite_style})"
                    logger.info(f"Prompt rewritten: {prompt[:80]}...")
                else:
                    logger.warning("Prompt rewrite enabled but no API key found")
                    status_message = "Prompt rewrite skipped (no API key)"
            except Exception as e:
                logger.warning(f"Prompt rewriting failed: {e}")
                prompt = original_prompt
                status_message = f"Prompt rewrite failed: {str(e)[:80]}"
        
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # ── Seed ──
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # ── CRITICAL: Apply efficient MoE patch ──
        logger.info("Applying memory-efficient MoE dispatch patch...")
        patch_moe_efficient_forward(model)
        
        # ── Patch resolution group ──
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
        
        # ── Build generation kwargs ──
        gen_kwargs = {
            "prompt": prompt,
            "diff_infer_steps": steps,
            "stream": True,
            "diff_guidance_scale": guidance_scale,
        }
        
        # Progress bar
        pbar = ProgressBar(steps)
        def callback(pipe, step, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs
        gen_kwargs["callback_on_step_end"] = callback
        
        if resolution_str is not None:
            gen_kwargs["image_size"] = resolution_str
            logger.info(f"Using resolution: {width}x{height}")
        
        logger.info(f"Guidance scale: {guidance_scale}, Steps: {steps}")
        
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"VRAM before generation: {(total-free)/1024**3:.1f}GB used, {free/1024**3:.1f}GB free")
            
            image = model.generate_image(**gen_kwargs)
            
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                logger.info(f"VRAM after generation: {(total-free)/1024**3:.1f}GB used, {free/1024**3:.1f}GB free")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU OOM even with efficient MoE! Try:")
                logger.error("  1. Set offload_mode to 'always'")
                logger.error("  2. Use a smaller resolution")
                logger.error("  3. Run Force Unload node first")
                logger.error("  4. For 4K+, you may need >96GB VRAM")
            raise
        finally:
            # Restore resolution patch
            if original_get_target_size is not None:
                model.image_processor.reso_group.get_target_size = original_get_target_size
                logger.info("Restored original resolution logic")
            
            # Note: We do NOT unpatch the MoE forward here.
            # The efficient forward is safe for all resolutions and avoids re-patching overhead.
            # It will be automatically replaced if the model is reloaded.
        
        # ── Process output ──
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
            raise RuntimeError(f"Unexpected return type from generate_image: {type(image)}")
        
        logger.info(f"Output image: mode={image.mode}, size={image.size}")
        
        if image.mode == "I":
            image = image.point(lambda i: i * (1 / 255))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # ── Post-generation action ──
        if post_action == "soft_unload_to_cpu":
            logger.info("Post-action: Moving model to CPU...")
            if HunyuanModelCache.soft_unload():
                logger.info("✓ Model moved to CPU RAM")
            else:
                logger.warning("Soft unload failed")
        elif post_action == "full_unload":
            logger.info("Post-action: Full unload...")
            HunyuanModelCache.clear()
            logger.info("✓ Model cleared")
        
        logger.info("=" * 60)
        logger.info(f"✓ High-res image generated: {image_tensor.shape}")
        logger.info("=" * 60)
        
        status_message = f"HighRes ({width}x{height}, {megapixels:.1f}MP, efficient MoE) - {status_message}"
        return (image_tensor, rewritten_prompt, status_message, True)


NODE_CLASS_MAPPINGS = {
    "HunyuanImage3GenerateHighRes": HunyuanImage3GenerateHighRes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage3GenerateHighRes": "Hunyuan 3 Generate (HighRes Efficient)",
}
