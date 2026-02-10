"""
HunyuanImage-3.0 MoE Efficient Test Node

A standalone test node that applies the memory-efficient MoE forward patch
during generation. Use this to compare quality/prompt-following side-by-side
with the regular generate node (which uses the standard MoE implementation).

The ONLY difference from the standard generate node is the MoE dispatch method:
- Standard: dispatch_mask-based (O(N²) VRAM, official implementation)
- This node: loop-based (O(N) VRAM, memory-efficient)

Both produce the same routing decisions — the loop dispatches tokens to the
same experts in the same order. The question is whether numerical precision
differences (summation order, etc.) affect output quality.

To test: Run both nodes with the SAME prompt, seed, steps, resolution,
and guidance_scale. Compare the results visually.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import gc
import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image

# Import shared utilities
try:
    from .hunyuan_shared import (
        HunyuanModelCache,
        patch_moe_efficient_forward,
        unpatch_moe_efficient_forward,
    )
    SHARED_AVAILABLE = True
except ImportError:
    try:
        from hunyuan_shared import (
            HunyuanModelCache,
            patch_moe_efficient_forward,
            unpatch_moe_efficient_forward,
        )
        SHARED_AVAILABLE = True
    except ImportError:
        SHARED_AVAILABLE = False
        patch_moe_efficient_forward = None
        unpatch_moe_efficient_forward = None

# ComfyUI imports
try:
    from comfy.utils import ProgressBar
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    ProgressBar = None

logger = logging.getLogger(__name__)


# Resolution presets — same as the standard generate node
STANDARD_RESOLUTIONS = {
    "1024x1024 (1:1)": (1024, 1024),
    "1024x576 (16:9)": (576, 1024),
    "576x1024 (9:16)": (1024, 576),
    "1024x672 (3:2)": (672, 1024),
    "672x1024 (2:3)": (1024, 672),
    "1024x768 (4:3)": (768, 1024),
    "768x1024 (3:4)": (1024, 768),
    "1024x864 (~6:5)": (864, 1024),
    "864x1024 (~5:6)": (1024, 864),
}


class HunyuanImage3MoETest:
    """
    Test node for memory-efficient MoE dispatch.
    
    Applies the efficient MoE forward patch before generation and restores
    the original after. Use this to A/B test against the standard generate
    node with identical parameters.
    
    The patch replaces dispatch_mask-based routing (O(N²) VRAM) with
    loop-based routing (O(N) VRAM). The routing decisions are identical —
    tokens go to the same experts. The only difference is the summation
    order, which can affect numerical precision.
    
    Connect to any Hunyuan 3 loader (NF4, INT8, or BF16).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "resolution": (list(STANDARD_RESOLUTIONS.keys()),),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "post_action": (["keep_loaded", "soft_unload_to_cpu", "full_unload"], {
                    "default": "keep_loaded",
                    "tooltip": "After generation: keep_loaded (fastest reruns), soft_unload_to_cpu (free VRAM), full_unload (free all)"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "generate"
    CATEGORY = "HunyuanImage3/Test"
    
    def generate(
        self,
        model,
        prompt: str,
        seed: int,
        steps: int,
        resolution: str,
        guidance_scale: float,
        post_action: str = "keep_loaded",
    ) -> Tuple[torch.Tensor, str]:
        """Generate image using efficient MoE dispatch for quality testing."""
        
        if not SHARED_AVAILABLE or patch_moe_efficient_forward is None:
            raise RuntimeError(
                "MoE patch functions not available. "
                "Ensure hunyuan_shared.py is present and contains "
                "patch_moe_efficient_forward / unpatch_moe_efficient_forward."
            )
        
        # Parse resolution
        height, width = STANDARD_RESOLUTIONS.get(resolution, (1024, 1024))
        
        # Set seed
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        logger.info("=" * 60)
        logger.info("MoE EFFICIENT TEST NODE")
        logger.info("  This uses the loop-based MoE dispatch (memory-efficient)")
        logger.info("  Compare results with the standard generate node")
        logger.info("=" * 60)
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  Steps: {steps}, Seed: {seed}, CFG: {guidance_scale}")
        
        # Apply efficient MoE patch
        logger.info("Applying efficient MoE forward patch...")
        patch_moe_efficient_forward(model)
        
        try:
            # Clear VRAM before generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log VRAM before
            if torch.cuda.is_available():
                free_before, _ = torch.cuda.mem_get_info(0)
                logger.info(f"  VRAM free before generation: {free_before / 1024**3:.1f}GB")
            
            # Build generation kwargs — identical to standard generate node
            gen_kwargs = {
                "prompt": prompt,
                "diff_infer_steps": steps,
                "stream": True,
                "diff_guidance_scale": guidance_scale,
                "image_size": f"{height}x{width}",
            }
            
            # Progress bar
            if ProgressBar:
                pbar = ProgressBar(steps)
                def callback(pipe, step, timestep, callback_kwargs):
                    pbar.update(1)
                    return callback_kwargs
                gen_kwargs["callback_on_step_end"] = callback
            
            # Patch resolution group to force exact size (bypass bucket snapping)
            original_get_target_size = None
            if hasattr(model, "image_processor") and hasattr(model.image_processor, "reso_group"):
                reso_group = model.image_processor.reso_group
                original_get_target_size = reso_group.get_target_size
                reso_group.get_target_size = lambda w, h: (int(w), int(h))
            
            try:
                image = model.generate_image(**gen_kwargs)
            finally:
                # Restore resolution group
                if original_get_target_size is not None:
                    model.image_processor.reso_group.get_target_size = original_get_target_size
            
            # Handle streaming iterator
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
                raise RuntimeError(f"Unexpected return type: {type(image)}")
            
            # Convert to ComfyUI format
            if image.mode == "I":
                image = image.point(lambda i: i * (1 / 255))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            status = (
                f"MoE Efficient Test | {width}x{height} | "
                f"Steps: {steps} | Seed: {seed} | CFG: {guidance_scale}"
            )
            logger.info(f"Generation complete: {status}")
            
        finally:
            # ALWAYS restore original MoE forward — this is critical
            # to not affect subsequent generations with the standard node
            logger.info("Restoring original MoE forward...")
            unpatch_moe_efficient_forward(model)
        
        # Handle post-generation action
        if post_action == "soft_unload_to_cpu":
            logger.info("Post-action: Soft unloading model to CPU RAM...")
            HunyuanModelCache.soft_unload()
        elif post_action == "full_unload":
            logger.info("Post-action: Full unloading model...")
            HunyuanModelCache.full_unload()
        
        return (image_tensor, status)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "HunyuanImage3MoETest": HunyuanImage3MoETest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage3MoETest": "Hunyuan MoE Efficient Test (A/B Quality)",
}
