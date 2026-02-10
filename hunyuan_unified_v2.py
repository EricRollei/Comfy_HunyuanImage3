"""
HunyuanImage-3.0 Unified Generation Node V2

A complete rewrite of the unified node using explicit block swapping
instead of accelerate hooks. Compatible with all ComfyUI extensions.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import gc
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# ComfyUI imports
try:
    import comfy.model_management as mm
    import folder_paths
    from comfy.utils import ProgressBar
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    ProgressBar = None

# Local imports - handle both package and standalone import
try:
    from .hunyuan_memory_budget import MemoryBudget, log_vram_status
    from .hunyuan_vae_simple import SimpleVAEManager, VAEPlacement
    from .hunyuan_loader_clean import CleanModelLoader
    from .hunyuan_block_swap import BlockSwapConfig, BlockSwapManager, calculate_blocks_to_swap
    from .hunyuan_cache_v2 import get_cache, CachedModel
    from .hunyuan_shared import (
        HUNYUAN_FOLDER_NAME,
        get_available_hunyuan_models,
        patch_dynamic_cache_dtype,
        patch_hunyuan_generate_image,
        resolve_hunyuan_model_path,
    )
except ImportError:
    from hunyuan_memory_budget import MemoryBudget, log_vram_status
    from hunyuan_vae_simple import SimpleVAEManager, VAEPlacement
    from hunyuan_loader_clean import CleanModelLoader
    from hunyuan_block_swap import BlockSwapConfig, BlockSwapManager, calculate_blocks_to_swap
    from hunyuan_cache_v2 import get_cache, CachedModel
    from hunyuan_shared import (
        HUNYUAN_FOLDER_NAME,
        get_available_hunyuan_models,
        patch_dynamic_cache_dtype,
        patch_hunyuan_generate_image,
        resolve_hunyuan_model_path,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Resolution Presets - Based on HunyuanImage3 Actual Supported Resolutions
# =============================================================================
# HunyuanImage3 uses a 1024 base with standard aspect ratios.
# Format: "WxH (ratio)": (height, width) - stored as (H,W) for API's HxW format

RESOLUTION_PRESETS = {
    # Auto - let model predict from prompt
    "Auto (model predicts)": "auto",
    
    # Portrait (tall) - base 1024
    "576x1024 (9:16 Portrait)": (1024, 576),
    "640x1024 (5:8 Portrait)": (1024, 640),
    "672x1024 (2:3 Portrait)": (1024, 672),
    "768x1024 (3:4 Portrait)": (1024, 768),
    "864x1024 (~5:6 Portrait)": (1024, 864),
    "912x1024 (9:10 Portrait)": (1024, 912),
    
    # Square - base 1024
    "1024x1024 (1:1 Square)": (1024, 1024),
    
    # Landscape (wide) - base 1024
    "1024x912 (10:9 Landscape)": (912, 1024),
    "1024x864 (~6:5 Landscape)": (864, 1024),
    "1024x768 (4:3 Landscape)": (768, 1024),
    "1024x672 (3:2 Landscape)": (672, 1024),
    "1024x640 (8:5 Landscape)": (640, 1024),
    "1024x576 (16:9 Landscape)": (576, 1024),
    
    # Large Portrait - 1.5x scale
    "864x1536 (9:16 Large)": (1536, 864),
    "1008x1536 (2:3 Large)": (1536, 1008),
    "1152x1536 (3:4 Large)": (1536, 1152),
    
    # Large Square
    "1536x1536 (1:1 Large)": (1536, 1536),
    
    # Large Landscape - 1.5x scale
    "1536x1152 (4:3 Large)": (1152, 1536),
    "1536x1008 (3:2 Large)": (1008, 1536),
    "1536x864 (16:9 Large)": (864, 1536),
    
    # XL - 2x scale (high VRAM required)
    "1152x2048 (9:16 XL)": (2048, 1152),
    "1536x2048 (3:4 XL)": (2048, 1536),
    "2048x2048 (1:1 XL)": (2048, 2048),
    "2048x1536 (4:3 XL)": (1536, 2048),
    "2048x1152 (16:9 XL)": (1152, 2048),
}

# List of resolution names for dropdown
RESOLUTION_LIST = list(RESOLUTION_PRESETS.keys())

# System prompt modes for base text-to-image generation
# - disabled: No system prompt
# - vanilla: Basic "you are an AI image generator" prompt
# For prompt enhancement (recaption, CoT), use the Instruct nodes
SYSTEM_PROMPT_MODES = ["disabled", "vanilla"]


def detect_quant_type(model_name: str) -> str:
    """
    Auto-detect quantization type from model folder name.
    
    Patterns:
        - *NF4*, *nf4* -> nf4
        - *INT8*, *int8* -> int8  
        - Otherwise -> bf16
    """
    name_lower = model_name.lower()
    
    if "nf4" in name_lower:
        return "nf4"
    elif "int8" in name_lower:
        return "int8"
    else:
        return "bf16"


def parse_resolution(resolution_name: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse resolution from dropdown selection.
    
    Returns:
        (mode, height, width) where mode is "auto" or "fixed"
    """
    value = RESOLUTION_PRESETS.get(resolution_name, "auto")
    
    if value == "auto":
        return ("auto", None, None)
    else:
        height, width = value
        return ("fixed", height, width)


class HunyuanUnifiedV2:
    """
    Unified generation node for HunyuanImage-3.0 base models.
    
    Features:
    - Multi-GPU support with device_map distribution (BF16/INT8)
    - Explicit block swapping for large generations (NF4)
    - Simple VAE management with auto-tiling
    - Soft unload / restore support (NF4 only)
    - Cache with automatic reuse and resolution-aware invalidation
    - Auto blocks_to_swap calculation
    - Downstream VRAM reserve
    - Auto quant_type detection from model name
    - Resolution presets matching actual HunyuanImage3 supported sizes
    
    System Prompt Modes:
    - disabled: No system prompt
    - vanilla: Basic image generation instructions
    
    For prompt enhancement (recaption, CoT reasoning), image editing,
    and multi-image fusion, use the Instruct nodes instead.
    """
    
    CATEGORY = "Hunyuan/V2"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "final_prompt")
    
    @classmethod
    def INPUT_TYPES(cls):
        # Find model directories
        model_dirs = get_available_hunyuan_models(
            name_filter=lambda n: "hunyuan" in n.lower(),
            fallback=["HunyuanImage-3-NF4", "HunyuanImage-3-INT8", "HunyuanImage-3"],
        )
        
        return {
            "required": {
                "model_name": (model_dirs, {
                    "default": model_dirs[0] if model_dirs else "",
                    "tooltip": "Model folder. Quant type is auto-detected from name (NF4/INT8/BF16)."}),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "a beautiful sunset over mountains",
                    "tooltip": "Text prompt for image generation."}),
                "resolution": (RESOLUTION_LIST, {
                    "default": "1024x1024 (1:1 Square)",
                    "tooltip": "Image resolution. These match HunyuanImage3's actual supported sizes."}),
                "num_inference_steps": ("INT", {
                    "default": 40, "min": 10, "max": 100,
                    "tooltip": "Number of diffusion steps. 40 recommended (very close to 50 in quality). 30-40 typical."}),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "CFG scale. Higher = more prompt adherence. 5.0-7.0 typical."}),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2147483647,
                    "tooltip": "-1 = random seed."}),
            },
            "optional": {
                "system_prompt_mode": (SYSTEM_PROMPT_MODES, {
                    "default": "disabled",
                    "tooltip": "disabled=no system prompt (best quality, matches model defaults), vanilla=basic T2I instruction. For prompt enhancement use the Instruct nodes."}),
                "blocks_to_swap": ("INT", {
                    "default": -1, "min": -1, "max": 31, 
                    "tooltip": "-1 = auto calculate. 0 = no swapping (needs ~50GB VRAM for NF4). 1-31 = manual swap count."}),
                "vae_placement": (["auto", "always_gpu", "managed"], {
                    "default": "auto",
                    "tooltip": "auto: decide based on VRAM. always_gpu: VAE stays on GPU. managed: VAE moves to CPU when not decoding."}),
                "post_action": (["keep_loaded", "soft_unload", "full_unload"], {
                    "default": "keep_loaded",
                    "tooltip": "keep_loaded: Keep model on GPU. soft_unload: Move to CPU, keep cached. full_unload: Remove from memory."}),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VAE tiling for large images. Reduces VRAM but slower."}),
                "flow_shift": ("FLOAT", {
                    "default": 2.8, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Flow shift for the diffusion scheduler. Default 2.8. Lower = more detail, higher = smoother."}),
                "reserve_vram_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 48.0, "step": 0.5,
                    "tooltip": "Reserve VRAM for downstream nodes (upscalers, other models)."}),
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force full reload: clears cache, empties VRAM, reloads model fresh. Use if orphaned VRAM from failed loads."}),
                "reset_pipeline": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reset scheduler/pipeline state before each run. Recommended ON to avoid memory fragmentation between runs."}),
            }
        }
    
    def __init__(self):
        self.cache = get_cache()
        self.budget = MemoryBudget()
    
    def _get_model_path(self, model_name: str) -> str:
        """Get full path to model directory."""
        return resolve_hunyuan_model_path(model_name)
    
    def _calculate_optimal_config(
        self,
        quant_type: str,
        width: int,
        height: int,
        blocks_to_swap: int,
        vae_placement: str,
        reserve_vram_gb: float,
        num_blocks: int = 32,
        gb_per_block: float = 1.22,
        device: str = "cuda:0"
    ) -> Tuple[int, str]:
        """
        Calculate optimal blocks_to_swap and vae_placement.
        
        Returns:
            Tuple of (blocks_to_swap, vae_placement)
        """
        # Get current VRAM state
        report = self.budget.get_vram_report()
        available_vram = report.free_gb
        
        # Get model size estimate
        model_est = self.budget.estimate_model_size(quant_type)
        
        # Estimate inference VRAM
        inference_vram = self.budget.estimate_inference_vram(width, height)
        
        # Non-block model VRAM (VAE, text encoder, etc.)
        non_block_vram = model_est.vae_gb + model_est.text_encoder_gb
        
        # Auto-calculate blocks_to_swap if -1
        if blocks_to_swap == -1:
            calculated_blocks, details = calculate_blocks_to_swap(
                available_vram_gb=available_vram,
                model_vram_gb=model_est.weights_gb + non_block_vram,
                inference_vram_gb=inference_vram,
                num_blocks=num_blocks,
                gb_per_block=gb_per_block,
                reserve_vram_gb=reserve_vram_gb + 2.0,  # Add 2GB safety margin
                safety_margin=0.9
            )
            
            logger.info(f"Auto blocks_to_swap: {calculated_blocks}")
            logger.info(f"  Reason: {details.get('reason', 'unknown')}")
            blocks_to_swap = calculated_blocks
        
        # Auto VAE placement
        if vae_placement == "auto":
            # If swapping many blocks, use managed VAE to save more VRAM
            if blocks_to_swap > num_blocks // 2:
                vae_placement = "managed"
            else:
                vae_placement = "always_gpu"
            logger.info(f"Auto VAE placement: {vae_placement}")
        
        return blocks_to_swap, vae_placement
    
    def _ensure_model_loaded(
        self,
        model_path: str,
        quant_type: str,
        blocks_to_swap: int,
        vae_placement: str,
        device: str = "cuda:0",
        reserve_vram_gb: float = 0.0,
        width: int = 1024,
        height: int = 1024
    ) -> CachedModel:
        """
        Ensure model is loaded and ready.
        
        Returns cached model if available, otherwise loads fresh.
        
        Args:
            reserve_vram_gb: User-specified downstream reserve
            width, height: Target resolution (used to calculate BF16 inference reserve)
        """
        # Calculate resolution-based VRAM reserve for BF16 models FIRST
        # We need this to check if cached model was loaded with enough reserve
        megapixels = (width * height) / 1_000_000
        
        if quant_type == "bf16":
            # Reserve calculation for BF16 with device_map="auto":
            #
            # How accelerate handles multi-GPU + CPU offload:
            #   - Layers assigned to GPU N run their forward pass on GPU N
            #   - CPU-offloaded (meta) layers execute on GPU 0 (main_device):
            #     weights are temporarily copied to GPU 0, forward runs, then freed
            #   - Only ONE layer runs at a time (no pipeline parallelism)
            #   - Activations are transferred between GPUs via .to() in hooks
            #
            # GPU 0 (primary) needs VRAM for:
            #   1. Per-layer activation peak (~5-8GB for MoE with CFG batch=2)
            #   2. Temp materialization of largest CPU-offloaded layer (~5GB)
            #   3. Latent tensors (scale with resolution: ~0.5-3GB)
            #   4. VAE decode (~0.5GB, happens after model cleanup)
            #   5. Scheduler/timestep state (~0.5GB)
            #   6. CUDA overhead and fragmentation (~3-5GB)
            #
            # Total primary reserve: ~15-22GB depending on resolution
            # (NOT 55GB+ like before — that assumed ALL activations pile up at once)
            
            # Per-layer activation scales mildly with resolution due to attention
            # MoE model: 64 experts but only 4 active per token, so activation
            # memory is dominated by attention (scales with sequence length)
            activation_per_layer = 6.0 + (megapixels * 1.0)  # ~7GB at 1MP, ~9GB at 3MP
            temp_offload_layer = 5.0   # Largest layer materialized from CPU
            latent_cost = 0.5 + (megapixels * 0.8)  # Latent tensors scale with resolution
            vae_and_overhead = 5.0     # VAE + scheduler + CUDA fragmentation
            
            inference_reserve = activation_per_layer + temp_offload_layer + latent_cost + vae_and_overhead
            
            # Add user-specified downstream reserve
            total_reserve = inference_reserve + reserve_vram_gb
            logger.info(f"BF16 reserve for {megapixels:.1f}MP: {activation_per_layer:.0f}GB activation + "
                       f"{temp_offload_layer:.0f}GB offload temp + {latent_cost:.1f}GB latent + "
                       f"{vae_and_overhead:.0f}GB overhead = {inference_reserve:.0f}GB + "
                       f"{reserve_vram_gb:.0f}GB downstream = {total_reserve:.0f}GB total reserve")
        else:
            # NF4/INT8 fit on GPU, use user-specified reserve only
            total_reserve = reserve_vram_gb if reserve_vram_gb > 0 else 10.0
            inference_reserve = 0.0
        
        # Check cache first
        logger.info(f"Checking cache for {model_path} / {quant_type}")
        cached = self.cache.get(model_path, quant_type)
        logger.info(f"Cache returned: {cached is not None}")
        
        if cached:
            logger.info(f"Cache hit: is_on_gpu={cached.is_on_gpu}, loaded_with_reserve_gb={cached.loaded_with_reserve_gb:.1f}")
            # For BF16 models: check if cached model was loaded with ENOUGH reserve
            # If current resolution needs more reserve than model was loaded with,
            # we MUST reload to get proper GPU/CPU split
            if quant_type == "bf16":
                cached_reserve = cached.loaded_with_reserve_gb
                
                # If loaded_with_reserve_gb is 0, this is an old cache entry
                # Check actual free VRAM to determine if we have enough headroom
                if cached_reserve == 0:
                    if torch.cuda.is_available():
                        free_bytes, _ = torch.cuda.mem_get_info(0)
                        current_free_gb = free_bytes / 1024**3
                        if current_free_gb < total_reserve:
                            logger.warning(f"Cached BF16 model has only {current_free_gb:.1f}GB free")
                            logger.warning(f"Current resolution ({megapixels:.1f}MP) needs {total_reserve:.1f}GB reserve")
                            logger.warning("Must reload model with larger reserve for this resolution!")
                            
                            # Force unload and reload
                            self.cache.full_unload()
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            log_vram_status(device, "After cache invalidation for larger resolution")
                            cached = None  # Force reload below
                        else:
                            logger.info(f"Cached BF16 model has {current_free_gb:.1f}GB free, sufficient for {total_reserve:.1f}GB needed")
                elif total_reserve > cached_reserve + 5.0:  # 5GB tolerance
                    logger.warning(f"Cached BF16 model was loaded with {cached_reserve:.1f}GB reserve")
                    logger.warning(f"Current resolution ({megapixels:.1f}MP) needs {total_reserve:.1f}GB reserve")
                    logger.warning("Must reload model with larger reserve for this resolution!")
                    
                    # Force unload and reload
                    self.cache.full_unload()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    log_vram_status(device, "After cache invalidation for larger resolution")
                    cached = None  # Force reload below
                else:
                    logger.info(f"Cached BF16 model has sufficient reserve ({cached_reserve:.1f}GB >= {total_reserve:.1f}GB needed)")
            
            if cached:
                # SANITY CHECK: For BF16, verify model is ACTUALLY in memory
                # This catches cases where cache entry exists but model was garbage collected
                if quant_type == "bf16":
                    if torch.cuda.is_available():
                        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                        # BF16 model should have at least 30GB on GPU
                        if allocated_gb < 30.0:
                            logger.warning(f"Cache says BF16 model loaded but only {allocated_gb:.1f}GB in VRAM!")
                            logger.warning("Model appears to have been garbage collected. Invalidating cache...")
                            self.cache.full_unload()
                            gc.collect()
                            torch.cuda.empty_cache()
                            cached = None
                
            if cached:
                # Model is cached and has enough reserve
                if not cached.is_on_gpu:
                    logger.info("Restoring cached model to GPU...")
                    self.cache.restore(model_path, quant_type, device)
                
                # Check if block swap config changed
                current_swap = cached.blocks_to_swap
                if current_swap != blocks_to_swap:
                    logger.info(f"Block swap config changed ({current_swap} -> {blocks_to_swap})")
                    
                    # Remove old hooks if any
                    if cached.block_swap_manager and cached.block_swap_manager.hooks_installed:
                        cached.block_swap_manager.remove_hooks()
                    
                    # Reconfigure
                    if cached.block_swap_manager:
                        cached.block_swap_manager.config.blocks_to_swap = blocks_to_swap
                        cached.block_swap_manager.setup_initial_placement()
                        if blocks_to_swap > 0:
                            cached.block_swap_manager.install_hooks()
                        cached.blocks_to_swap = blocks_to_swap
                
                return cached
        
        # Need to load model - NOT using cache
        logger.info("=" * 60)
        logger.info(f"LOADING NEW MODEL (cache miss or invalidated)")
        logger.info(f"Loading {quant_type} model from {model_path}...")
        logger.info(f"Reserve requested: {total_reserve:.1f}GB")
        logger.info("=" * 60)
        log_vram_status(device, "Before load")
        
        # Load model with calculated reserve
        result = CleanModelLoader.load(
            model_path=model_path,
            quant_type=quant_type,
            device=device,
            dtype=torch.bfloat16,
            reserve_vram_gb=total_reserve
        )
        
        log_vram_status(device, "After load")
        
        # Verify model loaded properly - check parameter distribution
        # For BF16 with device_map="auto", we expect: CUDA (on-GPU layers) + meta (CPU-offloaded)
        # Meta is NORMAL for CPU-offloaded layers - accelerate uses AlignDevicesHook to
        # materialize them on the execution device during forward pass.
        if quant_type == "bf16":
            meta_count = 0
            cuda_count = 0
            cpu_count = 0
            total_count = 0
            
            # Count ALL parameters (not just first N - that biases toward GPU-resident early layers)
            for param in result.model.parameters():
                total_count += 1
                if param.device.type == 'meta':
                    meta_count += 1
                elif param.device.type == 'cuda':
                    cuda_count += 1
                elif param.device.type == 'cpu':
                    cpu_count += 1
            
            logger.info(f"Model tensor placement ({total_count} total): {cuda_count} CUDA, {cpu_count} CPU, {meta_count} meta (CPU-offloaded)")
            
            # With device_map, cuda+meta is the expected pattern (meta = CPU-offloaded via hooks)
            # Only fail if EVERYTHING is on meta (no GPU layers at all)
            if cuda_count == 0:
                logger.error("=" * 60)
                logger.error("CRITICAL: No model tensors on GPU!")
                logger.error("The model failed to place any layers on GPU.")
                logger.error("Possible causes:")
                logger.error("  1. Not enough GPU VRAM (reserve too high?)")
                logger.error("  2. Other models consuming GPU memory")
                logger.error("  3. accelerate library version incompatibility")
                logger.error("=" * 60)
                raise RuntimeError("Model failed to place any layers on GPU")
        
        # Load tokenizer (required for generate_image)
        if hasattr(result.model, 'load_tokenizer'):
            result.model.load_tokenizer(model_path)
            logger.info("Tokenizer loaded")

        # Apply critical patches (matches working BF16 loader)
        logger.info("Applying dtype compatibility patches...")
        patch_dynamic_cache_dtype()
        patch_hunyuan_generate_image(result.model)
        
        # Apply pre-VAE cleanup patch - critical for BF16/device_map models
        # This clears KV cache before VAE decode to free VRAM
        try:
            from .hunyuan_shared import patch_pipeline_pre_vae_cleanup
        except ImportError:
            try:
                from hunyuan_shared import patch_pipeline_pre_vae_cleanup
            except ImportError:
                patch_pipeline_pre_vae_cleanup = None
        if patch_pipeline_pre_vae_cleanup is not None:
            patch_pipeline_pre_vae_cleanup(result.model, enabled=True)
        else:
            logger.warning("patch_pipeline_pre_vae_cleanup not available - pre-VAE cleanup disabled")
        
        # Check if model supports block swapping
        can_swap = result.is_moveable
        if not can_swap and blocks_to_swap > 0:
            logger.warning(f"{quant_type} model is not moveable (likely INT8 or BF16 with device_map)")
            logger.warning("Block swapping disabled for this model type")
            blocks_to_swap = 0
        
        # Setup block swap manager
        config = BlockSwapConfig(
            blocks_to_swap=blocks_to_swap,
            prefetch_blocks=2,  # Prefetch 2 blocks for better performance
            use_non_blocking=True
        )
        block_swap_manager = BlockSwapManager(result.model, config, target_device=device)
        block_swap_manager.setup_initial_placement()
        
        # Install hooks if swapping enabled
        if blocks_to_swap > 0 and can_swap:
            block_swap_manager.install_hooks()
            logger.info(f"Block swap enabled: {block_swap_manager.num_blocks} blocks, "
                       f"swapping {blocks_to_swap}, hooks installed")
        else:
            # For device_map models, blocks may be on GPU+CPU even without our swapping
            gpu_blocks = block_swap_manager.stats.blocks_currently_on_gpu
            cpu_blocks = block_swap_manager.stats.blocks_currently_on_cpu
            if cpu_blocks > 0:
                logger.info(f"Model uses device_map placement: {gpu_blocks} blocks on GPU, {cpu_blocks} on CPU (accelerate manages)")
            else:
                logger.info(f"Block swap disabled - all {block_swap_manager.num_blocks} blocks on GPU")
        
        # Setup VAE manager
        vae_mgr_placement = VAEPlacement.ALWAYS_GPU if vae_placement == "always_gpu" else VAEPlacement.MANAGED
        vae_manager = SimpleVAEManager(
            model=result.model,
            placement=vae_mgr_placement,
            device=device,
            dtype=torch.bfloat16
        )
        vae_manager.setup_initial_placement()
        
        # Cache the model with the reserve it was loaded with
        cached = self.cache.put(
            model_path=model_path,
            quant_type=quant_type,
            model=result.model,
            is_moveable=result.is_moveable,
            device=result.device,
            dtype=result.dtype,
            block_swap_manager=block_swap_manager,
            vae_manager=vae_manager,
            load_time=result.load_time_seconds,
            blocks_to_swap=blocks_to_swap,
            vae_placement=vae_placement,
            loaded_with_reserve_gb=total_reserve  # Track for resolution-based cache invalidation
        )
        
        return cached
    
    def _patch_resolution_group(self, model, height: int, width: int):
        """
        Patch model to allow exact resolution.
        
        Without this patch, the model may snap to predefined aspect ratios.
        This overrides get_target_size to return exact dimensions.
        """
        if hasattr(model, "image_processor") and hasattr(model.image_processor, "reso_group"):
            reso_group = model.image_processor.reso_group
            self._original_get_target_size = reso_group.get_target_size
            
            def new_get_target_size(w, h):
                return int(w), int(h)
            
            reso_group.get_target_size = new_get_target_size
            logger.debug(f"Patched resolution: {width}x{height} (WxH)")
    
    def _restore_resolution_group(self, model):
        """Restore original resolution handling after generation."""
        if hasattr(self, '_original_get_target_size'):
            if hasattr(model, "image_processor") and hasattr(model.image_processor, "reso_group"):
                model.image_processor.reso_group.get_target_size = self._original_get_target_size
            del self._original_get_target_size
            logger.debug("Restored original resolution handling")

    def _run_inference(
        self,
        cached: CachedModel,
        prompt: str,
        image_size: str,  # "auto" or "HxW" format
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        enable_vae_tiling: bool,
        flow_shift: float = 2.8,
        system_prompt_mode: str = "vanilla",
        reset_pipeline: bool = True,
    ) -> Tuple[List[Image.Image], str]:
        """
        Run the inference pipeline using generate_image() API.
        
        HunyuanImage3 models use generate_image() method.
        Block swapping is handled automatically by installed hooks.
        """
        model = cached.model
        vae_manager = cached.vae_manager
        block_swap_manager = cached.block_swap_manager
        
        # Enable VAE tiling if requested (helps with large images and BF16 VRAM constraints)
        if enable_vae_tiling:
            # Try model-level method first
            if hasattr(model, 'enable_vae_tiling'):
                model.enable_vae_tiling()
                logger.info("VAE tiling enabled (model method)")
            # Try VAE's enable_tiling method (simpler, calls enable_spatial_tiling)
            elif hasattr(model, 'vae') and hasattr(model.vae, 'enable_tiling'):
                model.vae.enable_tiling(True)
                logger.info("VAE tiling enabled (vae.enable_tiling)")
            # Try VAE's spatial tiling method directly
            elif hasattr(model, 'vae') and hasattr(model.vae, 'enable_spatial_tiling'):
                model.vae.enable_spatial_tiling(True)
                if hasattr(model.vae, 'enable_temporal_tiling'):
                    model.vae.enable_temporal_tiling(True)
                logger.info("VAE tiling enabled (spatial + temporal)")
            else:
                logger.warning("VAE tiling requested but model/VAE doesn't support it")
        else:
            if hasattr(model, 'disable_vae_tiling'):
                model.disable_vae_tiling()
            elif hasattr(model, 'vae') and hasattr(model.vae, 'disable_tiling'):
                model.vae.disable_tiling()
            elif hasattr(model, 'vae') and hasattr(model.vae, 'enable_spatial_tiling'):
                model.vae.enable_spatial_tiling(False)
                if hasattr(model.vae, 'enable_temporal_tiling'):
                    model.vae.enable_temporal_tiling(False)
        
        # Prepare VAE for generation
        if vae_manager:
            vae_manager.prepare_for_decode()
        
        # Reset block swap stats
        if block_swap_manager:
            block_swap_manager.reset_stats()
        
        try:
            # Reset pipeline state if requested (recommended ON to avoid fragmentation)
            if reset_pipeline:
                logger.info("Resetting pipeline state for fresh run...")
                # Reset scheduler state that might persist between runs
                if hasattr(model, '_pipeline') and model._pipeline is not None:
                    pipeline = model._pipeline
                    scheduler = pipeline.scheduler
                    
                    # Clear scheduler state
                    if hasattr(scheduler, '_step_index'):
                        scheduler._step_index = None
                    if hasattr(scheduler, 'sigmas'):
                        del scheduler.sigmas
                    if hasattr(scheduler, 'timesteps'):
                        scheduler.timesteps = None
                    if hasattr(scheduler, '_begin_index'):
                        scheduler._begin_index = None
                    
                    # Clear any cached model outputs from previous runs
                    if hasattr(pipeline, 'model_kwargs'):
                        # Clear past_key_values which can accumulate
                        if 'past_key_values' in pipeline.model_kwargs:
                            del pipeline.model_kwargs['past_key_values']
                        # Clear other cached states  
                        if 'output_hidden_states' in pipeline.model_kwargs:
                            del pipeline.model_kwargs['output_hidden_states']
                    
                    logger.debug("  Scheduler and pipeline state reset")
                
                # Also clear any model-level generation cache
                if hasattr(model, '_cache'):
                    model._cache = None
                if hasattr(model, 'past_key_values'):
                    model.past_key_values = None
            
            # Aggressive memory cleanup before inference to defragment CUDA memory
            # This is critical for BF16 models where VRAM is tight
            gc.collect()
            gc.collect()  # Double collect to catch cyclic refs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Log pre-inference memory state for all active GPUs
                for gpu_idx in range(torch.cuda.device_count()):
                    alloc_gb = torch.cuda.memory_allocated(gpu_idx) / 1024**3
                    if alloc_gb < 0.1:
                        continue  # Skip unused GPUs
                    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
                    free_gb = free_bytes / 1024**3
                    reserved_gb = torch.cuda.memory_reserved(gpu_idx) / 1024**3
                    logger.info(f"Pre-inference GPU {gpu_idx}: {alloc_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved, {free_gb:.1f}GB free")
                    
                    # If we have more reserved than allocated, try to release it
                    if reserved_gb - alloc_gb > 5.0:
                        logger.info(f"  Releasing {reserved_gb - alloc_gb:.1f}GB of reserved but unused CUDA memory...")
                
                # Release all unreserved CUDA memory across all devices
                torch.cuda.empty_cache()
                
                free_bytes, _ = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / 1024**3
                if free_gb < 25.0:
                    logger.warning(f"Low VRAM on primary GPU ({free_gb:.1f}GB free). MoE models need ~25GB headroom.")
            
            # Set seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            # CRITICAL: Set generation parameters on model.generation_config directly.
            # The upstream generate_image() reads diff_infer_steps, diff_guidance_scale,
            # and flow_shift from self.generation_config — NOT from **kwargs.
            # Passing them as kwargs to generate_image() has NO EFFECT.
            model.generation_config.diff_infer_steps = num_inference_steps
            model.generation_config.diff_guidance_scale = guidance_scale
            model.generation_config.flow_shift = flow_shift
            
            # Build generation kwargs for HunyuanImage3's generate_image() API
            gen_kwargs = {
                "prompt": prompt,
                "image_size": image_size,  # "auto" or "HxW" format
                "seed": seed,
                "stream": False,
            }
            
            # Setup progress bar callback
            if ProgressBar is not None:
                pbar = ProgressBar(num_inference_steps)
                def progress_callback(pipe, step, timestep, callback_kwargs):
                    pbar.update(1)
                    return callback_kwargs
                gen_kwargs["callback_on_step_end"] = progress_callback
            
            # Add system prompt options
            # V2 base node only supports disabled/vanilla modes.
            # Recaption, think_recaption, and custom modes (which require bot_task
            # and text generation) are available in the Instruct nodes.
            # 
            # How HunyuanImage-3 API works for base generation:
            # - use_system_prompt: which system prompt template to use
            # - For vanilla mode: just tells the model to generate a high-quality image
            # - No bot_task needed for base text-to-image generation
            if system_prompt_mode == "vanilla":
                gen_kwargs["use_system_prompt"] = "en_vanilla"
            
            # Log generation parameters
            logger.info(f"Calling generate_image:")
            logger.info(f"  size={image_size}, steps={num_inference_steps}, cfg={guidance_scale}, seed={seed}")
            if system_prompt_mode != "disabled":
                logger.info(f"  system_prompt_mode={system_prompt_mode}")
            if block_swap_manager:
                # Use actual stats from block placement detection
                gpu_blocks = block_swap_manager.stats.blocks_currently_on_gpu
                cpu_blocks = block_swap_manager.stats.blocks_currently_on_cpu
                swap_blocks = block_swap_manager.config.blocks_to_swap
                if cpu_blocks > 0 and swap_blocks == 0:
                    # device_map model - accelerate manages placement
                    logger.info(f"  blocks: {gpu_blocks} on GPU, {cpu_blocks} on CPU (device_map placement)")
                else:
                    # Our block swap is active
                    logger.info(f"  blocks: {gpu_blocks}/{block_swap_manager.num_blocks} on GPU, "
                               f"{swap_blocks} swapping")
            
            # Let generate_image() handle everything natively including CoT/recaption
            # Note: We can't easily capture the enhanced prompt without modifying generate_image()
            # For now, return the original prompt
            final_prompt = prompt
            
            # IMPORTANT: torch.inference_mode() is safe for NF4 models (loaded with
            # device_map={"":device}, no accelerate hooks). But it CONFLICTS with
            # accelerate's dispatch hooks used by BF16/INT8 models loaded with
            # device_map="auto", causing silent CUDA crashes. The upstream model
            # already uses @torch.no_grad() internally, so we only need
            # inference_mode() for the small performance benefit on NF4.
            uses_device_map = cached.model_path and (
                not cached.is_moveable or 
                getattr(cached.model, 'hf_device_map', None) is not None
            )
            
            if uses_device_map:
                # BF16/INT8 with device_map - do NOT wrap in inference_mode
                result = model.generate_image(**gen_kwargs)
            else:
                # NF4 without device_map - safe to use inference_mode
                with torch.inference_mode():
                    result = model.generate_image(**gen_kwargs)
            
            # Log block swap stats
            if block_swap_manager and block_swap_manager.config.blocks_to_swap > 0:
                logger.info(f"Block swap stats: {block_swap_manager.stats}")
            
            # Handle different return types
            images = []
            if isinstance(result, Image.Image):
                images = [result]
            elif isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, Image.Image):
                        images.append(item)
                if not images and result:
                    last = result[-1] if result else None
                    if isinstance(last, Image.Image):
                        images = [last]
            elif hasattr(result, "__iter__"):
                last_frame = None
                for frame in result:
                    last_frame = frame
                if last_frame is not None:
                    images = [last_frame]
            
            if not images:
                raise RuntimeError("Generation returned no images")
            
            return images, final_prompt
            
        finally:
            # Cleanup VAE
            if vae_manager and cached.vae_placement == "managed":
                vae_manager.cleanup_after_decode()
            
            # Clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()
    
    def _handle_post_action(
        self,
        cached: CachedModel,
        post_action: str,
        model_path: str,
        quant_type: str
    ):
        """Handle post-generation action."""
        if post_action == "keep_loaded":
            return
        
        elif post_action == "soft_unload":
            if cached.is_moveable:
                # Remove hooks before soft unload
                if cached.block_swap_manager and cached.block_swap_manager.hooks_installed:
                    cached.block_swap_manager.remove_hooks()
                self.cache.soft_unload(model_path, quant_type)
                log_vram_status("cuda:0", "After soft unload")
            else:
                logger.warning(f"Cannot soft unload {quant_type} model (not moveable)")
        
        elif post_action == "full_unload":
            # Remove hooks before full unload
            if cached.block_swap_manager and cached.block_swap_manager.hooks_installed:
                cached.block_swap_manager.remove_hooks()
            self.cache.full_unload(model_path, quant_type)
            log_vram_status("cuda:0", "After full unload")
    
    def _images_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert PIL images to ComfyUI tensor format."""
        import numpy as np
        
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(arr)
        
        batch = np.stack(tensors, axis=0)
        return torch.from_numpy(batch)
    
    def generate(
        self,
        model_name: str,
        prompt: str,
        resolution: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        system_prompt_mode: str = "vanilla",
        blocks_to_swap: int = -1,
        vae_placement: str = "auto",
        post_action: str = "keep_loaded",
        enable_vae_tiling: bool = False,
        flow_shift: float = 2.8,
        reserve_vram_gb: float = 0.0,
        force_reload: bool = False,
        reset_pipeline: bool = True,
    ) -> Tuple[torch.Tensor]:
        """
        Generate images using HunyuanImage-3.0.
        
        This is the main entry point called by ComfyUI.
        """
        start_time = time.time()
        device = "cuda:0"
        
        # Check baseline VRAM at start - detect if other extensions grabbed memory
        if torch.cuda.is_available():
            baseline_free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            baseline_free_gb = baseline_free_bytes / 1024**3
            baseline_allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"Baseline VRAM at generate() start: {baseline_allocated_gb:.1f}GB allocated, {baseline_free_gb:.1f}GB free")
        
        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()

        # Force reload: clear everything before loading
        if force_reload:
            logger.info("Force reload requested - clearing cache and VRAM...")
            try:
                # Get current cache status
                status = self.cache.get_status()
                if status.get("cached"):
                    # Remove hooks first
                    cached = self.cache.get(status["model_path"], status["quant_type"])
                    if cached and cached.block_swap_manager:
                        if cached.block_swap_manager.hooks_installed:
                            cached.block_swap_manager.remove_hooks()
                            logger.info("  Removed block swap hooks")
                    # Full unload
                    self.cache.full_unload()
                    logger.info("  Cache cleared")
                
                # Force garbage collection
                gc.collect()
                gc.collect()
                
                # Empty CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("  CUDA cache emptied")
                
                log_vram_status(device, "After force reload cleanup")
            except Exception as e:
                logger.warning(f"Force reload cleanup error (continuing anyway): {e}")
        
        # Auto-detect quant type from model name
        quant_type = detect_quant_type(model_name)
        
        # CRITICAL: INT8 and BF16 models use device_map and CANNOT use block swapping
        # Force blocks_to_swap=0 BEFORE loading to prevent issues
        if quant_type in ("int8", "bf16"):
            if blocks_to_swap != 0:
                logger.info(f"{quant_type.upper()} model detected - forcing blocks_to_swap=0 (device_map models cannot block swap)")
                blocks_to_swap = 0
            # Also force post_action since soft_unload won't work
            if post_action == "soft_unload":
                logger.info(f"{quant_type.upper()} model cannot soft_unload - using keep_loaded instead")
                post_action = "keep_loaded"
        
        # NF4 and BF16 models benefit from VAE tiling
        # NF4: Required for stable decoding
        # BF16: Needed because VRAM is tight after model load
        if quant_type in ("nf4", "bf16") and not enable_vae_tiling:
            logger.info(f"{quant_type.upper()} model detected - auto-enabling VAE tiling")
            enable_vae_tiling = True
        
        # Parse resolution
        res_mode, height, width = parse_resolution(resolution)
        if res_mode == "auto":
            image_size = "auto"
            # Use defaults for VRAM calculation
            calc_width, calc_height = 1024, 1024
        else:
            image_size = f"{height}x{width}"  # HxW format for HunyuanImage3
            calc_width, calc_height = width, height
        
        # Get model path
        model_path = self._get_model_path(model_name)
        
        logger.info(f"=== HunyuanUnifiedV2 Generation ===")
        logger.info(f"Model: {model_name} (auto-detected: {quant_type})")
        logger.info(f"Resolution: {resolution} -> {image_size}")
        logger.info(f"Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        if system_prompt_mode != "disabled":
            logger.info(f"System prompt mode: {system_prompt_mode}")
        
        # Log VRAM state
        report = self.budget.get_vram_report()
        logger.info(f"VRAM: {report.free_gb:.1f}GB free / {report.total_gb:.1f}GB total")
        
        # High-res BF16 feasibility info
        # With device_map, inference uses per-layer activations (~8GB peak) not
        # all-at-once, so high-res is more feasible than single-GPU estimates suggest.
        # The main concern is CPU-offloaded layers slowing inference.
        calc_mp = (calc_width * calc_height) / 1_000_000
        if quant_type == "bf16" and calc_mp > 2.5:
            logger.info("=" * 60)
            logger.info(f"High resolution: {calc_mp:.1f}MP with BF16 (device_map multi-GPU)")
            logger.info(f"Per-layer activation peak: ~{6 + calc_mp:.0f}GB (runs on each GPU in turn)")
            logger.info(f"CPU-offloaded layers will temporarily load to primary GPU during forward pass")
            logger.info(f"More model weight on GPU = faster inference (less CPU<->GPU transfers)")
            if calc_mp > 4.0:
                logger.warning(f"⚠️  Very high resolution ({calc_mp:.1f}MP) may still OOM during VAE decode.")
                logger.warning(f"  Consider NF4 quantized model or reduce resolution if OOM occurs.")
            logger.info("=" * 60)
        
        try:
            # Calculate optimal config if auto
            final_blocks, final_vae = self._calculate_optimal_config(
                quant_type=quant_type,
                width=calc_width,
                height=calc_height,
                blocks_to_swap=blocks_to_swap,
                vae_placement=vae_placement,
                reserve_vram_gb=reserve_vram_gb,
                device=device
            )
            
            logger.info(f"Config: blocks_to_swap={final_blocks}, VAE={final_vae}")
            if reserve_vram_gb > 0:
                logger.info(f"Reserving {reserve_vram_gb:.1f}GB for downstream nodes")
            
            # Ensure model is loaded with resolution-aware reserve
            cached = self._ensure_model_loaded(
                model_path=model_path,
                quant_type=quant_type,
                blocks_to_swap=final_blocks,
                vae_placement=final_vae,
                device=device,
                reserve_vram_gb=reserve_vram_gb,
                width=calc_width,
                height=calc_height
            )
            
            # Patch resolution to allow exact dimensions (not snapped to reso_group)
            if res_mode != "auto":
                self._patch_resolution_group(cached.model, height, width)
            
            # Run inference
            gen_start = time.time()
            try:
                images, final_prompt = self._run_inference(
                    cached=cached,
                    prompt=prompt,
                    image_size=image_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    enable_vae_tiling=enable_vae_tiling,
                    flow_shift=flow_shift,
                    system_prompt_mode=system_prompt_mode,
                    reset_pipeline=reset_pipeline,
                )
            finally:
                # Always restore original resolution handling
                if res_mode != "auto":
                    self._restore_resolution_group(cached.model)
            gen_time = time.time() - gen_start
            
            logger.info(f"Generation completed in {gen_time:.2f}s")
            logger.info(f"Generated {len(images)} image(s), size: {images[0].size}")
            
            # Convert to tensor
            output_tensor = self._images_to_tensor(images)
            
            # Handle post action
            self._handle_post_action(cached, post_action, model_path, quant_type)
            
            total_time = time.time() - start_time
            logger.info(f"Total time: {total_time:.2f}s")
            
            return (output_tensor, final_prompt)
            
        except torch.cuda.OutOfMemoryError as oom_error:
            logger.error(f"Generation failed: Out of VRAM - {oom_error}")
            logger.error("=" * 60)
            logger.error("OOM TROUBLESHOOTING:")
            logger.error("  1. Reduce image resolution (try 1024x1024 or smaller)")
            logger.error("  2. Enable VAE tiling (checkbox in node)")
            logger.error("  3. For BF16 models: increase reserve_vram_gb (try 30-35)")
            logger.error("  4. For repeated runs: restart ComfyUI to clear CUDA fragmentation")
            logger.error("  5. Try NF4 quantized model instead of BF16 (uses ~45GB vs ~65GB)")
            logger.error("=" * 60)
            
            # Try to recover VRAM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Return empty tensor on failure
            out_height = calc_height if height else 1024
            out_width = calc_width if width else 1024
            empty = torch.zeros((1, out_height, out_width, 3), dtype=torch.float32)
            return (empty, prompt)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty tensor on failure
            out_height = calc_height if height else 1024
            out_width = calc_width if width else 1024
            empty = torch.zeros((1, out_height, out_width, 3), dtype=torch.float32)
            return (empty, prompt)  # Return original prompt on error


class HunyuanUnloadV2:
    """
    Utility node to unload Hunyuan model from memory.
    
    Use this to free VRAM for other tasks.
    """
    
    CATEGORY = "Hunyuan/V2"
    FUNCTION = "unload"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_type": (["soft_unload", "full_unload"], {"default": "soft_unload"}),
            }
        }
    
    def unload(self, unload_type: str):
        """Unload the cached model."""
        cache = get_cache()
        status = cache.get_status()
        
        if not status.get("cached"):
            logger.info("No model cached to unload")
            return ()
        
        # Get cached model to clean up hooks
        cached = cache.get(status["model_path"], status["quant_type"])
        if cached and cached.block_swap_manager:
            if cached.block_swap_manager.hooks_installed:
                cached.block_swap_manager.remove_hooks()
        
        if unload_type == "soft_unload":
            if status.get("is_moveable"):
                cache.soft_unload(status["model_path"], status["quant_type"])
                logger.info("Model soft unloaded (moved to CPU)")
            else:
                logger.warning("Model is not moveable, cannot soft unload")
        else:
            cache.full_unload()
            logger.info("Model fully unloaded")
        
        log_vram_status("cuda:0", f"After {unload_type}")
        return ()


class HunyuanCacheStatusV2:
    """
    Utility node to check cache status.
    
    Useful for debugging and understanding memory usage.
    """
    
    CATEGORY = "Hunyuan/V2"
    FUNCTION = "get_status"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    def get_status(self) -> Tuple[str]:
        """Get cache status as string."""
        cache = get_cache()
        status = cache.get_status()
        
        if not status.get("cached"):
            result = "No model cached"
        else:
            result = (
                f"Model: {status['model_path']}\n"
                f"Type: {status['quant_type']}\n"
                f"On GPU: {status['is_on_gpu']}\n"
                f"Moveable: {status['is_moveable']}\n"
                f"Uses: {status['use_count']}\n"
                f"Block swap: {status['blocks_to_swap']}\n"
                f"VAE: {status['vae_placement']}\n"
                f"Load time: {status['load_time']:.1f}s"
            )
            
            # Add block swap details if available
            cached = cache.get(status["model_path"], status["quant_type"])
            if cached and cached.block_swap_manager:
                mgr = cached.block_swap_manager
                blocks_on_gpu = mgr.num_blocks - mgr.config.blocks_to_swap
                result += f"\n\nBlock Manager:\n"
                result += f"  Total blocks: {mgr.num_blocks}\n"
                result += f"  On GPU: {blocks_on_gpu}\n"
                result += f"  Swapping: {mgr.config.blocks_to_swap}\n"
                result += f"  Hooks: {'installed' if mgr.hooks_installed else 'not installed'}\n"
                result += f"  Stats: {mgr.stats}"
        
        # Add VRAM info
        budget = MemoryBudget()
        report = budget.get_vram_report()
        result += f"\n\nVRAM: {report.allocated_gb:.1f}GB / {report.total_gb:.1f}GB"
        result += f"\nFree: {report.free_gb:.1f}GB"
        
        return (result,)


class HunyuanVRAMCalculatorV2:
    """
    Utility node to calculate VRAM requirements.
    
    Helps determine optimal settings before generation.
    """
    
    CATEGORY = "Hunyuan/V2"
    FUNCTION = "calculate"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("report", "recommended_blocks", "recommended_vae")
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get model directories for auto-detection
        model_dirs = get_available_hunyuan_models(
            name_filter=lambda n: "hunyuan" in n.lower(),
            fallback=["HunyuanImage-3-NF4", "HunyuanImage-3-INT8", "HunyuanImage-3"],
        )
            
        return {
            "required": {
                "model_name": (model_dirs, {
                    "default": model_dirs[0] if model_dirs else "",
                    "tooltip": "Model folder. Quant type is auto-detected from name."}),
                "resolution": (RESOLUTION_LIST, {
                    "default": "1024x1024 (1:1 Square)"}),
            },
            "optional": {
                "reserve_vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 48.0, "step": 0.5}),
            }
        }
    
    def calculate(
        self,
        model_name: str,
        resolution: str,
        reserve_vram_gb: float = 0.0
    ) -> Tuple[str, int, str]:
        """Calculate VRAM requirements and recommendations."""
        budget = MemoryBudget()
        
        # Auto-detect quant type
        quant_type = detect_quant_type(model_name)
        
        # Parse resolution
        res_mode, height, width = parse_resolution(resolution)
        if res_mode == "auto":
            width, height = 1024, 1024  # Use default for calculation
        
        # Get current state
        report = budget.get_vram_report()
        model_est = budget.estimate_model_size(quant_type)
        inference_vram = budget.estimate_inference_vram(width, height)
        
        # Calculate optimal config
        config = budget.get_optimal_config(
            quant_type=quant_type,
            width=width,
            height=height,
            downstream_reserve_gb=reserve_vram_gb
        )
        
        # Format report
        megapixels = (width * height) / 1_000_000
        
        result = f"=== VRAM Calculation ===\n\n"
        result += f"Model: {model_name}\n"
        result += f"Detected type: {quant_type}\n"
        result += f"Resolution: {resolution}\n"
        if res_mode == "auto":
            result += f"  (Using 1024x1024 for calculation)\n"
        result += f"  {width}x{height} ({megapixels:.2f} MP)\n\n"
        
        result += f"--- Current VRAM ---\n"
        result += f"Total: {report.total_gb:.1f} GB\n"
        result += f"Free: {report.free_gb:.1f} GB\n"
        result += f"Allocated: {report.allocated_gb:.1f} GB\n\n"
        
        result += f"--- Estimates ---\n"
        result += f"Model size: {model_est.total_gb:.1f} GB\n"
        result += f"Inference: {inference_vram:.1f} GB\n"
        if reserve_vram_gb > 0:
            result += f"Reserved: {reserve_vram_gb:.1f} GB\n"
        result += f"Total needed: {model_est.total_gb + inference_vram + reserve_vram_gb:.1f} GB\n\n"
        
        result += f"--- Recommendation ---\n"
        result += f"blocks_to_swap: {config['blocks_to_swap']}\n"
        result += f"vae_placement: {config['vae_placement']}\n"
        result += f"Reason: {config['reason']}\n"
        
        if config['can_fit_entirely']:
            result += f"\n✓ Will fit entirely on GPU"
        else:
            result += f"\n⚠ Requires block swapping"
        
        return (result, config['blocks_to_swap'], config['vae_placement'])


class HunyuanEmergencyCleanup:
    """
    Emergency cleanup node to clear orphaned GPU memory.
    
    Use this if VRAM is stuck after a failed generation or crash.
    This will:
    1. Remove all hook handles
    2. Clear the model cache
    3. Force garbage collection
    4. Empty CUDA cache
    
    WARNING: This will unload any cached models!
    """
    
    CATEGORY = "Hunyuan/V2"
    FUNCTION = "cleanup"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Set to True to confirm cleanup. This will unload all cached models!"}),
            }
        }
    
    def cleanup(self, confirm: bool) -> Tuple[str]:
        """Emergency cleanup of GPU memory."""
        if not confirm:
            return ("Cleanup not confirmed. Set 'confirm' to True to proceed.",)
        
        results = []
        budget = MemoryBudget()
        
        # Get VRAM before cleanup
        before = budget.get_vram_report()
        results.append(f"Before cleanup: {before.allocated_gb:.1f}GB allocated, {before.free_gb:.1f}GB free")
        
        try:
            # Step 1: Clear the cache (removes hooks too)
            cache = get_cache()
            status = cache.get_status()
            
            if status.get("cached"):
                # Try to remove hooks first
                cached = cache.get(status["model_path"], status["quant_type"])
                if cached and cached.block_swap_manager:
                    try:
                        if cached.block_swap_manager.hooks_installed:
                            cached.block_swap_manager.remove_hooks()
                        results.append("Removed block swap hooks")
                    except Exception as e:
                        results.append(f"Warning: Failed to remove hooks: {e}")
                
                # Force unload
                cache.full_unload()
                results.append("Cache cleared")
            else:
                results.append("No cached model found")
            
            # Step 2: Force garbage collection
            gc.collect()
            gc.collect()  # Double collect to catch cycles
            results.append("Garbage collection completed")
            
            # Step 3: Empty CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                results.append("CUDA cache emptied")
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                results.append("Peak memory stats reset")
            
            # Step 4: Additional cleanup - clear any module hooks
            # This catches orphaned hooks from failed runs
            try:
                import sys
                cleaned_modules = 0
                for name, module in list(sys.modules.items()):
                    if hasattr(module, '_forward_hooks'):
                        hooks = getattr(module, '_forward_hooks', {})
                        if hooks:
                            hooks.clear()
                            cleaned_modules += 1
                if cleaned_modules > 0:
                    results.append(f"Cleared hooks from {cleaned_modules} modules")
            except Exception as e:
                results.append(f"Warning: Module hook cleanup failed: {e}")
            
            # Get VRAM after cleanup
            gc.collect()
            torch.cuda.empty_cache()
            after = budget.get_vram_report()
            
            freed = before.allocated_gb - after.allocated_gb
            results.append(f"\nAfter cleanup: {after.allocated_gb:.1f}GB allocated, {after.free_gb:.1f}GB free")
            results.append(f"Freed: {freed:.1f}GB")
            
            return ("\n".join(results),)
            
        except Exception as e:
            results.append(f"ERROR: {e}")
            import traceback
            results.append(traceback.format_exc())
            return ("\n".join(results),)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HunyuanUnifiedV2": HunyuanUnifiedV2,
    "HunyuanUnloadV2": HunyuanUnloadV2,
    "HunyuanCacheStatusV2": HunyuanCacheStatusV2,
    "HunyuanVRAMCalculatorV2": HunyuanVRAMCalculatorV2,
    "HunyuanEmergencyCleanup": HunyuanEmergencyCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanUnifiedV2": "Hunyuan Unified Generate V2",
    "HunyuanUnloadV2": "Hunyuan Unload V2",
    "HunyuanCacheStatusV2": "Hunyuan Cache Status V2",
    "HunyuanVRAMCalculatorV2": "Hunyuan VRAM Calculator V2",
    "HunyuanEmergencyCleanup": "Hunyuan Emergency Cleanup",
}
