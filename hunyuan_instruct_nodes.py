"""
HunyuanImage-3.0 Instruct Nodes for ComfyUI

Dedicated nodes for the HunyuanImage-3.0-Instruct and Instruct-Distil models.
These models support:
- Built-in prompt enhancement (no external API needed)
- Chain-of-Thought (CoT) reasoning
- Image-to-Image editing
- Multi-image fusion (up to 3 images)
- Block swap for NF4, INT8, and BF16 (CPU↔GPU block swapping)
- Experimental low-memory BF16 CUDA/CPU/disk offload

Original author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025-2026 Eric Hiss. All rights reserved.

This is an unofficial low-memory adaptation. It preserves the original
generation logic while adding portable disk-offload configuration and
compatibility fixes for offloaded HunyuanImage/SigLIP2 modules.
"""

import gc
import logging
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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

# Import shared utilities (patches that fix BF16 issues)
try:
    from .hunyuan_shared import (
        HUNYUAN_INSTRUCT_FOLDER_NAME,
        get_available_hunyuan_models,
        patch_dynamic_cache_dtype,
        patch_hunyuan_static_cache_device,
        patch_pipeline_pre_vae_cleanup,
        patch_static_cache_lazy_init,
        patch_to_device_for_instruct,
        clear_generation_cache,
        resolve_hunyuan_model_path,
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    try:
        from hunyuan_shared import (
            HUNYUAN_INSTRUCT_FOLDER_NAME,
            get_available_hunyuan_models,
            patch_dynamic_cache_dtype,
            patch_hunyuan_static_cache_device,
            patch_pipeline_pre_vae_cleanup,
            patch_static_cache_lazy_init,
            patch_to_device_for_instruct,
            clear_generation_cache,
            resolve_hunyuan_model_path,
        )
        SHARED_UTILS_AVAILABLE = True
    except ImportError:
        SHARED_UTILS_AVAILABLE = False
        patch_dynamic_cache_dtype = None
        patch_hunyuan_static_cache_device = None
        patch_pipeline_pre_vae_cleanup = None
        patch_static_cache_lazy_init = None
        patch_to_device_for_instruct = None
        clear_generation_cache = None

# Import block swap manager
try:
    from .hunyuan_block_swap import BlockSwapConfig, BlockSwapManager
    BLOCK_SWAP_AVAILABLE = True
except ImportError:
    try:
        from hunyuan_block_swap import BlockSwapConfig, BlockSwapManager
        BLOCK_SWAP_AVAILABLE = True
    except ImportError:
        BLOCK_SWAP_AVAILABLE = False
        BlockSwapConfig = None
        BlockSwapManager = None

logger = logging.getLogger(__name__)

# =============================================================================
# Portable BF16 SSD offload configuration
# =============================================================================

def _default_bf16_disk_offload_dir() -> str:
    """Return a writable, platform-appropriate default offload directory.

    The location can be overridden without editing this file by setting the
    ``HUNYUAN_DISK_OFFLOAD_DIR`` environment variable.
    """
    override = os.environ.get("HUNYUAN_DISK_OFFLOAD_DIR", "").strip()
    if override:
        return os.path.abspath(
            os.path.expanduser(os.path.expandvars(override))
        )

    home = os.path.expanduser("~")

    if os.name == "nt":
        cache_root = os.environ.get("LOCALAPPDATA")
        if not cache_root:
            cache_root = os.path.join(home, "AppData", "Local")
    elif sys.platform == "darwin":
        cache_root = os.path.join(home, "Library", "Caches")
    else:
        cache_root = os.environ.get("XDG_CACHE_HOME")
        if not cache_root:
            cache_root = os.path.join(home, ".cache")

    return os.path.join(
        cache_root,
        "ComfyUI",
        "HunyuanImage3",
        "disk_offload",
    )


# The full BF16 checkpoint is larger than the memory available on many
# consumer systems. Accelerate can keep most modules backed by safetensors on
# disk and materialize them only while each module is being executed.
DEFAULT_BF16_DISK_OFFLOAD_DIR = _default_bf16_disk_offload_dir()
DEFAULT_BF16_CPU_MEMORY_LIMIT_GIB = 32
LOW_VRAM_THRESHOLD_GIB = 24.0
LOW_VRAM_RESERVE_GIB = 5.0
# At least one indivisible Hunyuan transformer module needs about 4.65 GiB.
# A smaller budget yields a cpu+disk-only map, making CPU the execution device.
LOW_VRAM_GPU_MODEL_BUDGET_GIB = 5.0


def _prepare_bf16_disk_offload(
    model_path: str,
    offload_root: str,
    requested_cpu_limit_gib: int,
    requested_vram_reserve_gib: float,
) -> Tuple[str, Dict[Any, Any]]:
    """Prepare an Accelerate device map budget for BF16 SSD offload.

    The checkpoint remains in ``model_path``. Accelerate may use the original
    sharded safetensors directly or create auxiliary files under
    ``offload_root/<model-name>``. CPU RAM is deliberately capped so the
    operating system retains headroom; pagefile or swap may still be required.
    """
    import psutil
    import shutil

    model_name = os.path.basename(os.path.normpath(model_path)) or "HunyuanImage3"
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)

    normalized_offload_root = (offload_root or "").strip()
    if not normalized_offload_root:
        normalized_offload_root = DEFAULT_BF16_DISK_OFFLOAD_DIR
    normalized_offload_root = os.path.abspath(
        os.path.expanduser(os.path.expandvars(normalized_offload_root))
    )

    offload_dir = os.path.join(normalized_offload_root, safe_name)
    os.makedirs(offload_dir, exist_ok=True)

    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    available_ram_gib = ram.available / 1024**3
    total_ram_gib = ram.total / 1024**3
    pagefile_total_gib = swap.total / 1024**3
    pagefile_free_gib = swap.free / 1024**3
    estimated_commit_limit_gib = total_ram_gib + pagefile_total_gib

    # Accelerate's max_memory["cpu"] is a placement budget, not a requirement
    # that at least 16 GiB must be resident in physical RAM.  A small budget is
    # valid when most modules are assigned to the explicit SSD offload folder.
    # Keep roughly 8 GiB of currently available RAM outside the model budget for
    # Windows, ComfyUI, Python, CUDA bookkeeping and transient shard loading.
    if available_ram_gib < 8.0:
        raise RuntimeError(
            f"Too little free physical RAM to start BF16 disk offload: "
            f"{available_ram_gib:.1f} GiB available. Close other programs and retry."
        )

    max_safe_cpu_gib = max(4, int(available_ram_gib - 8.0))
    safe_cpu_limit = int(max(4, min(requested_cpu_limit_gib, max_safe_cpu_gib)))
    if safe_cpu_limit != requested_cpu_limit_gib:
        logger.warning(
            f"  CPU placement budget reduced from {requested_cpu_limit_gib} GiB "
            f"to {safe_cpu_limit} GiB to preserve host-memory headroom."
        )

    logger.info(
        f"  Physical RAM: {available_ram_gib:.1f} GiB available / "
        f"{total_ram_gib:.1f} GiB total"
    )
    virtual_memory_name = "Windows pagefile" if os.name == "nt" else "Swap"
    logger.info(
        f"  {virtual_memory_name}: {pagefile_free_gib:.1f} GiB free / "
        f"{pagefile_total_gib:.1f} GiB total; estimated RAM + virtual-memory "
        f"limit ~{estimated_commit_limit_gib:.1f} GiB"
    )
    if estimated_commit_limit_gib < 185.0:
        if os.name == "nt":
            logger.warning(
                "  The RAM + pagefile commit limit is below ~185 GiB. "
                "Windows may raise OS error 1455 while Transformers maps "
                "the ~160 GiB checkpoint and creates temporary loading objects."
            )
        else:
            logger.warning(
                "  RAM + swap is below ~185 GiB. Loading the full BF16 "
                "checkpoint may fail if temporary allocations exceed the "
                "available virtual-memory limit."
            )

    max_memory: Dict[Any, Any] = {}

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_vram_gib = free_bytes / 1024**3
        # Keep the resident model budget tiny.  Disk/CPU modules are copied to
        # the execution GPU only for their forward pass.
        usable_after_reserve = max(1.0, free_vram_gib - requested_vram_reserve_gib)
        gpu_budget_gib = max(1.0, min(LOW_VRAM_GPU_MODEL_BUDGET_GIB, usable_after_reserve))
        max_memory[0] = f"{gpu_budget_gib:.0f}GiB"
        logger.info(
            f"  SSD offload GPU budget: {gpu_budget_gib:.1f} GiB; "
            f"requested inference reserve: {requested_vram_reserve_gib:.1f} GiB"
        )
        if gpu_budget_gib < 4.7:
            logger.warning(
                "  GPU placement budget is below the ~4.65 GiB minimum module size. "
                "Accelerate may create a cpu+disk-only map. Lower vram_reserve_gb "
                "to 5 GiB and retry so CUDA becomes the execution device."
            )

    # Keep CPU last in max_memory so Accelerate considers CUDA before CPU.
    max_memory["cpu"] = f"{safe_cpu_limit}GiB"

    try:
        disk = shutil.disk_usage(offload_dir)
        disk_free_gib = disk.free / 1024**3
        logger.info(f"  SSD offload directory: {offload_dir}")
        logger.info(f"  SSD free space: {disk_free_gib:.1f} GiB")
        if disk_free_gib < 170:
            logger.warning(
                "  Less than 170 GiB is free in the SSD offload location. "
                "The BF16 offload cache may not fit."
            )
    except Exception as exc:
        logger.warning(f"  Could not inspect SSD free space: {exc}")

    logger.info(
        f"  SSD offload CPU budget: {safe_cpu_limit} GiB "
        f"({available_ram_gib:.1f} GiB currently available)"
    )
    return offload_dir, max_memory


def _patch_siglip2_position_embedding_for_disk_offload(model: Any) -> bool:
    """Fix SigLIP2 positional embeddings when their nn.Embedding is disk-offloaded.

    The official Siglip2VisionEmbeddings.forward() reads
    ``self.position_embedding.weight`` directly instead of calling the
    registered ``nn.Embedding`` submodule. Accelerate attaches the SSD/CPU
    materialization hook to that submodule, so direct weight access leaves the
    parameter on the meta device.

    This patch keeps Accelerate's hook wrapper intact and replaces only the
    wrapped implementation. Calling ``self.position_embedding(position_ids)``
    triggers the normal pre/post hooks: the tiny positional table is loaded
    for the operation and can be offloaded again immediately afterwards.
    """
    import types

    vision_model = getattr(model, "vision_model", None)
    embeddings = getattr(vision_model, "embeddings", None)
    position_embedding = getattr(embeddings, "position_embedding", None)

    if embeddings is None or position_embedding is None:
        logger.debug("  SigLIP2 disk-offload patch skipped: embeddings not found")
        return False

    if getattr(embeddings, "_hunyuan_position_embedding_offload_patch", False):
        return True

    def _disk_safe_embeddings_forward(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        # patch_embedding is called normally, so its Accelerate hook can
        # materialize an offloaded weight before the linear projection.
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # CRITICAL: invoke the nn.Embedding module instead of reading
        # position_embedding.weight directly. The module call activates its
        # Accelerate hook and returns a real tensor rather than a meta tensor.
        position_ids = torch.arange(
            self.num_patches,
            device=patch_embeds.device,
            dtype=torch.long,
        )
        positional_embeddings = self.position_embedding(position_ids)
        positional_embeddings = positional_embeddings.reshape(
            self.position_embedding_size,
            self.position_embedding_size,
            -1,
        )

        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings,
            spatial_shapes,
            max_length=pixel_values.shape[1],
        )
        if resized_positional_embeddings.device != patch_embeds.device:
            resized_positional_embeddings = resized_positional_embeddings.to(
                device=patch_embeds.device,
                dtype=patch_embeds.dtype,
            )
        elif resized_positional_embeddings.dtype != patch_embeds.dtype:
            resized_positional_embeddings = resized_positional_embeddings.to(
                dtype=patch_embeds.dtype
            )

        return patch_embeds + resized_positional_embeddings

    patched_forward = types.MethodType(_disk_safe_embeddings_forward, embeddings)

    # Accelerate wraps module.forward and stores the original callable in
    # _old_forward. Replacing _old_forward preserves the pre/post offload hook.
    if hasattr(embeddings, "_old_forward"):
        embeddings._old_forward = patched_forward
        patch_target = "_old_forward (Accelerate hook preserved)"
    else:
        embeddings.forward = patched_forward
        patch_target = "forward"

    embeddings._hunyuan_position_embedding_offload_patch = True

    mapped_device = None
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        full_name = "vision_model.embeddings.position_embedding"
        best_prefix = ""
        for module_name, device in device_map.items():
            if (
                full_name == module_name
                or full_name.startswith(module_name + ".")
                or module_name == ""
            ):
                if len(module_name) >= len(best_prefix):
                    best_prefix = module_name
                    mapped_device = device

    logger.warning(
        "  Compatibility fix: patched SigLIP2 positional embedding access for "
        f"SSD/CPU offload via {patch_target}"
        + (
            f" (device-map target: {mapped_device})."
            if mapped_device is not None
            else "."
        )
    )
    return True


def _patch_multihead_attention_offload_hooks(model: Any) -> int:
    """Make Accelerate preload child weights of nn.MultiheadAttention.

    PyTorch's MultiheadAttention.forward() uses ``out_proj.weight`` and
    ``out_proj.bias`` directly inside ``F.multi_head_attention_forward``.
    It does not call ``out_proj.forward()``, so an Accelerate hook attached
    only to the child Linear module never fires.  When that child is CPU/disk
    offloaded, its tensors therefore remain on the meta device.

    Setting ``place_submodules=True`` on the parent MHA AlignDevicesHook makes
    its pre-forward materialize both the direct MHA parameters and parameters
    of registered child modules.  The corresponding post-forward then
    offloads them again, preserving the low-memory execution model.
    """

    def _enable_on_hook(hook: Any) -> bool:
        changed = False

        if hasattr(hook, "place_submodules"):
            if not bool(getattr(hook, "place_submodules")):
                hook.place_submodules = True
                changed = True

        # Accelerate may wrap several hooks in a SequentialHook.
        nested = getattr(hook, "hooks", None)
        if nested:
            for child_hook in nested:
                changed = _enable_on_hook(child_hook) or changed

        return changed

    patched = 0
    already_enabled = 0
    meta_out_proj = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.MultiheadAttention):
            continue

        out_proj = getattr(module, "out_proj", None)
        if out_proj is not None:
            for param in out_proj.parameters(recurse=False):
                if getattr(param, "device", None) is not None and param.device.type == "meta":
                    meta_out_proj += 1
                    break

        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue

        if _enable_on_hook(hook):
            patched += 1
            logger.warning(
                "  Compatibility fix: enabled Accelerate submodule preloading "
                f"for MultiheadAttention '{module_name}'."
            )
        else:
            # Count hooks that already expose an enabled place_submodules flag.
            hook_stack = [hook]
            enabled = False
            while hook_stack:
                current = hook_stack.pop()
                if bool(getattr(current, "place_submodules", False)):
                    enabled = True
                    break
                nested = getattr(current, "hooks", None)
                if nested:
                    hook_stack.extend(nested)
            if enabled:
                already_enabled += 1

    if patched or already_enabled:
        logger.warning(
            "  MultiheadAttention offload compatibility: "
            f"{patched} hooks patched, {already_enabled} already compatible; "
            f"{meta_out_proj} modules had meta out_proj parameters."
        )
    else:
        logger.debug(
            "  MultiheadAttention offload patch found no Accelerate hooks to modify."
        )

    return patched

# =============================================================================
# Constants and Configuration
# =============================================================================

# Bot task modes for Instruct model
BOT_TASK_MODES = ["image", "recaption", "think_recaption"]

# System prompt options
SYSTEM_PROMPT_OPTIONS = [
    "dynamic",              # Auto-select best prompt for the chosen bot_task
    "en_unified",           # Comprehensive prompt for all modes
    "en_recaption",         # Focused on prompt rewriting
    "en_think_recaption",   # Focused on CoT + rewriting
    "en_vanilla",           # Minimal - just "generate a high-quality image"
    "none",                 # Truly disabled — no system prompt sent to model
]

# Dynamic system prompt mapping: bot_task → best system prompt
# For bot_task="image" (plain T2I), no system prompt is sent — the instruction
# text ("you are an expert...") interferes with pure image generation quality,
# causing the same artifacts (extra limbs, malformed faces) seen with the base model.
# For recaption/think modes, system prompts are needed to guide CoT reasoning.
DYNAMIC_PROMPT_MAP = {
    "image": "None",                    # No system prompt — clean T2I like base model
    "recaption": "en_recaption",        # Prompt rewriting focus
    "think_recaption": "en_unified",    # CoT + rewriting — unified handles all
}

# Resolution presets — all 33 bucket resolutions from ResolutionGroup(base_size=1024)
# Format: "WxH (ratio description)": (height, width) - stored as (H,W) for API's HxW format
#
# These are the exact resolutions the model was trained on (~1MP each).
# Ordered: tallest portrait → square → widest landscape.
#
# NOTE on larger sizes: The model's tokenizer has size tokens for base_size
# up to 8192, but the diffusion head was only trained at base_size=1024.
# Specifying larger sizes (2MP+) produces artifacts. Future Tencent releases
# may support higher base sizes.
INSTRUCT_RESOLUTION_PRESETS = {
    "Auto (model predicts)": "auto",
    # ---- Extreme portrait (1:4 to 1:3) ----
    "512x2048 (1:4 Tall)": (2048, 512),
    "512x1984 (~1:4 Tall)": (1984, 512),
    "512x1920 (4:15 Tall)": (1920, 512),
    "512x1856 (~1:4 Tall)": (1856, 512),
    "512x1792 (2:7 Tall)": (1792, 512),
    "512x1728 (~1:3 Tall)": (1728, 512),
    "512x1664 (4:13 Tall)": (1664, 512),
    "512x1600 (8:25 Tall)": (1600, 512),
    "512x1536 (1:3 Portrait)": (1536, 512),
    # ---- Tall portrait (9:23 to 3:5) ----
    "576x1472 (9:23 Portrait)": (1472, 576),
    "640x1408 (5:11 Portrait)": (1408, 640),
    "704x1344 (11:21 Portrait)": (1344, 704),
    "768x1280 (3:5 Portrait)": (1280, 768),
    # ---- Standard portrait (13:19 to 15:17) ----
    "832x1216 (13:19 Portrait)": (1216, 832),
    "896x1152 (7:9 Portrait)": (1152, 896),
    "960x1088 (15:17 Portrait)": (1088, 960),
    # ---- Square ----
    "1024x1024 (1:1 Square)": (1024, 1024),
    # ---- Standard landscape (17:15 to 19:13) ----
    "1088x960 (17:15 Landscape)": (960, 1088),
    "1152x896 (9:7 Landscape)": (896, 1152),
    "1216x832 (19:13 Landscape)": (832, 1216),
    # ---- Wide landscape (5:3 to 11:5) ----
    "1280x768 (5:3 Landscape)": (768, 1280),
    "1344x704 (21:11 Landscape)": (704, 1344),
    "1408x640 (11:5 Landscape)": (640, 1408),
    "1472x576 (23:9 Landscape)": (576, 1472),
    # ---- Extreme landscape (3:1 to 4:1) ----
    "1536x512 (3:1 Wide)": (512, 1536),
    "1600x512 (25:8 Wide)": (512, 1600),
    "1664x512 (13:4 Wide)": (512, 1664),
    "1728x512 (27:8 Wide)": (512, 1728),
    "1792x512 (7:2 Wide)": (512, 1792),
    "1856x512 (29:8 Wide)": (512, 1856),
    "1920x512 (15:4 Wide)": (512, 1920),
    "1984x512 (31:8 Wide)": (512, 1984),
    "2048x512 (4:1 Wide)": (512, 2048),
}

RESOLUTION_LIST = list(INSTRUCT_RESOLUTION_PRESETS.keys())


# =============================================================================
# GPU Memory Utilities
# =============================================================================

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPUs sorted by VRAM (largest first).
    
    Returns:
        List of dicts with keys: index, name, total_vram_gb, free_vram_gb
    """
    if not torch.cuda.is_available():
        return []
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_vram = props.total_memory
        # Use mem_get_info for accurate free memory (accounts for fragmentation)
        try:
            free_vram, _ = torch.cuda.mem_get_info(i)
        except Exception:
            free_vram = total_vram - torch.cuda.memory_allocated(i)
        
        gpus.append({
            "index": i,
            "name": props.name,
            "total_vram_gb": total_vram / (1024**3),
            "free_vram_gb": free_vram / (1024**3),
        })
    
    # Sort by TOTAL VRAM descending (largest GPU first for layer priority)
    gpus.sort(key=lambda x: x["total_vram_gb"], reverse=True)
    return gpus


def create_device_map_for_instruct(
    reserve_min_gb: float = 30.0,
    model_size_gb: float = 80.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create optimal device_map + max_memory for Instruct model on multi-GPU.
    
    CRITICAL INSIGHT: The MoE dispatch_mask is allocated on whichever device
    a transformer layer executes on. If ANY transformer layer runs on a
    secondary GPU (e.g. 48GB RTX 5000), the dispatch_mask (~6-12GB for
    multi-fusion, ~28GB for think_recaption+CFG) can OOM that GPU.
    
    Therefore, ALL transformer layers must execute on the primary GPU (GPU 0).
    Layers that don't fit on GPU 0 go to CPU — accelerate's CPU-offloaded
    layers execute on main_device (GPU 0), so dispatch_mask is still on GPU 0.
    
    Secondary GPUs only get lightweight pre/post-processing components:
    VAE, vision_model, vision_aligner. These don't create dispatch_masks.
    
    Model structure (HunyuanImage3ForCausalMM):
      COMPUTE (must be on GPU 0 or CPU-offloaded to GPU 0):
        model.layers.0..31    - 32 transformer blocks (dispatch_mask here!)
        model.wte             - word token embedding
        model.ln_f            - final layer norm
        lm_head               - language model head
        timestep_emb          - timestep embedding (used every diffusion step)
        guidance_emb          - guidance embedding (every step, CFG-distilled)
        timestep_r_emb        - meanflow timestep embedding (every step)
        patch_embed           - patch embedding (used every diffusion step)
        time_embed            - time embedding (used every diffusion step)
        time_embed_2          - time embedding 2 (used in final_layer)
        final_layer           - output projection (used every diffusion step)
      
      PRE/POST-PROCESSING (safe to put on secondary GPU):
        vae                   - VAE encoder/decoder (~600MB bf16)
        vision_model          - Siglip2 vision transformer (~800MB bf16)
        vision_aligner        - Light projector (tiny)
    
    Args:
        reserve_min_gb: Minimum GB to reserve on primary GPU for inference.
            Default 30GB covers dispatch_mask (~14GB) + KV cache + activations.
        model_size_gb: Estimated total model size in GB (for budget calculation).
        
    Returns:
        Tuple of (device_map, max_memory) for from_pretrained().
        device_map is an explicit dict mapping module names to devices,
        OR "auto" with max_memory if we can't build an explicit map.
    """
    gpus = get_gpu_info()
    
    if not gpus:
        logger.warning("No CUDA GPUs found, model will load on CPU only")
        return "auto", {"cpu": "200GiB"}
    
    # Log detected GPUs
    logger.info("=" * 60)
    logger.info("GPU Detection for Instruct Model:")
    for gpu in gpus:
        logger.info(f"  GPU {gpu['index']}: {gpu['name']}")
        logger.info(f"    Total: {gpu['total_vram_gb']:.1f}GB, Free: {gpu['free_vram_gb']:.1f}GB")
    
    primary_gpu = gpus[0]  # Sorted by total VRAM descending
    primary_idx = primary_gpu["index"]
    primary_free = primary_gpu["free_vram_gb"]
    
    # Reserve on primary GPU for inference headroom
    reserve_gb = max(reserve_min_gb, primary_gpu["total_vram_gb"] * 0.30)
    primary_budget = max(0, primary_free - reserve_gb)
    
    logger.info(f"  GPU {primary_idx} (PRIMARY): {primary_free:.1f}GB free, "
               f"reserving {reserve_gb:.1f}GB, budget: {primary_budget:.1f}GB")
    
    # Pre/post-processing components that are safe for secondary GPUs.
    # These don't create dispatch_mask tensors and run independently
    # of the transformer forward pass.
    safe_for_secondary = ["vae", "vision_model", "vision_aligner"]
    
    # Compute components that MUST be on GPU 0 (or CPU-offloaded to GPU 0).
    # These are used during every diffusion step and create/consume tensors
    # that must be on the same device as the transformer layers.
    compute_components = [
        "timestep_emb", "guidance_emb", "timestep_r_emb",
        "patch_embed", "time_embed", "time_embed_2",
        "final_layer", "model.wte", "model.ln_f", "lm_head",
        "cached_rope",       # RoPE cache — internal tensors need GPU
        "image_processor",   # No parameters but accelerate may need to map it
    ]
    
    # Estimate component sizes (approximate, in GB)
    # These are rough estimates; the actual sizes depend on quantization
    vae_size_gb = 0.6        # VAE is always bf16 (skipped from quantization)
    vision_size_gb = 0.8     # Siglip2 vision transformer (bf16)
    aligner_size_gb = 0.05   # Light projector (tiny)
    secondary_total_gb = vae_size_gb + vision_size_gb + aligner_size_gb  # ~1.45GB
    
    # Everything else goes to GPU 0 or CPU
    compute_total_gb = model_size_gb - secondary_total_gb
    
    # Check if we have a secondary GPU with enough room for VAE/vision
    secondary_gpu = None
    for gpu in gpus[1:]:  # Skip primary
        if gpu["free_vram_gb"] > secondary_total_gb + 2.0:  # 2GB margin
            secondary_gpu = gpu
            break
    
    # Build explicit device map
    device_map = {}
    
    if secondary_gpu is not None:
        secondary_idx = secondary_gpu["index"]
        logger.info(f"  GPU {secondary_idx} (SECONDARY): hosting VAE + vision "
                   f"(~{secondary_total_gb:.1f}GB, {secondary_gpu['free_vram_gb']:.1f}GB free)")
        
        # Put VAE/vision on secondary GPU
        for component in safe_for_secondary:
            device_map[component] = secondary_idx
    else:
        # No secondary GPU or not enough room — put VAE/vision on CPU
        logger.info("  No secondary GPU available — VAE/vision will load on CPU")
        for component in safe_for_secondary:
            device_map[component] = "cpu"
    
    # All compute components go to GPU 0
    for component in compute_components:
        device_map[component] = primary_idx
    
    # Transformer layers: fit as many on GPU 0 as possible, rest to CPU.
    # CPU-offloaded layers execute on main_device (GPU 0) via accelerate hooks,
    # so dispatch_mask is ALWAYS allocated on GPU 0 — never on secondary GPUs.
    num_layers = 32
    layer_size_gb = (compute_total_gb - 3.0) / num_layers  # Subtract non-layer compute (~3GB)
    
    # How many layers fit on GPU 0?
    # Budget for layers = primary_budget - non-layer compute components (~3GB)
    layer_budget_gb = primary_budget - 3.0
    layers_on_gpu = max(1, min(num_layers, int(layer_budget_gb / layer_size_gb)))
    layers_on_cpu = num_layers - layers_on_gpu
    
    for i in range(num_layers):
        if i < layers_on_gpu:
            device_map[f"model.layers.{i}"] = primary_idx
        else:
            device_map[f"model.layers.{i}"] = "cpu"
    
    # Build max_memory for accelerate (needed even with explicit device_map)
    max_memory = {
        primary_idx: int(primary_budget * 1024**3),
        "cpu": "200GiB",
    }
    if secondary_gpu is not None:
        secondary_budget = secondary_gpu["free_vram_gb"] - 2.0  # 2GB safety margin
        max_memory[secondary_idx] = int(max(0, secondary_budget) * 1024**3)
    
    logger.info(f"  Layers on GPU {primary_idx}: {layers_on_gpu} "
               f"({layers_on_gpu * layer_size_gb:.1f}GB)")
    if layers_on_cpu > 0:
        logger.info(f"  Layers on CPU: {layers_on_cpu} "
                   f"({layers_on_cpu * layer_size_gb:.1f}GB, offloaded to GPU {primary_idx} during forward)")
    logger.info(f"  Inference headroom on GPU {primary_idx}: ~{reserve_gb:.1f}GB "
               f"(for MoE dispatch_mask + KV cache)")
    logger.info("=" * 60)
    
    return device_map, max_memory


# =============================================================================
# Utility Functions
# =============================================================================

def _estimate_model_size_gb(model_path: str) -> float:
    """
    Estimate model size in GB from safetensors/bin files on disk.
    
    This is the size on disk (compressed for INT8/NF4). The in-memory 
    footprint will be similar for pre-quantized models since they're
    already stored in their quantized format.
    
    Returns:
        Estimated model size in GB, or 80.0 as fallback for INT8.
    """
    total_bytes = 0
    try:
        for f in os.listdir(model_path):
            if f.endswith(('.safetensors', '.bin')):
                total_bytes += os.path.getsize(os.path.join(model_path, f))
    except Exception:
        pass
    
    if total_bytes > 0:
        return total_bytes / (1024 ** 3)
    
    # Fallback estimates based on typical model sizes
    logger.warning(f"Could not estimate model size from {model_path}, using default")
    return 80.0  # Conservative INT8 estimate


def _fix_int8_module_devices(module: Any, device: torch.device, label: str = "", verbose: int = 1) -> int:
    """
    Fix INT8 weight.CB/SCB after calling module.to(device).
    
    bitsandbytes Linear8bitLt.to() and Module._apply() both fail to move
    weight.CB and weight.SCB to the target device. This function walks all
    child modules and fixes any Linear8bitLt layers.
    
    Returns: number of tensors fixed
    """
    fixed = 0
    try:
        from bitsandbytes.nn import Linear8bitLt
    except ImportError:
        return 0
    
    for name, child in module.named_modules():
        if isinstance(child, Linear8bitLt):
            full_name = f"{label}.{name}" if name else label
            weight = child.weight
            # Fix weight.CB
            if hasattr(weight, 'CB') and weight.CB is not None and weight.CB.device != device:
                if verbose >= 1:
                    logger.info(f"  [INT8 fix] {full_name}: weight.CB {weight.CB.device} -> {device}")
                weight.CB = weight.CB.to(device)
                fixed += 1
            # Fix weight.SCB
            if hasattr(weight, 'SCB') and weight.SCB is not None and weight.SCB.device != device:
                if verbose >= 1:
                    logger.info(f"  [INT8 fix] {full_name}: weight.SCB {weight.SCB.device} -> {device}")
                weight.SCB = weight.SCB.to(device)
                fixed += 1
            # Fix state.CB/SCB (may exist after first forward)
            if hasattr(child, 'state'):
                state = child.state
                if state.CB is not None and state.CB.device != device:
                    state.CB = state.CB.to(device)
                    fixed += 1
                if state.SCB is not None and state.SCB.device != device:
                    state.SCB = state.SCB.to(device)
                    fixed += 1
    return fixed


def _move_non_block_components_to_gpu(
    model: Any,
    target_device: str = "cuda:0",
    verbose: int = 1,
) -> float:
    """
    Move all non-transformer-block components of the model to GPU.
    
    After loading with device_map="cpu", this function moves the embedding,
    vision, timestep, patch, final, and head components to GPU while leaving
    the 32 transformer blocks (model.model.layers) on CPU for BlockSwapManager.
    
    The model structure is:
      HunyuanImage3ForCausalMM:
        .vae                  - VAE encoder/decoder (bf16, needed for encode/decode)
        .vision_model         - Siglip2 vision transformer (for image encoding)
        .vision_aligner       - LightProjector (vision alignment)
        .timestep_emb         - Timestep embedding (every diffusion step)
        .guidance_emb         - Guidance embedding (every step, CFG-distilled models)
        .timestep_r_emb       - Meanflow timestep embedding (every step)
        .patch_embed          - Patch embedding (UNetDown)
        .time_embed           - Time embedding
        .time_embed_2         - Time embedding 2
        .final_layer          - Output projection (UNetUp)
        .model.wte            - Word token embedding
        .model.layers[0..31]  - 32 transformer blocks (LEFT ON CPU for block swap)
        .model.ln_f           - Final layer norm
        .lm_head              - Language model head
    
    Args:
        model: The HunyuanImage3ForCausalMM model loaded on CPU
        target_device: GPU device to move components to
        verbose: Logging verbosity (0=silent, 1=info, 2=debug)
        
    Returns:
        Total GB moved to GPU
    """
    device = torch.device(target_device)
    total_bytes_moved = 0
    
    # Top-level components (on the ForCausalMM wrapper)
    top_level_components = [
        "vae", "vision_model", "vision_aligner",
        "timestep_emb", "guidance_emb", "timestep_r_emb",
        "patch_embed", "time_embed", "time_embed_2",
        "final_layer",
        "cached_rope",  # RoPE cache — no parameters but internal tensors must be on GPU
    ]
    
    # Inner model components (on model.model = HunyuanImage3Model)
    inner_components = [
        "wte", "ln_f",
    ]
    
    # Move top-level components
    for name in top_level_components:
        component = getattr(model, name, None)
        if component is not None and hasattr(component, 'to'):
            size_bytes = sum(
                p.numel() * p.element_size() for p in component.parameters()
            ) if hasattr(component, 'parameters') else 0
            component.to(device)
            # Fix INT8 CB/SCB that .to() fails to move (bitsandbytes bug)
            _fix_int8_module_devices(component, device, label=name, verbose=verbose)
            total_bytes_moved += size_bytes
            if verbose >= 2:
                logger.info(f"  Moved {name} to {device} ({size_bytes / 1024**3:.2f}GB)")
    
    # Move inner model components (skip layers — those are for BlockSwapManager)
    inner_model = getattr(model, 'model', None)
    if inner_model is not None:
        for name in inner_components:
            component = getattr(inner_model, name, None)
            if component is not None and hasattr(component, 'to'):
                size_bytes = sum(
                    p.numel() * p.element_size() for p in component.parameters()
                ) if hasattr(component, 'parameters') else 0
                component.to(device)
                # Fix INT8 CB/SCB that .to() fails to move (bitsandbytes bug)
                _fix_int8_module_devices(component, device, label=f"model.{name}", verbose=verbose)
                total_bytes_moved += size_bytes
                if verbose >= 2:
                    logger.info(f"  Moved model.{name} to {device} ({size_bytes / 1024**3:.2f}GB)")
    
    # Move lm_head (INT8 quantized — needs CB/SCB fix after .to())
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is not None and hasattr(lm_head, 'to'):
        size_bytes = sum(
            p.numel() * p.element_size() for p in lm_head.parameters()
        ) if hasattr(lm_head, 'parameters') else 0
        lm_head.to(device)
        # Fix INT8 CB/SCB that .to() fails to move (bitsandbytes bug)
        _fix_int8_module_devices(lm_head, device, label="lm_head", verbose=verbose)
        total_bytes_moved += size_bytes
        if verbose >= 2:
            logger.info(f"  Moved lm_head to {device} ({size_bytes / 1024**3:.2f}GB)")
    
    total_gb = total_bytes_moved / 1024**3
    if verbose >= 1:
        logger.info(f"  Moved non-block components to {device}: {total_gb:.2f}GB total")
    
    return total_gb


def detect_model_type(model_path: str) -> Dict[str, Any]:
    """
    Detect model type from path name AND config.json.
    
    Reads config.json to detect cfg_distilled and use_meanflow, which
    determine whether CFG (batch=2) is used at inference. This has huge
    VRAM implications:
    - CFG-distilled (Distil): batch=1, single-pass inference
    - Non-CFG (full Instruct): batch=2, ALL tensors doubled (dispatch_mask,
      KV cache, activations) → roughly 2× inference VRAM
    
    Returns:
        dict with keys: is_instruct, is_distil, quant_type, default_steps,
        cfg_distilled, use_meanflow, uses_cfg, cfg_factor
    """
    name_lower = os.path.basename(model_path).lower()
    
    is_instruct = "instruct" in name_lower
    is_distil = "distil" in name_lower
    
    # Detect quantization type from folder name
    if "nf4" in name_lower or "4bit" in name_lower:
        quant_type = "nf4"
    elif "int8" in name_lower or "8bit" in name_lower:
        quant_type = "int8"
    else:
        quant_type = "bf16"
    
    # Read config.json for cfg_distilled and use_meanflow
    # These determine whether CFG batching (batch=2) is used at inference
    cfg_distilled = is_distil  # Fallback: assume distil = cfg_distilled
    use_meanflow = is_distil   # Fallback: assume distil = meanflow
    try:
        import json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            cfg_distilled = config.get("cfg_distilled", False)
            use_meanflow = config.get("use_meanflow", False)
            logger.info(f"  Config: cfg_distilled={cfg_distilled}, use_meanflow={use_meanflow}")
    except Exception as e:
        logger.warning(f"  Could not read config.json: {e}, using defaults from folder name")
    
    # CFG factor: determines inference batch multiplier
    # Non-CFG-distilled models use CFG with batch=2 (conditional + unconditional)
    # This DOUBLES all inference tensors: dispatch_mask, KV cache, activations
    uses_cfg = not cfg_distilled
    cfg_factor = 2 if uses_cfg else 1
    
    # Default steps: 8 for Distil, 40 for full Instruct/Base
    # (40 steps is very close in quality to 50, with ~20% faster generation)
    if is_distil:
        default_steps = 8
    elif is_instruct:
        default_steps = 40
    else:
        default_steps = 40
    
    return {
        "is_instruct": is_instruct,
        "is_distil": is_distil,
        "quant_type": quant_type,
        "default_steps": default_steps,
        "model_type": "distil" if is_distil else ("instruct" if is_instruct else "base"),
        "cfg_distilled": cfg_distilled,
        "use_meanflow": use_meanflow,
        "uses_cfg": uses_cfg,
        "cfg_factor": cfg_factor,
    }


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI image tensor to PIL Image.
    
    ComfyUI format: [B, H, W, C] float 0-1
    PIL format: (H, W, C) uint8 0-255
    """
    if image_tensor.dim() == 4:
        # Take first image in batch
        img = image_tensor[0]
    else:
        img = image_tensor
    
    # Convert to numpy uint8
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def tensor_to_temp_path(image_tensor: torch.Tensor, suffix: str = ".png") -> str:
    """
    Convert ComfyUI image tensor to temporary file path.
    
    The Instruct model API expects file paths for input images.
    This creates a temp file that can be passed to the model.
    
    Returns:
        str: Path to temporary image file
    """
    pil_img = tensor_to_pil(image_tensor)
    
    # Create temp file
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    pil_img.save(path)
    return path


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI tensor format.
    
    PIL format: (H, W, C) or (H, W) for grayscale
    ComfyUI format: [B, H, W, C] float 0-1
    """
    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Convert to numpy then tensor
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)
    
    # Add batch dimension
    return img_tensor.unsqueeze(0)


def cleanup_temp_files(paths: List[str]) -> None:
    """Clean up temporary image files."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {path}: {e}")


def parse_resolution(resolution_name: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse resolution from dropdown selection.
    
    Returns:
        (mode, height, width) where mode is "auto" or "fixed"
    """
    value = INSTRUCT_RESOLUTION_PRESETS.get(resolution_name, "auto")
    
    if value == "auto":
        return ("auto", None, None)
    else:
        height, width = value
        return ("fixed", height, width)


def _aggressive_vram_cleanup(model: Any, context: str = "generation") -> None:
    """
    Aggressive VRAM cleanup before generation to maximize free memory.
    
    Clears stale KV cache AND flushes PyTorch's CUDA caching allocator.
    The caching allocator can hold 50-80GB of reserved-but-unused VRAM
    after model loading and block swap setup, causing OOM even when
    actual tensor allocations are well within budget.
    """
    # Clear any stale generation cache from previous runs
    if SHARED_UTILS_AVAILABLE and clear_generation_cache:
        clear_generation_cache(model)
    
    # Always run gc + empty_cache regardless of the above
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Log VRAM state after cleanup
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            if allocated > 0.1:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                logger.info(f"  Pre-{context} VRAM GPU {i}: "
                           f"{allocated:.1f}GB allocated, "
                           f"{free_bytes/1024**3:.1f}GB free / "
                           f"{total_bytes/1024**3:.1f}GB total")

def _flush_pinned_memory_cache() -> bool:
    """Flush PyTorch's CachingHostAllocator to return pinned pages to OS.
    
    After pinned tensors are freed (refcount → 0), their backing memory
    goes to PyTorch's internal host allocator free list rather than back
    to the OS. This can hold ~93-130GB of 'freed' pinned pages.
    
    This function tries multiple API paths to flush that free list,
    calling cudaFreeHost() on each block → VirtualFree() on Windows.
    
    Returns True if cache was flushed, False if no API available.
    """
    if not torch.cuda.is_available():
        return False
    
    # Try every known API path for flushing the host allocator cache
    api_attempts = [
        ("torch._C._cuda_CachingHostAllocator_emptyCache",
         lambda: torch._C._cuda_CachingHostAllocator_emptyCache()),
        ("torch._C._host_emptyCache",
         lambda: torch._C._host_emptyCache()),
        ("torch.cuda.memory._host_emptyCache",
         lambda: torch.cuda.memory._host_emptyCache()),
    ]
    
    for name, api_call in api_attempts:
        try:
            api_call()
            logger.info(f"  Flushed PyTorch pinned memory (host) cache via {name}")
            return True
        except AttributeError:
            logger.debug(f"  {name}: not found (AttributeError)")
        except (RuntimeError, TypeError) as e:
            logger.warning(f"  {name}: call failed: {e}")
    
    # Diagnostic: list all potentially relevant torch._C symbols
    host_syms = [s for s in dir(torch._C) if any(
        k in s.lower() for k in ('host', 'pin', 'cachinghost'))]
    if host_syms:
        logger.info(f"  Available host-related torch._C symbols: {host_syms}")
    else:
        logger.info("  No host/pin/cachinghost symbols found in torch._C")
    
    # Fallback: try to call via ctypes into the PyTorch shared library
    # The C++ function c10::cuda::CachingHostAllocator_emptyCache() is
    # exported from torch_cuda.dll / libtorch_cuda.so
    flushed_via_ctypes = _flush_pinned_via_ctypes()
    if flushed_via_ctypes:
        return True
    
    logger.warning("  CachingHostAllocator.emptyCache() not available — "
                   "pinned memory may not be returned to OS")
    return False


def _flush_pinned_via_ctypes() -> bool:
    """Try to call CachingHostAllocator_emptyCache via ctypes symbol lookup.
    
    PyTorch may not expose this function to Python, but the C++ symbol
    is exported from the shared library. We look for it using ctypes.
    """
    import ctypes
    import sys
    import os
    
    if sys.platform != 'win32':
        return False  # TODO: Linux .so lookup
    
    # Find PyTorch's CUDA shared library
    torch_lib_dir = os.path.dirname(torch.__file__)
    lib_dir = os.path.join(torch_lib_dir, 'lib')
    
    # Candidate DLLs that might export the function
    dll_candidates = ['torch_cuda.dll', 'c10_cuda.dll', 'torch_cuda_cu.dll']
    
    for dll_name in dll_candidates:
        dll_path = os.path.join(lib_dir, dll_name)
        if not os.path.exists(dll_path):
            continue
        try:
            lib = ctypes.CDLL(dll_path)
            # Try to find the exported symbol
            # C++ name mangling varies, but the C-linkage wrapper (if any) is clean
            for sym_name in [
                'CachingHostAllocator_emptyCache',
                '_ZN3c104cuda31CachingHostAllocator_emptyCacheEv',  # GCC/Clang mangling
                '?CachingHostAllocator_emptyCache@cuda@c10@@YAXXZ',  # MSVC mangling
            ]:
                try:
                    func = getattr(lib, sym_name)
                    func.restype = None
                    func.argtypes = []
                    func()
                    logger.info(f"  Flushed pinned cache via ctypes: {dll_name}::{sym_name}")
                    return True
                except (AttributeError, OSError):
                    continue
        except OSError:
            continue
    
    logger.debug("  ctypes: could not find CachingHostAllocator_emptyCache in PyTorch DLLs")
    return False

def get_instruct_model_dirs() -> List[str]:
    """Find Instruct model directories in registered model paths.

    Uses the centralized ``get_available_hunyuan_models`` from
    ``hunyuan_shared``, filtered to directories whose name contains both
    'hunyuan' and 'instruct'.
    """
    if SHARED_UTILS_AVAILABLE:
        dirs = get_available_hunyuan_models(
            category=HUNYUAN_INSTRUCT_FOLDER_NAME,
            name_filter=lambda n: "hunyuan" in n.lower() and "instruct" in n.lower(),
            fallback=[
                "HunyuanImage-3.0-Instruct-Distil-INT8",
                "HunyuanImage-3.0-Instruct-Distil-NF4",
            ],
        )
        # Stash path map for resolve_model_path()
        get_instruct_model_dirs._path_map = getattr(
            get_available_hunyuan_models, "_path_map", {}
        )
        return dirs

    # Fallback when shared utils are not available
    return [
        "HunyuanImage-3.0-Instruct-Distil-INT8",
        "HunyuanImage-3.0-Instruct-Distil-NF4",
    ]


def resolve_model_path(model_name: str) -> str:
    """Resolve a model name to its full path.

    Delegates to the centralized ``resolve_hunyuan_model_path`` when
    available, with a local fallback that checks the scanner path map.
    """
    if SHARED_UTILS_AVAILABLE:
        return resolve_hunyuan_model_path(
            model_name, category=HUNYUAN_INSTRUCT_FOLDER_NAME
        )

    # Fallback: check local path map from get_instruct_model_dirs
    path_map = getattr(get_instruct_model_dirs, '_path_map', {})
    if model_name in path_map:
        return path_map[model_name]

    if os.path.exists(model_name):
        return model_name

    return model_name


# =============================================================================
# Model Cache for Instruct Models
# =============================================================================

class InstructModelCache:
    """
    Simple cache for Instruct models.
    
    Keeps one model loaded at a time to avoid memory issues.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_path = None
            cls._instance.model_info = None
        return cls._instance
    
    def get(self, model_path: str) -> Optional[Any]:
        """Get cached model if it matches the requested path."""
        if self.model is not None and self.model_path == model_path:
            # Check if model was gutted by unload (all param storage freed)
            try:
                first_param = next(self.model.parameters())
                if first_param.numel() == 0:
                    logger.info("Cached model is gutted (was unloaded) — treating as cache miss")
                    self.model = None
                    self.model_path = None
                    self.model_info = None
                    return None
            except StopIteration:
                logger.info("Cached model has no parameters — treating as cache miss")
                self.model = None
                self.model_path = None
                self.model_info = None
                return None
            logger.info(f"Using cached Instruct model: {model_path}")
            return self.model
        return None
    
    def put(self, model_path: str, model: Any, model_info: Dict[str, Any]) -> None:
        """Cache a model, clearing any existing cached model."""
        if self.model is not None and self.model_path != model_path:
            self.clear()
        
        self.model = model
        self.model_path = model_path
        self.model_info = model_info
        logger.info(f"Cached Instruct model: {model_path}")
    
    def clear(self) -> None:
        """Clear the cached model and free all RAM/VRAM.
        
        This method carefully breaks all circular references before deleting
        the model to ensure both CPU RAM and GPU VRAM are actually freed.
        
        Without explicit cleanup, these circular refs prevent gc from freeing
        ~80GB+ of transformer block tensors in RAM:
          model → _block_swap_manager → model (via manager.model)
          model → _block_swap_manager → blocks[] → model.model.layers[i]
          model → vae → decode closure → model (via monkey-patch)
        """
        if self.model is not None:
            logger.info(f"Clearing cached model: {self.model_path}")
            
            # Track RAM before cleanup
            import psutil
            ram_before = psutil.Process().memory_info().rss / 1024**3
            
            # Step 1: Clean up BlockSwapManager (MUST happen before del model)
            # This breaks circular refs and releases block tensor references
            if hasattr(self.model, '_block_swap_manager'):
                manager = self.model._block_swap_manager
                if manager is not None:
                    manager.cleanup()  # Full cleanup: hooks + refs + streams
                    logger.info("BlockSwapManager cleaned up")
                self.model._block_swap_manager = None
            
            # Step 2: Unpatch VAE decode closure (holds ref to model)
            try:
                if SHARED_UTILS_AVAILABLE:
                    from .hunyuan_shared import unpatch_pipeline_pre_vae_cleanup
                    unpatch_pipeline_pre_vae_cleanup(self.model)
            except ImportError:
                try:
                    from hunyuan_shared import unpatch_pipeline_pre_vae_cleanup
                    unpatch_pipeline_pre_vae_cleanup(self.model)
                except ImportError:
                    pass
            except Exception as e:
                logger.debug(f"VAE unpatch during cleanup: {e}")
            
            # Step 2a: Remove external monkey-patches on submodules.
            # Other custom nodes (e.g. seedvr2_videoupscaler) may patch
            # methods on our model's submodules with closures that capture
            # the model, keeping the entire tree alive after unload.
            import types as _types
            import ctypes as _ct
            ext_broken = 0
            try:
                for _, submod in self.model.named_modules():
                    for attr_name in list(vars(submod).keys()):
                        attr = vars(submod).get(attr_name)
                        if attr is None:
                            continue
                        closure = None
                        if isinstance(attr, _types.FunctionType):
                            closure = attr.__closure__
                        elif isinstance(attr, _types.MethodType):
                            closure = getattr(attr.__func__, '__closure__', None)
                        if closure:
                            for cell in closure:
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
                logger.info(f"  Broke {ext_broken} external monkey-patch "
                            f"closure cells on submodules")
            
            # Step 2b: Cleanly reverse all monkey-patches on the model.
            # RESTORE originals instead of breaking closure cells to avoid
            # corrupting the class for future loads.
            try:
                try:
                    from .hunyuan_shared import unpatch_hunyuan_generate_image
                except ImportError:
                    from hunyuan_shared import unpatch_hunyuan_generate_image
                unpatch_hunyuan_generate_image(self.model)
            except ImportError:
                # Fallback: just delete the instance-level generate_image
                if hasattr(self.model, 'generate_image'):
                    try:
                        del self.model.generate_image
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error during unpatch_hunyuan_generate_image: {e}")
            
            try:
                try:
                    from .hunyuan_shared import unpatch_pipeline_pre_vae_cleanup
                except ImportError:
                    from hunyuan_shared import unpatch_pipeline_pre_vae_cleanup
                unpatch_pipeline_pre_vae_cleanup(self.model)
            except (ImportError, Exception):
                pass
            
            # Step 2c: Nuclear gc closure scan — break ANY closure cell
            # that still references this model or its submodules
            _cell_sentinel = None
            cell_type = type((lambda: _cell_sentinel).__closure__[0])
            gc.collect()
            nuked = 0
            model_ref = self.model  # local ref before we del later
            # Pre-build set of all module ids for O(1) lookup
            model_module_ids = set()
            try:
                for _, _sub in model_ref.named_modules():
                    model_module_ids.add(id(_sub))
            except Exception:
                pass
            model_module_ids.add(id(model_ref))
            
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
            del model_ref
            
            # Step 3: Clear generation caches (KV cache, etc.)
            if SHARED_UTILS_AVAILABLE and clear_generation_cache:
                try:
                    clear_generation_cache(self.model)
                except Exception:
                    pass
            
            # Step 4: Remove any remaining accelerate hooks.
            # Also clean up instance-level `forward` attributes left by
            # remove_hook_from_module — they shadow the class method and
            # would be found by the cache-clear monkey-patch remover,
            # which could nuke their __class__ closure cell.
            try:
                from accelerate.hooks import remove_hook_from_module
                for name, module in self.model.named_modules():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module)
                    if 'forward' in vars(module):
                        try:
                            delattr(module, 'forward')
                        except Exception:
                            pass
            except ImportError:
                pass
            except Exception:
                pass
            
            # Step 5: Clear attached metadata (break any remaining refs)
            for attr in ('_hunyuan_info', '_hunyuan_path', '_block_swap_manager'):
                if hasattr(self.model, attr):
                    try:
                        setattr(self.model, attr, None)
                    except Exception:
                        pass
                    
            # Step 6: Free tensor storage in-place (replaces model.to('cpu'))
            # model.to('cpu') would create new CRT heap allocations for every
            # GPU tensor, which Windows never returns to the OS. Instead,
            # replace every param/buffer .data with empty(0) to free storage
            # directly — both GPU and CPU tensors.
            #
            # CRITICAL: Also handles bitsandbytes INT8 internals (.CB, .SCB,
            # .state) and hunts ALL reachable CUDA tensors via gc referent walk.
            try:
                # Checkpoint: VRAM before gutting
                vram_before_gut = 0
                if torch.cuda.is_available():
                    vram_before_gut = torch.cuda.memory_allocated(0)
                    logger.info(f"  VRAM allocated before gutting: "
                               f"{vram_before_gut/1024**3:.1f}GB")
                
                gpu_freed = 0
                cpu_freed = 0
                
                # Phase 1a: Clear bitsandbytes INT8 internals FIRST.
                # Linear8bitLt stores weight data in weight.CB/SCB and
                # child.state which param.data= doesn't touch.
                bnb_cleared = 0
                try:
                    from bitsandbytes.nn import Linear8bitLt
                    for mod_name, module in self.model.named_modules():
                        if isinstance(module, Linear8bitLt):
                            w = module.weight
                            for attr in ('CB', 'SCB'):
                                t = getattr(w, attr, None)
                                if t is not None and isinstance(t, torch.Tensor):
                                    nb = t.numel() * t.element_size()
                                    if t.device.type == 'cuda':
                                        gpu_freed += nb
                                    else:
                                        cpu_freed += nb
                                    setattr(w, attr, None)
                                    bnb_cleared += 1
                            # Clear matmul state (holds CxB, CB, SCB copies)
                            if hasattr(module, 'state'):
                                state = module.state
                                for attr in ('CxB', 'CB', 'SCB', 'SB',
                                             'idx', 'outlier_cols'):
                                    t = getattr(state, attr, None)
                                    if t is not None and isinstance(t, torch.Tensor):
                                        nb = t.numel() * t.element_size()
                                        if t.device.type == 'cuda':
                                            gpu_freed += nb
                                        else:
                                            cpu_freed += nb
                                        setattr(state, attr, None)
                                        bnb_cleared += 1
                except ImportError:
                    pass
                if bnb_cleared:
                    logger.info(f"  Cleared {bnb_cleared} bitsandbytes INT8 "
                               f"internal tensors")
                
                # Phase 1b: Gut registered parameters and buffers
                for param in self.model.parameters():
                    nbytes = param.data.numel() * param.data.element_size()
                    if param.data.device.type == 'cuda':
                        gpu_freed += nbytes
                    else:
                        cpu_freed += nbytes
                    param.data = torch.empty(0)
                    if param.grad is not None:
                        param.grad = None
                for buf in self.model.buffers():
                    nbytes = buf.data.numel() * buf.data.element_size()
                    if buf.data.device.type == 'cuda':
                        gpu_freed += nbytes
                    else:
                        cpu_freed += nbytes
                    buf.data = torch.empty(0)
                
                logger.info(f"  Phase 1: {gpu_freed/1024**3:.1f}GB GPU + "
                           f"{cpu_freed/1024**3:.1f}GB CPU from "
                           f"params/buffers/bnb internals")
                
                # Checkpoint: flush and measure after Phase 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    vram_after_p1 = torch.cuda.memory_allocated(0)
                    logger.info(f"  VRAM after Phase 1 + gc: "
                               f"{vram_after_p1/1024**3:.1f}GB "
                               f"(released {(vram_before_gut - vram_after_p1)/1024**3:.1f}GB)")
                
                # Phase 2: Hunt non-parameter tensor attributes on modules
                attr_gpu_freed = 0
                attr_cpu_freed = 0
                attr_count = 0
                param_ids = set(id(p) for p in self.model.parameters())
                buffer_ids = set(id(b) for b in self.model.buffers())
                skip_ids = param_ids | buffer_ids
                
                for mod_name, module in self.model.named_modules():
                    for attr_name in list(vars(module).keys()):
                        if attr_name in ('_parameters', '_buffers', '_modules',
                                        '_backward_hooks', '_forward_hooks',
                                        '_forward_pre_hooks', '_state_dict_hooks',
                                        '_load_state_dict_pre_hooks', 'training'):
                            continue
                        try:
                            val = getattr(module, attr_name)
                        except Exception:
                            continue
                        if isinstance(val, torch.Tensor) and id(val) not in skip_ids:
                            nbytes = val.numel() * val.element_size()
                            if val.device.type == 'cuda':
                                attr_gpu_freed += nbytes
                                attr_count += 1
                            elif val.device.type == 'cpu' and nbytes > 0:
                                attr_cpu_freed += nbytes
                                attr_count += 1
                            try:
                                setattr(module, attr_name, None)
                            except Exception:
                                pass
                
                if attr_count > 0:
                    logger.info(f"  Phase 2: {attr_gpu_freed/1024**3:.1f}GB GPU + "
                               f"{attr_cpu_freed/1024**3:.1f}GB CPU from "
                               f"{attr_count} module tensor attributes")
                
                # Phase 3: Nuclear walk — find ALL CUDA tensors reachable from
                # the model via gc referent traversal. This catches tensors
                # stored in nested dicts, lists, GenerationConfig, custom
                # cache objects, etc. that named_modules() can't reach.
                gc.collect()
                if torch.cuda.is_available():
                    vram_after_p2 = torch.cuda.memory_allocated(0)
                    if vram_after_p2 > 1 * 1024**3:  # >1GB still on GPU
                        logger.info(f"  Phase 3: {vram_after_p2/1024**3:.1f}GB "
                                   f"still allocated — hunting reachable tensors...")
                        
                        nuked_gpu = 0
                        nuked_count = 0
                        visited = set()
                        # Walk all objects reachable from the model
                        stack = [self.model]
                        while stack:
                            obj = stack.pop()
                            obj_id = id(obj)
                            if obj_id in visited:
                                continue
                            visited.add(obj_id)
                            
                            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                                nb = obj.numel() * obj.element_size()
                                if nb > 0:
                                    nuked_gpu += nb
                                    nuked_count += 1
                                    obj.data = torch.empty(0)
                                continue  # Don't recurse into tensor internals
                            
                            # Recurse into containers and object attributes
                            try:
                                if isinstance(obj, dict):
                                    stack.extend(obj.values())
                                elif isinstance(obj, (list, tuple)):
                                    stack.extend(obj)
                                elif hasattr(obj, '__dict__') and not isinstance(obj, type):
                                    stack.extend(vars(obj).values())
                            except Exception:
                                pass
                        
                        del visited
                        if nuked_count:
                            logger.info(f"  Phase 3: Nuked {nuked_count} reachable "
                                       f"CUDA tensors ({nuked_gpu/1024**3:.1f}GB)")
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        vram_after_p3 = torch.cuda.memory_allocated(0)
                        logger.info(f"  VRAM after Phase 3 + gc: "
                                   f"{vram_after_p3/1024**3:.1f}GB")
                
                # Final summary
                if torch.cuda.is_available():
                    vram_final = torch.cuda.memory_allocated(0)
                    total_vram_released = vram_before_gut - vram_final
                    logger.info(f"  Total VRAM actually released: "
                               f"{total_vram_released/1024**3:.1f}GB "
                               f"({vram_before_gut/1024**3:.1f} → "
                               f"{vram_final/1024**3:.1f}GB)")
                
            except Exception as e:
                logger.warning(f"  Tensor gutting error: {e}")
                import traceback
                traceback.print_exc()
            
            # Step 7: Delete model and clear cache state
            del self.model
            self.model = None
            self.model_path = None
            self.model_info = None
            
            # Step 8: Aggressive garbage collection + VRAM release
            # Multiple rounds to handle nested reference cycles
            gc.collect()
            gc.collect()
            gc.collect()
            
            if torch.cuda.is_available():
                # Diagnostic: check reserved vs allocated before empty_cache
                alloc_before = torch.cuda.memory_allocated(0) / 1024**3
                reserved_before = torch.cuda.memory_reserved(0) / 1024**3
                free_before, _ = torch.cuda.mem_get_info(0)
                free_before_gb = free_before / 1024**3
                logger.info(f"  Pre-empty_cache: allocated={alloc_before:.1f}GB, "
                           f"reserved={reserved_before:.1f}GB, "
                           f"CUDA free={free_before_gb:.1f}GB")
                
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                alloc_after = torch.cuda.memory_allocated(0) / 1024**3
                reserved_after = torch.cuda.memory_reserved(0) / 1024**3
                free_after, _ = torch.cuda.mem_get_info(0)
                free_after_gb = free_after / 1024**3
                logger.info(f"  Post-empty_cache: allocated={alloc_after:.1f}GB, "
                           f"reserved={reserved_after:.1f}GB, "
                           f"CUDA free={free_after_gb:.1f}GB")
                
                if reserved_after > 1.0:
                    # Reserved pool didn't release — try nuclear options
                    logger.warning(f"  empty_cache left {reserved_after:.1f}GB reserved!")
                    
                    # Option A: Reset the CUDA caching allocator entirely
                    # This forces ALL reserved memory back to CUDA
                    try:
                        torch.cuda.memory.empty_cache()  # redundant but try both paths
                    except Exception:
                        pass
                    
                    # Option B: Force a CUDA context reset on all devices
                    # by creating and immediately destroying a small tensor
                    for i in range(torch.cuda.device_count()):
                        try:
                            tmp = torch.zeros(1, device=f'cuda:{i}')
                            del tmp
                        except Exception:
                            pass
                    
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    reserved_final = torch.cuda.memory_reserved(0) / 1024**3
                    free_final, _ = torch.cuda.mem_get_info(0)
                    logger.info(f"  After nuclear: reserved={reserved_final:.1f}GB, "
                               f"CUDA free={free_final/1024**3:.1f}GB")

            # Step 8a: Flush pinned memory (host) cache
            # After gutting tensor storage (Step 6) and gc (Step 8), freed
            # pinned buffers (~93GB from BlockSwapManager) sit in PyTorch's
            # CachingHostAllocator free list. Without flushing, these pages
            # are never returned to the OS via cudaFreeHost/VirtualFree.
            rss_before_pin = psutil.Process().memory_info().rss / 1024**3
            _flush_pinned_memory_cache()
            rss_after_pin = psutil.Process().memory_info().rss / 1024**3
            logger.info(f"  Pinned cache flush: RSS {rss_before_pin:.1f}GB → "
                       f"{rss_after_pin:.1f}GB "
                       f"(freed {rss_before_pin - rss_after_pin:.1f}GB)")

# Global cache instance
_instruct_cache = InstructModelCache()


# =============================================================================
# Node: HunyuanInstructLoader
# =============================================================================

class HunyuanInstructLoader:
    """
    Load HunyuanImage-3.0-Instruct or Instruct-Distil model.
    
    These models support all features of the base model plus:
    - Built-in prompt enhancement (no external API needed)
    - Chain-of-Thought (CoT) reasoning
    - Image-to-Image editing
    - Multi-image fusion (up to 3 images)
    - Block swap for all quant types (NF4, INT8, BF16)
    
    The Distil variant uses only 8 inference steps vs 50 for full Instruct.
    """
    
    CATEGORY = "Hunyuan/Instruct"
    FUNCTION = "load_model"
    RETURN_TYPES = ("HUNYUAN_INSTRUCT_MODEL",)
    RETURN_NAMES = ("model",)
    
    @classmethod
    def IS_CHANGED(cls, force_reload=False, **kwargs):
        if force_reload:
            return float("nan")
        # If model was unloaded or gutted, force re-execution to reload
        if _instruct_cache.model is None:
            return float("nan")
        try:
            first_param = next(_instruct_cache.model.parameters())
            if first_param.numel() == 0:
                return float("nan")
        except (StopIteration, Exception):
            return float("nan")
        return False
    
    @classmethod
    def INPUT_TYPES(cls):
        model_dirs = get_instruct_model_dirs()
        
        return {
            "required": {
                "model_name": (model_dirs, {
                    "default": model_dirs[0] if model_dirs else "",
                    "tooltip": "Select Instruct or Instruct-Distil model folder"
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force reload model even if cached"
                }),
            },
            "optional": {
                "attention_impl": (["sdpa", "flash_attention_2"], {
                    "default": "sdpa",
                    "tooltip": "Attention implementation. flash_attention_2 requires flash-attn package."
                }),
                "moe_impl": (["eager", "flashinfer"], {
                    "default": "eager",
                    "tooltip": "MoE implementation. flashinfer is faster but requires flashinfer package."
                }),
                "vram_reserve_gb": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 80.0,
                    "step": 1.0,
                    "tooltip": (
                        "VRAM to reserve on primary GPU for inference (BF16 and INT8, "
                        "only when blocks_to_swap=0). The MoE dispatch_mask during "
                        "think_recaption can use 14-28GB (Distil) or 28-56GB (full Instruct). "
                        "Default 30GB is auto-boosted to 40GB for full Instruct models "
                        "(which use CFG with batch=2, doubling all inference tensors). "
                        "Increase to 40-50 if you see OOM during generation. "
                        "For INT8: if the model + reserve exceeds GPU memory, the model "
                        "is automatically split across GPUs. "
                        "Ignored when blocks_to_swap > 0 (block swap manages VRAM instead). "
                        "Has no effect on NF4 models."
                    )
                }),
                "blocks_to_swap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 31,
                    "tooltip": (
                        "Number of transformer blocks to swap CPU↔GPU during inference. "
                        "Enables running large models on GPUs with limited VRAM. "
                        "0 = no swapping (all blocks on GPU, fastest). "
                        "Higher values save more VRAM but run slower. "
                        "Quality is NOT affected — only memory management changes.\n"
                        "Block sizes by quant type:\n"
                        "  NF4: ~0.75GB/block (10 blocks → saves ~7.5GB)\n"
                        "  INT8: ~2.5GB/block (10 blocks → saves ~25GB)\n"
                        "  BF16: ~5.0GB/block (22 blocks → saves ~110GB)\n"
                        "IMPORTANT for full Instruct models (non-Distil):\n"
                        "  These use CFG (batch=2) which DOUBLES all inference tensors.\n"
                        "  For INT8 full Instruct, recommend blocks_to_swap=10-15 to\n"
                        "  leave 40GB+ free for CFG dispatch_mask + KV cache.\n"
                        "When block swap is enabled for INT8/BF16, the model loads to "
                        "CPU first then swaps blocks to GPU as needed, bypassing "
                        "device_map=auto. This avoids accelerate hook conflicts."
                    )
                }),
                "moe_drop_tokens": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True (default): MoE drops tokens that exceed expert capacity "
                        "(lower VRAM, ~1–3% quality cost on dense regions). "
                        "False: route every token through its top-K experts "
                        "(best quality, higher VRAM peak — recommended only on ≥48GB cards)."
                    )
                }),
                "vae_dtype": (["bfloat16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": (
                        "VAE decode precision. bfloat16 (default) matches model dtype. "
                        "float32 reduces banding/chroma noise on smooth gradients with "
                        "negligible cost on big cards. (Some users may already force this "
                        "via ComfyUI launch flag.)"
                    )
                }),
                "use_disk_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "For BF16 with blocks_to_swap=0, keep most weights as memory-mapped "
                        "files on SSD instead of requiring 150+ GB of physical RAM. "
                        "This is experimental and extremely slow."
                    )
                }),
                "disk_offload_dir": ("STRING", {
                    "default": DEFAULT_BF16_DISK_OFFLOAD_DIR,
                    "tooltip": "SSD directory used by Accelerate for BF16 memory-mapped offload files."
                }),
                "cpu_memory_limit_gb": ("INT", {
                    "default": DEFAULT_BF16_CPU_MEMORY_LIMIT_GIB,
                    "min": 8,
                    "max": 192,
                    "step": 1,
                    "tooltip": (
                        "Maximum physical-RAM placement budget for BF16 loading. "
                        "Use 8–16 GB for maximum SSD offload, or 24–32 GB for "
                        "better speed when enough RAM is free. Pagefile capacity is "
                        "reported separately and is not counted as physical RAM."
                    )
                }),
            }
        }
    
    def load_model(
        self,
        model_name: str,
        force_reload: bool = False,
        attention_impl: str = "sdpa",
        moe_impl: str = "eager",
        vram_reserve_gb: float = 30.0,
        blocks_to_swap: int = 0,
        moe_drop_tokens: bool = True,
        vae_dtype: str = "bfloat16",
        use_disk_offload: bool = True,
        disk_offload_dir: str = DEFAULT_BF16_DISK_OFFLOAD_DIR,
        cpu_memory_limit_gb: int = DEFAULT_BF16_CPU_MEMORY_LIMIT_GIB,
    ) -> Tuple[Any]:
        """Load the Instruct model (BF16, INT8, or NF4)."""
        from transformers import AutoModelForCausalLM
        
        # Resolve model path (handles both models_dir and external locations)
        model_path = resolve_model_path(model_name)
        
        # Validate path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Check cache
        if not force_reload:
            cached = _instruct_cache.get(model_path)
            if cached is not None:
                return (cached,)
        else:
            logger.info("Force reload: cleaning up previous model...")
            import psutil
            ram_before = psutil.Process().memory_info().rss / 1024**3
            
            # Step 1: Our cache clear — breaks circular refs (9-step cleanup)
            _instruct_cache.clear()
            
            # Step 2: Multiple gc rounds to collect broken cycles
            gc.collect()
            gc.collect()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Step 3: Final gc (orphan hunt removed — it was destroying
            # other models' parameters across the process)
            gc.collect()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force Windows to return freed memory to OS
            try:
                from .hunyuan_shared import force_windows_memory_release
            except ImportError:
                from hunyuan_shared import force_windows_memory_release
            force_windows_memory_release()
            
            ram_after = psutil.Process().memory_info().rss / 1024**3
            logger.info(f"  Force reload cleanup: RAM {ram_before:.1f}GB → {ram_after:.1f}GB "
                       f"(freed {ram_before - ram_after:.1f}GB)")
        
        # Detect model type (includes quant_type and CFG detection from config.json)
        model_info = detect_model_type(model_path)
        quant_type = model_info["quant_type"]
        uses_cfg = model_info.get("uses_cfg", False)
        cfg_factor = model_info.get("cfg_factor", 1)
        
        logger.info(f"Loading Instruct model: {model_name}")
        logger.info(f"  Path: {model_path}")
        logger.info(f"  Type: {model_info['model_type']}")
        logger.info(f"  Quantization: {quant_type}")
        logger.info(f"  Default steps: {model_info['default_steps']}")

        total_vram_gib = 0.0
        if torch.cuda.is_available():
            total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        low_vram_mode = 0 < total_vram_gib < LOW_VRAM_THRESHOLD_GIB
        if low_vram_mode and vram_reserve_gb > LOW_VRAM_RESERVE_GIB:
            logger.warning(
                f"  Low-VRAM GPU detected ({total_vram_gib:.1f} GiB). "
                f"Clamping vram_reserve_gb from {vram_reserve_gb:.1f} to "
                f"{LOW_VRAM_RESERVE_GIB:.1f} GiB."
            )
            vram_reserve_gb = LOW_VRAM_RESERVE_GIB
        
        # CRITICAL: Full Instruct models (cfg_distilled=False) use CFG with batch=2.
        # This DOUBLES all inference tensors:
        #   - MoE dispatch_mask: ~14GB (image) → ~28GB, or ~28GB (think) → ~56GB
        #   - KV cache: doubled token count → 2× cache size
        #   - All attention activations: 2× larger
        # The user's vram_reserve_gb must account for this.
        if uses_cfg:
            logger.info(f"  ⚠ CFG model (cfg_distilled=False): inference uses batch=2")
            logger.info(f"    All inference tensors are DOUBLED vs Distil models.")
            logger.info(f"    dispatch_mask for 'image' mode: ~8GB (vs ~4GB for Distil)")
            logger.info(f"    dispatch_mask for 'think_recaption': ~56GB (vs ~28GB for Distil)")
            
            # Auto-boost vram_reserve if user left it at default and it's too low for CFG.
            # For CFG models, 30GB reserve is barely enough for bot_task=image at 1024x1024.
            # We recommend at least 40GB for safety.
            if low_vram_mode:
                logger.warning(
                    "    Full Instruct CFG on a sub-24-GiB GPU is below the supported "
                    "memory envelope. The loader will keep the reserve at the low-VRAM "
                    "value and use SSD offload, but generation may still OOM."
                )
                logger.info("    TIP: Use bot_task='image', 512x512 input, and the fewest practical steps.")
            elif vram_reserve_gb <= 30.0:
                original_reserve = vram_reserve_gb
                vram_reserve_gb = 40.0
                logger.info(f"    Auto-boosted vram_reserve: {original_reserve:.0f}GB → {vram_reserve_gb:.0f}GB "
                           f"(CFG needs more headroom for doubled dispatch_mask)")
                logger.info(f"    TIP: Use bot_task='image' (not think_recaption) to minimize dispatch_mask.")
                logger.info(f"    TIP: Use blocks_to_swap > 0 to free more VRAM for CFG inference.")
        else:
            logger.info(f"  CFG-distilled model: inference uses batch=1 (efficient)")
        
        if quant_type in ("bf16", "int8") and blocks_to_swap == 0:
            logger.info(f"  VRAM reserve: {vram_reserve_gb:.1f}GB (for MoE dispatch_mask + inference)")
        if blocks_to_swap > 0:
            # Approximate GB per block for each quant type
            gb_per_block_est = {"nf4": 0.75, "int8": 2.5, "bf16": 5.0}.get(quant_type, 2.5)
            logger.info(f"  Block swap: {blocks_to_swap} blocks "
                       f"(~{gb_per_block_est:.1f}GB/block, saves ~{blocks_to_swap * gb_per_block_est:.1f}GB VRAM)")
            if uses_cfg and blocks_to_swap < 10 and quant_type == "int8":
                logger.warning(f"  ⚠ CFG model with only {blocks_to_swap} blocks swapped. "
                             f"For INT8 CFG inference, recommend blocks_to_swap >= 10-15 "
                             f"to leave enough VRAM for the doubled dispatch_mask.")
        
        # Clear VRAM before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load model with type-specific strategy
        start_time = time.time()
        
        if quant_type == "nf4":
            # NF4 pre-quantized models: ~24GB. Strategy depends on blocks_to_swap.
            if blocks_to_swap > 0 and BLOCK_SWAP_AVAILABLE:
                # Block swap mode: load entirely to CPU, then place non-block
                # components on GPU. BlockSwapManager handles the 32 transformer
                # blocks via synchronous per-parameter movement (Params4bit.to
                # crashes with non_blocking=True on CUDA->CPU).
                logger.info(f"  Block swap mode: loading NF4 Instruct model to CPU first...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attention_impl,
                    moe_impl=moe_impl,
                    moe_drop_tokens=moe_drop_tokens,
                    low_cpu_mem_usage=True,
                )
                logger.info("  Moving non-block components to GPU...")
                _move_non_block_components_to_gpu(model, target_device="cuda:0", verbose=1)
                # Strip accelerate hooks installed by device_map="cpu"
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for _n, _m in model.named_modules():
                        if hasattr(_m, "_hf_hook"):
                            remove_hook_from_module(_m)
                        if "forward" in vars(_m):
                            try:
                                delattr(_m, "forward")
                            except Exception:
                                pass
                except Exception:
                    pass
                if hasattr(model, "hf_device_map"):
                    try:
                        delattr(model, "hf_device_map")
                    except Exception:
                        pass
            else:
                # NF4 fits on a single GPU (~24GB) — direct placement.
                logger.info("Loading pre-quantized NF4 Instruct model to single GPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": "cuda:0"},  # Single GPU, moveable
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attention_impl,
                    moe_impl=moe_impl,
                    moe_drop_tokens=moe_drop_tokens,
                    low_cpu_mem_usage=True,
                    # No quantization_config - model is pre-quantized on disk
                )

            # Apply transformers 5.x compat shims for NF4 (issues #24, #27, #34)
            try:
                from .hunyuan_shared import apply_nf4_transformers_compat
            except ImportError:
                from hunyuan_shared import apply_nf4_transformers_compat
            apply_nf4_transformers_compat(model)

            model_info["is_moveable"] = True
            
        elif quant_type == "int8":
            # INT8 pre-quantized models: ~80GB. Strategy depends on blocks_to_swap.
            logger.info("Loading pre-quantized INT8 Instruct model...")
            
            if blocks_to_swap > 0 and BLOCK_SWAP_AVAILABLE:
                # Block swap mode: load entirely to CPU, then manually place
                # non-block components on GPU. BlockSwapManager handles the
                # 32 transformer blocks. This avoids accelerate's AlignDevicesHook
                # which conflicts with our block swap hooks.
                # Int8Params.to(device) fully supports GPU↔CPU movement.
                logger.info(f"  Block swap mode: loading INT8 model to CPU first...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attention_impl,
                    moe_impl=moe_impl,
                    moe_drop_tokens=moe_drop_tokens,
                    low_cpu_mem_usage=True,
                )
                # Move non-block components to GPU (VAE, vision, embeddings, etc.)
                logger.info("  Moving non-block components to GPU...")
                _move_non_block_components_to_gpu(model, target_device="cuda:0", verbose=1)
                model_info["is_moveable"] = True
            else:
                # No block swap: load INT8 model to GPU(s).
                #
                # IMPORTANT: Pre-quantized INT8 models CANNOT have CPU entries
                # in the device_map. Transformers converts CPU-mapped layers to
                # regular nn.Linear (not Linear8bitLt), but the checkpoint has
                # int8 weights which can't be loaded into nn.Linear (int8 tensors
                # can't require gradients). This is a fundamental limitation of
                # the bitsandbytes/transformers INT8 loading path for pre-quantized
                # checkpoints.
                #
                # NOTE: This limitation ONLY affects pre-quantized INT8 checkpoints
                # (weights stored as int8 on disk with SCB scales). Models that
                # quantize at load time (float16/bf16 checkpoint + load_in_8bit=True)
                # do NOT hit this issue because CPU-mapped layers stay in fp16/fp32
                # and only GPU-mapped layers get quantized to int8 on-the-fly.
                #
                # Strategy:
                # 1. If model fits on primary GPU with headroom → device_map="cuda:0"
                # 2. If model doesn't fit → use create_device_map_for_instruct() to
                #    build an explicit map that keeps ALL transformer layers on GPU 0
                #    and only puts VAE/vision on secondary GPU. Any layers that don't
                #    fit go to CPU — but we must filter those out for INT8 (no CPU).
                #    If we can't fit all layers on GPU(s), fall back to block swap
                #    recommendation.
                model_size_gb = _estimate_model_size_gb(model_path)
                
                if torch.cuda.is_available():
                    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                    free_gb = free_bytes / 1024**3
                    headroom_after_load = free_gb - model_size_gb
                    
                    logger.info(f"  INT8 model size: ~{model_size_gb:.1f}GB")
                    logger.info(f"  GPU 0 free: {free_gb:.1f}GB, "
                               f"headroom after load: {headroom_after_load:.1f}GB, "
                               f"required reserve: {vram_reserve_gb:.1f}GB")
                else:
                    free_gb = 0
                    headroom_after_load = -1
                
                if headroom_after_load >= vram_reserve_gb:
                    # Model fits on single GPU with good headroom — best case.
                    logger.info(f"  INT8 fits on single GPU with {headroom_after_load:.1f}GB "
                               f"headroom (>= {vram_reserve_gb:.1f}GB reserve). Using direct placement.")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="cuda:0",
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=attention_impl,
                        moe_impl=moe_impl,
                        moe_drop_tokens=moe_drop_tokens,
                        low_cpu_mem_usage=True,
                    )
                elif headroom_after_load > 0:
                    # Model fits on single GPU but headroom is tight.
                    # Still use single GPU because dispatch_mask must be on GPU 0.
                    logger.info(f"  INT8 fits on single GPU with {headroom_after_load:.1f}GB "
                               f"headroom (< {vram_reserve_gb:.1f}GB reserve, but single-GPU is optimal). "
                               f"Consider using blocks_to_swap to free more VRAM if OOM occurs.")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="cuda:0",
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=attention_impl,
                        moe_impl=moe_impl,
                        moe_drop_tokens=moe_drop_tokens,
                        low_cpu_mem_usage=True,
                    )
                else:
                    # Model doesn't fit on primary GPU.
                    # Use create_device_map_for_instruct() which puts ALL transformer
                    # layers on GPU 0, VAE/vision on secondary GPU, and any overflow
                    # to CPU. But for INT8, we must NOT have CPU entries in the map.
                    # So we remap any "cpu" entries to the primary GPU and hope it fits,
                    # or recommend block swap.
                    logger.info(f"  INT8 model ({model_size_gb:.1f}GB) exceeds GPU 0 free space "
                               f"({free_gb:.1f}GB). Building explicit device map...")
                    
                    explicit_map, max_memory = create_device_map_for_instruct(
                        reserve_min_gb=vram_reserve_gb,
                        model_size_gb=model_size_gb,
                    )
                    
                    # INT8 can't use CPU entries — remap any "cpu" entries to GPU 0.
                    # This may cause OOM at load time, but it's the only option
                    # without block swap. We warn the user.
                    cpu_entries = [k for k, v in explicit_map.items() if v == "cpu"]
                    if cpu_entries:
                        logger.warning(f"  INT8 cannot offload to CPU! Remapping {len(cpu_entries)} "
                                     f"CPU entries to GPU 0. If this OOMs, use blocks_to_swap > 0.")
                        for key in cpu_entries:
                            gpus = get_gpu_info()
                            primary_idx = gpus[0]["index"] if gpus else 0
                            explicit_map[key] = primary_idx
                    
                    # Remove CPU from max_memory for INT8 and give GPUs more room
                    max_memory.pop("cpu", None)
                    max_memory["cpu"] = "0GiB"
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=explicit_map,
                        max_memory=max_memory,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=attention_impl,
                        moe_impl=moe_impl,
                        moe_drop_tokens=moe_drop_tokens,
                        low_cpu_mem_usage=True,
                    )
                model_info["is_moveable"] = False
            
        else:
            # BF16 model: ~160GB. Strategy depends on blocks_to_swap.
            
            if blocks_to_swap > 0 and BLOCK_SWAP_AVAILABLE:
                # Block swap mode: load entirely to CPU, then manually place
                # non-block components on GPU. BlockSwapManager handles the
                # 32 transformer blocks (~5GB each). This avoids accelerate's
                # AlignDevicesHook which conflicts with our block swap hooks.
                # BF16 benefits greatly: blocks_to_swap=22 → ~50GB on GPU,
                # leaving ~46GB free for inference on a 96GB card.
                logger.info("Loading BF16 Instruct model to CPU for block swap...")
                logger.info(f"  Block swap: {blocks_to_swap} blocks will be managed on CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation=attention_impl,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    moe_impl=moe_impl,
                    moe_drop_tokens=moe_drop_tokens,
                )
                # Move non-block components to GPU (VAE, vision, embeddings, etc.)
                logger.info("  Moving non-block components to GPU...")
                _move_non_block_components_to_gpu(model, target_device="cuda:0", verbose=1)
                model_info["is_moveable"] = True
            else:
                # No block swap.  On this user's 12 GB GPU / 64 GB RAM machine,
                # physical RAM cannot hold the ~160 GB BF16 model.  Use Accelerate's
                # real SSD offload rather than Windows paging.
                if use_disk_offload:
                    logger.info("Loading BF16 Instruct model with SSD disk offload...")
                    offload_dir, max_memory = _prepare_bf16_disk_offload(
                        model_path=model_path,
                        offload_root=disk_offload_dir,
                        requested_cpu_limit_gib=cpu_memory_limit_gb,
                        requested_vram_reserve_gib=vram_reserve_gb,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        attn_implementation=attention_impl,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_memory=max_memory,
                        offload_folder=offload_dir,
                        offload_state_dict=True,
                        low_cpu_mem_usage=True,
                        moe_impl=moe_impl,
                        moe_drop_tokens=moe_drop_tokens,
                    )
                    model_info["disk_offload"] = True
                    model_info["offload_dir"] = offload_dir
                else:
                    # Original high-memory path: explicit GPU + CPU placement.
                    logger.info("Loading BF16 Instruct model with explicit device map...")
                    logger.info(f"  Primary GPU reserve: {vram_reserve_gb:.1f}GB "
                               f"(covers MoE dispatch_mask + KV cache for think_recaption)")
                    explicit_map, max_memory = create_device_map_for_instruct(
                        reserve_min_gb=vram_reserve_gb,
                        model_size_gb=160.0,  # BF16 model is ~160GB
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        attn_implementation=attention_impl,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map=explicit_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True,
                        moe_impl=moe_impl,
                        moe_drop_tokens=moe_drop_tokens,
                    )
                model_info["is_moveable"] = False
        
        # Log actual device placement
        if hasattr(model, 'hf_device_map'):
            devices_used = set(str(v) for v in model.hf_device_map.values())
            logger.info(f"Model distributed across: {', '.join(sorted(devices_used))}")
            disk_modules = [name for name, dev in model.hf_device_map.items() if str(dev) == "disk"]
            if disk_modules:
                logger.info(f"  SSD disk offload active: {len(disk_modules)} modules mapped to disk")
                logger.info(f"  Offload directory: {model_info.get('offload_dir', disk_offload_dir)}")
        
        # Load tokenizer.  The official Instruct config currently omits
        # ``model_version`` while its remote ``load_tokenizer`` implementation
        # unconditionally reads ``self.config.model_version``.  Transformers 4.x
        # therefore raises AttributeError after the weights have loaded.  The
        # base HunyuanImage-3.0 config uses this exact value.
        logger.info("Loading tokenizer...")
        if not hasattr(model.config, "model_version"):
            model.config.model_version = "HunyuanImage-3.0"
            logger.warning(
                "  Compatibility fix: config.model_version was missing; "
                "using 'HunyuanImage-3.0' for tokenizer initialization."
            )
        model.load_tokenizer(model_path)

        # Compatibility shim for the official Instruct remote code.  At the
        # time of writing, modeling_hunyuan_image_3.py calls the legacy/short
        # method name ``build_img_ratio_slice_logits_proc`` while
        # image_processor.py exposes
        # ``build_img_ratio_slice_logits_processor``.  Add an instance alias
        # instead of editing the Hugging Face module cache, so the fix survives
        # cache refreshes and remains limited to this loaded model.
        image_processor = getattr(model, "image_processor", None)
        if image_processor is not None:
            legacy_ratio_builder = "build_img_ratio_slice_logits_proc"
            current_ratio_builder = "build_img_ratio_slice_logits_processor"
            if (
                not hasattr(image_processor, legacy_ratio_builder)
                and hasattr(image_processor, current_ratio_builder)
            ):
                setattr(
                    image_processor,
                    legacy_ratio_builder,
                    getattr(image_processor, current_ratio_builder),
                )
                logger.warning(
                    "  Compatibility fix: aliased image_processor."
                    "build_img_ratio_slice_logits_proc to "
                    "build_img_ratio_slice_logits_processor."
                )


        # Accelerate cannot materialize the SigLIP2 positional embedding when
        # the official forward reads ``position_embedding.weight`` directly.
        # Replace that one implementation with a module call so its disk/CPU
        # offload hook runs normally.
        if model_info.get("disk_offload"):
            _patch_siglip2_position_embedding_for_disk_offload(model)
            _patch_multihead_attention_offload_hooks(model)

        # Apply critical patches for memory management and dtype compatibility
        if SHARED_UTILS_AVAILABLE:
            logger.info("Applying dtype compatibility patches...")
            if patch_dynamic_cache_dtype:
                patch_dynamic_cache_dtype()
            if patch_static_cache_lazy_init:
                patch_static_cache_lazy_init()
            
            # CRITICAL: Fix upstream to_device() to handle dict inputs.
            # Without this, image editing fails because the vision encoder's
            # attention_mask stays on CPU while the model runs on CUDA.
            if patch_to_device_for_instruct:
                patch_to_device_for_instruct(model)
            
            # CRITICAL: Apply pre-VAE cleanup patch to clear KV cache before VAE decode
            # This prevents OOM during VAE decode by freeing transformer cache first
            if patch_pipeline_pre_vae_cleanup:
                patch_pipeline_pre_vae_cleanup(model, enabled=True)
                logger.info("Applied pre-VAE memory cleanup patch")
            
            # Fix HunyuanStaticCache device mismatch on multi-GPU setups
            if patch_hunyuan_static_cache_device:
                patch_hunyuan_static_cache_device(model)
        
        # Setup block swap for models with blocks_to_swap > 0
        # Works for NF4 (proven), INT8 (Int8Params.to() supports device movement),
        # and BF16 (standard tensor .to()). For INT8/BF16, the model was loaded
        # to CPU above and non-block components already moved to GPU.
        block_swap_manager = None
        if blocks_to_swap > 0 and BLOCK_SWAP_AVAILABLE:
            logger.info(f"Setting up block swap for {quant_type.upper()} model ({blocks_to_swap} blocks)...")
            swap_config = BlockSwapConfig(
                blocks_to_swap=blocks_to_swap,
                prefetch_blocks=2,
                use_non_blocking=True,
            )
            block_swap_manager = BlockSwapManager(model, swap_config, target_device="cuda:0")
            block_swap_manager.setup_initial_placement()
            block_swap_manager.install_hooks()
            
            summary = block_swap_manager.get_memory_summary()
            logger.info(f"  Block swap active: {summary['blocks_on_gpu']} on GPU, "
                       f"{summary['blocks_on_cpu']} on CPU")
            logger.info(f"  VRAM saved: ~{summary['cpu_memory_gb']:.1f}GB")
            logger.info(f"  Block size: ~{summary['gb_per_block']:.2f}GB each")
            logger.info(f"  Quality is NOT affected — only memory management changes")
            
            # Clean up stale hf_device_map from device_map="cpu" loading.
            # After block swap setup, we manage device placement ourselves.
            # The stale map would confuse accelerate and logging.
            if hasattr(model, 'hf_device_map'):
                delattr(model, 'hf_device_map')
                logger.debug("  Removed stale hf_device_map (block swap manages placement)")
            
            # Remove any accelerate dispatch hooks that device_map="cpu" may have installed.
            # With device_map="cpu", accelerate might install minimal hooks even though
            # everything is on CPU. Our block swap hooks replace them.
            # Also clean up instance-level `forward` attributes left by hook removal
            # to prevent the cache-clear monkey-patch remover from nuking their
            # __class__ closure cell (which permanently breaks super()).
            try:
                from accelerate.hooks import remove_hook_from_module
                for name, module in model.named_modules():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module)
                    if 'forward' in vars(module):
                        try:
                            delattr(module, 'forward')
                        except Exception:
                            pass
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"  Note: could not clean accelerate hooks: {e}")
            
        elif blocks_to_swap > 0 and not BLOCK_SWAP_AVAILABLE:
            logger.warning("Block swap requested but hunyuan_block_swap module not available!")

        # Optional VAE precision override (quality: cleaner gradients on big cards)
        if vae_dtype == "float32" and hasattr(model, "vae"):
            try:
                _vae_dev = next(model.vae.parameters()).device
                model.vae = model.vae.to(dtype=torch.float32)
                logger.info(f"VAE cast to float32 (device={_vae_dev}) for higher decode precision")
            except Exception as e:
                logger.warning(f"Could not cast VAE to float32: {e}")

        # Diagnostic: verify device placement after setup
        if blocks_to_swap > 0:
            try:
                vae_dev = next(model.vae.parameters()).device if hasattr(model, 'vae') else "N/A"
                model_dev_prop = model.device  # PreTrainedModel.device property
                first_param = next(model.parameters())
                first_param_dev = first_param.device
                first_param_name = next(n for n, p in model.named_parameters() if p is first_param)
                block0_dev = next(model.model.layers[0].parameters()).device
                logger.info(f"  Device check: model.device={model_dev_prop}, "
                           f"first_param=({first_param_name}, {first_param_dev}), "
                           f"vae={vae_dev}, block0={block0_dev}")
                # Check if accelerate hooks remain
                has_hooks = any(hasattr(m, '_hf_hook') for _, m in model.named_modules())
                logger.info(f"  Accelerate hooks remaining: {has_hooks}")
            except Exception as e:
                logger.warning(f"  Device diagnostic failed: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.1f}s")
        
        # Log memory usage across all GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                if allocated > 0.1:
                    free_bytes, _ = torch.cuda.mem_get_info(i)
                    logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {free_bytes/1024**3:.1f}GB free")
        
        # Attach model info for downstream nodes
        model._hunyuan_info = model_info
        model._hunyuan_path = model_path
        
        # Attach block swap manager if active
        if block_swap_manager is not None:
            model._block_swap_manager = block_swap_manager
        
        # Cache the model
        _instruct_cache.put(model_path, model, model_info)
        
        return (model,)


# =============================================================================
# Node: HunyuanInstructGenerate
# =============================================================================

class HunyuanInstructGenerate:
    """
    Text-to-Image generation with Instruct model.
    
    Features built-in prompt enhancement via bot_task modes:
    - image: Direct generation (no enhancement)
    - recaption: Enhance prompt then generate
    - think_recaption: Think (CoT) → enhance → generate (best quality)
    
    Returns the generated image and optional CoT reasoning text.
    """
    
    CATEGORY = "Hunyuan/Instruct"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "cot_reasoning", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_INSTRUCT_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful sunset over mountains with dramatic clouds",
                    "tooltip": "Text prompt for image generation"
                }),
                "bot_task": (BOT_TASK_MODES, {
                    "default": "image",
                    "tooltip": (
                        "Controls how the model processes your prompt before generating.\n"
                        "• image: Direct generation — your prompt is used as-is with no enhancement.\n"
                        "• recaption: The model rewrites your prompt into a detailed, structured description "
                        "(photography terms, composition, lighting, etc.) then generates from that enhanced prompt.\n"
                        "• think_recaption: (BEST QUALITY) The model first reasons about your prompt using "
                        "Chain-of-Thought (CoT), analyzing intent, style, composition, then rewrites the prompt, "
                        "then generates. Slower but produces the highest quality results.\n"
                        "The CoT reasoning text is returned via the cot_reasoning output.\n\n"
                        "WARNING: recaption and think_recaption add several minutes of autoregressive text "
                        "generation before the image starts. Even after recent decode-path fixes the cost is "
                        "significant on long prompts. Use 'image' for direct generation when you already have "
                        "a well-structured prompt."
                    )
                }),
                "system_prompt": (SYSTEM_PROMPT_OPTIONS, {
                    "default": "dynamic",
                    "tooltip": (
                        "System prompt that guides how the model interprets and processes your input.\n"
                        "• dynamic: (RECOMMENDED) Auto-selects the best prompt for your bot_task — "
                        "no system prompt for image (clean T2I), en_recaption for recaption, "
                        "en_unified for think_recaption.\n"
                        "• en_unified: Comprehensive prompt covering T2I, editing, recaption, and CoT.\n"
                        "• en_recaption: Focused on prompt rewriting — best with bot_task=recaption.\n"
                        "• en_think_recaption: Focused on CoT + rewriting — best with think_recaption.\n"
                        "• en_vanilla: Minimal prompt — just 'generate a high-quality image'.\n"
                        "• none: No system prompt sent to the model at all.\n"
                        "For bot_task=image, system prompt text can interfere with image quality. "
                        "For recaption/think_recaption, system prompts guide CoT reasoning and are beneficial."
                    )
                }),
                "resolution": (RESOLUTION_LIST, {
                    "default": "1024x1024 (1:1 Square)",
                    "tooltip": "Output image resolution"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "-1 for random seed"
                }),
            },
            "optional": {
                "steps": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": (
                        "-1 for auto (8 for Distil, 40 for full Instruct). "
                        "For full Instruct, higher step counts (50–80) reduce flow-matching artifacts "
                        "at 2K+ resolutions but generation time scales linearly — expect a much longer wait."
                    )
                }),
                "guidance_scale": ("FLOAT", {
                    "default": -1,
                    "min": -1,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG scale. -1 = auto (uses model's recommended value, typically 2.5)"
                }),
                "flow_shift": ("FLOAT", {
                    "default": 2.8,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Flow shift for the diffusion scheduler. Controls denoising schedule shape.\n"
                        "Default 2.8 is balanced. Presets:\n"
                        "  Portraits / faces: 2.0–2.5 (sharper detail).\n"
                        "  Landscapes / illustrations: 3.5–5.0 (cleaner gradients, less high-frequency noise)."
                    )
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "tooltip": (
                        "Maximum tokens the model can generate for CoT reasoning and prompt rewriting. "
                        "Only used when bot_task is recaption or think_recaption. "
                        "Higher values allow more detailed reasoning but use more time and memory. "
                        "2048 is usually sufficient; increase if CoT output is getting truncated."
                    )
                }),
                "verbose": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Verbosity level. 0=silent (recommended), 1=info (shows full system prompt), 2=debug"
                }),
                "vae_tiling": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": (
                        "VAE decode tiling.\n"
                        "• auto: Enable tiling automatically when free VRAM is below the safety threshold (post block-swap release).\n"
                        "• on: Always tile (slower but lowest VRAM, recommended for >=2K outputs on ≤24GB cards).\n"
                        "• off: Never tile (fastest, may OOM at high resolution)."
                    )
                }),
                "vae_offload": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": (
                        "VAE GPU↔CPU offload around decode.\n"
                        "• auto: Move VAE to GPU just for decode and back to CPU after, when VRAM is tight.\n"
                        "• on: Always offload VAE to CPU when idle.\n"
                        "• off: Keep VAE on GPU at all times (fastest)."
                    )
                }),
            }
        }
    
    def generate(
        self,
        model,
        prompt: str,
        bot_task: str = "think_recaption",
        system_prompt: str = "dynamic",
        resolution: str = "1024x1024 (1:1 Square)",
        seed: int = -1,
        steps: int = -1,
        guidance_scale: float = -1,
        flow_shift: float = 2.8,
        max_new_tokens: int = 2048,
        verbose: int = 0,
        vae_tiling: str = "auto",
        vae_offload: str = "auto",
    ) -> Tuple[torch.Tensor, str, str]:
        """Generate image from text prompt."""
        
        # Parse resolution
        res_mode, height, width = parse_resolution(resolution)
        if res_mode == "auto":
            image_size = "auto"
        else:
            image_size = f"{height}x{width}"  # HxW format for HunyuanImage3 API
        
        # Get model info
        model_info = getattr(model, "_hunyuan_info", {"default_steps": 40})
        
        # Determine steps
        if steps == -1:
            steps = model_info.get("default_steps", 40)
        
        # Determine guidance scale - use model's recommended value when auto
        if guidance_scale < 0:
            guidance_scale = getattr(
                model.generation_config, "diff_guidance_scale", 2.5
            )
        
        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()
        
        # Handle system prompt selection.
        # The upstream instruct model's generate_image calls get_system_prompt()
        # and then does system_prompt.strip() WITHOUT a None guard.
        # get_system_prompt("None", ...) returns Python None → crashes on .strip().
        # Workaround: use sys_type="custom" with system_prompt="" for no-prompt mode.
        # This returns "" which survives .strip() and is falsy → effectively no prompt.
        use_system_prompt_value = None   # What we pass as use_system_prompt kwarg
        custom_system_prompt = None      # What we pass as system_prompt kwarg (only for "custom" mode)
        
        if system_prompt == "dynamic":
            resolved = DYNAMIC_PROMPT_MAP.get(bot_task, "en_unified")
            if resolved == "None":
                # Want no system prompt — use "custom" with empty string to avoid .strip() crash
                use_system_prompt_value = "custom"
                custom_system_prompt = ""
                logger.info(f"  Dynamic system prompt: disabled (for bot_task='{bot_task}')")
            else:
                use_system_prompt_value = resolved
                logger.info(f"  Dynamic system prompt: '{resolved}' (for bot_task='{bot_task}')")
        elif system_prompt == "none":
            # User explicitly wants no system prompt
            use_system_prompt_value = "custom"
            custom_system_prompt = ""
            logger.info(f"  System prompt: disabled (none)")
        else:
            use_system_prompt_value = system_prompt
            logger.info(f"  System prompt: '{use_system_prompt_value}'")
        
        logger.info(f"Generating image:")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Bot task: {bot_task}")
        logger.info(f"  Resolution: {image_size}")
        logger.info(f"  Steps: {steps}, Seed: {seed}")
        
        start_time = time.time()
        
        # Aggressive VRAM cleanup before generation - clears stale KV cache
        # from previous runs which can hold 1-4GB of VRAM
        _aggressive_vram_cleanup(model, context="T2I generation")

        # Propagate user VAE preferences to the patched decode wrapper (read
        # by patch_pipeline_pre_vae_cleanup at decode time).
        model._vae_tiling_mode = vae_tiling
        model._vae_offload_mode = vae_offload
        
        try:
            # Call model's generate_image method
            # This matches the official run_image_gen.py calling convention exactly.
            # The upstream method handles bot_task decomposition internally:
            #   - "image": direct text-to-image
            #   - "recaption": prompt enhancement → image generation
            #   - "think_recaption": CoT reasoning → prompt enhancement → image generation
            # For recaption/think_recaption, autoregressive text generation runs first
            # (can take several minutes and uses significant KV cache memory).
            logger.info(f"Starting generation with bot_task='{bot_task}' "
                        f"(this may take several minutes for recaption modes)...")
            
            # Set generation_config attributes directly — the upstream pipeline
            # reads diff_infer_steps and diff_guidance_scale from generation_config,
            # NOT from kwargs passed to generate_image().
            model.generation_config.diff_infer_steps = steps
            model.generation_config.diff_guidance_scale = guidance_scale
            model.generation_config.flow_shift = flow_shift
            
            # NOTE: Do NOT wrap in torch.inference_mode() — the upstream already
            # uses @torch.no_grad(), and inference_mode() conflicts with
            # accelerate's device dispatch hooks when device_map='auto' is used,
            # causing silent CUDA crashes.
            result = model.generate_image(
                prompt=prompt,
                image=None,  # No input image for T2I
                seed=seed,
                image_size=image_size,
                use_system_prompt=use_system_prompt_value,
                system_prompt=custom_system_prompt,  # Only used when use_system_prompt="custom"
                bot_task=bot_task,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            
            # Handle return value - Instruct always returns (cot_text, samples_list)
            if isinstance(result, tuple) and len(result) == 2:
                cot_text, samples = result
            else:
                # Fallback for unexpected return format
                cot_text = ""
                samples = [result] if isinstance(result, Image.Image) else result
            
            # Convert PIL to tensor
            if samples and len(samples) > 0:
                pil_image = samples[0]
                image_tensor = pil_to_tensor(pil_image)
            else:
                raise RuntimeError("No image generated")
            
            elapsed = time.time() - start_time
            status = f"Generated in {elapsed:.1f}s | Seed: {seed} | Steps: {steps} | Mode: {bot_task}"
            
            logger.info(f"Generation complete: {elapsed:.1f}s")
            if cot_text:
                logger.info(f"CoT reasoning: {cot_text[:200]}...")
            
            # CRITICAL: Clear KV cache and intermediate state after generation.
            # Without this, the KV cache from the text generation phase persists
            # in GPU memory, causing OOM or stale-state errors on subsequent runs
            # when force_reload=False (model is reused from cache).
            if SHARED_UTILS_AVAILABLE and clear_generation_cache:
                clear_generation_cache(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return (image_tensor, cot_text or "", status)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory during generation: {e}")
            # Try to recover - wrap empty_cache in try/except because
            # after OOM the CUDA context may be in error state
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache after OOM")
            
            error_msg = (
                f"Out of GPU memory during '{bot_task}' generation. "
                f"The MoE dispatch_mask grows O(N²) with token count. "
            )
            # Add CFG-specific guidance
            is_cfg_model = not getattr(model.config, 'cfg_distilled', True)
            if is_cfg_model:
                error_msg += (
                    f"This is a full Instruct model (CFG batch=2) — all inference "
                    f"tensors are DOUBLED vs Distil. Dispatch_mask for '{bot_task}': "
                    f"{'~56GB' if bot_task == 'think_recaption' else '~8-16GB'}. "
                )
            else:
                error_msg += (
                    f"think_recaption uses ~28GB for the dispatch mask alone. "
                )
            error_msg += (
                f"Try: (1) Increase blocks_to_swap in the Loader "
                f"({'to 10-15 for INT8 CFG' if is_cfg_model else 'works for NF4, INT8, and BF16'}), "
                f"(2) Use bot_task='image' (much lower dispatch memory), "
                f"(3) Use Instruct-Distil (no CFG, halves all tensors), "
                f"(4) Increase vram_reserve_gb in the Loader."
            )
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, "", f"Error: {error_msg}")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to recover GPU memory
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache after error")
            
            # Return empty image on failure
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, "", f"Error: {str(e)}")


# =============================================================================
# Node: HunyuanInstructImageEdit
# =============================================================================

class HunyuanInstructImageEdit:
    """
    Image-to-Image editing with Instruct model.
    
    Takes an input image and an instruction to modify it.
    Supports operations like:
    - Adding/removing elements
    - Style changes
    - Background replacement
    - Object modifications
    """
    
    CATEGORY = "Hunyuan/Instruct"
    FUNCTION = "edit"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "cot_reasoning", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_INSTRUCT_MODEL",),
                "image": ("IMAGE",),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Change the background to a sunset scene",
                    "tooltip": "Instruction for how to modify the image"
                }),
                "bot_task": (BOT_TASK_MODES, {
                    "default": "image",
                    "tooltip": (
                        "Controls how the model processes your edit instruction.\n"
                        "• image: Direct edit — your instruction is applied as-is.\n"
                        "• recaption: The model rewrites your instruction into a detailed edit description "
                        "(preserving elements, specifying changes) then applies the edit.\n"
                        "• think_recaption: (BEST QUALITY) The model first reasons about what to change "
                        "and what to preserve using CoT analysis, rewrites the instruction, then edits. "
                        "Best for complex edits like style transfer, element replacement, or multi-step changes.\n\n"
                        "WARNING: recaption and think_recaption add several minutes of autoregressive text "
                        "generation before the edit starts. Use 'image' for direct editing when your "
                        "instruction is already clear."
                    )
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "-1 for random seed"
                }),
            },
            "optional": {
                "system_prompt": (SYSTEM_PROMPT_OPTIONS, {
                    "default": "dynamic",
                    "tooltip": (
                        "System prompt guiding the model's edit behavior. "
                        "dynamic (recommended) auto-selects the best prompt for your bot_task. "
                        "en_unified covers all editing modes including replacement, "
                        "addition, removal, style transfer, and text editing. "
                        "The model understands both English and Chinese instructions natively."
                    )
                }),
                "align_output_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match output size to input image size (ignored unless resolution=auto)"
                }),
                "resolution": (RESOLUTION_LIST, {
                    "default": "auto",
                    "tooltip": (
                        "Output resolution. 'auto' lets the model pick (or matches input if "
                        "align_output_size=True). Otherwise overrides with a fixed preset."
                    )
                }),
                "steps": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": (
                        "-1 for auto (8 for Distil, 50 for full Instruct). "
                        "Higher step counts (60–80) reduce flow-matching artifacts at 2K+ resolutions "
                        "but generation time scales linearly — expect a much longer wait."
                    )
                }),
                "guidance_scale": ("FLOAT", {
                    "default": -1,
                    "min": -1,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG scale. -1 = auto (model's recommended value, typically 2.5)"
                }),
                "flow_shift": ("FLOAT", {
                    "default": 2.8,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Flow shift for the diffusion scheduler. Controls denoising schedule shape.\n"
                        "Default 2.8 is balanced. Presets:\n"
                        "  Portraits / faces: 2.0–2.5 (sharper detail).\n"
                        "  Landscapes / illustrations: 3.5–5.0 (cleaner gradients)."
                    )
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "tooltip": (
                        "Maximum tokens for CoT reasoning and instruction rewriting. "
                        "Only used with recaption or think_recaption modes. "
                        "2048 is usually sufficient."
                    )
                }),
                "verbose": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Verbosity level. 0=silent (recommended), 1=info (shows full system prompt), 2=debug"
                }),
            }
        }
    
    def edit(
        self,
        model,
        image: torch.Tensor,
        instruction: str,
        bot_task: str = "think_recaption",
        seed: int = -1,
        system_prompt: str = "dynamic",
        align_output_size: bool = True,
        resolution: str = "auto",
        steps: int = -1,
        guidance_scale: float = -1,
        flow_shift: float = 2.8,
        max_new_tokens: int = 2048,
        verbose: int = 0,
    ) -> Tuple[torch.Tensor, str, str]:
        """Edit image based on instruction."""
        
        # Convert input image to temp file
        temp_path = tensor_to_temp_path(image)
        temp_files = [temp_path]
        
        try:
            # Get model info
            model_info = getattr(model, "_hunyuan_info", {"default_steps": 40})
            
            # Determine steps
            if steps == -1:
                steps = model_info.get("default_steps", 40)
            
            # Determine guidance scale - use model's recommended value when auto
            if guidance_scale < 0:
                guidance_scale = getattr(
                    model.generation_config, "diff_guidance_scale", 2.5
                )
            
            # Handle seed
            if seed == -1:
                seed = torch.randint(0, 2147483647, (1,)).item()
            
            # Handle system prompt selection (same logic as generate node).
            # The upstream instruct model does system_prompt.strip() without a None guard,
            # so we use sys_type="custom" + empty string instead of "None" to avoid crash.
            use_system_prompt_value = None
            custom_system_prompt = None
            
            if system_prompt == "dynamic":
                resolved = DYNAMIC_PROMPT_MAP.get(bot_task, "en_unified")
                if resolved == "None":
                    use_system_prompt_value = "custom"
                    custom_system_prompt = ""
                    logger.info(f"  Dynamic system prompt: disabled (for bot_task='{bot_task}')")
                else:
                    use_system_prompt_value = resolved
                    logger.info(f"  Dynamic system prompt: '{resolved}' (for bot_task='{bot_task}')")
            elif system_prompt == "none":
                use_system_prompt_value = "custom"
                custom_system_prompt = ""
                logger.info(f"  System prompt: disabled (none)")
            else:
                use_system_prompt_value = system_prompt
                logger.info(f"  System prompt: '{use_system_prompt_value}'")
            
            logger.info(f"Editing image:")
            logger.info(f"  Instruction: {instruction[:100]}...")
            logger.info(f"  Bot task: {bot_task}")
            logger.info(f"  Steps: {steps}, Seed: {seed}")
            
            start_time = time.time()
            
            # Aggressive VRAM cleanup before image edit - clears stale KV cache
            _aggressive_vram_cleanup(model, context="image edit")
            
            logger.info(f"Starting image edit with bot_task='{bot_task}' "
                        f"(this may take several minutes for recaption modes)...")
            
            # Set generation_config attributes directly — upstream reads from
            # generation_config, not from kwargs.
            model.generation_config.diff_infer_steps = steps
            model.generation_config.diff_guidance_scale = guidance_scale
            model.generation_config.flow_shift = flow_shift
            
            # Call model's generate_image with input image
            # NOTE: No torch.inference_mode() — conflicts with accelerate hooks
            res_mode, height, width = parse_resolution(resolution)
            if res_mode == "auto":
                image_size = "auto"
            else:
                image_size = f"{height}x{width}"
            logger.info(f"  Resolution: {image_size}")
            result = model.generate_image(
                prompt=instruction,
                image=temp_path,  # Single image path
                seed=seed,
                image_size=image_size,
                use_system_prompt=use_system_prompt_value,
                system_prompt=custom_system_prompt,
                bot_task=bot_task,
                infer_align_image_size=align_output_size and image_size == "auto",
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            
            # Handle return value
            if isinstance(result, tuple) and len(result) == 2:
                cot_text, samples = result
            else:
                cot_text = ""
                samples = [result] if isinstance(result, Image.Image) else result
            
            # Convert PIL to tensor
            if samples and len(samples) > 0:
                pil_image = samples[0]
                image_tensor = pil_to_tensor(pil_image)
            else:
                raise RuntimeError("No image generated")
            
            elapsed = time.time() - start_time
            status = f"Edited in {elapsed:.1f}s | Seed: {seed} | Steps: {steps} | Mode: {bot_task}"
            
            logger.info(f"Edit complete: {elapsed:.1f}s")
            
            # CRITICAL: Clear KV cache after generation to prevent OOM/stale state on next run
            if SHARED_UTILS_AVAILABLE and clear_generation_cache:
                clear_generation_cache(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return (image_tensor, cot_text or "", status)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory during image edit: {e}")
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache after OOM")
            
            error_msg = (
                f"Out of GPU memory during '{bot_task}' image edit. "
                f"Image editing concatenates image tokens + text tokens, making the "
                f"MoE dispatch_mask much larger (O(N²)). "
            )
            is_cfg_model = not getattr(model.config, 'cfg_distilled', True)
            if is_cfg_model:
                error_msg += (
                    f"This is a full Instruct model (CFG batch=2) — all inference "
                    f"tensors are DOUBLED. For image edit with '{bot_task}', "
                    f"dispatch_mask can exceed 30GB. "
                )
            error_msg += (
                f"Try: (1) Increase blocks_to_swap in the Loader "
                f"({'to 10-15 for INT8 CFG' if is_cfg_model else 'works for NF4, INT8, and BF16'}), "
                f"(2) Use bot_task='image' (much lower dispatch memory), "
                f"(3) Use Instruct-Distil, (4) Reduce max_new_tokens, "
                f"(5) Increase vram_reserve_gb in the Loader."
            )
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, "", f"Error: {error_msg}")
            
        except Exception as e:
            logger.error(f"Image edit failed: {e}")
            import traceback
            traceback.print_exc()
            
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache after error")
            
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, "", f"Error: {str(e)}")
            
        finally:
            # Clean up temp files
            cleanup_temp_files(temp_files)


# =============================================================================
# Node: HunyuanInstructMultiFusion
# =============================================================================

class HunyuanInstructMultiFusion:
    """
    Multi-image fusion with Instruct model.
    
    Combines elements from 2-5 reference images based on instruction.
    The model officially supports up to 3 images; 4-5 are experimental
    and require more VRAM.
    Examples:
    - "Based on image 1's logo, create a fridge magnet with image 2's material"
    - "Combine the style of image 1 with the subject from image 2"
    - "Place the object from image 1 into the scene from image 2"
    """
    
    CATEGORY = "Hunyuan/Instruct"
    FUNCTION = "fuse"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "cot_reasoning", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUNYUAN_INSTRUCT_MODEL",),
                "image_1": ("IMAGE",),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Combine the style of image 1 with elements from image 2",
                    "tooltip": (
                        "Instruction describing how to combine the images. "
                        "Reference images as 'image 1', 'image 2', 'image 3', etc. "
                        "Examples: 'Based on image 1 logo, create a fridge magnet with image 2 material', "
                        "'Let the cat from image 1 take a selfie with the cat from image 2, "
                        "with image 3 as background'. Works with both English and Chinese prompts."
                    )
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "-1 for random seed"
                }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE", {"tooltip": "Experimental — model officially supports up to 3, but the pipeline accepts more. Increases VRAM usage significantly."}),
                "image_5": ("IMAGE", {"tooltip": "Experimental — model officially supports up to 3, but the pipeline accepts more. Increases VRAM usage significantly."}),
                "bot_task": (BOT_TASK_MODES, {
                    "default": "image",
                    "tooltip": (
                        "Controls how the model processes your fusion instruction.\\n"
                        "\u2022 image: Direct fusion \u2014 instruction applied as-is.\\n"
                        "\u2022 recaption: Rewrites instruction into detailed fusion description then generates.\\n"
                        "\u2022 think_recaption: (BEST) CoT reasoning about how to combine elements, "
                        "rewrites instruction, then generates. Recommended for complex multi-image fusion."
                    )
                }),
                "system_prompt": (SYSTEM_PROMPT_OPTIONS, {
                    "default": "dynamic",
                    "tooltip": (
                        "System prompt guiding multi-image fusion. "
                        "dynamic (recommended) auto-selects the best prompt for your bot_task. "
                        "en_unified includes reference editing guidelines for extracting "
                        "and combining elements across images. "
                        "The model understands both English and Chinese instructions."
                    )
                }),
                "resolution": (RESOLUTION_LIST, {
                    "default": "1024x1024 (1:1 Square)",
                    "tooltip": (
                        "Output image resolution. Auto lets the model predict aspect ratio "
                        "(often unpredictable with multiple input images of different sizes).\n"
                        "Selecting a specific resolution gives deterministic output dimensions."
                    )
                }),
                "steps": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": (
                        "-1 for auto (8 for Distil, 50 for full Instruct). "
                        "Higher step counts (60–80) reduce flow-matching artifacts at 2K+ resolutions "
                        "but generation time scales linearly."
                    )
                }),
                "guidance_scale": ("FLOAT", {
                    "default": -1,
                    "min": -1,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG scale. -1 = auto (model's recommended value, typically 2.5)"
                }),
                "flow_shift": ("FLOAT", {
                    "default": 2.8,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": (
                        "Flow shift for the diffusion scheduler. Controls denoising schedule shape.\n"
                        "Default 2.8 is balanced. Presets:\n"
                        "  Portraits / faces: 2.0–2.5 (sharper detail).\n"
                        "  Landscapes / illustrations: 3.5–5.0 (cleaner gradients)."
                    )
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "tooltip": (
                        "Maximum tokens for CoT reasoning and instruction rewriting. "
                        "Only used with recaption or think_recaption modes. "
                        "Multi-image fusion may benefit from higher values (e.g. 3072) for complex instructions."
                    )
                }),
                "verbose": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "tooltip": "Verbosity level. 0=silent (recommended), 1=info (shows full system prompt), 2=debug"
                }),
            }
        }
    
    def fuse(
        self,
        model,
        image_1: torch.Tensor,
        instruction: str,
        seed: int = -1,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
        image_5: Optional[torch.Tensor] = None,
        bot_task: str = "think_recaption",
        system_prompt: str = "dynamic",
        resolution: str = "1024x1024 (1:1 Square)",
        steps: int = -1,
        guidance_scale: float = -1,
        flow_shift: float = 2.8,
        max_new_tokens: int = 2048,
        verbose: int = 0,
    ) -> Tuple[torch.Tensor, str, str]:
        """Fuse multiple images based on instruction."""
        
        # Convert input images to temp files
        temp_files = []
        image_paths = []
        
        # Image 1 is required
        path1 = tensor_to_temp_path(image_1)
        temp_files.append(path1)
        image_paths.append(path1)
        
        # Images 2-5 are optional
        for img in [image_2, image_3, image_4, image_5]:
            if img is not None:
                path = tensor_to_temp_path(img)
                temp_files.append(path)
                image_paths.append(path)
        
        try:
            # Get model info
            model_info = getattr(model, "_hunyuan_info", {"default_steps": 40})
            
            # Determine steps
            if steps == -1:
                steps = model_info.get("default_steps", 40)
            
            # Determine guidance scale - use model's recommended value when auto
            if guidance_scale < 0:
                guidance_scale = getattr(
                    model.generation_config, "diff_guidance_scale", 2.5
                )
            
            # Handle seed
            if seed == -1:
                seed = torch.randint(0, 2147483647, (1,)).item()
            
            # Handle system prompt selection (same logic as generate node).
            # The upstream instruct model does system_prompt.strip() without a None guard,
            # so we use sys_type="custom" + empty string instead of "None" to avoid crash.
            use_system_prompt_value = None
            custom_system_prompt = None
            
            if system_prompt == "dynamic":
                resolved = DYNAMIC_PROMPT_MAP.get(bot_task, "en_unified")
                if resolved == "None":
                    use_system_prompt_value = "custom"
                    custom_system_prompt = ""
                    logger.info(f"  Dynamic system prompt: disabled (for bot_task='{bot_task}')")
                else:
                    use_system_prompt_value = resolved
                    logger.info(f"  Dynamic system prompt: '{resolved}' (for bot_task='{bot_task}')")
            elif system_prompt == "none":
                use_system_prompt_value = "custom"
                custom_system_prompt = ""
                logger.info(f"  System prompt: disabled (none)")
            else:
                use_system_prompt_value = system_prompt
                logger.info(f"  System prompt: '{use_system_prompt_value}'")
            
            # Parse resolution
            res_mode, height, width = parse_resolution(resolution)
            if res_mode == "auto":
                image_size = "auto"
            else:
                image_size = f"{height}x{width}"  # HxW format for HunyuanImage3 API
            
            logger.info(f"Multi-image fusion:")
            logger.info(f"  Instruction: {instruction[:100]}...")
            logger.info(f"  Number of images: {len(image_paths)}")
            logger.info(f"  Bot task: {bot_task}")
            logger.info(f"  Resolution: {image_size}")
            logger.info(f"  Steps: {steps}, Seed: {seed}")
            
            start_time = time.time()
            
            # Aggressive VRAM cleanup before fusion - clears stale KV cache
            _aggressive_vram_cleanup(model, context="multi-fusion")
            
            logger.info(f"Starting multi-image fusion with bot_task='{bot_task}' "
                        f"(this may take several minutes for recaption modes)...")
            
            # Set generation_config attributes directly — upstream reads from
            # generation_config, not from kwargs.
            model.generation_config.diff_infer_steps = steps
            model.generation_config.diff_guidance_scale = guidance_scale
            model.generation_config.flow_shift = flow_shift
            
            # Call model's generate_image with list of images
            # NOTE: No torch.inference_mode() — conflicts with accelerate hooks
            result = model.generate_image(
                prompt=instruction,
                image=image_paths,  # List of image paths
                seed=seed,
                image_size=image_size,
                use_system_prompt=use_system_prompt_value,
                system_prompt=custom_system_prompt,
                bot_task=bot_task,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            
            # Handle return value
            if isinstance(result, tuple) and len(result) == 2:
                cot_text, samples = result
            else:
                cot_text = ""
                samples = [result] if isinstance(result, Image.Image) else result
            
            # Convert PIL to tensor
            if samples and len(samples) > 0:
                pil_image = samples[0]
                image_tensor = pil_to_tensor(pil_image)
            else:
                raise RuntimeError("No image generated")
            
            elapsed = time.time() - start_time
            status = f"Fused {len(image_paths)} images in {elapsed:.1f}s | Seed: {seed} | Mode: {bot_task}"
            
            logger.info(f"Fusion complete: {elapsed:.1f}s")
            
            # CRITICAL: Clear KV cache after generation to prevent OOM/stale state on next run
            if SHARED_UTILS_AVAILABLE and clear_generation_cache:
                clear_generation_cache(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return (image_tensor, cot_text or "", status)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory during multi-image fusion: {e}")
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache after OOM")
            
            error_msg = (
                f"Out of GPU memory during '{bot_task}' multi-image fusion. "
                f"Multiple images increase token count significantly, causing the "
                f"MoE dispatch_mask to grow O(N²). "
            )
            is_cfg_model = not getattr(model.config, 'cfg_distilled', True)
            if is_cfg_model:
                error_msg += (
                    f"This is a full Instruct model (CFG batch=2) — all inference "
                    f"tensors are DOUBLED. Multi-image fusion with CFG can exceed 40GB. "
                )
            error_msg += (
                f"Try: (1) Increase blocks_to_swap in the Loader "
                f"({'to 15+ for INT8 CFG fusion' if is_cfg_model else 'works for NF4, INT8, and BF16'}), "
                f"(2) Use bot_task='image' mode, "
                f"(3) Use fewer input images, (4) Use a smaller resolution, "
                f"(5) Use Instruct-Distil (no CFG, halves all tensors)."
            )
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, "", f"Error: {error_msg}")
            
        except Exception as e:
            logger.error(f"Multi-image fusion failed: {e}")
            import traceback
            traceback.print_exc()
            
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache after error")
            
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, "", f"Error: {str(e)}")
            
        finally:
            # Clean up temp files
            cleanup_temp_files(temp_files)


# =============================================================================
# Node: HunyuanInstructUnload
# =============================================================================

class HunyuanInstructUnload:
    """
    Unload Instruct model from memory.
    
    Use this to free up VRAM when done with the Instruct model.
    """
    
    CATEGORY = "Hunyuan/Instruct"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Toggle unload on/off. When False, node passes through without unloading. "
                               "Useful for keeping the node wired in your workflow but skipping unload "
                               "when you want to keep the model loaded for multiple runs."
                }),
            },
            "optional": {
                "trigger": ("*", {"default": None}),
            }
        }
    
    def unload(self, enabled: bool = True, trigger=None) -> Tuple[str]:
        """Unload the cached model and hunt orphaned RAM tensors."""
        if not enabled:
            logger.info("Instruct unload skipped (disabled)")
            return ("Unload skipped (disabled)",)
        
        import psutil
        ram_before = psutil.Process().memory_info().rss / 1024**3
        
        # Step 1: Clear our instruct cache (9-step circular ref cleanup)
        _instruct_cache.clear()
        
        # Step 2: Also clear base model cache if loaded
        if SHARED_UTILS_AVAILABLE:
            try:
                from .hunyuan_shared import HunyuanModelCache
                if HunyuanModelCache._cached_model is not None:
                    logger.info("Also clearing base HunyuanModelCache...")
                    HunyuanModelCache.clear()
            except ImportError:
                try:
                    from hunyuan_shared import HunyuanModelCache
                    if HunyuanModelCache._cached_model is not None:
                        logger.info("Also clearing base HunyuanModelCache...")
                        HunyuanModelCache.clear()
                except ImportError:
                    pass
        
        # Step 3: Extra gc rounds to catch remaining cycles
        gc.collect()
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Step 4: Final gc + VRAM flush (orphan hunt removed — it was gutting
        # ALL nn.Module instances in the process including downstream models
        # like Marigold/segmentation, causing "weight should have at least
        # three dimensions" errors)
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force Windows to return freed memory to OS
        try:
            from .hunyuan_shared import force_windows_memory_release
        except ImportError:
            from hunyuan_shared import force_windows_memory_release
        force_windows_memory_release()
        
        if torch.cuda.is_available():
            # Report VRAM status
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            
            # Report RAM status
            ram_after = psutil.Process().memory_info().rss / 1024**3
            ram_freed = ram_before - ram_after
            
            status = (f"Model unloaded. "
                     f"VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total. "
                     f"RAM: {ram_after:.1f}GB (freed {ram_freed:.1f}GB)")
        else:
            ram_after = psutil.Process().memory_info().rss / 1024**3
            ram_freed = ram_before - ram_after
            status = f"Model unloaded (no CUDA). RAM: {ram_after:.1f}GB (freed {ram_freed:.1f}GB)"
        
        logger.info(status)
        return (status,)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "HunyuanInstructLoader": HunyuanInstructLoader,
    "HunyuanInstructGenerate": HunyuanInstructGenerate,
    "HunyuanInstructImageEdit": HunyuanInstructImageEdit,
    "HunyuanInstructMultiFusion": HunyuanInstructMultiFusion,
    "HunyuanInstructUnload": HunyuanInstructUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanInstructLoader": "Hunyuan Instruct Loader",
    "HunyuanInstructGenerate": "Hunyuan Instruct Generate",
    "HunyuanInstructImageEdit": "Hunyuan Instruct Image Edit",
    "HunyuanInstructMultiFusion": "Hunyuan Instruct Multi-Image Fusion",
    "HunyuanInstructUnload": "Hunyuan Instruct Unload",
}

