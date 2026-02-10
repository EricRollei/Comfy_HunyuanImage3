"""
HunyuanImage-3.0 Memory Manager
Uses accelerate's native device_map and max_memory for VRAM management.

This replaces the broken soft_unload/restore approach for INT8 models.

Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
import torch
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_available_vram_gb(device_index: int = 0) -> Tuple[float, float]:
    """
    Get available and total VRAM in GB.
    
    Returns:
        Tuple of (free_gb, total_gb)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return (free_bytes / 1024**3, total_bytes / 1024**3)


def compute_model_max_memory(
    total_vram_gb: float,
    downstream_reserve_gb: float = 15.0,
    system_overhead_gb: float = 2.0,
    cpu_memory_gb: float = 100.0,
) -> Dict:
    """
    Compute max_memory dict for accelerate's infer_auto_device_map.
    
    Args:
        total_vram_gb: Total GPU VRAM in GB
        downstream_reserve_gb: VRAM to reserve for downstream models (VAE decode, 
                               face detailer, upscaler, etc.)
        system_overhead_gb: VRAM for CUDA overhead, fragmentation
        cpu_memory_gb: Max CPU RAM for overflow layers
        
    Returns:
        Dict suitable for max_memory parameter: {0: bytes, "cpu": "XGiB"}
    """
    # Calculate how much VRAM the model can use
    model_vram_gb = total_vram_gb - downstream_reserve_gb - system_overhead_gb
    
    # Ensure minimum viable allocation (at least 40GB for INT8 to be useful)
    model_vram_gb = max(model_vram_gb, 40.0)
    
    # Convert to bytes for GPU, string for CPU
    model_vram_bytes = int(model_vram_gb * 1024**3)
    
    max_memory = {
        0: model_vram_bytes,
        "cpu": f"{int(cpu_memory_gb)}GiB"
    }
    
    logger.info(f"Memory budget: {model_vram_gb:.1f}GB for model on GPU, "
                f"{downstream_reserve_gb:.1f}GB reserved for downstream, "
                f"{cpu_memory_gb:.0f}GB CPU overflow")
    
    return max_memory


def compute_device_map_with_reservation(
    model_or_config,
    max_memory: Dict,
    no_split_modules: list = None,
) -> Dict:
    """
    Compute optimal device map using accelerate's infer_auto_device_map.
    
    This places as many layers on GPU as fit within max_memory,
    with overflow going to CPU. Accelerate handles JIT loading during forward.
    
    Args:
        model_or_config: Model instance or config for empty model creation
        max_memory: Dict from compute_model_max_memory()
        no_split_modules: Module classes that shouldn't be split across devices
        
    Returns:
        device_map dict for dispatch_model or from_pretrained
    """
    from accelerate import infer_auto_device_map
    
    if no_split_modules is None:
        # Default: don't split transformer layers
        no_split_modules = ["HunyuanImage3DecoderLayer"]
    
    device_map = infer_auto_device_map(
        model_or_config,
        max_memory=max_memory,
        no_split_module_classes=no_split_modules,
        dtype=torch.bfloat16,  # Use bfloat16 for size estimation
    )
    
    # Log the distribution
    gpu_modules = [k for k, v in device_map.items() if v == 0]
    cpu_modules = [k for k, v in device_map.items() if v == "cpu"]
    
    logger.info(f"Device map: {len(gpu_modules)} modules on GPU, {len(cpu_modules)} on CPU")
    
    if cpu_modules:
        # Show which layers went to CPU
        layer_nums = []
        other_cpu = []
        for m in cpu_modules:
            if "layers." in m:
                try:
                    layer_num = int(m.split("layers.")[1].split(".")[0])
                    layer_nums.append(layer_num)
                except:
                    other_cpu.append(m)
            else:
                other_cpu.append(m)
        
        if layer_nums:
            layer_nums.sort()
            logger.info(f"  Transformer layers on CPU: {min(layer_nums)}-{max(layer_nums)} "
                       f"({len(layer_nums)} layers)")
        if other_cpu:
            logger.info(f"  Other modules on CPU: {other_cpu[:5]}{'...' if len(other_cpu) > 5 else ''}")
    
    return device_map


def estimate_inference_vram_gb(megapixels: float) -> float:
    """
    Estimate additional VRAM needed for inference at a given resolution.
    
    Based on empirical measurements:
    - 1MP: ~12GB overhead
    - 2MP: ~22GB overhead  
    - 3MP: ~45GB overhead
    
    Args:
        megapixels: Image resolution in megapixels
        
    Returns:
        Estimated VRAM needed in GB
    """
    # Empirical formula based on attention's O(n²) complexity
    base_vram = 12.0
    return base_vram * (megapixels ** 1.4)


def get_recommended_reserve_gb(
    target_resolution_mp: float = 1.0,
    include_vae_decode: bool = True,
    include_face_detailer: bool = True,
    include_upscaler: bool = False,
) -> float:
    """
    Get recommended VRAM reservation based on planned downstream operations.
    
    Args:
        target_resolution_mp: Target image resolution in megapixels
        include_vae_decode: Reserve for VAE decode (~3GB)
        include_face_detailer: Reserve for face detection/SDXL (~6GB)
        include_upscaler: Reserve for upscaler like SeedVR2 (~8GB)
        
    Returns:
        Recommended reservation in GB
    """
    reserve = 2.0  # Base overhead
    
    # Inference overhead scales with resolution
    reserve += estimate_inference_vram_gb(target_resolution_mp) * 0.3  # 30% of estimate
    
    if include_vae_decode:
        reserve += 3.0
    
    if include_face_detailer:
        reserve += 6.0
        
    if include_upscaler:
        reserve += 8.0
    
    return reserve


def get_vram_status() -> Dict:
    """
    Get current VRAM status for diagnostics.
    
    Returns:
        Dict with allocated, reserved, free, and total VRAM in GB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free, total = torch.cuda.mem_get_info(0)
    
    return {
        "available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free / 1024**3,
        "total_gb": total / 1024**3,
    }


def log_vram_status(prefix: str = "") -> None:
    """Log current VRAM status."""
    status = get_vram_status()
    if not status["available"]:
        logger.info(f"{prefix}CUDA not available")
        return
    
    logger.info(f"{prefix}VRAM: {status['allocated_gb']:.2f}GB allocated, "
                f"{status['free_gb']:.2f}GB free / {status['total_gb']:.2f}GB total")
