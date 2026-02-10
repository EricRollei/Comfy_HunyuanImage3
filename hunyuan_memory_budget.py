"""
HunyuanImage-3.0 Memory Budget Calculator - V2 Unified Node Support

Calculates VRAM availability and allocation for clean, hook-free model loading.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

import torch

logger = logging.getLogger(__name__)


@dataclass
class VRAMReport:
    """VRAM status report."""
    total_bytes: int
    free_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    device_name: str
    device_index: int
    
    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)
    
    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)
    
    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / (1024**3)
    
    @property
    def reserved_gb(self) -> float:
        return self.reserved_bytes / (1024**3)
    
    def __str__(self) -> str:
        return (
            f"VRAM [{self.device_name}]: "
            f"{self.allocated_gb:.1f}GB allocated, "
            f"{self.free_gb:.1f}GB free, "
            f"{self.total_gb:.1f}GB total"
        )


@dataclass 
class ModelSizeEstimate:
    """Estimated model memory requirements."""
    quant_type: str
    weights_gb: float
    vae_gb: float
    text_encoder_gb: float
    total_gb: float
    num_transformer_blocks: int
    bytes_per_block: int
    
    def __str__(self) -> str:
        return (
            f"Model [{self.quant_type}]: "
            f"{self.total_gb:.1f}GB total "
            f"(weights: {self.weights_gb:.1f}GB, VAE: {self.vae_gb:.1f}GB, "
            f"text_enc: {self.text_encoder_gb:.1f}GB)"
        )


class MemoryBudget:
    """
    Calculate VRAM availability and allocation for Hunyuan models.
    
    This class provides utilities for:
    - Querying current VRAM state
    - Estimating model memory requirements
    - Calculating inference VRAM needs based on resolution
    - Determining optimal block swap configuration
    """
    
    # Model size estimates (empirically measured)
    # HunyuanImage-3.0 has 32 hidden layers (num_hidden_layers=32 in config.json)
    # Each layer is a MoE transformer block with 64 experts
    MODEL_SIZES = {
        "bf16": {
            "weights_gb": 80.0,  # Full BF16 transformer weights
            "vae_gb": 3.0,
            "text_encoder_gb": 2.0,
            "num_blocks": 32,
        },
        "int8": {
            "weights_gb": 40.0,  # INT8 quantized
            "vae_gb": 3.0,
            "text_encoder_gb": 2.0,
            "num_blocks": 32,
        },
        "nf4": {
            "weights_gb": 24.0,  # NF4 quantized (double quant)
            "vae_gb": 3.0,
            "text_encoder_gb": 2.0,
            "num_blocks": 32,
        },
        "gguf": {
            "weights_gb": 20.0,  # Placeholder - depends on quant level
            "vae_gb": 3.0,
            "text_encoder_gb": 2.0,
            "num_blocks": 32,
        },
    }
    
    # Inference VRAM estimates by resolution (megapixels -> GB)
    # Based on empirical measurements
    INFERENCE_VRAM = {
        0.5: 6.0,   # 0.5MP
        1.0: 12.0,  # 1MP
        1.5: 17.0,  # 1.5MP
        2.0: 24.0,  # 2MP
        2.5: 32.0,  # 2.5MP
        3.0: 45.0,  # 3MP
    }
    
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize memory budget calculator.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        """
        self.device = device
        self.device_index = int(device.split(":")[-1]) if ":" in device else 0
        
    def get_vram_report(self) -> VRAMReport:
        """Get current VRAM status."""
        if not torch.cuda.is_available():
            return VRAMReport(
                total_bytes=0,
                free_bytes=0,
                allocated_bytes=0,
                reserved_bytes=0,
                device_name="No CUDA",
                device_index=0
            )
        
        try:
            free, total = torch.cuda.mem_get_info(self.device_index)
            allocated = torch.cuda.memory_allocated(self.device_index)
            reserved = torch.cuda.memory_reserved(self.device_index)
            name = torch.cuda.get_device_name(self.device_index)
            
            return VRAMReport(
                total_bytes=total,
                free_bytes=free,
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                device_name=name,
                device_index=self.device_index
            )
        except Exception as e:
            logger.error(f"Failed to get VRAM info: {e}")
            return VRAMReport(
                total_bytes=0,
                free_bytes=0,
                allocated_bytes=0,
                reserved_bytes=0,
                device_name="Error",
                device_index=self.device_index
            )
    
    def get_total_vram_gb(self) -> float:
        """Get total VRAM in GB."""
        report = self.get_vram_report()
        return report.total_gb
    
    def get_free_vram_gb(self) -> float:
        """Get currently free VRAM in GB."""
        report = self.get_vram_report()
        return report.free_gb
    
    def estimate_model_size(self, quant_type: str) -> ModelSizeEstimate:
        """
        Estimate model memory requirements.
        
        Args:
            quant_type: One of "bf16", "int8", "nf4", "gguf"
            
        Returns:
            ModelSizeEstimate with memory breakdown
        """
        quant_type = quant_type.lower()
        if quant_type not in self.MODEL_SIZES:
            logger.warning(f"Unknown quant_type '{quant_type}', defaulting to nf4 estimate")
            quant_type = "nf4"
        
        sizes = self.MODEL_SIZES[quant_type]
        weights = sizes["weights_gb"]
        vae = sizes["vae_gb"]
        text_enc = sizes["text_encoder_gb"]
        num_blocks = sizes["num_blocks"]
        
        total = weights + vae + text_enc
        bytes_per_block = int((weights * 1024**3) / num_blocks)
        
        return ModelSizeEstimate(
            quant_type=quant_type,
            weights_gb=weights,
            vae_gb=vae,
            text_encoder_gb=text_enc,
            total_gb=total,
            num_transformer_blocks=num_blocks,
            bytes_per_block=bytes_per_block
        )
    
    def estimate_inference_vram(self, width: int, height: int) -> float:
        """
        Estimate VRAM required for inference at given resolution.
        
        Based on empirical measurements. Attention has O(n^2) complexity
        so VRAM scales roughly with megapixels^1.4.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Estimated VRAM in GB
        """
        megapixels = (width * height) / 1_000_000
        
        # Use formula: 12 * MP^1.4
        # This gives: 1MP->12GB, 2MP->24GB, 3MP->42GB
        base_vram = 12.0
        estimated = base_vram * (megapixels ** 1.4)
        
        logger.debug(f"Resolution {width}x{height} ({megapixels:.2f}MP) -> {estimated:.1f}GB estimated")
        return estimated
    
    def calculate_blocks_to_swap(
        self,
        quant_type: str,
        width: int,
        height: int,
        vae_on_gpu: bool = True,
        safety_margin_gb: float = 2.0
    ) -> Tuple[int, str]:
        """
        Calculate how many transformer blocks need to be swapped to CPU.
        
        Args:
            quant_type: Model quantization type
            width: Target image width
            height: Target image height
            vae_on_gpu: Whether VAE will be on GPU during inference
            safety_margin_gb: Extra VRAM headroom to maintain
            
        Returns:
            Tuple of (blocks_to_swap, reason_string)
        """
        vram = self.get_vram_report()
        model_size = self.estimate_model_size(quant_type)
        inference_vram = self.estimate_inference_vram(width, height)
        
        # Calculate available VRAM for model
        available = vram.free_gb - safety_margin_gb
        
        # Calculate model footprint
        model_footprint = model_size.weights_gb
        if vae_on_gpu:
            model_footprint += model_size.vae_gb
        model_footprint += model_size.text_encoder_gb
        
        # Total needed
        total_needed = model_footprint + inference_vram
        
        logger.info(f"VRAM Budget: {available:.1f}GB available, {total_needed:.1f}GB needed")
        logger.info(f"  Model: {model_footprint:.1f}GB, Inference: {inference_vram:.1f}GB")
        
        # If we fit entirely, no swap needed
        if total_needed <= available:
            return 0, f"Fits entirely ({total_needed:.1f}GB needed, {available:.1f}GB available)"
        
        # Calculate how much we need to offload
        deficit_gb = total_needed - available
        bytes_per_block = model_size.bytes_per_block
        blocks_needed = int((deficit_gb * 1024**3) / bytes_per_block) + 1
        
        # Clamp to valid range
        max_blocks = model_size.num_transformer_blocks
        blocks_needed = min(blocks_needed, max_blocks)
        
        reason = (
            f"Need to swap {blocks_needed} blocks "
            f"(deficit: {deficit_gb:.1f}GB, {bytes_per_block/1024**3:.2f}GB/block)"
        )
        
        return blocks_needed, reason
    
    def can_fit_entirely(
        self,
        quant_type: str,
        width: int,
        height: int,
        safety_margin_gb: float = 2.0
    ) -> bool:
        """Check if model + inference fits entirely on GPU."""
        blocks_to_swap, _ = self.calculate_blocks_to_swap(
            quant_type, width, height, 
            vae_on_gpu=True, 
            safety_margin_gb=safety_margin_gb
        )
        return blocks_to_swap == 0
    
    def get_optimal_config(
        self,
        quant_type: str,
        width: int,
        height: int,
        downstream_reserve_gb: float = 0.0
    ) -> Dict:
        """
        Get optimal memory configuration for a generation.
        
        Args:
            quant_type: Model quantization type
            width: Target image width  
            height: Target image height
            downstream_reserve_gb: VRAM to reserve for downstream nodes
            
        Returns:
            Dict with configuration recommendations
        """
        vram = self.get_vram_report()
        model_size = self.estimate_model_size(quant_type)
        inference_vram = self.estimate_inference_vram(width, height)
        
        # Calculate with safety margin + downstream reserve
        safety = 2.0 + downstream_reserve_gb
        blocks_to_swap, reason = self.calculate_blocks_to_swap(
            quant_type, width, height,
            vae_on_gpu=True,
            safety_margin_gb=safety
        )
        
        # Determine VAE placement
        # If we're already swapping a lot, consider keeping VAE on CPU too
        vae_placement = "always_gpu"
        if blocks_to_swap > model_size.num_transformer_blocks // 2:
            vae_placement = "managed"  # Move VAE to GPU only for decode
        
        config = {
            "blocks_to_swap": blocks_to_swap,
            "vae_placement": vae_placement,
            "reason": reason,
            "vram_total_gb": vram.total_gb,
            "vram_free_gb": vram.free_gb,
            "model_size_gb": model_size.total_gb,
            "inference_vram_gb": inference_vram,
            "can_fit_entirely": blocks_to_swap == 0,
        }
        
        return config


def log_vram_status(device: str = "cuda:0", prefix: str = ""):
    """Utility function to log current VRAM status."""
    budget = MemoryBudget(device)
    report = budget.get_vram_report()
    if prefix:
        logger.info(f"{prefix}: {report}")
    else:
        logger.info(str(report))
    return report
