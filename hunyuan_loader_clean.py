"""
HunyuanImage-3.0 Clean Model Loader - V2 Unified Node Support

Load models without accelerate hooks, using explicit device placement.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of model loading."""
    model: Any
    quant_type: str
    is_moveable: bool  # False for INT8 and BF16 with device_map (cannot be moved after loading)
    device: str
    dtype: torch.dtype
    load_time_seconds: float
    uses_device_map: bool = False  # True if device_map="auto" was used
    
    def __str__(self) -> str:
        moveable = "moveable" if self.is_moveable else "fixed"
        device_map_str = " (device_map)" if self.uses_device_map else ""
        return f"LoadResult({self.quant_type}, {self.device}, {moveable}{device_map_str})"


class CleanModelLoader:
    """
    Load Hunyuan models without accelerate's automatic hooks.
    
    This loader uses explicit device placement instead of device_map="auto"
    where possible. This avoids hook conflicts with other ComfyUI extensions
    and provides predictable device placement.
    
    Supported quantization types:
    - bf16: Full precision BFloat16 (uses device_map="auto" for GPU+CPU split)
    - nf4: 4-bit NormalFloat quantization (fits on single GPU)
    - int8: 8-bit integer quantization (LIMITED: cannot be moved after load)
    - gguf: GGUF quantization (placeholder for future)
    """
    
    # Modules to skip during quantization (keep in full precision)
    SKIP_MODULES = [
        # VAE components
        "vae",
        "model.vae",
        "vae.decoder",
        "vae.encoder",
        "autoencoder",
        "model.autoencoder",
        # Embedding layers
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
        # Attention projections (critical for quality)
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "attn.qkv_proj",
        "attn.out_proj",
        "self_attn",
        "cross_attn",
    ]
    
    @classmethod
    def load(
        cls,
        model_path: str,
        quant_type: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        reserve_vram_gb: float = 0.0,
    ) -> LoadResult:
        """
        Load a model with specified quantization.
        
        Args:
            model_path: Path to model directory
            quant_type: One of "bf16", "nf4", "int8", "gguf"
            device: Target device (e.g., "cuda:0")
            dtype: Compute dtype for non-quantized layers
            reserve_vram_gb: VRAM to reserve for inference (BF16 only)
            
        Returns:
            LoadResult with loaded model
        """
        import time
        start_time = time.time()
        
        quant_type = quant_type.lower()
        
        if quant_type == "bf16":
            result = cls._load_bf16(model_path, device, dtype, reserve_vram_gb)
        elif quant_type == "nf4":
            result = cls._load_nf4(model_path, device, dtype)
        elif quant_type == "int8":
            result = cls._load_int8(model_path, device, dtype)
        elif quant_type == "gguf":
            result = cls._load_gguf(model_path, device, dtype)
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}. Expected bf16, nf4, int8, or gguf")
        
        elapsed = time.time() - start_time
        result.load_time_seconds = elapsed
        
        logger.info(f"Model loaded in {elapsed:.1f}s: {result}")
        return result
    
    @classmethod
    def _load_bf16(
        cls,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        reserve_vram_gb: float = 0.0
    ) -> LoadResult:
        """
        Load BF16 model using device_map="auto" with GPU+CPU RAM split.
        
        This matches the working HunyuanImage3FullLoader implementation exactly.
        
        BF16 Hunyuan models are ~160GB and won't fit on a single GPU.
        We use device_map="auto" to automatically split across GPU and CPU RAM.
        
        CRITICAL for MoE models:
        - Reserve enough VRAM for inference (MoE routing needs activation memory)
        - Use moe_impl="eager" and moe_drop_tokens=True
        - Use torch_dtype="auto" not explicit bfloat16
        - Use integer device indices (0) not strings ("cuda:0")
        
        IMPORTANT: Models loaded with device_map cannot be easily moved,
        so is_moveable=False and block swapping is NOT supported.
        """
        logger.info(f"Loading BF16 model from {model_path}")
        logger.info("BF16 model (~160GB) will be split across GPU and CPU RAM using device_map='auto'")
        logger.warning("BF16 models with device_map CANNOT use block swapping or soft_unload")
        
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Parse device index - this is the PRIMARY GPU for inference
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        
        # Calculate max memory for PRIMARY GPU ONLY + CPU offload
        #
        # BF16 model (~160GB) is too large for any single GPU, so we use
        # device_map="auto" to split across GPU 0 + CPU RAM.
        #
        # IMPORTANT: We only use the PRIMARY GPU, matching the proven working
        # HunyuanImage3FullLoader approach. Secondary GPUs are explicitly blocked.
        # Reasons:
        #   1. Secondary GPU often drives the display (Windows DWM needs VRAM)
        #   2. Multi-GPU with device_map caused TDR crashes and OOM in testing
        #   3. CPU-offloaded layers always run on GPU 0 anyway (accelerate main_device)
        #   4. The old single-GPU loader worked reliably, even for high resolutions
        #
        # CPU-offloaded layers are served by accelerate's AlignDevicesHook which
        # temporarily copies weights to GPU 0 during forward pass, then frees them.
        # This is slower than having everything on GPU, but stable and proven.
        
        max_memory = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_memory = {}
            
            # Primary GPU: reserve VRAM for inference activations
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            
            reserve_gb = reserve_vram_gb if reserve_vram_gb > 0 else 20.0
            max_gpu_bytes = free_bytes - int(reserve_gb * 1024**3)
            
            # Minimum 4GB for model on primary GPU
            if max_gpu_bytes < int(4 * 1024**3):
                logger.warning(f"Primary GPU {device_idx}: Reserved memory ({reserve_gb:.1f}GB) leaves only {max_gpu_bytes/1024**3:.1f}GB for model")
                max_gpu_bytes = int(4 * 1024**3)
            
            max_memory[device_idx] = max_gpu_bytes
            logger.info(f"Primary GPU {device_idx} ({torch.cuda.get_device_name(device_idx)}): "
                       f"{free_gb:.1f}GB free, reserving {reserve_gb:.1f}GB for inference -> "
                       f"{max_gpu_bytes/1024**3:.1f}GB for model")
            
            # Block ALL other GPUs (prevents accelerate from using display GPU, etc.)
            for i in range(num_gpus):
                if i != device_idx:
                    max_memory[i] = 0
                    logger.info(f"Blocking GPU {i} ({torch.cuda.get_device_name(i)}) from model loading")
            
            # CPU RAM for the remainder (~80-100GB typically)
            total_gpu_budget = max_gpu_bytes / 1024**3
            model_size_gb = 160.0
            cpu_needed_gb = max(0, model_size_gb - total_gpu_budget)
            cpu_allocation_gb = max(100, int(cpu_needed_gb * 1.2))
            max_memory["cpu"] = f"{cpu_allocation_gb}GiB"
            
            logger.info(f"GPU budget: {total_gpu_budget:.1f}GB on GPU {device_idx}")
            logger.info(f"Estimated CPU RAM needed: {cpu_needed_gb:.1f}GB (allocating {cpu_allocation_gb}GiB)")
        
        # Log system RAM status before loading
        try:
            import psutil
            ram = psutil.virtual_memory()
            logger.info(f"System RAM: {ram.available/1024**3:.1f}GB available / {ram.total/1024**3:.1f}GB total ({ram.percent:.1f}% used)")
        except ImportError:
            logger.warning("psutil not available - cannot check system RAM")
        
        # Load with parameters matching the working HunyuanImage3FullLoader
        load_kwargs = dict(
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype="auto",  # Use "auto" not explicit dtype - matches working loader
            attn_implementation="sdpa",
            moe_impl="eager",  # Critical for MoE models
            moe_drop_tokens=True,  # Critical for MoE models
            offload_folder=None,  # Explicitly disable disk offload
        )
        
        logger.info("=" * 60)
        logger.info("Loading BF16 model with MoE-optimized settings...")
        if max_memory:
            for k, v in max_memory.items():
                if isinstance(v, int):
                    logger.info(f"  max_memory[{k}] = {v/1024**3:.1f} GiB")
                else:
                    logger.info(f"  max_memory[{k}] = {v}")
        logger.info("=" * 60)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
        except TypeError as exc:
            # Fallback for older transformers versions
            logger.warning(f"Falling back to legacy load args due to: {exc}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        
        # Check if accelerate used disk offload (which would explain meta tensors)
        if hasattr(model, 'hf_device_map'):
            device_map = model.hf_device_map
            disk_modules = [k for k, v in device_map.items() if v == "disk"]
            cpu_modules = [k for k, v in device_map.items() if v == "cpu"]
            cuda_modules = [k for k, v in device_map.items() if isinstance(v, int) or (isinstance(v, str) and v.startswith("cuda"))]
            
            logger.info(f"Device map: {len(cuda_modules)} on GPU, {len(cpu_modules)} on CPU, {len(disk_modules)} on disk")
            
            if disk_modules:
                logger.error("=" * 60)
                logger.error("CRITICAL: accelerate is using DISK OFFLOAD!")
                logger.error(f"  {len(disk_modules)} modules offloaded to disk")
                logger.error("This happens when max_memory budget can't fit the model in GPU+CPU RAM.")
                logger.error("This will be EXTREMELY slow and may cause meta tensor issues.")
                logger.error("")
                logger.error("Solutions:")
                logger.error("  1. Reduce reserve_vram_gb to allow more GPU space")
                logger.error("  2. Check system RAM availability (need ~100GB+ free)")
                logger.error("  3. Use a smaller model (NF4 or INT8)")
                logger.error("=" * 60)
        
        # Load tokenizer
        if hasattr(model, 'load_tokenizer'):
            model.load_tokenizer(model_path)
            logger.info("Tokenizer loaded")
        
        # Ensure VAE is on GPU and in correct dtype
        if hasattr(model, 'vae'):
            target_device = torch.device(device)
            model.vae = model.vae.to(device=target_device, dtype=torch.bfloat16)
            logger.info("VAE moved to GPU in bfloat16")
        
        # NOTE: The comprehensive distribution summary below iterates ALL params.
        # We don't need a separate biased sample check here.
        
        # Comprehensive device distribution summary (like legacy loader)
        logger.info("=" * 60)
        logger.info("MODEL DISTRIBUTION SUMMARY")
        logger.info("=" * 60)
        
        device_usage = {}
        total_size_gb = 0.0
        for name, param in model.named_parameters():
            device_str = str(param.device)
            if device_str not in device_usage:
                device_usage[device_str] = {"params": 0, "size_gb": 0.0}
            device_usage[device_str]["params"] += 1
            param_size = param.numel() * param.element_size() / 1024**3
            device_usage[device_str]["size_gb"] += param_size
            total_size_gb += param_size
        
        for device_str, stats in sorted(device_usage.items()):
            logger.info(
                "  %-12s: %5d parameters, %.2f GiB",
                device_str, stats["params"], stats["size_gb"]
            )
        
        logger.info("  %-12s: %5s parameters, %.2f GiB", "TOTAL", "", total_size_gb)
        logger.info("=" * 60)
        
        # Check distribution health
        # With device_map="auto", meta = CPU-offloaded via accelerate hooks (NORMAL).
        # accelerate keeps weights as meta tensors and installs AlignDevicesHook
        # to copy weights to the execution device on-the-fly during forward pass.
        cuda_params = device_usage.get('cuda:0', {}).get('params', 0) + device_usage.get('cuda:1', {}).get('params', 0)
        meta_params = device_usage.get('meta', {}).get('params', 0)
        meta_gb = device_usage.get('meta', {}).get('size_gb', 0)
        
        if cuda_params == 0:
            logger.error("=" * 60)
            logger.error("CRITICAL: No model parameters on GPU!")
            logger.error("The model cannot run inference without GPU-resident layers.")
            logger.error("Increase GPU memory budget or use a smaller model (NF4/INT8).")
            logger.error("=" * 60)
        elif meta_params > 0:
            logger.info(f"Note: {meta_params} params ({meta_gb:.1f} GiB) are CPU-offloaded via accelerate hooks")
            logger.info("  These will be loaded to GPU on-the-fly during forward pass (slower but functional)")
        
        # Log actual memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            free_after = (torch.cuda.get_device_properties(device_idx).total_memory - torch.cuda.memory_allocated(device_idx)) / (1024**3)
            logger.info(f"GPU memory after load: {allocated:.1f}GB allocated, {free_after:.1f}GB free")
        
        logger.info(f"BF16 model loaded with device_map='auto' (GPU+CPU RAM split)")
        
        return LoadResult(
            model=model,
            quant_type="bf16",
            is_moveable=False,  # Cannot move device_map models
            device=device,
            dtype=torch.bfloat16,
            load_time_seconds=0.0,
            uses_device_map=True
        )
    
    @classmethod
    def _load_nf4(
        cls,
        model_path: str,
        device: str,
        dtype: torch.dtype
    ) -> LoadResult:
        """
        Load NF4 quantized model.
        
        NF4 models CAN be moved after loading (unlike INT8), so we use
        direct device placement without accelerate's device_map="auto".
        
        NF4 Hunyuan is ~29GB and fits on most high-end GPUs.
        """
        logger.info(f"Loading NF4 model from {model_path}")
        
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=cls.SKIP_MODULES,
        )
        
        # Use direct device placement - NO device_map="auto"
        # This avoids accelerate's hook system and allows block swapping
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map={"": device},  # Direct placement
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            moe_impl="eager",       # Critical: use eager MoE (avoids massive pre-allocation)
            moe_drop_tokens=True,   # Critical: route to top-K experts only, not all 64
        )
        
        # Load tokenizer
        if hasattr(model, 'load_tokenizer'):
            model.load_tokenizer(model_path)
        
        # Ensure VAE is in full precision
        if hasattr(model, 'vae'):
            model.vae = model.vae.to(device=device, dtype=dtype)
            logger.info("VAE configured in full precision (bfloat16)")
        
        logger.info(f"NF4 model loaded to {device}")
        
        return LoadResult(
            model=model,
            quant_type="nf4",
            is_moveable=True,  # NF4 can be moved and block-swapped
            device=device,
            dtype=dtype,
            load_time_seconds=0.0,
            uses_device_map=False
        )
    
    @classmethod
    def _load_int8(
        cls,
        model_path: str,
        device: str,
        dtype: torch.dtype
    ) -> LoadResult:
        """
        Load INT8 quantized model.
        
        IMPORTANT: INT8 models CANNOT be moved after loading!
        The quantization state is tied to the original device.
        We must use device_map="auto" for INT8.
        
        INT8 Hunyuan is ~45GB and needs a high-VRAM GPU.
        """
        logger.info(f"Loading INT8 model from {model_path}")
        logger.warning("INT8 models cannot be moved after loading - soft_unload and block_swap will NOT work")
        
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=cls.SKIP_MODULES,
            llm_int8_enable_fp32_cpu_offload=False,  # Keep on GPU
        )
        
        # INT8 loading strategy:
        # The pre-quantized INT8 model should load directly to GPU without quantization_config
        # since weights are already INT8 on disk. Using device_map="cuda:0" is faster and 
        # avoids accelerate overhead.
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        
        # Try direct GPU loading first (faster, no accelerate overhead)
        # This matches the working GitHub INT8 Budget loader approach
        try:
            logger.info("Loading pre-quantized INT8 model directly to GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=f"cuda:{device_idx}",  # Direct GPU placement
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
                low_cpu_mem_usage=True,
                # No quantization_config needed - model is pre-quantized
            )
        except Exception as e:
            # Fallback to device_map="auto" if direct loading fails
            logger.warning(f"Direct GPU load failed ({e}), falling back to device_map=auto...")
            
            # Build multi-GPU max_memory map (INT8 model ~85GB)
            max_memory = {}
            total_gpu_budget = 0
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                    free_gb = free_bytes / 1024**3
                    
                    if i == device_idx:
                        # Primary GPU: reserve headroom for MoE inference activations
                        reserve_gb = 15.0
                        max_gpu_bytes = free_bytes - int(reserve_gb * 1024**3)
                    else:
                        # Secondary GPU: small reserve for overhead
                        reserve_gb = 2.0
                        max_gpu_bytes = free_bytes - int(reserve_gb * 1024**3)
                    
                    if max_gpu_bytes > int(2 * 1024**3):  # Only if >2GB usable
                        max_memory[i] = max_gpu_bytes
                        total_gpu_budget += max_gpu_bytes / 1024**3
                        logger.info(f"GPU {i}: {free_gb:.1f}GB free -> "
                                   f"{max_gpu_bytes/1024**3:.1f}GB for INT8 model "
                                   f"({'primary' if i == device_idx else 'secondary'})")
                    else:
                        max_memory[i] = 0
                
                max_memory["cpu"] = "50GiB"  # Allow CPU spillover
                logger.info(f"Total GPU budget for INT8: {total_gpu_budget:.1f}GB")
            else:
                max_memory = {"cpu": "100GiB"}
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                moe_impl="eager",       # Critical: use eager MoE
                moe_drop_tokens=True,   # Critical: route to top-K experts only
            )
        
        # Load tokenizer
        if hasattr(model, 'load_tokenizer'):
            model.load_tokenizer(model_path)
        
        # Ensure VAE is in full precision on correct device
        if hasattr(model, 'vae'):
            target_device = torch.device(device)
            model.vae = model.vae.to(device=target_device, dtype=dtype)
            logger.info("VAE configured in full precision (bfloat16)")
        
        # Log memory usage across all GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / 1024**3
                if allocated > 0.1:  # Only log GPUs actually in use
                    logger.info(f"INT8 model on GPU {i}: {allocated:.1f}GB allocated, {free_gb:.1f}GB free")
        
        return LoadResult(
            model=model,
            quant_type="int8",
            is_moveable=False,  # INT8 CANNOT be moved or block-swapped
            device=device,
            dtype=dtype,
            load_time_seconds=0.0,
            uses_device_map=True
        )
    
    @classmethod
    def _load_gguf(
        cls,
        model_path: str,
        device: str,
        dtype: torch.dtype
    ) -> LoadResult:
        """
        Load GGUF quantized model.
        
        PLACEHOLDER: GGUF loading not yet implemented.
        """
        raise NotImplementedError(
            "GGUF loading is not yet implemented. "
            "Please use NF4 or INT8 quantization for now."
        )


def apply_model_patches(model: Any) -> None:
    """
    Apply necessary patches to the model for generation.
    
    This applies the same patches used by the working loaders,
    but without the hook-based memory management.
    """
    from .hunyuan_shared import (
        patch_dynamic_cache_dtype,
        patch_hunyuan_generate_image,
        patch_pipeline_pre_vae_cleanup,
    )
    
    # Apply dtype patches
    patch_dynamic_cache_dtype()
    
    # Apply generation patches
    patch_hunyuan_generate_image(model)
    
    # Apply pre-VAE cleanup patch for BF16/device_map models
    # This clears KV cache before VAE decode to free VRAM
    patch_pipeline_pre_vae_cleanup(model, enabled=True)
    
    logger.debug("Applied model patches for generation")


def validate_model_placement(model: Any, expected_device: str) -> Tuple[bool, str]:
    """
    Validate that model is on expected device.
    
    Args:
        model: The loaded model
        expected_device: Expected device string
        
    Returns:
        Tuple of (is_valid, message)
    """
    expected = torch.device(expected_device)
    
    cuda_count = 0
    cpu_count = 0
    meta_count = 0
    total_count = 0
    
    for i, param in enumerate(model.parameters()):
        if i >= 100:  # Sample first 100 params
            break
        total_count += 1
        
        if param.device.type == "cuda":
            cuda_count += 1
        elif param.device.type == "cpu":
            cpu_count += 1
        elif param.device.type == "meta":
            meta_count += 1
    
    message = f"Sampled {total_count} params: {cuda_count} CUDA, {cpu_count} CPU, {meta_count} meta"
    
    if expected.type == "cuda":
        is_valid = cuda_count > 0 and meta_count == 0
    else:
        is_valid = cpu_count > 0 and meta_count == 0
    
    return is_valid, message
