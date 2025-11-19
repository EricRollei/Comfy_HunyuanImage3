"""
HunyuanImage-3.0 ComfyUI Custom Nodes - Full BF16 Loaders
Multiple loader strategies for full precision BF16 and multi-GPU configurations

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

import math
import torch
from transformers import AutoModelForCausalLM
import folder_paths
from pathlib import Path
import logging

from .hunyuan_shared import (
    HunyuanModelCache,
    ensure_model_on_device,
    patch_dynamic_cache_dtype,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HunyuanImage3FullLoader:
    """Loads FULL HunyuanImage-3.0 model in BF16 (no quantization)."""
    
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
            # Look for full model directories (exclude quantized ones)
            if item.is_dir() and not (item / "quantization_metadata.json").exists():
                # Check if it has model files
                if any(item.glob("*.safetensors")) or any(item.glob("*.bin")) or (item / "config.json").exists():
                    available.append(item.name)
        
        return available if available else ["HunyuanImage-3"]
    
    def load_model(self, model_name):
        model_path = Path(folder_paths.models_dir) / model_name
        model_path_str = str(model_path)

        cached = HunyuanModelCache.get(model_path_str)
        if cached is not None:
            self._apply_dtype_patches()
            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=False)
            return (cached,)
        
        logger.info(f"Loading FULL BF16 model: {model_name}")
        logger.info("This uses ~80GB VRAM but avoids quantization complexity")
        logger.info("Loading model (this may take 30-60 seconds)...")

        model = None
        try:
            load_kwargs = dict(
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
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
                    device_map={'': 0},
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
            
            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()

            logger.info("Verifying device placement...")
            ensure_model_on_device(model, torch.device("cuda:0"), skip_quantized_params=False)

            HunyuanModelCache.store(model_path_str, model)
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"✓ Model loaded - VRAM Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            logger.info("✓ Model ready for generation")
            return (model,)
        
        except Exception as e:
            logger.error("Failed to load model: %s", e)
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


NODE_CLASS_MAPPINGS = {
    "HunyuanImage3FullLoader": HunyuanImage3FullLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage3FullLoader": "Hunyuan 3 Loader (Full BF16)",
}


class HunyuanImage3FullGPULoader:
    """Loads the BF16 model onto a single GPU with an explicit device map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls._get_available_models(),),
                "max_gpu_fraction": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 0.99, "step": 0.01}),
                "reserve_memory_gb": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 32.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    @classmethod
    def _get_available_models(cls):
        return HunyuanImage3FullLoader._get_available_models()

    def load_model(self, model_name, max_gpu_fraction, reserve_memory_gb):
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model_path = Path(folder_paths.models_dir) / model_name
        model_path_str = str(model_path)

        cached = HunyuanModelCache.get(model_path_str)
        if cached is not None:
            self._apply_dtype_patches()
            ensure_model_on_device(cached, target_device, skip_quantized_params=False)
            return (cached,)

        logger.info("Loading FULL BF16 model (single GPU forcing): %s", model_name)
        if target_device.type != "cuda":
            logger.warning("CUDA not available; falling back to CPU load")

        max_gpu_fraction = float(max_gpu_fraction)
        max_gpu_fraction = min(max(max_gpu_fraction, 0.5), 0.99)
        reserve_memory_gb = float(reserve_memory_gb)
        reserve_memory_gb = min(max(reserve_memory_gb, 2.0), 32.0)

        max_memory = None
        device_map = "auto"
        if target_device.type == "cuda":
            cuda_index = target_device.index or 0
            device_key = f"cuda:{cuda_index}"
            total_mem = torch.cuda.get_device_properties(target_device).total_memory
            
            # Check currently free memory (accounts for Chrome, VSCode, etc.)
            free_bytes, _ = torch.cuda.mem_get_info(cuda_index)
            currently_used = (total_mem - free_bytes) / 1024**3
            
            reserve_bytes = int(reserve_memory_gb * (1024 ** 3))
            
            # Calculate usable memory from what's actually free
            # Option 1: Use fraction of free memory
            fraction_of_free = int(math.floor(free_bytes * max_gpu_fraction))
            # Option 2: Use free memory minus reserve with 2GB safety margin
            free_minus_reserve = free_bytes - reserve_bytes - int(2.0 * (1024 ** 3))
            
            usable_mem = min(fraction_of_free, free_minus_reserve)
            
            # Safety check - ensure at least 40% of total memory
            min_allowed = int(total_mem * 0.4)
            if usable_mem < min_allowed:
                usable_mem = min_allowed
                logger.warning(
                    "Requested reserve (%.2f GiB) too large given current usage; clamping to use at least 40%% of GPU memory",
                    reserve_bytes / 1024**3
                )
            
            device_map = {"": cuda_index}
            max_memory = {device_key: usable_mem, "cpu": "10GiB"}
            logger.info(
                "Requesting single-GPU load on %s: %.2f GiB total, %.2f GiB currently used, %.2f GiB available for model (%.2f GiB reserved for inference)",
                device_key,
                total_mem / 1024**3,
                currently_used,
                usable_mem / 1024**3,
                reserve_bytes / 1024**3,
            )

        model = None
        try:
            load_kwargs = dict(
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
                low_cpu_mem_usage=True,
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    **load_kwargs,
                )
            except TypeError as exc:
                logger.warning("Falling back to legacy load args due to: %s", exc)
                fallback_map = {"":
0} if target_device.type == "cuda" else {"":
"cpu"}
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
                    device_map=fallback_map,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()

            logger.info("Verifying device placement...")
            ensure_model_on_device(model, target_device, skip_quantized_params=False)

            HunyuanModelCache.store(model_path_str, model, keep_in_cache)

            if torch.cuda.is_available():
                device_index = target_device.index or 0
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                allocated = (total_bytes - free_bytes) / 1024**3
                logger.info(
                    "✓ Model loaded - VRAM in use %.2fGB / %.2fGB (free %.2fGB)",
                    allocated,
                    total_bytes / 1024**3,
                    free_bytes / 1024**3,
                )

            logger.info("✓ Model ready for generation")
            return (model,)
        
        except Exception as e:
            logger.error("Failed to load model: %s", e)
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


NODE_CLASS_MAPPINGS.update({
    "HunyuanImage3FullGPULoader": HunyuanImage3FullGPULoader,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "HunyuanImage3FullGPULoader": "Hunyuan 3 Loader (Full BF16 GPU)",
})


class HunyuanImage3DualGPULoader:
    """Loads the BF16 model across multiple GPUs with automatic distribution."""

    @classmethod
    def INPUT_TYPES(cls):
        import os
        # Temporarily clear CUDA_VISIBLE_DEVICES to see all GPUs
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if original_cuda_visible is not None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            if torch.cuda.is_available():
                torch.cuda.init()
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Restore original setting
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        
        gpu_info = []
        if num_gpus > 0:
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        
        return {
            "required": {
                "model_name": (HunyuanImage3FullLoader._get_available_models(),),
                "primary_gpu": ("INT", {"default": 0, "min": 0, "max": max(0, num_gpus - 1)}),
                "reserve_memory_gb": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 32.0, "step": 0.5}),
                "keep_in_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "info": ("STRING", {"default": "\n".join(gpu_info) if gpu_info else "No CUDA GPUs detected", "multiline": True}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    @classmethod
    def _get_available_models(cls):
        return HunyuanImage3FullLoader._get_available_models()

    def load_model(self, model_name, primary_gpu, reserve_memory_gb, keep_in_cache, info=None):
        import os
        
        model_path = Path(folder_paths.models_dir) / model_name
        model_path_str = str(model_path)

        cached = HunyuanModelCache.get(model_path_str)
        if cached is not None:
            logger.info("Using cached model (already loaded)")
            self._apply_dtype_patches()
            # Don't re-enforce device placement for multi-GPU - model already distributed
            return (cached,)

        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU loader requires CUDA GPUs")

        # Check if CUDA_VISIBLE_DEVICES is limiting GPU visibility
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None:
            logger.warning("CUDA_VISIBLE_DEVICES is set to: %s", cuda_visible)
            logger.warning("This may limit GPU detection. Temporarily clearing for multi-GPU load...")
            # Temporarily clear to see all GPUs
            original_cuda_visible = cuda_visible
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            # Force re-initialization
            torch.cuda.init()

        num_gpus = torch.cuda.device_count()
        logger.info("Detected %d CUDA device(s)", num_gpus)
        
        # Log all available GPUs
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info("  GPU %d: %s (Compute %d.%d, %.2f GiB)", 
                       i, props.name, props.major, props.minor, props.total_memory / 1024**3)
        
        if num_gpus < 2:
            logger.warning("Multi-GPU loader works best with 2+ GPUs, but only %d detected", num_gpus)
            logger.warning("Falling back to single-GPU mode on cuda:0")
            # Restore original setting if we changed it
            if cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            # Don't raise error - fall through to use single GPU

        primary_gpu = int(primary_gpu)
        if primary_gpu >= num_gpus:
            primary_gpu = 0
            logger.warning("Invalid primary_gpu index, defaulting to GPU 0")

        reserve_memory_gb = float(reserve_memory_gb)
        reserve_memory_gb = min(max(reserve_memory_gb, 2.0), 32.0)

        # Build max_memory dict for all available GPUs
        max_memory = {}
        total_available = 0
        
        logger.info("=" * 60)
        logger.info("MULTI-GPU CONFIGURATION")
        logger.info("=" * 60)
        
        # Handle single GPU fallback
        gpus_to_use = max(num_gpus, 1)
        
        for i in range(gpus_to_use):
            try:
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory
                
                # Check currently free memory (accounts for Chrome, VSCode, etc.)
                free_bytes, _ = torch.cuda.mem_get_info(i)
                currently_used = (total_mem - free_bytes) / 1024**3
                
                # Reserve memory on primary GPU (where inference happens)
                if i == primary_gpu:
                    reserve_bytes = int(reserve_memory_gb * (1024 ** 3))
                    # Use free memory minus reserve, with 2GB safety margin
                    usable_mem = free_bytes - reserve_bytes - int(2.0 * (1024 ** 3))
                    usable_mem = max(usable_mem, int(total_mem * 0.3))  # Use at least 30% of GPU
                    logger.info(
                        "GPU %d (PRIMARY): %s - %.2f GiB total, %.2f GiB currently used, %.2f GiB available for model (%.2f GiB reserved for inference)",
                        i, props.name, total_mem / 1024**3, currently_used, usable_mem / 1024**3, reserve_bytes / 1024**3
                    )
                else:
                    # Use free memory with 2GB safety margin on secondary GPUs
                    usable_mem = free_bytes - int(2.0 * (1024 ** 3))
                    usable_mem = max(usable_mem, int(total_mem * 0.3))  # Use at least 30% of GPU
                    logger.info(
                        "GPU %d: %s - %.2f GiB total, %.2f GiB currently used, %.2f GiB available for model",
                        i, props.name, total_mem / 1024**3, currently_used, usable_mem / 1024**3
                    )
                
                max_memory[f"cuda:{i}"] = usable_mem
                total_available += usable_mem
            except Exception as e:
                logger.warning("Could not access GPU %d: %s", i, e)
                # If we can't access GPU 1, provide a reasonable default for dual-GPU systems
                if i == 1 and num_gpus == 1:
                    # User reports having a second GPU - use conservative estimate
                    estimated_mem = int(8.0 * (1024 ** 3))  # 8GB conservative
                    max_memory[f"cuda:{i}"] = estimated_mem
                    total_available += estimated_mem
                    logger.info("GPU 1: Using estimated 8.0 GiB (detection failed, assuming secondary GPU exists)")
        
        max_memory["cpu"] = "20GiB"  # Fallback to CPU if needed
        
        logger.info("=" * 60)
        logger.info("Total VRAM available for model: %.2f GiB across %d GPUs", total_available / 1024**3, num_gpus)
        logger.info("Loading FULL BF16 model: %s", model_name)
        logger.info("This will distribute layers across available GPUs automatically")
        logger.info("=" * 60)

        model = None
        try:
            load_kwargs = dict(
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
                low_cpu_mem_usage=True,
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
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            self._apply_dtype_patches()

            # Log device distribution
            logger.info("=" * 60)
            logger.info("MODEL DISTRIBUTION SUMMARY")
            logger.info("=" * 60)
            
            device_usage = {}
            for name, param in model.named_parameters():
                device_str = str(param.device)
                if device_str not in device_usage:
                    device_usage[device_str] = {"params": 0, "size_gb": 0.0}
                device_usage[device_str]["params"] += 1
                device_usage[device_str]["size_gb"] += param.numel() * param.element_size() / 1024**3
            
            for device_str, stats in sorted(device_usage.items()):
                logger.info(
                    "  %-12s: %5d parameters, %.2f GiB",
                    device_str, stats["params"], stats["size_gb"]
                )
            
            logger.info("=" * 60)

            # Log actual VRAM usage
            if torch.cuda.is_available():
                total_allocated = 0
                total_capacity = 0
                for i in range(num_gpus):
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                    allocated = (total_bytes - free_bytes) / 1024**3
                    total_allocated += allocated
                    total_capacity += total_bytes / 1024**3
                    marker = " (PRIMARY)" if i == primary_gpu else ""
                    logger.info(
                        "GPU %d%s: %.2f GiB used / %.2f GiB total (%.2f GiB free)",
                        i, marker, allocated, total_bytes / 1024**3, free_bytes / 1024**3
                    )
                logger.info("=" * 60)
                logger.info("TOTAL: %.2f GiB used / %.2f GiB total", total_allocated, total_capacity)

            HunyuanModelCache.store(model_path_str, model, keep_in_cache)
            
            logger.info("=" * 60)
            logger.info("✓ Model loaded and ready for generation")
            logger.info("✓ Inference will run on GPU %d (primary)", primary_gpu)
            logger.info("=" * 60)
            return (model,)
        
        except Exception as e:
            logger.error("Failed to load model: %s", e)
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


NODE_CLASS_MAPPINGS.update({
    "HunyuanImage3DualGPULoader": HunyuanImage3DualGPULoader,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "HunyuanImage3DualGPULoader": "Hunyuan 3 Loader (Multi-GPU BF16)",
})


class HunyuanImage3GPUInfo:
    """Diagnostic node that displays GPU information and CUDA environment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gpu_info",)
    FUNCTION = "get_info"
    CATEGORY = "HunyuanImage3"
    OUTPUT_NODE = True

    def get_info(self):
        import os
        info_lines = []
        
        info_lines.append("=" * 70)
        info_lines.append("CUDA ENVIRONMENT")
        info_lines.append("=" * 70)
        
        # Check environment variables
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        info_lines.append(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        if not torch.cuda.is_available():
            info_lines.append("\n⚠️  CUDA is not available!")
            info_lines.append("PyTorch was not built with CUDA support or no NVIDIA drivers found.")
            result = "\n".join(info_lines)
            logger.info(result)
            return (result,)
        
        info_lines.append(f"PyTorch CUDA Version: {torch.version.cuda}")
        info_lines.append(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        num_gpus = torch.cuda.device_count()
        info_lines.append(f"\nDetected GPUs: {num_gpus}")
        info_lines.append("=" * 70)
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            
            info_lines.append(f"\nGPU {i}: {props.name}")
            info_lines.append(f"  Compute Capability: {props.major}.{props.minor}")
            info_lines.append(f"  Total Memory: {total / 1024**3:.2f} GiB")
            info_lines.append(f"  Used Memory: {used / 1024**3:.2f} GiB")
            info_lines.append(f"  Free Memory: {free / 1024**3:.2f} GiB")
            info_lines.append(f"  Multi-Processor Count: {props.multi_processor_count}")
        
        info_lines.append("=" * 70)
        
        result = "\n".join(info_lines)
        logger.info("\n" + result)
        return (result,)


NODE_CLASS_MAPPINGS.update({
    "HunyuanImage3GPUInfo": HunyuanImage3GPUInfo,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "HunyuanImage3GPUInfo": "Hunyuan 3 GPU Info",
})


class HunyuanImage3SingleGPU88GB:
    """Optimized loader for single 88-96GB GPU (e.g., RTX 6000 Blackwell)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (HunyuanImage3FullLoader._get_available_models(),),
                "reserve_memory_gb": ("FLOAT", {"default": 14.0, "min": 8.0, "max": 24.0, "step": 0.5}),
                "keep_in_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage3"

    def load_model(self, model_name, reserve_memory_gb, keep_in_cache):
        model_path = Path(folder_paths.models_dir) / model_name
        model_path_str = str(model_path)

        cached = HunyuanModelCache.get(model_path_str)
        if cached is not None:
            logger.info("Using cached model")
            patch_dynamic_cache_dtype()
            ensure_model_on_device(cached, torch.device("cuda:0"), skip_quantized_params=False)
            return (cached,)

        if not torch.cuda.is_available():
            raise RuntimeError("This loader requires CUDA")

        target_device = torch.device("cuda:0")
        reserve_memory_gb = float(reserve_memory_gb)
        
        logger.info("=" * 60)
        logger.info("OPTIMIZED 88GB GPU LOADER")
        logger.info("=" * 60)

        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory
        free_bytes, _ = torch.cuda.mem_get_info(0)
        currently_used = (total_mem - free_bytes) / 1024**3

        reserve_bytes = int(reserve_memory_gb * (1024 ** 3))
        usable_mem = free_bytes - reserve_bytes - int(3.0 * (1024 ** 3))  # 3GB extra safety

        logger.info("GPU 0: %s", props.name)
        logger.info("  Total: %.2f GiB", total_mem / 1024**3)
        logger.info("  Currently used: %.2f GiB", currently_used)
        logger.info("  Available for model: %.2f GiB", usable_mem / 1024**3)
        logger.info("  Reserved for inference: %.2f GiB", reserve_bytes / 1024**3)
        logger.info("=" * 60)

        model = None
        try:
            logger.info("Loading FULL BF16 model: %s", model_name)
            
            load_kwargs = dict(
                device_map={"": 0},
                max_memory={"cuda:0": usable_mem, "cpu": "10GiB"},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                moe_impl="eager",
                moe_drop_tokens=True,
                low_cpu_mem_usage=True,
            )

            model = AutoModelForCausalLM.from_pretrained(model_path_str, **load_kwargs)
            
            logger.info("Loading tokenizer...")
            model.load_tokenizer(model_path_str)

            logger.info("Applying dtype compatibility patches...")
            patch_dynamic_cache_dtype()

            logger.info("Verifying device placement...")
            ensure_model_on_device(model, target_device, skip_quantized_params=False)

            HunyuanModelCache.store(model_path_str, model, keep_in_cache)

            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            allocated = (total_bytes - free_bytes) / 1024**3
            logger.info("=" * 60)
            logger.info("✓ Model loaded successfully")
            logger.info("  VRAM used: %.2f GiB / %.2f GiB", allocated, total_bytes / 1024**3)
            logger.info("  VRAM free: %.2f GiB", free_bytes / 1024**3)
            logger.info("=" * 60)

            return (model,)

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            logger.info("Cleaning up VRAM...")
            if model is not None:
                try:
                    del model
                except:
                    pass
            HunyuanModelCache.clear()
            raise


NODE_CLASS_MAPPINGS.update({
    "HunyuanImage3SingleGPU88GB": HunyuanImage3SingleGPU88GB,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "HunyuanImage3SingleGPU88GB": "Hunyuan 3 Loader (88GB GPU Optimized)",
})