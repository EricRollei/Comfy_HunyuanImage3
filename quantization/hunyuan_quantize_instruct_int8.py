"""
HunyuanImage-3.0-Instruct INT8 Quantization Script
Quantize the full BF16 Instruct model to INT8 format (~95-100GB)

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

Note: HunyuanImage-3.0-Instruct model is subject to Tencent's Apache 2.0 license.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HunyuanInstructQuantizerINT8:
    """
    Handles quantization of HunyuanImage-3.0-Instruct model to INT8 format.
    
    INT8 provides significantly better quality than NF4 with ~2x memory usage.
    This class provides a clean interface for quantizing the Instruct model and
    saving it in a format optimized for fast loading on GPU.
    
    Key differences from base model:
    - Includes vision_model and vision_aligner (kept at BF16 for image understanding)
    - Same overall architecture as base model
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        int8_threshold: float = 6.0,
        device_map: str = "auto"
    ):
        """
        Initialize the quantizer.
        
        Args:
            model_path: Path to the source model (or HuggingFace model ID)
            output_path: Path where quantized model will be saved
            compute_dtype: Computation dtype (bfloat16 recommended for quality)
            int8_threshold: Threshold for outlier detection (default: 6.0)
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
        """
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.compute_dtype = compute_dtype
        self.int8_threshold = int8_threshold
        self.device_map = device_map
        
        # Validate paths
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Instruct INT8 quantizer:")
        logger.info(f"  Source: {model_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Compute dtype: {compute_dtype}")
        logger.info(f"  INT8 threshold: {int8_threshold}")
    
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create the BitsAndBytes quantization configuration for INT8.
        
        Returns:
            BitsAndBytesConfig object configured for INT8 quantization
        """
        # Modules to keep at BF16 for quality
        # Includes all base model modules PLUS Instruct-specific modules
        skip_modules = [
            # === VAE (critical for image quality) ===
            "vae",
            "model.vae",
            "vae.decoder",
            "vae.encoder",
            "autoencoder",
            "model.autoencoder",
            "autoencoder.decoder",
            "autoencoder.encoder",
            
            # === Vision Model (Instruct - for image understanding) ===
            "vision_model",
            "model.vision_model",
            "vision_aligner",
            "model.vision_aligner",
            
            # === Diffusion components ===
            "patch_embed",
            "model.patch_embed",
            "final_layer",
            "model.final_layer",
            
            # === Time embeddings ===
            "time_embed",
            "model.time_embed",
            "time_embed_2",
            "model.time_embed_2",
            "timestep_emb",
            "model.timestep_emb",
            
            # === Attention projections (critical for quality) ===
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "attn.qkv_proj",
            "attn.out_proj",
            "self_attn",
            "cross_attn",
        ]

        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=self.int8_threshold,
            llm_int8_skip_modules=skip_modules,
            llm_int8_enable_fp32_cpu_offload=True,  # CPU modules stay FP32 (avoids SCB bug)
        )
        
        logger.info("Created INT8 quantization config for Instruct model:")
        logger.info(f"  Quantization type: INT8")
        logger.info(f"  Outlier threshold: {self.int8_threshold}")
        logger.info(f"  Compute dtype: {self.compute_dtype}")
        logger.info(f"  Skipped modules: {len(skip_modules)} module patterns")
        logger.info(f"  Note: vision_model and vision_aligner kept at BF16")
        logger.info(f"  CPU offload: FP32 (avoids INT8 SCB serialization bug)")
        
        return config
    
    def _build_device_map(self) -> Dict[str, int]:
        """
        Build an explicit device_map that places ALL modules on GPU.
        
        Accelerate's 'auto' device_map estimates module sizes at BF16
        (before quantization) and overestimates memory needs, causing
        modules to spill to CPU as meta tensors. Since INT8 is ~2x
        smaller than BF16, the model actually fits entirely on GPU.
        
        This method reads the model config to determine the number of
        layers, then distributes them proportionally across GPUs based
        on available memory (accounting for non-layer overhead on GPU 0).
        """
        config_path = Path(self.model_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        num_layers = config.get("num_hidden_layers", 32)
        
        # Get GPU memory
        gpu_count = torch.cuda.device_count()
        gpu_memories = []
        for i in range(gpu_count):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_memories.append(total)
            logger.info(f"  GPU {i}: {total:.1f} GB")
        
        # Non-layer modules always go on GPU 0
        device_map = {
            "vae": 0,
            "vision_model": 0,
            "vision_aligner": 0,
            "timestep_emb": 0,
            "patch_embed": 0,
            "time_embed": 0,
            "final_layer": 0,
            "time_embed_2": 0,
            "model.wte": 0,
            "model.ln_f": 0,
            "lm_head": 0,
        }
        
        if gpu_count == 1:
            # Single GPU — all layers on GPU 0
            for i in range(num_layers):
                device_map[f"model.layers.{i}"] = 0
        else:
            # Multi-GPU — split layers proportionally.
            # GPU 0 has ~15-20GB of non-layer overhead (VAE, vision, embeddings)
            # so it gets fewer layers.
            gpu0_overhead_gb = 20  # conservative estimate for non-layer modules
            gpu0_available = gpu_memories[0] - gpu0_overhead_gb
            gpu1_available = gpu_memories[1] if gpu_count > 1 else 0
            total_available = gpu0_available + gpu1_available
            
            # Ratio of layers for GPU 0
            gpu0_ratio = gpu0_available / total_available
            split_point = int(num_layers * gpu0_ratio)
            # Ensure at least a few layers on each GPU
            split_point = max(4, min(split_point, num_layers - 4))
            
            logger.info(f"  Layer split: GPU 0 gets layers 0-{split_point-1}, "
                        f"GPU 1 gets layers {split_point}-{num_layers-1}")
            
            for i in range(num_layers):
                device_map[f"model.layers.{i}"] = 0 if i < split_point else 1
        
        return device_map

    def load_and_quantize(self) -> AutoModelForCausalLM:
        """
        Load the model and apply INT8 quantization.
        
        Uses an explicit device_map to force all modules onto GPU,
        bypassing accelerate's auto planner which overestimates memory
        at BF16 sizes and incorrectly spills modules to CPU/meta.
        
        Returns:
            Quantized model ready for inference
            
        Raises:
            RuntimeError: If model loading or quantization fails
        """
        try:
            logger.info("Starting Instruct INT8 model quantization...")
            logger.info("This will take several minutes on first run...")
            
            # Create quantization config
            quant_config = self.create_quantization_config()
            
            # Build explicit device_map — accelerate's 'auto' overestimates
            # at BF16 sizes and spills to CPU as meta tensors that can't be saved.
            # The INT8 model is ~76GB and fits entirely on GPU.
            device_map = self._build_device_map()
            logger.info(f"Explicit device map: {device_map}")
            
            # Load model with quantization
            logger.info(f"Loading Instruct model from {self.model_path}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            
            logger.info("Instruct model loaded and quantized successfully!")
            logger.info(f"Model device map: {model.hf_device_map}")
            
            # Verify nothing ended up on CPU/meta
            if hasattr(model, 'hf_device_map'):
                cpu_modules = [
                    k for k, v in model.hf_device_map.items()
                    if v in ('cpu', 'disk')
                ]
                if cpu_modules:
                    logger.warning(
                        f"WARNING: {len(cpu_modules)} modules still on CPU: {cpu_modules}. "
                        f"Save may fail."
                    )
                else:
                    logger.info("All modules on GPU — save should succeed.")
            
            # Get memory usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i}: {allocated:.1f} / {total:.1f} GB "
                                f"({allocated/total*100:.0f}% used)")
            
            return model
            
        except Exception as ex:
            logger.error(f"Failed to load and quantize model: {str(ex)}")
            logger.error(f"\n[ERROR] Quantization failed: {str(ex)}")
            raise
    
    def save_quantized_model(self, model: AutoModelForCausalLM) -> None:
        """
        Save the quantized model and configuration.
        
        Args:
            model: The quantized model to save
        """
        try:
            logger.info(f"Saving INT8 quantized Instruct model to {self.output_path}...")
            
            # Save the model
            model.save_pretrained(
                self.output_path,
                safe_serialization=True,
            )
            
            # Save quantization metadata
            quant_metadata = {
                "model_type": "HunyuanImage-3.0-Instruct",
                "quantization_method": "bitsandbytes_int8",
                "load_in_8bit": True,
                "llm_int8_threshold": self.int8_threshold,
                "expected_vram_gb": 95,
                "expected_total_memory_gb": 100,
                "modules_kept_bf16": [
                    "vae", "vision_model", "vision_aligner",
                    "patch_embed", "final_layer", "time_embed",
                    "time_embed_2", "timestep_emb", "attention_projections"
                ],
                "notes": "Instruct model with vision encoder kept at BF16 for image understanding quality.",
                "attention_layers_quantized": False,
                "quality_vs_nf4": "Significantly better - approximately 2x memory for ~98% quality retention"
            }
            
            metadata_path = self.output_path / "quantization_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(quant_metadata, f, indent=2)
            
            logger.info("Saved quantization metadata")
            self._copy_support_assets()

            logger.info(f"Model saved successfully to {self.output_path}")
            
        except Exception as ex:
            logger.error(f"Failed to save model: {str(ex)}")
            raise RuntimeError(f"Save failed: {str(ex)}") from ex
    
    def _copy_support_assets(self) -> None:
        """Copy tokenizer and VAE artifacts that are not captured by save_pretrained."""
        source_root = Path(self.model_path)
        dest_root = self.output_path

        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "tokenizer_wrapper.py",
            "tokenization_hunyuan.py",
        ]

        copied_any = False
        for filename in tokenizer_files:
            src_file = source_root / filename
            if src_file.exists():
                dest_file = dest_root / filename
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
                logger.info("Copied tokenizer asset: %s", filename)
                copied_any = True

        if not copied_any:
            logger.info("No standalone tokenizer files detected; assuming embedded tokenizer")

        for folder_name in ["tokenizer", "vae"]:
            src_dir = source_root / folder_name
            dest_dir = dest_root / folder_name
            if src_dir.exists():
                shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                logger.info("Mirrored directory %s", src_dir.name)

    def create_loading_script(self) -> None:
        """
        Create a helper script for loading the INT8 quantized model.
        """
        loading_script = f'''"""
Quick loader for INT8 quantized HunyuanImage-3.0-Instruct model.
Generated automatically by hunyuan_quantize_instruct_int8.py
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_quantized_instruct_int8(model_path="{self.output_path}"):
    """Load the INT8 quantized HunyuanImage-3.0-Instruct model."""
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold={self.int8_threshold},
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    # Load tokenizer
    model.load_tokenizer(model_path)
    
    return model

if __name__ == "__main__":
    print("Loading INT8 quantized Instruct model...")
    model = load_quantized_instruct_int8()
    print("Model loaded successfully!")
    print(f"Device map: {{model.hf_device_map}}")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB")
        print(f"GPU memory reserved: {{torch.cuda.memory_reserved() / 1024**3:.2f}} GB")
'''
        
        script_path = self.output_path / "load_quantized_instruct_int8.py"
        with open(script_path, 'w') as f:
            f.write(loading_script)
        
        logger.info(f"Created loading helper script: {script_path}")
    
    def quantize_and_save(self) -> None:
        """
        Complete quantization workflow: load, quantize, and save.
        """
        logger.info("=" * 60)
        logger.info("Starting HunyuanImage-3.0-Instruct INT8 Quantization")
        logger.info("=" * 60)
        
        model = self.load_and_quantize()
        self.save_quantized_model(model)
        self.create_loading_script()
        
        logger.info("=" * 60)
        logger.info("Instruct INT8 Quantization complete!")
        logger.info("Quality: ~98% of full precision (much better than NF4)")
        logger.info("Memory: ~95-100GB, should fit on 144GB dual-GPU setup")
        logger.info("=" * 60)


def main():
    """Main entry point for the Instruct INT8 quantization script."""
    parser = argparse.ArgumentParser(
        description="Quantize HunyuanImage-3.0-Instruct to INT8 format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="H:/Testing/HunyuanImage-3.0-Instruct",
        help="Path to source Instruct model directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="H:/Testing/HunyuanImage-3.0-Instruct-INT8",
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--int8-threshold",
        type=float,
        default=6.0,
        help="Outlier detection threshold for INT8 (default: 6.0)"
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Computation dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: auto)"
    )
    
    args = parser.parse_args()
    
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bfloat16" else torch.float16
    
    quantizer = HunyuanInstructQuantizerINT8(
        model_path=args.model_path,
        output_path=args.output_path,
        compute_dtype=compute_dtype,
        int8_threshold=args.int8_threshold,
        device_map=args.device_map
    )
    
    try:
        quantizer.quantize_and_save()
        logger.info("\n[OK] Success! Your INT8 quantized Instruct model is ready.")
        logger.info("Use with HunyuanInstructLoader node in ComfyUI.")
    except Exception as ex:
        logger.error(f"\n[ERROR] Quantization failed: {str(ex)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
