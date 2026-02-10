"""
HunyuanImage-3.0-Instruct-Distil INT8 Quantization Script
Quantize the Distilled Instruct model to INT8 format (~15-18GB)

The Instruct-Distil model is already much smaller (80B params, 13B active due to MoE).
INT8 quantization will make it fit comfortably in a single GPU.

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

Note: HunyuanImage-3.0-Instruct-Distil model is subject to Tencent's Apache 2.0 license.
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


class HunyuanInstructDistilQuantizerINT8:
    """
    Handles quantization of HunyuanImage-3.0-Instruct-Distil model to INT8 format.
    
    The Instruct-Distil model has:
    - 80B total params, but only 13B active (MoE architecture)
    - CFG distillation (guidance_emb module)
    - Meanflow support (timestep_r_emb module)
    - Much smaller memory footprint than full Instruct model
    
    INT8 quantization should result in ~15-18GB model that fits on a single GPU.
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
        
        logger.info(f"Initialized Instruct-Distil INT8 quantizer:")
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
        # Includes all base + Instruct modules PLUS Distil-specific modules
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
            
            # === Distil-specific modules (CFG distillation & meanflow) ===
            "guidance_emb",
            "model.guidance_emb",
            "timestep_r_emb",
            "model.timestep_r_emb",
            
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
        
        logger.info("Created INT8 quantization config for Instruct-Distil model:")
        logger.info(f"  Quantization type: INT8")
        logger.info(f"  Outlier threshold: {self.int8_threshold}")
        logger.info(f"  Compute dtype: {self.compute_dtype}")
        logger.info(f"  Skipped modules: {len(skip_modules)} module patterns")
        logger.info(f"  Note: vision_model, guidance_emb, timestep_r_emb kept at BF16")
        logger.info(f"  CPU offload: FP32 (avoids INT8 SCB serialization bug)")
        
        return config
    
    def _build_device_map(self) -> Dict[str, int]:
        """
        Build an explicit device_map that places ALL modules on GPU.
        
        Accelerate's 'auto' device_map estimates module sizes at BF16
        (before quantization) and overestimates memory needs, causing
        modules to spill to CPU as meta tensors. Since INT8 is ~2x
        smaller than BF16, the model actually fits entirely on GPU.
        
        The Distil model is small (~15-18GB INT8) and easily fits on
        a single GPU, so this primarily puts everything on GPU 0.
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
        # Distil has extra modules: guidance_emb, timestep_r_emb
        device_map = {
            "vae": 0,
            "vision_model": 0,
            "vision_aligner": 0,
            "timestep_emb": 0,
            "guidance_emb": 0,
            "timestep_r_emb": 0,
            "patch_embed": 0,
            "time_embed": 0,
            "final_layer": 0,
            "time_embed_2": 0,
            "model.wte": 0,
            "model.ln_f": 0,
            "lm_head": 0,
        }
        
        # Distil model is small — put all layers on GPU 0
        # (it's only ~15-18GB INT8, easily fits on any modern GPU)
        for i in range(num_layers):
            device_map[f"model.layers.{i}"] = 0
        
        return device_map

    def load_and_quantize(self) -> AutoModelForCausalLM:
        """
        Load the model and apply INT8 quantization.
        
        Uses an explicit device_map to force all modules onto GPU,
        bypassing accelerate's auto planner which overestimates memory.
        
        Returns:
            Quantized model ready for inference
            
        Raises:
            RuntimeError: If model loading or quantization fails
        """
        try:
            logger.info("Starting Instruct-Distil INT8 model quantization...")
            logger.info("This model is already small (13B active params)...")
            logger.info("Expected memory: ~15-18GB (should fit on single GPU)")
            
            # Create quantization config
            quant_config = self.create_quantization_config()
            
            # Build explicit device_map — accelerate's 'auto' overestimates
            # at BF16 sizes and may spill to CPU as meta tensors.
            device_map = self._build_device_map()
            logger.info(f"Explicit device map: {device_map}")
            
            # Load model with quantization
            logger.info(f"Loading Instruct-Distil model from {self.model_path}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            
            logger.info("Instruct-Distil model loaded and quantized successfully!")
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
            logger.info(f"Saving INT8 quantized Instruct-Distil model to {self.output_path}...")
            
            # Save the model
            model.save_pretrained(
                self.output_path,
                safe_serialization=True,
            )
            
            # Save quantization metadata
            quant_metadata = {
                "model_type": "HunyuanImage-3.0-Instruct-Distil",
                "quantization_method": "bitsandbytes_int8",
                "load_in_8bit": True,
                "llm_int8_threshold": self.int8_threshold,
                "expected_vram_gb": 18,
                "expected_total_memory_gb": 20,
                "total_params": "80B",
                "active_params": "13B (MoE)",
                "modules_kept_bf16": [
                    "vae", "vision_model", "vision_aligner",
                    "patch_embed", "final_layer", "time_embed",
                    "time_embed_2", "timestep_emb", "guidance_emb",
                    "timestep_r_emb", "attention_projections"
                ],
                "distil_features": {
                    "cfg_distilled": True,
                    "meanflow": True,
                    "description": "Single-step CFG-free generation with meanflow"
                },
                "notes": "Distilled Instruct model - fast inference, single GPU friendly.",
                "attention_layers_quantized": False,
                "quality_vs_nf4": "Better quality than NF4 with reasonable memory"
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
Quick loader for INT8 quantized HunyuanImage-3.0-Instruct-Distil model.
Generated automatically by hunyuan_quantize_instruct_distil_int8.py

This model is optimized for fast inference:
- CFG distillation: No classifier-free guidance needed
- Meanflow: Improved sampling
- Only 13B active params despite 80B total (MoE)
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_quantized_instruct_distil_int8(model_path="{self.output_path}"):
    """Load the INT8 quantized HunyuanImage-3.0-Instruct-Distil model."""
    
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
    print("Loading INT8 quantized Instruct-Distil model...")
    model = load_quantized_instruct_distil_int8()
    print("Model loaded successfully!")
    print(f"Device map: {{model.hf_device_map}}")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB")
        print(f"GPU memory reserved: {{torch.cuda.memory_reserved() / 1024**3:.2f}} GB")
'''
        
        script_path = self.output_path / "load_quantized_instruct_distil_int8.py"
        with open(script_path, 'w') as f:
            f.write(loading_script)
        
        logger.info(f"Created loading helper script: {script_path}")
    
    def quantize_and_save(self) -> None:
        """
        Complete quantization workflow: load, quantize, and save.
        """
        logger.info("=" * 60)
        logger.info("Starting HunyuanImage-3.0-Instruct-Distil INT8 Quantization")
        logger.info("=" * 60)
        
        model = self.load_and_quantize()
        self.save_quantized_model(model)
        self.create_loading_script()
        
        logger.info("=" * 60)
        logger.info("Instruct-Distil INT8 Quantization complete!")
        logger.info("Model size: ~15-18GB (fits on single GPU!)")
        logger.info("Features: CFG distillation, meanflow, MoE")
        logger.info("=" * 60)


def main():
    """Main entry point for the Instruct-Distil INT8 quantization script."""
    parser = argparse.ArgumentParser(
        description="Quantize HunyuanImage-3.0-Instruct-Distil to INT8 format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="H:/Testing/HunyuanImage-3.0-Instruct-Distil",
        help="Path to source Instruct-Distil model directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="H:/Testing/HunyuanImage-3.0-Instruct-Distil-INT8",
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
    
    quantizer = HunyuanInstructDistilQuantizerINT8(
        model_path=args.model_path,
        output_path=args.output_path,
        compute_dtype=compute_dtype,
        int8_threshold=args.int8_threshold,
        device_map=args.device_map
    )
    
    try:
        quantizer.quantize_and_save()
        logger.info("\n[OK] Success! Your INT8 quantized Instruct-Distil model is ready.")
        logger.info("Use with HunyuanInstructLoader node in ComfyUI.")
        logger.info("This model should fit entirely on a single 24GB+ GPU!")
    except Exception as ex:
        logger.error(f"\n[ERROR] Quantization failed: {str(ex)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
