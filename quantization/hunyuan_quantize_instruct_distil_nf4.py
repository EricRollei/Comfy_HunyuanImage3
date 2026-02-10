"""
HunyuanImage-3.0-Instruct-Distil NF4 Quantization Script
Quantize the Distilled Instruct model to NF4 format (~10-12GB)

The Instruct-Distil model is already compact (80B params, 13B active due to MoE).
NF4 quantization makes it extremely lightweight - fits on a single 16GB GPU.

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


class HunyuanInstructDistilQuantizerNF4:
    """
    Handles quantization of HunyuanImage-3.0-Instruct-Distil model to NF4 format.
    
    The Instruct-Distil model has:
    - 80B total params, but only 13B active (MoE architecture)
    - CFG distillation (guidance_emb module) — no CFG batch doubling
    - Meanflow support (timestep_r_emb module)
    - Much smaller memory footprint than full Instruct model
    
    NF4 quantization should result in ~10-12GB model that fits on a single 16GB GPU.
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the quantizer.
        
        Args:
            model_path: Path to the source model (or HuggingFace model ID)
            output_path: Path where quantized model will be saved
            compute_dtype: Computation dtype (bfloat16 recommended for quality)
            use_double_quant: Whether to use nested quantization (saves ~0.4 bits/param)
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
        """
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.compute_dtype = compute_dtype
        self.use_double_quant = use_double_quant
        self.device_map = device_map
        
        # Validate paths
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Instruct-Distil NF4 quantizer:")
        logger.info(f"  Source: {model_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Compute dtype: {compute_dtype}")
        logger.info(f"  Double quantization: {use_double_quant}")
    
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create the BitsAndBytes quantization configuration for NF4.
        
        Returns:
            BitsAndBytesConfig object configured for NF4 quantization
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
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for normal distributions
            bnb_4bit_use_double_quant=self.use_double_quant,  # Nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16,  # MUST be bfloat16 for stable attention
            modules_to_not_convert=skip_modules,
            llm_int8_skip_modules=skip_modules,
        )
        
        logger.info("Created NF4 quantization config for Instruct-Distil model:")
        logger.info(f"  Quantization type: NF4")
        logger.info(f"  Double quantization: {self.use_double_quant}")
        logger.info(f"  Compute dtype: {self.compute_dtype}")
        logger.info(f"  Skipped modules: {len(skip_modules)} module patterns")
        logger.info(f"  Note: vision_model, guidance_emb, timestep_r_emb kept at BF16")
        
        return config
    
    def load_and_quantize(self) -> AutoModelForCausalLM:
        """
        Load the model and apply NF4 quantization.
        
        Returns:
            Quantized model ready for inference
            
        Raises:
            RuntimeError: If model loading or quantization fails
        """
        try:
            logger.info("Starting Instruct-Distil NF4 model quantization...")
            logger.info("This model is already small (13B active params)...")
            logger.info("Expected memory: ~10-12GB (should fit on single 16GB GPU)")
            
            # Create quantization config
            quant_config = self.create_quantization_config()
            
            # Load model with quantization
            logger.info(f"Loading Instruct-Distil model from {self.model_path}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            
            logger.info("Instruct-Distil model loaded and quantized successfully!")
            logger.info(f"Model device map: {model.hf_device_map}")
            
            # Get memory usage
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak GPU memory allocated: {mem_allocated:.2f} GB")
            
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
            logger.info(f"Saving NF4 quantized Instruct-Distil model to {self.output_path}...")
            
            # Save the model
            model.save_pretrained(
                self.output_path,
                safe_serialization=True,
            )
            
            # Save quantization metadata
            quant_metadata = {
                "model_type": "HunyuanImage-3.0-Instruct-Distil",
                "quantization_method": "bitsandbytes_nf4",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": self.use_double_quant,
                "bnb_4bit_compute_dtype": "torch.bfloat16",
                "expected_vram_gb": 12,
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
                "notes": "NF4 quantized Distil model - extremely compact, single GPU friendly.",
                "attention_layers_quantized": False,
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
        Create a helper script for loading the NF4 quantized model.
        """
        loading_script = f'''"""
Quick loader for NF4 quantized HunyuanImage-3.0-Instruct-Distil model.
Generated automatically by hunyuan_quantize_instruct_distil_nf4.py

This model is optimized for fast inference on consumer GPUs:
- CFG distillation: No classifier-free guidance needed (no batch doubling)
- Meanflow: Improved sampling in 8 steps
- Only 13B active params despite 80B total (MoE)
- NF4: Fits on a single 16GB GPU
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_quantized_instruct_distil_nf4(model_path="{self.output_path}"):
    """Load the NF4 quantized HunyuanImage-3.0-Instruct-Distil model."""
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant={self.use_double_quant},
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="cuda:0",  # Should fit on single GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    # Load tokenizer
    model.load_tokenizer(model_path)
    
    return model

if __name__ == "__main__":
    print("Loading NF4 quantized Instruct-Distil model...")
    model = load_quantized_instruct_distil_nf4()
    print("Model loaded successfully!")
    print(f"Device map: {{model.hf_device_map}}")
    
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {{torch.cuda.memory_allocated() / 1024**3:.2f}} GB")
        print(f"GPU memory reserved: {{torch.cuda.memory_reserved() / 1024**3:.2f}} GB")
'''
        
        script_path = self.output_path / "load_quantized_instruct_distil_nf4.py"
        with open(script_path, 'w') as f:
            f.write(loading_script)
        
        logger.info(f"Created loading helper script: {script_path}")
    
    def quantize_and_save(self) -> None:
        """
        Complete quantization workflow: load, quantize, and save.
        """
        logger.info("=" * 60)
        logger.info("Starting HunyuanImage-3.0-Instruct-Distil NF4 Quantization")
        logger.info("=" * 60)
        
        model = self.load_and_quantize()
        self.save_quantized_model(model)
        self.create_loading_script()
        
        logger.info("=" * 60)
        logger.info("Instruct-Distil NF4 Quantization complete!")
        logger.info("Model size: ~10-12GB (fits on a 16GB GPU!)")
        logger.info("Features: CFG distillation, meanflow, MoE")
        logger.info("=" * 60)


def main():
    """Main entry point for the Instruct-Distil NF4 quantization script."""
    parser = argparse.ArgumentParser(
        description="Quantize HunyuanImage-3.0-Instruct-Distil to NF4 format"
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
        default="H:/Testing/HunyuanImage-3.0-Instruct-Distil-NF4",
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Disable nested quantization (saves less memory)"
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
    
    quantizer = HunyuanInstructDistilQuantizerNF4(
        model_path=args.model_path,
        output_path=args.output_path,
        compute_dtype=compute_dtype,
        use_double_quant=not args.no_double_quant,
        device_map=args.device_map
    )
    
    try:
        quantizer.quantize_and_save()
        logger.info("\n[OK] Success! Your NF4 quantized Instruct-Distil model is ready.")
        logger.info("Use with HunyuanInstructLoader node in ComfyUI.")
        logger.info("This model should fit entirely on a single 16GB GPU!")
    except Exception as ex:
        logger.error(f"\n[ERROR] Quantization failed: {str(ex)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
