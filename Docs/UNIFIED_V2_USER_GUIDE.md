# Hunyuan Unified V2 - User Guide

**Version:** 2.0
**Date:** December 12, 2025

---

## Overview

The Hunyuan Unified V2 nodes provide a clean, efficient way to generate images with Tencent's HunyuanImage-3.0 model. Key features:

- **No accelerate hooks** - Clean, predictable memory management
- **Block swapping** - Generate larger images by swapping transformer blocks to CPU
- **Auto configuration** - Automatically calculates optimal settings for your GPU
- **Resolution presets** - Sorted by aspect ratio for easy selection
- **Auto model detection** - Quant type detected from folder name

---

## Important Notes

### No Negative Prompts
HunyuanImage-3.0 is an **autoregressive model**, not a traditional diffusion model. It does NOT support negative prompts. The model uses intelligent world-knowledge reasoning to interpret your prompts.

### Auto Mode for Resolution
The model can **automatically predict the optimal aspect ratio** based on your prompt! Select "Auto (model predicts)" and let the model decide.

---

## Nodes

### 1. Hunyuan Unified Generate V2 (Main Node)

The primary generation node. Handles loading, generation, and memory management.

#### Required Parameters

| Parameter | Description |
|-----------|-------------|
| **model_name** | Select your Hunyuan model folder. Quant type is auto-detected from the name (NF4/INT8/BF16). |
| **prompt** | Your generation prompt. No negative prompt support. |
| **resolution** | Choose from preset resolutions sorted by aspect ratio, or "Auto" to let the model decide. |
| **num_inference_steps** | Denoising steps (30-50 recommended) |
| **seed** | Random seed (-1 = random) |

#### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **blocks_to_swap** | -1 | -1=auto, 0=none, 1-31=manual |
| **vae_placement** | auto | auto, always_gpu, or managed |
| **post_action** | keep_loaded | What to do after generation |
| **enable_vae_tiling** | False | Enable for large images |
| **reserve_vram_gb** | 0.0 | Reserve VRAM for downstream nodes |

#### Resolution Presets

Sorted from portrait (tall) to landscape (wide):

**Portrait:**
- 768x1360 (9:16)
- 832x1216 (2:3)
- 896x1152 (3:4)

**Square:**
- 1024x1024 (1:1)

**Landscape:**
- 1152x896 (4:3)
- 1216x832 (3:2)
- 1360x768 (16:9)

**Large:**
- 1536x1536 (1:1 Large)
- 1792x1024 (16:9 Large)
- 2048x1152 (16:9 XL)

---

### 2. Hunyuan VRAM Calculator V2

Preview VRAM requirements before generation. Now auto-detects quant type from model name.

**Outputs:**
- **report** - Detailed breakdown of VRAM needs
- **recommended_blocks** - Suggested blocks_to_swap value
- **recommended_vae** - Suggested VAE placement

---

### 3. Hunyuan Emergency Cleanup (NEW)

Use this if VRAM is stuck after a failed generation or crash.

**Actions:**
1. Removes all hook handles
2. Clears the model cache
3. Forces garbage collection
4. Empties CUDA cache

**Warning:** This will unload any cached models!

---

### 4. Hunyuan Unload V2

Force unload the cached model.

---

### 5. Hunyuan Cache Status V2

Check what's cached, VRAM usage, and block swap statistics.

---

## Model Auto-Detection

The node automatically detects quantization from folder name:

| Folder Name Contains | Detected Type |
|---------------------|---------------|
| nf4 or NF4 | nf4 |
| int8 or INT8 | int8 |
| anything else | bf16 |

Examples:
- HunyuanImage-3-NF4 → nf4
- HunyuanImage_3_int8 → int8
- HunyuanImage-3 → bf16

---

## Recommended Workflows

### Single Generation (Plenty of VRAM)
`
Settings:
  blocks_to_swap: 0 (or -1 for auto)
  vae_placement: auto
  post_action: keep_loaded
`

### Large Image Generation
`
Settings:
  resolution: 2048x1152 (16:9 XL)
  blocks_to_swap: -1 (auto)
  vae_placement: managed
  enable_vae_tiling: true
`

### Running Other Models After
`
Settings:
  blocks_to_swap: -1 (auto)
  reserve_vram_gb: 10-20
  post_action: soft_unload
`

### Memory Constrained System
`
Settings:
  blocks_to_swap: 10-20
  vae_placement: managed
  post_action: soft_unload
`

---

## Troubleshooting

### "CUDA out of memory"
1. Set locks_to_swap = -1 (auto)
2. Enable ae_tiling for large images
3. Use smaller resolution
4. Use soft_unload or ull_unload between generations

### Stuck VRAM After Crash
Use the **Hunyuan Emergency Cleanup** node with confirm = True

### INT8 Won't Soft Unload
This is expected - INT8 uses bitsandbytes which doesn't support device movement.

### Node Not Appearing
1. Restart ComfyUI
2. Check console for import errors

---

## VRAM Requirements

| Model | Min VRAM | Comfortable | Notes |
|-------|----------|-------------|-------|
| NF4 @ 1024x1024 | 24 GB* | 48 GB | *With block swapping |
| NF4 @ 1536x1536 | 32 GB* | 64 GB | |
| INT8 @ 1024x1024 | 48 GB | 64 GB | No block swap |
| BF16 | 48 GB** | 96 GB | **With device_map |

---

## Questions?

Check the GitHub repository for updates and issue tracking.
