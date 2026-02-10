# 🎉 Release v1.1.0 — Instruct Models, Block Swap & Flexible Model Paths

Professional ComfyUI custom nodes for Tencent HunyuanImage-3.0, the powerful 80B parameter (13B active) native multimodal MoE image generation model.

## ✨ What's New in v1.1.0

### 🤖 Instruct Model Support
- **5 new nodes**: Instruct Loader, Generate, Image Edit, Multi-Image Fusion, and Unload
- **Built-in prompt enhancement** with Chain-of-Thought (CoT) reasoning — no external LLM API needed
- **Image editing** with natural language instructions ("Add sunglasses to the person")
- **Multi-image fusion** — combine elements from 2–3 reference images
- **Distil variant** — 8-step fast inference with CFG-distilled model
- Pre-quantized INT8 and NF4 Instruct models on [Hugging Face](https://huggingface.co/EricRollei)

### ⚡ Block Swap for All Loaders
- Run BF16 (~160GB) and INT8 (~81GB) models on 48–96GB GPUs
- Async prefetching swaps transformer blocks between GPU↔CPU during inference
- Configurable `blocks_to_swap` and `vram_reserve_gb` per loader

### 📁 Flexible Model Paths
- All loaders now use ComfyUI's `folder_paths` system
- Store models anywhere — add custom paths via `extra_model_paths.yaml`:
  ```yaml
  comfyui:
      hunyuan: |
          models/
          H:/MyModels/
      hunyuan_instruct: |
          models/
          H:/MyModels/
  ```
- No more hardcoded paths — works out of the box on any machine

### 🚀 HighRes Efficient Generation
- Generate 3MP–4K+ images on 96GB GPUs
- Loop-based MoE expert routing uses ~75× less VRAM than standard dispatch_mask
- New dedicated HighRes Efficient generate node

### 🧠 Unified V2 Node
- Single generate node that auto-detects NF4/INT8/BF16 from model folder name
- Integrated block swap, VAE management, and VRAM budget calculation
- Post-action options: keep_loaded, soft_unload, full_unload

## 📋 Previous Features (v1.0.0)

- **Multiple Loading Modes**: Full BF16, INT8, NF4 Quantized, Single GPU, Multi-GPU
- **Smart Memory Management**: Automatic VRAM tracking and cleanup
- **High-Quality Generation**: Standard (<2MP) and Large (2MP-8MP+) image support
- **INT8 Quantized Loader**: bitsandbytes LLM.int8() with selective quantization
- **NF4 Low VRAM Loader**: Custom device map for 24–32GB cards
- **Advanced Prompting**: Official HunyuanImage-3.0 prompt enhancement with OpenAI-compatible LLM APIs
- **Professional Resolution Control**: Organized presets with megapixel indicators

## 📦 Requirements

- **ComfyUI** installed and working
- **NVIDIA GPU**: 24GB+ VRAM for NF4, 48GB+ for INT8, 96GB+ for BF16
- **Python 3.10+**, **PyTorch 2.8+**, **CUDA 12.8+**
- **bitsandbytes 0.48+** for INT8/NF4 quantized models

## 🚀 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Comfy_HunyuanImage3.git
cd Comfy_HunyuanImage3
pip install -r requirements.txt
```

See [README.md](https://github.com/EricRollei/Comfy_HunyuanImage3#readme) for full documentation.

## ⚠️ Known Limitations

1. **Instruct (full) INT8** — OOM during inference with block swap. Use Instruct-Distil-INT8 instead.
2. **RAM accumulation** — Successive model loads may leak RAM. Restart ComfyUI if needed.

## 🙏 Credits

This project integrates the **HunyuanImage-3.0** model developed by **Tencent Hunyuan Team** under Apache 2.0 license.
