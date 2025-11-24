# 🎨 HunyuanImage-3.0 ComfyUI Custom Nodes

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

Professional ComfyUI custom nodes for [Tencent HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0), the powerful 80B parameter native multimodal image generation model.

> **🙏 Acknowledgment**: This project integrates the HunyuanImage-3.0 model developed by **Tencent Hunyuan Team** and uses their official system prompts. The model and original code are licensed under [Apache 2.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE). This integration code is separately licensed under CC BY-NC 4.0 for non-commercial use.

## 🎯 Features

- **Multiple Loading Modes**: Full BF16, NF4 Quantized, Single GPU, Multi-GPU
- **Smart Memory Management**: Automatic VRAM tracking, cleanup, and optimization
- **High-Quality Image Generation**: 
  - Standard generation (<2MP) - Fast, GPU-only
  - Large image generation (2MP-8MP+) - CPU offload support
- **Advanced Prompting**:
  - Optional prompt enhancement using official HunyuanImage-3.0 system prompts
  - Supports any OpenAI-compatible LLM API (DeepSeek, OpenAI, Claude, local LLMs)
  - Two professional rewriting modes: en_recaption (structured) and en_think_recaption (advanced)
- **Professional Resolution Control**:
  - Organized dropdown with portrait/landscape/square labels
  - Megapixel indicators and size categories
  - Auto resolution detection based on prompt
- **Production Ready**: Comprehensive error handling, detailed logging, VRAM monitoring

## 📦 Installation

### Prerequisites

- ComfyUI installed and working
- NVIDIA GPU with CUDA support
- **Minimum 24GB VRAM** for NF4 quantized model
- **Minimum 80GB VRAM** (or multi-GPU) for full BF16 model
- Python 3.10+
- PyTorch 2.7+ with CUDA 12.8+

### System Requirements & Hardware Recommendations

The hardware requirements depend heavily on which model version you use.

#### 1. Full BF16 Model (Original)
This is the uncompressed 80B parameter model. It is massive.
- **Model Size**: ~160GB on disk.
- **VRAM**: 
  - **Ideal**: 80GB+ (A100, H100, RTX 6000 Ada). Runs entirely on GPU.
  - **Minimum**: 24GB (RTX 3090/4090). *Requires massive System RAM.*
- **System RAM (CPU Memory)**:
  - If you have <80GB VRAM, the model weights that don't fit on GPU are stored in RAM.
  - **Requirement**: **192GB+ System RAM** is recommended if using a 24GB card.
  - *Example*: On a 24GB card, ~140GB of weights will live in RAM.
- **Performance**:
  - On low VRAM cards, generation will be slow due to swapping data between RAM and VRAM.

#### 2. NF4 Quantized Model (Recommended)
This version is compressed to 4-bit, reducing size by ~4x with minimal quality loss.
- **Model Size**: ~45GB on disk.
- **VRAM**:
  - **Ideal**: 48GB+ (RTX 6000, A6000). Runs entirely on GPU.
  - **Minimum**: 24GB (RTX 3090/4090).
    - *Note*: Since 45GB > 24GB, about half the model will live in System RAM.
    - **Performance**: Slower than 48GB cards, but functional.
- **System RAM**: 
  - **64GB+ recommended** (especially for 24GB VRAM cards to hold the offloaded weights).
- **Performance**: Much faster on consumer hardware.

### Quick Install

1. **Clone this repository** into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ericRollei/Eric_Hunyuan3.git
```

2. **Install dependencies**:
```bash
cd Eric_Hunyuan3
pip install -r requirements.txt
```

3. **Download model weights**:

**Option A: Full BF16 Model (~80GB)**
```bash
# Download to ComfyUI/models/
cd ../../models
huggingface-cli download tencent/HunyuanImage-3.0 --local-dir HunyuanImage-3
```

**Option B: Download Pre-Quantized NF4 Model (~20GB)** - *Recommended for single GPU <96GB*
You can download the pre-quantized weights directly from Hugging Face:
[EricRollei/HunyuanImage-3-NF4-ComfyUI](https://huggingface.co/EricRollei/HunyuanImage-3-NF4-ComfyUI)

```bash
# Download to ComfyUI/models/
cd ../../models
huggingface-cli download EricRollei/HunyuanImage-3-NF4-ComfyUI --local-dir HunyuanImage-3-NF4
```

**Option C: Quantize Yourself (from Full Model)**
If you prefer to quantize it yourself:
```bash
# First download full model (Option A), then quantize
cd path/to/Eric_Hunyuan3/quantization
python hunyuan_quantize_nf4.py \
  --model-path "../../models/HunyuanImage-3" \
  --output-path "../../models/HunyuanImage-3-NF4"
```

4. **Restart ComfyUI**

## 🚀 Usage

### Node Overview

| Node Name | Purpose | VRAM Required | Speed |
|-----------|---------|---------------|-------|
| **Hunyuan 3 Loader (NF4)** | Load quantized model | ~45GB | Fast load |
| **Hunyuan 3 Loader (Full BF16)** | Load full precision model | ~80GB | Moderate |
| **Hunyuan 3 Loader (Full BF16 GPU)** | Single GPU with memory control | ~75GB+ | Moderate |
| **Hunyuan 3 Loader (Multi-GPU BF16)** | Distribute across GPUs | 80GB total | Fast |
| **Hunyuan 3 Loader (88GB GPU Optimized)** | **DEPRECATED** - Use Full BF16 Loader | - | - |
| **Hunyuan 3 Generate** | Standard generation (<2MP) | Varies | **Fast** ⚡ |
| **Hunyuan 3 Generate (Large/Offload)** | Large images (2-8MP+) | Varies | Moderate |
| **Hunyuan 3 Unload** | Free VRAM | - | Instant |
| **Hunyuan 3 GPU Info** | Diagnostic/GPU detection | - | Instant |

### Node Compatibility Guide

**⚠️ IMPORTANT**: Match the Loader to the correct Generate node for stability.

| Loader Node | Compatible Generate Node | Why? |
|-------------|--------------------------|------|
| **Hunyuan 3 Loader (NF4)** | **Hunyuan 3 Generate** | Keeps model on GPU. Best for standard sizes (<2MP). |
| **Hunyuan 3 Loader (Full BF16)** | **Hunyuan 3 Generate (Large/Offload)** | Keeps model in RAM. Allows CPU offloading for massive images (4K+). |

> **Do not mix them!** Using the NF4 Loader with the Large/Offload node will likely cause errors because the quantized model cannot be moved to CPU correctly.

### Basic Workflow

```
┌─────────────────────────┐
│ Hunyuan 3 Loader (NF4)  │
│  model_name: HunyuanImage-3-NF4 │
│  keep_in_cache: True    │
└───────────┬─────────────┘
            │ HUNYUAN_MODEL
            ▼
┌─────────────────────────┐
│ Hunyuan 3 Generate      │
│  prompt: "..."          │
│  steps: 50              │
│  resolution: 1024x1024  │
│  guidance_scale: 7.5    │
└───────────┬─────────────┘
            │ IMAGE
            ▼
     ┌──────────────┐
     │ Save Image   │
     └──────────────┘
```

### Advanced: Prompt Rewriting

**✨ Feature**: Uses official HunyuanImage-3.0 system prompts to professionally expand your prompts for better results.

**⚠️ Note**: Requires a paid LLM API. This feature is optional - you can use the nodes without it.

**Supported APIs** (any OpenAI-compatible endpoint):
- DeepSeek (default, recommended for cost)
- OpenAI GPT-4/GPT-3.5
- Claude (via OpenAI-compatible proxy)
- Local LLMs (via LM Studio, Ollama with OpenAI API)

**Setup (Secure)**:
1. Rename `api_config.ini.example` to `api_config.ini` in the custom node folder.
2. Add your API key to the file:
   ```ini
   [API]
   api_key = sk-your-key-here
   ```
3. Alternatively, set environment variables: `HUNYUAN_API_KEY`, `HUNYUAN_API_URL`.

**Usage**:
- **Option 1 (Integrated)**: Enable `enable_prompt_rewrite` in the Generate node.
- **Option 2 (Standalone)**: Use the **Hunyuan Prompt Rewriter** node to rewrite prompts before passing them to any model.

```
┌─────────────────────────┐      ┌─────────────────────────┐
│ Hunyuan Prompt Rewriter │      │ Hunyuan 3 Generate      │
│  prompt: "dog running"  │ ───► │  prompt: (rewritten)    │
│  rewrite_style: ...     │      │  ...                    │
└─────────────────────────┘      └─────────────────────────┘
```

**Result**: Automatically expands to:
> "An energetic brown and white border collie running across a sun-drenched meadow filled with wildflowers, motion blur on legs showing speed, golden hour lighting, shallow depth of field, professional photography, high detail, 8k quality"

### Large Image Generation

For high-resolution outputs (2K, 4K, 6MP+):

```
┌─────────────────────────────────┐
│ Hunyuan 3 Generate (Large)      │
│  resolution: 3840x2160 - 4K UHD │
│  cpu_offload: True              │
│  steps: 50                      │
└─────────────────────────────────┘
```

## 📐 Resolution Guide

### Standard Generation Node
- **832x1280 - Portrait (1.0MP) [<2MP]** ✅ Safe, fast
- **1024x1024 - Square (1.0MP) [<2MP]** ✅ Safe, fast
- **1280x832 - Landscape (1.0MP) [<2MP]** ✅ Safe, fast
- **1536x1024 - Landscape (1.5MP) [<2MP]** ✅ Safe, fast
- **2048x2048 - Square (4.0MP) [>2MP]** ⚠️ May OOM

### Large/Offload Node
- **2560x1440 - Landscape 2K (3.7MP)** ✅ With CPU offload
- **3840x2160 - Landscape 4K UHD (8.3MP)** ✅ With CPU offload
- **3072x2048 - Landscape 6MP (6.3MP)** ✅ With CPU offload

**Tip**: Test prompts at small resolutions (fast), then render finals in large node.

## 🔧 Configuration

### Memory Management

**Single GPU (24-48GB VRAM):**
```
Use: Hunyuan 3 Loader (NF4)
Settings:
  - keep_in_cache: True (for multiple generations)
  - Use standard Generate node for <2MP
```

**Single GPU (80-96GB VRAM):**
```
Use: Hunyuan 3 Loader (88GB GPU Optimized)
Settings:
  - reserve_memory_gb: 14.0 (leaves room for inference)
  - Full BF16 quality
```

**Multi-GPU Setup:**
```
Use: Hunyuan 3 Loader (Multi-GPU BF16)
Settings:
  - primary_gpu: 0 (where inference runs)
  - reserve_memory_gb: 12.0
  - Automatically distributes across all GPUs
```

### ⚡ Performance Optimization Guide

To get the maximum speed and avoid unnecessary offloading (which slows down generation):

1.  **Reserve Enough VRAM**:
    *   Use the `reserve_memory_gb` slider in the Loader.
    *   Set it high enough to cover the generation overhead for your target resolution (e.g., **30GB+ for 4K**).
    *   *Why?* If you reserve space upfront, the model stays on the GPU. If you don't, the "Smart Offload" might panic and move everything to RAM to prevent a crash.

2.  **Select Specific Resolutions**:
    *   Avoid using "Auto (model default)" in the Large Generate node if you are optimizing for speed.
    *   **Auto Mode Safety**: When "Auto" is selected, the node assumes a large resolution (~2.5MP) to be safe. This might trigger offloading even if your actual image is small.
    *   **Specific Mode**: Selecting "1024x1024" tells the node *exactly* how much VRAM is needed, allowing it to skip offload if you have the space.

### LLM Prompt Rewriting (Optional)

**✨ Feature**: Uses official HunyuanImage-3.0 system prompts to professionally expand your prompts for better results.

**⚠️ Note**: Requires a paid LLM API. This feature is optional - you can use the nodes without it.

**Supported APIs** (any OpenAI-compatible endpoint):
- DeepSeek (default, recommended for cost)
- OpenAI GPT-4/GPT-3.5
- Claude (via OpenAI-compatible proxy)
- Local LLMs (via LM Studio, Ollama with OpenAI API)

**Setup (Secure)**:
1. Rename `api_config.ini.example` to `api_config.ini` in the custom node folder.
2. Add your API key to the file:
   ```ini
   [API]
   api_key = sk-your-key-here
   ```
3. Alternatively, set environment variables: `HUNYUAN_API_KEY`, `HUNYUAN_API_URL`.

**Usage**:
- **Option 1 (Integrated)**: Enable `enable_prompt_rewrite` in the Generate node.
- **Option 2 (Standalone)**: Use the **Hunyuan Prompt Rewriter** node to rewrite prompts before passing them to any model.

```
┌─────────────────────────┐      ┌─────────────────────────┐
│ Hunyuan Prompt Rewriter │      │ Hunyuan 3 Generate      │
│  prompt: "dog running"  │ ───► │  prompt: (rewritten)    │
│  rewrite_style: ...     │      │  ...                    │
└─────────────────────────┘      └─────────────────────────┘
```

**Result**: Automatically expands to:
> "An energetic brown and white border collie running across a sun-drenched meadow filled with wildflowers, motion blur on legs showing speed, golden hour lighting, shallow depth of field, professional photography, high detail, 8k quality"

**Rewrite Styles** (Official HunyuanImage-3.0 system prompts):
- **none**: (Default) Use your original prompt without modification
- **en_recaption**: Structured professional expansion with detailed descriptions (recommended)
  - Adds objective, physics-consistent details
  - Enhances lighting, composition, color descriptions
  - Best for general use
- **en_think_recaption**: Advanced mode with thinking phase + detailed expansion
  - LLM analyzes your intent first, then creates detailed prompt
  - More comprehensive but uses more tokens
  - Best for complex or ambiguous prompts

**Example Results**:
```
Original: "a cat on a table"

en_recaption: "A domestic short-hair cat with orange and white fur sits 
upright on a wooden dining table. Soft afternoon sunlight streams through 
a nearby window, casting warm highlights on the cat's fur and creating 
gentle shadows on the table surface. The background shows a blurred 
kitchen interior with neutral tones. Ultra-realistic, photographic style, 
sharp focus on the cat, shallow depth of field, 8k resolution."
```

If you get a "402 Payment Required" error, add credits to your API account or disable prompt rewriting.

## 🛠️ Quantization

Create your own NF4 quantized model:

```bash
cd quantization
python hunyuan_quantize_nf4.py \
  --model-path "/path/to/HunyuanImage-3" \
  --output-path "/path/to/HunyuanImage-3-NF4"
```

**Benefits:**
- ~4x smaller (80GB → 20GB model size)
- ~45GB VRAM usage (vs 80GB+ for BF16)
- Minimal quality loss
- Attention layers kept in full precision for stability

## 📊 Performance Benchmarks

**RTX 6000 Ada (48GB) - NF4 Quantized:**
- Load time: ~35 seconds
- 1024x1024 @ 50 steps: ~4 seconds/step
- VRAM usage: ~45GB

**2x RTX 4090 (48GB each) - Multi-GPU BF16:**
- Load time: ~60 seconds
- 1024x1024 @ 50 steps: ~3.5 seconds/step
- VRAM usage: ~70GB + 10GB distributed

**RTX 6000 Blackwell (96GB) - Full BF16:**
- Load time: ~25 seconds
- 1024x1024 @ 50 steps: ~3 seconds/step
- VRAM usage: ~80GB

## 🐛 Troubleshooting

### Out of Memory Errors

**Solutions:**
1. Use NF4 quantized model instead of full BF16
2. Reduce resolution (pick options marked `[<2MP]`)
3. Lower `steps` (try 30-40 instead of 50)
4. Use "Hunyuan 3 Generate (Large/Offload)" node with `cpu_offload: True`
5. Run "Hunyuan 3 Unload" node before generating
6. Set `keep_in_cache: False` in loader

### Pixelated/Corrupted Output

**If using NF4 quantization:**
- Re-quantize with the updated script (includes attention layer fix)
- Old quantized models may produce artifacts

### Multi-GPU Not Detecting Second GPU

**Check:**
1. Run "Hunyuan 3 GPU Info" node
2. Look for `CUDA_VISIBLE_DEVICES` environment variable
3. Ensure ComfyUI can see all GPUs: `torch.cuda.device_count()`

**Fix:**
```bash
# Remove GPU visibility restrictions
unset CUDA_VISIBLE_DEVICES
# Restart ComfyUI
```

### Slow Generation

**Optimizations:**
1. Use NF4 quantized model (faster than BF16)
2. Reduce `steps` (30-40 is often sufficient)
3. Keep model in cache (`keep_in_cache: True`)
4. Use smaller resolutions for testing

## 📝 Advanced Tips

### Prompt Engineering

**Good prompts include:**
1. **Subject**: What is the main focus
2. **Action**: What is happening
3. **Environment**: Where it takes place
4. **Style**: Artistic style, mood, atmosphere
5. **Technical**: Lighting, composition, quality keywords

**Example:**
```
A majestic snow leopard prowling through a misty mountain forest at dawn,
dappled golden light filtering through pine trees, shallow depth of field,
wildlife photography, National Geographic style, 8k, highly detailed fur texture
```

**Note**: HunyuanImage-3.0 uses an autoregressive architecture (like GPT) rather than diffusion, so it doesn't support negative prompts. Instead, be explicit in your prompt about what you want to include.

### Reproducible Results

Set a specific seed (0-18446744073709551615) to get the same image:
```
seed: 42  # Use any number, same seed = same image
```

## 📚 Model Information

**HunyuanImage-3.0:**
- **Architecture**: Native multimodal autoregressive transformer
- **Parameters**: 80B total (13B active per token)
- **Experts**: 64 experts (Mixture of Experts architecture)
- **Training**: Text-to-image with RLHF post-training
- **License**: Apache 2.0 (see Tencent repo for details)

**Paper**: [HunyuanImage 3.0 Technical Report](https://arxiv.org/pdf/2509.23951)

**Official Repo**: [Tencent-Hunyuan/HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

## 📄 License

**Dual License** (Non-Commercial and Commercial Use):

1. **Non-Commercial Use**: Licensed under [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/)
2. **Commercial Use**: Requires separate license. Contact [eric@historic.camera](mailto:eric@historic.camera) or [eric@rollei.us](mailto:eric@rollei.us)

See [LICENSE](LICENSE) for full details.

**Note**: The HunyuanImage-3.0 model itself is licensed under Apache 2.0 by Tencent. This license only covers the ComfyUI integration code.

Copyright (c) 2025 Eric Hiss. All rights reserved.

## 🙏 Credits

### ComfyUI Integration
- **Author**: Eric Hiss ([GitHub: EricRollei](https://github.com/ericRollei/))
- **License**: CC BY-NC 4.0 (Non-Commercial) / Commercial License Available

### HunyuanImage-3.0 Model
- **Developed by**: Tencent Hunyuan Team
- **Official Repository**: [Tencent-Hunyuan/HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- **Model License**: [Apache License 2.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE)
- **Paper**: [HunyuanImage 3.0 Technical Report](https://arxiv.org/pdf/2509.23951)
- This integration uses the official HunyuanImage-3.0 system prompts and model architecture developed by Tencent

### Special Thanks
- **Tencent Hunyuan Team** for creating and open-sourcing the incredible HunyuanImage-3.0 model
- **ComfyUI Community** for the excellent extensible framework
- All contributors and testers

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ericRollei/Eric_Hunyuan3/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ericRollei/Eric_Hunyuan3/discussions)
- **Email**: [eric@historic.camera](mailto:eric@historic.camera) or [eric@rollei.us](mailto:eric@rollei.us)
- **Tencent Official**: [WeChat](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/assets/WECHAT.md) | [Discord](https://discord.gg/ehjWMqF5wY)

## 🔄 Changelog

### v1.0.0 (2025-11-17)
- Initial release
- Full BF16 and NF4 quantized model support
- Multi-GPU loading support
- Optional prompt rewriting with DeepSeek API
- Improved resolution organization
- Large image generation with CPU offload
- Comprehensive error handling and VRAM management

---

**Made with ❤️ for the ComfyUI community**
