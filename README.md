# 🎨 HunyuanImage-3.0 ComfyUI Custom Nodes

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

Professional ComfyUI custom nodes for [Tencent HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0), the powerful 80B parameter native multimodal image generation model.

> **Latest**
>
> - ✅ **NF4 Low VRAM Loader + Low VRAM Budget generator are verified on 24‑32 GB cards** thanks to the custom device-map strategy that pins every quantized layer on GPU.
> - ✅ **INT8 Budget loader works end-to-end** and a pre-quantized checkpoint is available on Hugging Face: [EricRollei/Hunyuan_Image_3_Int8](https://huggingface.co/EricRollei/Hunyuan_Image_3_Int8) for anyone who wants INT8 quality without running the quantizer locally.
> - 📸 Added a reference workflow (below) showing the recommended node pair for Low VRAM setups.

> **🙏 Acknowledgment**: This project integrates the HunyuanImage-3.0 model developed by **Tencent Hunyuan Team** and uses their official system prompts. The model and original code are licensed under [Apache 2.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE). This integration code is separately licensed under CC BY-NC 4.0 for non-commercial use.

## 📋 TODO / Known Issues

- [x] **NF4 Low VRAM Loader**: Custom device map keeps NF4 layers on GPU so 24‑32 GB cards can use the Low VRAM Budget workflow without bitsandbytes errors.
- [x] Provide example workflow/screenshot for Low VRAM users (see below).
- [ ] Add screenshots/documentation for every node (in progress).
- [ ] Test and document multi-GPU setup.
- [ ] Continue long-run stability testing on the INT8 Budget loader with CPU offload edge cases.

## 🎯 Features

- **Multiple Loading Modes**: Full BF16, INT8/NF4 Quantized, Single GPU, Multi-GPU
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

#### 3. INT8 Quantized Model (High Quality)

> ⚠️ **Warning**: INT8 loader is currently non-functional due to bitsandbytes validation issues when CPU offload is required. The INT8 quantization format does not save quantized weights to disk - it only saves metadata and re-quantizes on load, which triggers validation errors for large models requiring CPU offload. **Use NF4 or BF16 loaders instead.**

This version is compressed to 8-bit, offering near-original quality with reduced memory usage.
- **Model Size**: ~85GB on disk.
- **VRAM**:
  - **Ideal**: 80GB+ (A100, H100). Runs entirely on GPU.
  - **Minimum**: 24GB (RTX 3090/4090).
    - *Note*: Significant CPU offloading required (~60GB of weights in RAM).
- **System RAM**: 
  - **128GB+ recommended**.
- **Performance**: 
  - **Quality**: ~98% of full precision (better than NF4).
  - **Speed**: Faster inference than NF4 (less dequantization overhead) but requires more memory transfer if offloading.
  - **Practical Requirement**: For the selective INT8 checkpoints shipped here, you realistically need a 96GB-class GPU (RTX 6000 Pro Blackwell, H100, etc.). Forcing CPU offload on smaller cards makes each step take minutes because the quantized tensors continually stream over PCIe.

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

**Option C: Download Pre-Quantized INT8 Model (~85GB)**
If you want maximum fidelity without running the INT8 quantizer locally, grab the ready-to-go checkpoint:
[EricRollei/Hunyuan_Image_3_Int8](https://huggingface.co/EricRollei/Hunyuan_Image_3_Int8)

```bash
# Download to ComfyUI/models/
cd ../../models
huggingface-cli download EricRollei/Hunyuan_Image_3_Int8 --local-dir HunyuanImage-3-INT8
```

**Option D: Quantize Yourself (from Full Model)**
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

### 📸 Low VRAM Workflow Reference

![Low VRAM Workflow](Docs/LowVRAM-Workflow-Image.png)

Pair **Hunyuan 3 Loader (NF4 Low VRAM+)** with **Hunyuan 3 Generate (Low VRAM Budget)** for 24‑32 GB cards. The loader’s GPU budget slider keeps 18–20 GB free for inference while the generator adds smart telemetry and optional prompt rewriting. The screenshot above shows a full working chain (loader → rewriter → Low VRAM Budget generator → save/display) producing a 1600×1600 render on a 24 GB test rig without triggering bitsandbytes validation.

### Node Overview

| Node Name | Purpose | VRAM Required | Speed |
|-----------|---------|---------------|-------|
| **Hunyuan 3 Loader (NF4)** | Load quantized model | ~45GB | Fast load |
| **Hunyuan 3 Loader (NF4 Low VRAM+)** | NF4 loader with GPU budget slider for 24‑32GB GPUs | 18‑28GB (configurable) | Balanced |
| **Hunyuan 3 Loader (INT8 Budget)** | INT8 loader with GPU budget & telemetry metadata | 20‑55GB (configurable) | Balanced |
| **Hunyuan 3 Loader (Full BF16)** | Load full precision model | ~80GB | Moderate |
| **Hunyuan 3 Loader (Full BF16 GPU)** | Single GPU with memory control | ~75GB+ | Moderate |
| **Hunyuan 3 Loader (Multi-GPU BF16)** | Distribute across GPUs | 80GB total | Fast |
| **Hunyuan 3 Loader (88GB GPU Optimized)** | **DEPRECATED** - Use Full BF16 Loader | - | - |
| **Hunyuan 3 Generate** | Standard generation (<2MP) | Varies | **Fast** ⚡ |
| **Hunyuan 3 Generate (Telemetry)** | Adds RAM/VRAM stats to status output | Varies | Fast |
| **Hunyuan 3 Generate (Large/Offload)** | Large images (2-8MP+) | Varies | Moderate |
| **Hunyuan 3 Generate (Large Budget)** | Large/offload with GPU budget override + telemetry | Varies | Moderate |
| **Hunyuan 3 Generate (Low VRAM)** | Quantized-friendly large generation | Varies | Moderate |
| **Hunyuan 3 Generate (Low VRAM Budget)** | Low VRAM mode with smart heuristics + telemetry | Varies | Moderate |
| **Hunyuan 3 Unload** | Free VRAM | - | Instant |
| **Hunyuan 3 Soft Unload (Fast)** | ⚠️ Future use - see limitations below | - | - |
| **Hunyuan 3 Force Unload (Nuclear)** | Aggressive VRAM clearing | - | Instant |
| **Hunyuan 3 Clear Downstream Models** | Clear other models, keep Hunyuan | - | Instant |
| **Hunyuan 3 GPU Info** | Diagnostic/GPU detection | - | Instant |

> 🆕 **Budget/Telemetry Nodes**: New loader/generator variants keep the legacy nodes untouched while exposing GPU budget sliders, smarter "smart" mode logic, and MemoryTracker telemetry in the node status. Mix and match them only if you need the extra controls; existing workflows can stay on the classic nodes.

### 🎯 Choosing the Right Generate Node

| Node | Has offload_mode? | Best For |
|------|-------------------|----------|
| **Hunyuan 3 Generate** | No | Standard use, ≤2MP, plenty of VRAM (48GB+). Fastest option. |
| **Hunyuan 3 Generate (Telemetry)** | No | Same as above + RAM/VRAM stats in status output |
| **Hunyuan 3 Generate (Large/Offload)** | Yes | Large images (>2MP) that need CPU offload during inference |
| **Hunyuan 3 Generate (Large Budget)** | Yes | Large images + GPU budget control + telemetry |
| **Hunyuan 3 Generate (Low VRAM)** | Yes | Quantized models (INT8/NF4) on limited VRAM (24-48GB) |
| **Hunyuan 3 Generate (Low VRAM Budget)** | Yes | Quantized + GPU budget control + telemetry |

**Quick Guide:**
- **96GB GPU + NF4/INT8**: Use **"Hunyuan 3 Generate"** (fastest, no offload overhead)
- **24-48GB GPU + NF4**: Use **"Hunyuan 3 Generate (Low VRAM)"** or **"Low VRAM Budget"**
- **Large images (>2MP)**: Use **"Large/Offload"** or **"Large Budget"** variants
- **Want telemetry?**: Use the **"Budget"** variants for memory stats in output

### 🧠 VRAM Management & Workflow Strategies

Managing VRAM is critical when running Hunyuan3 alongside other models. Here are the recommended workflows:

#### Scenario 1: Hunyuan Only (Simple)
```
[Loader] → [Generate] → [Save Image]
```
- Set `keep_model_loaded: True` for successive runs
- Model stays cached, fast subsequent generations

#### Scenario 2: Hunyuan with Downstream Models (Flux, SAM2, Florence)

**Option A: Keep Hunyuan Loaded (Recommended for successive runs)**
```
[Hunyuan Loader] → [Generate] → [Flux Detailer] → [SAM2] → [Save] → [Clear Downstream Models]
       ↑                                                                      ↓
       └──────────────────────── next run ────────────────────────────────────┘
```
- Place **"Hunyuan 3 Clear Downstream Models"** at the END of workflow
- Connect its `trigger` input to your final output
- This clears Flux/SAM2/Florence VRAM while **keeping Hunyuan loaded**
- Next run: Hunyuan is already loaded (fast!), downstream models reload (small)

**Option B: Clear Everything Between Runs**
```
[Hunyuan Loader] → [Generate] → [Unload] → [Flux Detailer] → [SAM2] → [Save]
                                   ↑
                    enable "clear_for_downstream"
```
- Use standard **Unload** node with `clear_for_downstream: True`
- Hunyuan unloads after generation, freeing VRAM for downstream
- Hunyuan reloads from disk on next run (slower but simpler)

#### Scenario 3: Cross-Tab VRAM Pollution

When running workflows in multiple browser tabs, models from other tabs stay in VRAM but ComfyUI "forgets" about them.

**Solution**: Use the BF16 Loader's `clear_vram_before_load` option:
```
[Hunyuan Loader (BF16)]  ← enable "clear_vram_before_load"
         ↓
    [Generate]
```
- Clears ALL orphaned VRAM before loading Hunyuan
- Helps when VRAM is "mysteriously" full

#### VRAM Management Node Summary

| Node | When to Use |
|------|-------------|
| **Unload** | Clear Hunyuan model between runs |
| **Clear Downstream Models** | Keep Hunyuan, clear Flux/SAM2/etc. |
| **Force Unload (Nuclear)** | After OOM errors, stuck VRAM |
| **Soft Unload** | ⚠️ See limitations below |

#### ⚠️ Soft Unload Node - Future/Limited Use

The **Soft Unload** node is currently included for **future compatibility** but has significant limitations with current hardware and libraries:

**Does NOT work with (most common cases):**
- **INT8/NF4 quantized models** - bitsandbytes locks tensors to their device
- **BF16 with device_map="auto"** - creates meta tensors that can't be moved

**Only works with:**
- BF16 model loaded entirely on a single GPU with NO offloading
- This requires **~160GB+ VRAM** (no consumer/prosumer GPU exists yet)

**Why it exists:**
- Future GPUs may have 160GB+ VRAM
- Future versions of bitsandbytes/accelerate may support moving quantized models to CPU
- The infrastructure is ready when these limitations are resolved

**For current use cases, use instead:**
- **"Clear Downstream Models"** - clears Flux/SAM2/etc while keeping Hunyuan cached
- **"Force Unload"** - completely removes model when needed

#### BF16 Loader Target Resolution Options

The BF16 loader has a `target_resolution` dropdown that reserves VRAM for inference:

| Option | VRAM Reserved | Use When |
|--------|---------------|----------|
| **Auto (safe default)** | 35GB | Unknown resolution, auto-detection |
| **1MP Fast (96GB+)** | 8GB | 96GB+ cards, max speed at 1MP |
| **1MP (1024x1024)** | 15GB | Standard 1MP generation |
| **2MP (1920x1080)** | 30GB | HD/2K generation |
| **3MP (2048x1536)** | 55GB | Large images |
| **4MP+ (2560x1920)** | 75GB | Very large images |

### INT8 Budget Workflow Controls

When using the **Hunyuan 3 Loader (INT8 Budget)** + **Hunyuan 3 Generate (Large Budget)** path (recommended for 96 GB RTX 6000 Pro / similar cards), keep these controls in mind:

- `reserve_memory_gb` (loader): Amount of VRAM the loader leaves free for inference overhead. Set to ~20 GB for 1.5–2 MP work; increase when targeting 4K+ so Smart mode has headroom without forcing CPU offload.
- `gpu_memory_target_gb` (loader): How much VRAM the INT8 weights are allowed to occupy. 80 GB is a good sweet spot on 96 GB GPUs. Lower values keep more VRAM free but may require CPU spillover during load.
- `offload_mode` (Large Budget generator):
  - `smart` (default) reads the loader’s reserve/budget metadata, estimates per-megapixel cost (~15 GB/MP + 6 GB), and only enables accelerate CPU offload when VRAM is critically low.
  - `disabled` forces fully-on-GPU sampling (fastest) – use this for ≤2 MP jobs when you already fit in VRAM.
  - `always` moves the model into accelerate offload before sampling. Only select this for extreme resolutions where you expect to exceed available VRAM; otherwise it slows INT8 inference to CPU speeds.
- `keep_model_loaded` (Large Budget generator): When `False`, the node calls the same cache-clearing logic as **Hunyuan 3 Unload** after a run, freeing the INT8 weights from VRAM/RAM. Leave it `True` for iterative workflows to avoid reload times.
- `gpu_budget_gb` override (Large Budget generator): Temporarily overrides the loader’s `_loader_gpu_budget_gb` value for Smart-mode math without remapping tensors. Set it if you want Smart mode to assume a different baseline (e.g., 65 GB) without reloading the model; leave at `-1` to use the loader’s recorded budget.

These generator settings never remap the model by themselves—they only influence whether accelerate offload engages and whether we clear the cached model afterward. Adjust the loader slider first (20 GB reserve / 80 GB target recommended), then use the generator controls to decide when to keep models in memory or trigger CPU offload for very large renders.

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

### RTX 6000 Pro Blackwell (96GB) - INT8 Quantized ⭐ RECOMMENDED

**Configuration**: INT8 Loader + Large Generate (Budget) with `offload_mode=disabled`

| Resolution | Steps | Time | Speed | VRAM |
|------------|-------|------|-------|------|
| 1152x864 (~1MP) | 40 | **2:35** | 3.9s/step | 85GB → 95GB |

This is the **fastest configuration** for high-VRAM cards. Key settings:
- Use **INT8 Loader (Budget)** with ~80GB GPU target
- Use **Large Generate (Budget)** with `offload_mode=disabled`
- Keep model loaded between runs for successive generations

### RTX 6000 Ada (48GB) - NF4 Quantized

- Load time: ~35 seconds
- 1024x1024 @ 50 steps: ~4 seconds/step
- VRAM usage: ~45GB

### 2x RTX 4090 (48GB each) - Multi-GPU BF16

- Load time: ~60 seconds
- 1024x1024 @ 50 steps: ~3.5 seconds/step
- VRAM usage: ~70GB + 10GB distributed

### RTX 6000 Blackwell (96GB) - Full BF16

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
