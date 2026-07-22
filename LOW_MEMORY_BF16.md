# Experimental Low-Memory BF16 Disk Offload

This document describes the unofficial low-memory loading path added by the
[foxidermist fork](https://github.com/foxidermist/Comfy_HunyuanImage3) of
`EricRollei/Comfy_HunyuanImage3`.

The goal is to run the full
`tencent/HunyuanImage-3.0-Instruct` BF16 checkpoint using a combination of:

- a small CUDA model budget;
- a capped physical-RAM model budget;
- disk-backed sharded `safetensors`;
- Windows pagefile or operating-system swap;
- targeted compatibility fixes for modules that otherwise remain on the
  PyTorch `meta` device.

This is an experimental compatibility mode, not a fast consumer-GPU mode.

## Status

A complete image-edit generation has been successfully produced with the
configuration below.

| Component | Tested value |
|---|---|
| Operating system | Windows |
| GPU | NVIDIA RTX 5070 |
| VRAM | 12 GB |
| System RAM | 64 GB |
| Windows pagefile | 96 GB |
| Checkpoint | Full `HunyuanImage-3.0-Instruct` BF16 |
| PyTorch | `2.13.0+cu130` |
| Transformers | `4.57.6` |
| Accelerate | `1.14.0` |
| Initial successful test | 4 diffusion steps, `bot_task=image` |
| Approximate speed | 2.5–3 minutes per diffusion step |

Results on other systems are not guaranteed.

## Why This Mode Exists

The full BF16 checkpoint is approximately 160 GB. Normal execution expects a
large GPU, multiple GPUs, or enough physical RAM to keep the weights resident.

The low-memory path instead asks Accelerate to distribute modules across:

```text
cuda:0
cpu
disk
```

Disk-mapped modules are materialized only when they are needed. This reduces
persistent VRAM and RAM use, but makes inference dominated by SSD reads and
PCIe transfers.

Low average GPU utilization is therefore normal. The GPU often finishes a
short computation and waits for the next module's weights.

## Scope of the Modifications

The fork adds or changes the following behavior in
`hunyuan_instruct_nodes.py`:

1. BF16 CUDA/CPU/disk loading with `device_map="auto"`.
2. Configurable CPU model-memory budget.
3. Portable disk-offload directory selection.
4. Low-VRAM GPU budget handling.
5. Physical RAM, virtual memory, free disk space, and final device-map logs.
6. Fallback for a missing `config.model_version`.
7. Compatibility alias for image-processor method-name drift.
8. SigLIP2 positional-embedding materialization through its module hook.
9. Accelerate submodule preloading for `torch.nn.MultiheadAttention`.

The original prompt processing, CFG behavior, VAE generation path, NF4/INT8
loading, and existing block-swap implementation are otherwise preserved.

## Important Security Note

Instruct checkpoints require:

```python
trust_remote_code=True
```

This executes Python code supplied with the model repository. Download model
files only from a source you trust and review unexpected remote-code changes.

## Installation

### 1. Clone the Fork

From the `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/foxidermist/Comfy_HunyuanImage3.git
```

Restart ComfyUI after installation.

### 2. Install the Standard Requirements

Using the Python environment that runs ComfyUI:

```bash
cd Comfy_HunyuanImage3
python -m pip install -r requirements.txt
```

For the Windows portable build, use the embedded interpreter rather than a
different system Python.

Example:

```bat
C:\ComfyUI\python_embeded\python.exe -m pip install -r requirements.txt
```

### 3. Install the Tested Transformers and Accelerate Versions

Completely close ComfyUI first.

If the repository includes the helper script, run:

```bat
tools\fix_hunyuan_accelerate_transformers.bat
```

Equivalent manual command for the standard portable path:

```bat
C:\ComfyUI\python_embeded\python.exe -m pip uninstall -y accelerate transformers
C:\ComfyUI\python_embeded\python.exe -m pip install --no-cache-dir "accelerate==1.14.0" "transformers==4.57.6"
```

Verify:

```bat
C:\ComfyUI\python_embeded\python.exe -c "import accelerate, transformers; print('accelerate:', accelerate.__version__); print('transformers:', transformers.__version__)"
```

Expected:

```text
accelerate: 1.14.0
transformers: 4.57.6
```

Transformers 5.x may work for other project modes, but it caused native
checkpoint-loader crashes in the tested full-BF16 disk-offload environment.
The pin above documents the verified combination rather than a universal
requirement for every ComfyUI setup.

## Model Installation

Download the official checkpoint:

```bash
huggingface-cli download tencent/HunyuanImage-3.0-Instruct \
  --local-dir HunyuanImage-3.0-Instruct
```

The model can be stored directly under:

```text
ComfyUI/models/HunyuanImage-3.0-Instruct
```

A separate fast drive is usually preferable because the checkpoint is large
and disk traffic is continuous during generation.

For an external model directory, add a category to
`extra_model_paths.yaml`:

```yaml
comfyui:
  hunyuan_instruct: |
    D:/Models
```

With this example, the model directory is:

```text
D:/Models/HunyuanImage-3.0-Instruct
```

The Python loader does not require a hard-coded checkpoint path.

## Portable Disk-Offload Directory

The default auxiliary offload directory is selected by platform.

### Windows

```text
%LOCALAPPDATA%\ComfyUI\HunyuanImage3\disk_offload
```

### Linux

```text
$XDG_CACHE_HOME/ComfyUI/HunyuanImage3/disk_offload
```

or, when `XDG_CACHE_HOME` is unset:

```text
~/.cache/ComfyUI/HunyuanImage3/disk_offload
```

### macOS

```text
~/Library/Caches/ComfyUI/HunyuanImage3/disk_offload
```

Override the default without modifying Python.

### Windows BAT

```bat
set "HUNYUAN_DISK_OFFLOAD_DIR=D:\HunyuanDiskOffload"
```

### PowerShell

```powershell
$env:HUNYUAN_DISK_OFFLOAD_DIR = "D:\HunyuanDiskOffload"
```

### Linux or macOS

```bash
export HUNYUAN_DISK_OFFLOAD_DIR=/mnt/fastssd/HunyuanDiskOffload
```

A path entered directly in the loader's `disk_offload_dir` field takes
priority for that workflow.

## Why the Offload Directory May Be Empty

An empty auxiliary offload folder does not prove that disk offload is
inactive.

When the checkpoint consists of sharded `safetensors`, Transformers and
Accelerate can use the original checkpoint files as the disk backing store.
In that case:

- the configured offload directory may contain no large files;
- the original model drive shows substantial read activity;
- parameters not currently active remain on the PyTorch `meta` device;
- the log still reports modules mapped to `disk`.

Confirm the active mode from logs such as:

```text
Model distributed across: 0, cpu, disk
SSD disk offload active: 33 modules mapped to disk
```

The exact module count can vary with library versions, available memory, and
checkpoint structure.

## Recommended Loader Settings

For the tested 12 GB VRAM / 64 GB RAM machine:

| Loader option | Value |
|---|---|
| `quant_type` | `bf16` |
| `attention_impl` | `sdpa` |
| `moe_impl` | `eager` |
| `vram_reserve_gb` | `5` |
| `blocks_to_swap` | `0` |
| `moe_drop_tokens` | `true` |
| `vae_dtype` | `bfloat16` |
| `use_disk_offload` | `true` |
| `cpu_memory_limit_gb` | `16` |
| `disk_offload_dir` | Default or a fast local SSD path |

The disk-offload path is used only for BF16 when `blocks_to_swap=0`.
Existing quantized and block-swap modes follow their original loading paths.

## Initial Workflow Test

Use conservative settings for the first run:

| Generation option | Initial test value |
|---|---|
| `bot_task` | `image` |
| Steps | `4` |
| Input image | Approximately `512×512` where practical |
| Resolution | `Auto` or a modest model-native bucket |
| Number of input images | `1` |

The full Instruct model is not CFG-distilled. It uses CFG with batch size 2,
which doubles major inference tensors compared with the distilled model.

Increasing the number of diffusion steps increases runtime but usually does
not reduce peak memory. Test stability before increasing resolution or using
recaption modes.

## Expected Startup Logs

A successful low-memory load should include messages similar to:

```text
Loading BF16 Instruct model with SSD disk offload...
SSD offload GPU budget: 5.0 GiB
SSD offload CPU budget: 16 GiB
Loading checkpoint shards: 100%
Model distributed across: 0, cpu, disk
SSD disk offload active: ... modules mapped to disk
Loading tokenizer...
Compatibility fix: config.model_version was missing
Compatibility fix: aliased image_processor...
Compatibility fix: patched SigLIP2 positional embedding access...
Compatibility fix: enabled Accelerate submodule preloading for MultiheadAttention...
```

The important device-map condition is the presence of all three targets:

```text
0, cpu, disk
```

A `cpu, disk` map without GPU target `0` makes CPU the main execution device
and is not the intended tested configuration.

## Windows Pagefile Guidance

The tested machine used:

```text
98304 MB
```

for both the initial and maximum Windows pagefile size.

This is not a universal minimum. Required virtual memory depends on physical
RAM, checkpoint mapping, library versions, and other running applications.

Reducing the pagefile can work, but it also reduces the Windows commit limit.
During a cold model load, monitor:

```text
Task Manager → Performance → Memory → Committed
```

Keep substantial headroom between committed memory and the commit limit.
Windows error 1455 indicates that the paging file or commit limit is too
small for the requested allocation.

Restart Windows after changing the pagefile size.

## Resource Utilization and Performance

The tested configuration is I/O-bound.

Typical observations include:

- low or bursty GPU utilization;
- VRAM use far below the physical 12 GB capacity between module calls;
- moderate physical-RAM use despite a 160 GB checkpoint;
- heavy reads from the drive containing the checkpoint;
- several minutes per diffusion step.

This behavior is expected. Accelerate repeatedly materializes the active
module, performs its computation, and releases or offloads it.

Increasing `cpu_memory_limit_gb` from 16 to 24 or 32 may reduce disk traffic
when enough RAM is available. Test one change at a time and preserve virtual
memory headroom.

Do not increase the GPU model budget merely because average VRAM usage appears
low. Short-lived MoE, CFG, attention, and VAE peaks may occur between
monitoring samples.

## Compatibility Fixes Explained

### Missing `config.model_version`

Some Instruct configurations omit `model_version`, while the tokenizer loader
expects it. The fork assigns:

```text
HunyuanImage-3.0
```

only when the attribute is absent.

### Image-Processor Method Name

Some model-code revisions call:

```text
build_img_ratio_slice_logits_proc
```

while the loaded processor provides:

```text
build_img_ratio_slice_logits_processor
```

The fork adds a compatibility alias.

### SigLIP2 Positional Embeddings on `meta`

The model code reads positional-embedding weights directly. Direct access can
bypass an Accelerate child-module hook, leaving the tensor on `meta`.

The fork invokes the embedding module normally so Accelerate can materialize
and offload it correctly.

### `MultiheadAttention.out_proj` on `meta`

PyTorch's `MultiheadAttention` accesses the child `out_proj` weights inside
its functional implementation rather than calling `out_proj.forward()`.

The fork enables Accelerate submodule preloading on the parent attention
module, ensuring the child weights are real tensors during the operation.

## Troubleshooting

### `AttributeError: ... model_version`

Confirm that this fork's `hunyuan_instruct_nodes.py` is installed and that
ComfyUI was fully restarted.

### `build_img_ratio_slice_logits_proc` Is Missing

The public fork includes an alias for the longer processor method name.
A stale Python file or cached old custom-node installation is the usual cause.

### `Tensor on device meta is not on the expected device cuda:0`

Check the traceback.

If it references:

```text
position_embedding
```

the SigLIP2 compatibility patch did not load.

If it references:

```text
MultiheadAttention
out_proj_weight
```

the Accelerate submodule-preloading patch did not load.

Confirm the corresponding compatibility messages appear during model startup.

### Windows Error 1455

Increase the Windows pagefile, close memory-heavy applications, restart
Windows, and retry a cold load.

### CUDA Out of Memory

Start with:

- `bot_task=image`;
- one `512×512` reference image;
- four steps;
- `moe_drop_tokens=true`;
- a modest output bucket.

Recaption and `think_recaption` modes can require dramatically more memory
than direct image mode.

### Model Loads but Generation Appears Frozen

Check disk activity. A single step may take several minutes. Disk-backed
execution can remain silent between progress updates while tens of gigabytes
of weights are read.

### The Offload Folder Is Empty

This can be normal with directly mapped sharded safetensors. Verify the final
device map and disk-module count in the log.

## Unsupported or Discouraged Combinations

Avoid the following unless you are deliberately testing a new path:

- BF16 disk offload with `blocks_to_swap > 0`;
- Soft Unload on a model containing `meta` parameters;
- `think_recaption` on a 12 GB GPU;
- very large input or output resolutions on the first run;
- updating Transformers without preserving the last working environment;
- copying model weights into the Git repository.

## Publishing and Contributions

When publishing changes based on this fork:

- retain the original copyright and license notices;
- credit Eric Hiss / EricRollei as the original integration author;
- identify the low-memory work as an unofficial adaptation;
- document tested hardware and dependency versions;
- do not include Tencent model weights in the repository;
- avoid implying endorsement by Tencent, Hugging Face, ComfyUI, Accelerate, or
  the upstream project.

A clean contribution should include:

```text
hunyuan_instruct_nodes.py
LOW_MEMORY_BF16.md
README.md
tools/fix_hunyuan_accelerate_transformers.bat
patches/hunyuan_instruct_nodes_original_to_public.patch
```

## Credits

- Original ComfyUI integration: Eric Hiss / EricRollei
- HunyuanImage-3.0 model: Tencent Hunyuan Team
- Low-memory fork and testing: foxidermist
- Development and debugging assistance: OpenAI ChatGPT

This fork is unofficial and experimental.
