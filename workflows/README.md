# Example Workflows

These workflows demonstrate the experimental full-BF16 CUDA/CPU/disk loading
path in the `foxidermist/Comfy_HunyuanImage3` fork.

## Files

- `HunyuanImage3_Instruct_BF16_DiskOffload_TextToImage.json`
- `HunyuanImage3_Instruct_BF16_DiskOffload_ImageToImage.json`

Import either JSON by dragging it into the ComfyUI canvas or using
**Workflow → Open**.

## Saved Loader Settings

| Setting | Value |
|---|---|
| `vram_reserve_gb` | `8` |
| `cpu_memory_limit_gb` | `32` |
| `blocks_to_swap` | `0` |
| `use_disk_offload` | `true` |
| `moe_drop_tokens` | `true` |
| `vae_dtype` | `bfloat16` |

The `disk_offload_dir` widget is intentionally saved as an empty string. The
public loader interprets an empty value as its platform-specific default cache
directory. Users can enter a different fast local SSD directory directly in
the loader.

On GPUs below 24 GiB, the loader may clamp the requested 8 GiB VRAM reserve to
its low-memory value so that at least one indivisible model module can remain
assigned to CUDA. This is expected and is reported in the console.

## Saved Generation Settings

| Setting | Value |
|---|---|
| `bot_task` | `image` |
| `steps` | `40` |
| `guidance_scale` | `-1` (model default) |
| `flow_shift` | `2.8` |
| `system_prompt` | `dynamic` |

With `bot_task=image`, the dynamic system-prompt selection disables the
recaption prompt and uses direct image generation/editing.

Forty steps are intended as a full run, not a smoke test. Disk-backed
generation can take well over an hour on a 12 GB GPU.

## Model Selection

The workflow stores the folder name:

```text
HunyuanImage-3.0-Instruct
```

If ComfyUI displays a location suffix such as:

```text
HunyuanImage-3.0-Instruct [Models]
```

select that entry manually in the loader after importing the workflow.

## Text-to-Image Workflow

The text-to-image example uses the model-native:

```text
1024x1024 (1:1 Square)
```

resolution. Full Instruct uses CFG batch size 2, so this can exceed the memory
available on some 12 GB GPUs. Test with fewer steps first when validating a new
machine. Step count changes runtime, not the main resolution-dependent memory
peak.

## Image-to-Image Workflow

The input image is scaled to `512×512` before editing to reduce peak memory.
The edit node uses `Auto (model predicts)` with `align_output_size=true`, so the
output is aligned to the scaled input where supported.

Remove or modify the `ImageScale` node when using larger hardware.

## Expected Loader Log

A working disk-backed load should include:

```text
Model distributed across: 0, cpu, disk
SSD disk offload active: ... modules mapped to disk
```

The auxiliary offload directory may remain empty when Accelerate reads tensors
directly from the original sharded safetensors.

See the repository-level `LOW_MEMORY_BF16.md` for setup and troubleshooting.
