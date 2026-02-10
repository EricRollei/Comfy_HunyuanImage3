# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-02-09

### Added
- **Instruct Model Nodes**: 5 new nodes for HunyuanImage-3.0-Instruct and Instruct-Distil models
  - **Hunyuan Instruct Loader**: Load any Instruct variant (BF16/INT8/NF4, Distil/Full). Auto-detects quant type from folder name.
  - **Hunyuan Instruct Generate**: Text-to-image with bot_task modes (image/recaption/think_recaption). Returns CoT reasoning text.
  - **Hunyuan Instruct Image Edit**: Edit images with natural language instructions.
  - **Hunyuan Instruct Multi-Image Fusion**: Combine 2–3 reference images with instructions.
  - **Hunyuan Instruct Unload**: Free cached Instruct model from VRAM/RAM.
- **Block Swap**: Async GPU↔CPU transformer block swapping for all loaders. Enables running BF16 (~160GB) and INT8 (~81GB) models on 48–96GB GPUs.
- **HighRes Efficient Node**: Loop-based MoE expert routing uses ~75× less VRAM than dispatch_mask. Generates 3MP–4K+ images on 96GB GPUs.
- **Unified V2 Node**: Single auto-detecting generate node with integrated block swap, VAE management, and VRAM budget.
- **Flexible Model Paths**: All loaders now use ComfyUI's `folder_paths` system. Models can be stored anywhere via `extra_model_paths.yaml` (`hunyuan` and `hunyuan_instruct` categories).
- **Pre-quantized Instruct models** on Hugging Face: INT8 and NF4 variants for both Instruct and Instruct-Distil.
- **INT8 bitsandbytes fix**: Guard hooks that fix `Module._apply` discarding `Int8Params.CB/SCB` during `.to()` calls. Enables block swap with INT8 models.
- **Soft Unload node**: Move model to CPU (keep cached) for fast restore without full reload.
- **Force Unload node**: Complete VRAM + RAM cleanup with aggressive garbage collection.
- **Clear Downstream node**: Clear other models from VRAM while preserving cached Hunyuan model.

### Changed
- Instruct Loader model discovery uses `folder_paths.get_folder_paths()` instead of hardcoded paths
- All base loaders (NF4, INT8, BF16, Multi-GPU, HighRes) migrated to centralized `get_available_hunyuan_models()` and `resolve_hunyuan_model_path()` in `hunyuan_shared.py`
- Updated README with comprehensive Instruct documentation, HuggingFace links, hardware tables, and workflow diagrams

### Known Issues
- **Instruct (full) INT8 with block swap**: OOM during inference. Distil-INT8 works fine. Under investigation.
- **RAM accumulation**: Successive model loads may leak RAM. Restart ComfyUI if needed.

## [Unreleased]

### Added
- **Rewritten Prompt Output**: Both `HunyuanImage3Generate` and `HunyuanImage3GenerateLarge` now output the rewritten prompt used for generation
  - Useful for saving to EXIF metadata
  - Can be reused for regeneration or variations
  - Contains the LLM-enhanced prompt when prompt rewriting is enabled
- **Status Output**: Both generation nodes now provide a status message indicating:
  - Whether prompt rewriting was used and which style
  - If prompt rewriting failed with error message
  - Large image mode settings (CPU offload status)

### Changed
- Generation nodes now return 3 outputs: `(image, rewritten_prompt, status)` instead of just `(image,)`
- Status messages provide better feedback about generation settings

### Fixed
- **Low VRAM NF4 Loader**: Resolved validation errors on 24GB/32GB cards by implementing a custom device map strategy that forces NF4 layers to GPU while allowing other components to offload to CPU.
- **Device Mapping**: Added logic to prevent `bitsandbytes` from seeing 4-bit layers on CPU, which was causing crashes in Low VRAM mode.

### Technical Details
- `rewritten_prompt`: STRING - The final prompt used for generation (either original or LLM-rewritten)
- `status`: STRING - Human-readable status message about the generation process

## [1.0.0] - 2024-11-18

### Initial Release
- Full BF16 and NF4 quantized model loading
- Multi-GPU support with smart memory management
- Official HunyuanImage-3.0 prompt enhancement with LLM APIs
- Large image generation with CPU offload
- Professional resolution presets with megapixel indicators

## [Low VRAM Fix] - 2024-11-19

### Fixed Low VRAM NF4 Loader
- Resolved validation errors on 24GB/32GB cards by implementing a custom device map strategy that forces NF4 layers to GPU while allowing other components to offload to CPU.

### Enhanced Device Mapping
- Added logic to prevent `bitsandbytes` from seeing 4-bit layers on CPU, which was causing crashes in Low VRAM mode.
