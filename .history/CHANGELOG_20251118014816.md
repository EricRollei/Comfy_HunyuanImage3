# Changelog

All notable changes to this project will be documented in this file.

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
