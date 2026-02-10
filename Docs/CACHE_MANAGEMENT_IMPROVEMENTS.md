# Cache Management Improvements

## Summary
This update improves the cache management system for Hunyuan Image 3 custom nodes, making it more intuitive and allowing better control over when models are cleared from VRAM.

## Changes Made

### 1. Removed Non-Functional `keep_in_cache` Parameter
**Problem**: The `keep_in_cache` boolean parameter on loader nodes didn't actually control caching behavior - models were always kept in cache regardless of the setting.

**Solution**: Removed the `keep_in_cache` parameter from all loader nodes:
- `HunyuanImage3QuantizedLoader` (NF4 loader)
- `HunyuanImage3FullLoader`
- `HunyuanImage3FullGPULoader`
- `HunyuanImage3MultiGPULoader`
- `HunyuanImage3SingleGPU88GB`

Models are now always cached for reuse, which is the desired behavior for performance.

### 2. Simplified Cache Storage Logic
**Updated**: `HunyuanModelCache.store()` method in `hunyuan_shared.py`
- Removed the `keep` parameter
- Method now always caches models for reuse
- Simplified the logic and removed unnecessary branching

### 3. Added Trigger Output to Generation Nodes
**Updated**: Both generation nodes now output a trigger signal
- `HunyuanImage3Generate`: Added `trigger` output (4th output)
- `HunyuanImage3GenerateLarge`: Added `trigger` output (4th output)

**Purpose**: The trigger output allows you to sequence the execution of other nodes (like cache clearing) to run after image generation completes.

**Return Values**:
- `image`: The generated image tensor
- `rewritten_prompt`: The prompt used (original or rewritten)
- `status`: Status message about generation
- `trigger`: Boolean value (True) to signal completion

### 4. Added Optional Input to Unload Node
**Updated**: `HunyuanImage3Unload` node in `hunyuan_shared.py`
- Added optional `trigger` input parameter
- Node can now be connected after generation nodes in the workflow
- Allows precise control over when VRAM is cleared

**Usage**: Connect the `trigger` output from a generation node to the `trigger` input of the Unload node. This ensures the cache is only cleared after image generation completes.

## Benefits

1. **Clearer Intent**: Removed confusing non-functional parameters
2. **Better Workflow Control**: You can now control exactly when cache clearing happens in your workflow
3. **Proper Sequencing**: Use the trigger output/input to ensure unload operations happen at the right time
4. **Consistent Behavior**: All loaders now behave the same way regarding caching

## Migration Guide

### For Existing Workflows

**Before**:
```
[Loader Node] -> [Generation Node]
  └─ keep_in_cache: True/False (had no effect)
```

**After**:
```
[Loader Node] -> [Generation Node] -> [Unload Node]
                        └─ trigger ────────┘
```

### Example Workflow Pattern

1. **Load Model**: Use any loader node (no more `keep_in_cache` parameter)
2. **Generate Image**: Use `HunyuanImage3Generate` or `HunyuanImage3GenerateLarge`
3. **Clear Cache** (optional): Connect the `trigger` output from the generation node to the `trigger` input of `HunyuanImage3Unload` to clear VRAM after generation

### When to Use the Unload Node

- **Use it**: When you need to free up VRAM for other models or processes after generation
- **Skip it**: If you're generating multiple images in sequence - keeping the model cached improves performance
- **Use with trigger**: Connect it after generation nodes to ensure proper execution order

## Technical Details

### Trigger System
- The trigger output uses the special `"*"` type in ComfyUI, which accepts any input type
- This allows flexible connections in the node graph
- The trigger value is `True` when generation completes successfully
- The optional input on the Unload node ensures it waits for the connected node to execute first

### Cache Behavior
- Models remain in cache by default for best performance on multiple generations
- Use the Unload node to explicitly clear cache when needed
- Cache clearing includes:
  - Moving model to CPU
  - Deleting model references
  - Running garbage collection
  - Clearing CUDA cache on all GPUs
