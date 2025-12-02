Caching Hunyuan model for reuse: A:\Comfy25\ComfyUI_windows_portable\ComfyUI\models\HunyuanImage-3-INT8
✓ INT8 model loaded - VRAM Allocated: 80.81GB, Reserved: 81.09GB
  Model distributed across GPU + CPU as needed
✓ INT8 model ready for generation with budget-aware nodes
============================================================
LARGE IMAGE GENERATION MODE (Budget)
Resolution: 1152x864
Offload Mode: disabled -> Disabled
Starting Memory: RAM=2.16GB, VRAM=85.61GB / 95.26GB
============================================================
Applied resolution patch to bypass bucket snapping
Generating image with 40 steps
Rewriting prompt using LLM API (style: en_recaption)...
Calling LLM API: https://api.deepseek.com/v1/chat/completions (model: deepseek-chat)
Original prompt: A detailed painting in the Italian Renaissance style, reminiscent of Hieronymous...
Rewritten prompt: A detailed painting in the Italian Renaissance style, reminiscent of Hieronymous...
Prompt: A detailed painting in the Italian Renaissance style, reminiscent of Hieronymous Bosch's art style. ...
Applied resolution patch to bypass bucket snapping
Using custom resolution: 1152x864
Guidance scale: 7.0
100%|██████████████████████████████████████████████████████████████████████████████████| 40/40 [02:35<00:00,  3.90s/it]
Restored original resolution logic
Converting image (mode: RGB, size: (1152, 864))
✓ Image generated successfully - tensor shape: torch.Size([1, 864, 1152, 3])
GPU memory after generation: 95.26GB used, 0.00GB free
============================================================
✓ Large image generated successfully (budget node)
MEMORY: Peak VRAM 92.2GB | VRAM Start 85.6GB → End 95.3GB | Peak RAM 331.3GB
============================================================