# Repository Setup Guide

## Current File Structure

```
Eric_Hunyuan3/
├── README.md                      # Main documentation
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── __init__.py                    # ComfyUI node registration
├── hunyuan_shared.py             # Shared utilities and cache management
├── hunyuan_full_bf16_nodes.py    # Full BF16 loader nodes
├── hunyuan_quantized_nodes.py    # NF4 quantized loader and generation nodes
└── quantization/
    ├── README.md                  # Quantization documentation
    ├── hunyuan_quantize_nf4.py   # Quantization script
    └── requirements_quantization.txt  # Quantization dependencies
```

## Files to Move to GitHub

### From `Eric_Hunyuan3/` folder:
✅ Already created:
- README.md
- LICENSE
- CONTRIBUTING.md
- requirements.txt
- .gitignore

✅ Existing (copy these):
- `__init__.py`
- `hunyuan_shared.py`
- `hunyuan_full_bf16_nodes.py`
- `hunyuan_quantized_nodes.py`

### From `hunyuan_quantization/` folder:
Create a `quantization/` subfolder and copy:
- `hunyuan_quantize_nf4.py`
- `requirements_quantization.txt` (if exists)
- Add the new `quantization/README.md` created above

## Setup Steps

### 1. Create GitHub Repository

```bash
# On GitHub website:
1. Click "New repository"
2. Name: Eric_Hunyuan3 (or ComfyUI-HunyuanImage3)
3. Description: "Professional ComfyUI nodes for HunyuanImage-3.0"
4. Public repository
5. Don't initialize with README (we have our own)
```

### 2. Initialize Local Repo

```bash
cd a:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Hunyuan3

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial release: HunyuanImage-3.0 ComfyUI nodes v1.0.0"

# Connect to GitHub
git remote add origin https://github.com/ericRollei/Eric_Hunyuan3.git

# Push
git branch -M main
git push -u origin main
```

### 3. Create Release

On GitHub website:
1. Go to "Releases" → "Create a new release"
2. Tag: `v1.0.0`
3. Title: "Initial Release - HunyuanImage-3.0 ComfyUI Nodes"
4. Description:
```markdown
## 🎉 Initial Release

Professional ComfyUI nodes for Tencent HunyuanImage-3.0.

### Features
- ✅ Multiple loading modes (BF16, NF4, Multi-GPU)
- ✅ Smart memory management
- ✅ Automatic prompt enhancement via DeepSeek
- ✅ Organized resolution controls
- ✅ Large image generation with CPU offload
- ✅ Comprehensive error handling

### Requirements
- ComfyUI
- 24GB+ VRAM (NF4) or 80GB+ (BF16)
- CUDA 12.8+

See README for full documentation.
```

### 4. Add Topics (on GitHub)

Click "⚙️ Settings" → "Topics":
- `comfyui`
- `comfyui-custom-nodes`
- `stable-diffusion`
- `image-generation`
- `hunyuan`
- `text-to-image`
- `pytorch`
- `quantization`

### 5. Optional: Add Screenshots

Create an `examples/` folder with:
- Example workflow screenshots
- Sample generated images
- UI screenshots

Add to README:
```markdown
## 📸 Examples

![Example Generation](examples/example_1.png)
*Prompt: "..."*
```

### 6. Optional: Add Badges

Top of README (already included):
```markdown
![HunyuanImage-3.0](https://img.shields.io/badge/HunyuanImage-3.0-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)
![License](https://img.shields.io/badge/License-MIT-orange)
```

## Before Publishing

### Checklist:
- [x] Update GitHub username in README links (✅ Set to ericRollei)
- [ ] Test installation from scratch
- [ ] Verify all imports work
- [ ] Check .gitignore excludes model files
- [ ] Review README for clarity
- [ ] Test quantization script
- [ ] Add example workflows (optional)
- [ ] Add screenshots (optional)

## After Publishing

### Promote Your Repo:
1. **ComfyUI Custom Node List**: Submit PR to [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. **Reddit**: Post on r/StableDiffusion, r/comfyui
3. **Discord**: Share in ComfyUI Discord server
4. **Twitter**: Tag @ComfyUI and @TencentHunyuan

### Maintain:
- Respond to issues promptly
- Accept helpful pull requests
- Keep README updated
- Tag new versions for updates

## Quick Commands Reference

```bash
# Check status
git status

# Add new files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push

# Create new release branch
git checkout -b release/v1.1.0

# Tag a version
git tag -a v1.1.0 -m "Version 1.1.0"
git push --tags
```

## Need Help?

- Git tutorial: https://guides.github.com/
- GitHub docs: https://docs.github.com/
- ComfyUI custom nodes guide: https://docs.comfy.org/

---

Good luck with your GitHub release! 🚀
