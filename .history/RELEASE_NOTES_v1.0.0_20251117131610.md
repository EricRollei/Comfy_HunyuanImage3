# 🎉 Initial Release - HunyuanImage-3.0 ComfyUI Nodes

Professional ComfyUI custom nodes for Tencent HunyuanImage-3.0, the powerful 80B parameter native multimodal image generation model.

## ✨ Features

- **Multiple Loading Modes**: Full BF16, NF4 Quantized, Single GPU, Multi-GPU
- **Smart Memory Management**: Automatic VRAM tracking and cleanup
- **High-Quality Generation**: Standard (<2MP) and Large (2MP-8MP+) image support
- **Advanced Prompting**: Official HunyuanImage-3.0 prompt enhancement with OpenAI-compatible LLM APIs
- **Professional Resolution Control**: Organized presets with megapixel indicators

## 📋 Requirements

- **ComfyUI** installed and working
- **NVIDIA GPU**: 24GB+ VRAM for NF4, 80GB+ for BF16
- **Python 3.10+**, **PyTorch 2.7+**, **CUDA 12.8+**

## 🚀 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Comfy_HunyuanImage3.git
cd Comfy_HunyuanImage3
pip install -r requirements.txt
```

See [README.md](https://github.com/EricRollei/Comfy_HunyuanImage3#readme) for full documentation.

## 🙏 Credits

This project integrates the **HunyuanImage-3.0** model developed by **Tencent Hunyuan Team** under Apache 2.0 license.
