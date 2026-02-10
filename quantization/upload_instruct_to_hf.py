"""
Upload quantized HunyuanImage-3.0 Instruct models to Hugging Face.

Uploads 4 models:
  1. HunyuanImage-3.0-Instruct-INT8
  2. HunyuanImage-3.0-Instruct-NF4
  3. HunyuanImage-3.0-Instruct-Distil-INT8
  4. HunyuanImage-3.0-Instruct-Distil-NF4

Usage:
    python upload_instruct_to_hf.py                    # Upload all 4
    python upload_instruct_to_hf.py --model instruct-nf4  # Upload one
    python upload_instruct_to_hf.py --dry-run           # Preview without uploading

Expects HF_TOKEN in environment variables (or will prompt for login).
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, create_repo
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_USERNAME = "EricRollei"
MODEL_BASE_DIR = Path(r"H:\Testing")

MODELS = {
    "instruct-int8": {
        "folder": "HunyuanImage-3.0-Instruct-INT8",
        "repo_name": "HunyuanImage-3.0-Instruct-INT8",
        "display_name": "Hunyuan Image 3.0 Instruct — INT8 Quantized",
        "quant_type": "INT8",
        "quant_method": "INT8 per-channel quantization via bitsandbytes",
        "model_variant": "Instruct (Full)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct",
        "steps": 50,
        "cfg_distilled": False,
        "disk_size": "~81 GB",
        "vram_usage": "~80 GB weights + ~12-20 GB inference",
        "total_vram": "~92-100 GB",
        "description": (
            "INT8 quantization of the HunyuanImage-3.0 Instruct model. "
            "Supports text-to-image, image editing, multi-image fusion, "
            "and Chain-of-Thought prompt enhancement (recaption/think_recaption)."
        ),
        "tags": ["int8"],
    },
    "instruct-nf4": {
        "folder": "HunyuanImage-3.0-Instruct-NF4",
        "repo_name": "HunyuanImage-3.0-Instruct-NF4",
        "display_name": "Hunyuan Image 3.0 Instruct — NF4 Quantized",
        "quant_type": "NF4",
        "quant_method": "4-bit NormalFloat (NF4) quantization via bitsandbytes with double quantization",
        "model_variant": "Instruct (Full)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct",
        "steps": 50,
        "cfg_distilled": False,
        "disk_size": "~45 GB",
        "vram_usage": "~29 GB weights + ~12-20 GB inference",
        "total_vram": "~41-49 GB",
        "description": (
            "NF4 (4-bit) quantization of the HunyuanImage-3.0 Instruct model. "
            "Fits on a single 48GB GPU. Supports text-to-image, image editing, "
            "multi-image fusion, and Chain-of-Thought prompt enhancement."
        ),
        "tags": ["nf4", "4bit"],
    },
    "instruct-distil-int8": {
        "folder": "HunyuanImage-3.0-Instruct-Distil-INT8",
        "repo_name": "HunyuanImage-3.0-Instruct-Distil-INT8",
        "display_name": "Hunyuan Image 3.0 Instruct Distil — INT8 Quantized",
        "quant_type": "INT8",
        "quant_method": "INT8 per-channel quantization via bitsandbytes",
        "model_variant": "Instruct Distil (CFG-Distilled, 8-step)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct-Distil",
        "steps": 8,
        "cfg_distilled": True,
        "disk_size": "~81 GB",
        "vram_usage": "~80 GB weights + ~12-20 GB inference",
        "total_vram": "~92-100 GB",
        "description": (
            "INT8 quantization of the HunyuanImage-3.0 Instruct Distil model. "
            "CFG-distilled for ~6x faster generation (8 steps vs 50). "
            "Same quality as the full Instruct model with dramatically faster inference."
        ),
        "tags": ["int8", "distilled"],
    },
    "instruct-distil-nf4": {
        "folder": "HunyuanImage-3.0-Instruct-Distil-NF4",
        "repo_name": "HunyuanImage-3.0-Instruct-Distil-NF4",
        "display_name": "Hunyuan Image 3.0 Instruct Distil — NF4 Quantized",
        "quant_type": "NF4",
        "quant_method": "4-bit NormalFloat (NF4) quantization via bitsandbytes with double quantization",
        "model_variant": "Instruct Distil (CFG-Distilled, 8-step)",
        "base_model": "tencent/HunyuanImage-3.0-Instruct-Distil",
        "steps": 8,
        "cfg_distilled": True,
        "disk_size": "~45 GB",
        "vram_usage": "~29 GB weights + ~12-20 GB inference",
        "total_vram": "~41-49 GB",
        "description": (
            "NF4 (4-bit) quantization of the HunyuanImage-3.0 Instruct Distil model. "
            "The most accessible option — fits on a single 48GB GPU with ~6x faster "
            "generation (8 steps vs 50). Best balance of speed, quality, and VRAM."
        ),
        "tags": ["nf4", "4bit", "distilled"],
    },
}


# ── Model Card Template ───────────────────────────────────────────────────────

def make_model_card(info: dict) -> str:
    """Generate a HF model card (README.md) for a quantized model."""

    distil_section = ""
    if info["cfg_distilled"]:
        distil_section = (
            "\n### Distillation\n\n"
            "This is the **CFG-Distilled** variant, which means:\n"
            "- Only **8 diffusion steps** needed (vs 50 for the full Instruct model)\n"
            "- **~6x faster** image generation\n"
            "- No quality loss — distilled to match the full model's output\n"
            "- `cfg_distilled: true` in config means no classifier-free guidance needed\n"
        )

    if info["quant_type"] == "NF4":
        hw_note = (
            "- **Fits on a single 48GB GPU** (RTX 6000 Ada, RTX PRO 5000, A6000)\n"
            "- Consumer GPUs (RTX 4090/5090 24GB) — not enough VRAM\n"
        )
    else:
        hw_note = (
            "- **NVIDIA RTX 6000 Blackwell (96GB)** — fits entirely in VRAM ✅\n"
            "- **NVIDIA RTX 6000 Ada (48GB)** — requires CPU offloading\n"
            "- Multi-GPU setups with 80GB+ combined VRAM\n"
        )

    # Build YAML tags block (deduplicate)
    base_tags = [
        "Hunyuan", "hunyuan", "quantization", info["quant_type"].lower(),
        "comfyui", "custom nodes", "autoregressive", "Dit",
        "HunyuanImage-3.0", "instruct", "image-editing", "bitsandbytes",
    ]
    all_tags = list(dict.fromkeys(base_tags + info.get("tags", [])))  # preserve order, dedupe
    tags_yaml = "\n".join(f"- {t}" for t in all_tags)

    vram_weight = info["vram_usage"].split("+")[0].strip()
    vram_infer = info["vram_usage"].split("+")[1].strip()
    speed_note = "(CFG-distilled for speed)" if info["cfg_distilled"] else "(full quality)"

    card = f"""---
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt
base_model: {info['base_model']}
pipeline_tag: text-to-image
library_name: transformers
tags:
{tags_yaml}
---

# {info['display_name']}

{info['description']}

## Key Features

- 🎯 **Instruct model** — supports text-to-image, image editing, multi-image fusion
- 🧠 **Chain-of-Thought** — built-in `think_recaption` mode for highest quality
- 💾 **{info['quant_type']} quantized** — {info['disk_size']} on disk
- ⚡ **{info['steps']} diffusion steps** {speed_note}
- 🔧 **ComfyUI ready** — works with [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3) nodes

## VRAM Requirements

| Component | Memory |
|-----------|--------|
| Weight Loading | {vram_weight} |
| Inference (additional) | {vram_infer} |
| **Total** | **{info['total_vram']}** |

**Recommended Hardware:**

{hw_note}
## Model Details

- **Architecture:** HunyuanImage-3.0 Mixture-of-Experts Diffusion Transformer
- **Parameters:** 80B total, 13B active per token (top-K MoE routing)
- **Variant:** {info['model_variant']}
- **Quantization:** {info['quant_method']}
- **Diffusion Steps:** {info['steps']}
- **Default Guidance Scale:** 2.5
- **Resolution:** Up to 2048x2048
- **Language:** English and Chinese prompts
{distil_section}
## Quantization Details

**Layers quantized to {info['quant_type']}:**
- Feed-forward networks (FFN/MLP layers)
- Expert layers in MoE architecture (64 experts per layer)
- Large linear transformations

**Kept in full precision (BF16):**
- VAE encoder/decoder (critical for image quality)
- Attention projection layers (q_proj, k_proj, v_proj, o_proj)
- Patch embedding layers
- Time embedding layers
- Vision model (SigLIP2)
- Final output layers

## Usage

### ComfyUI (Recommended)

This model is designed to work with the [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3) custom nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Comfy_HunyuanImage3
```

1. Download this model to your ComfyUI models directory
2. Use the **"Hunyuan 3 Instruct Loader"** node
3. Select this model folder and choose `{info['quant_type'].lower()}` precision
4. Connect to the **"Hunyuan 3 Instruct Generate"** node for text-to-image
5. Or use **"Hunyuan 3 Instruct Edit"** for image editing
6. Or use **"Hunyuan 3 Instruct Multi-Fusion"** for combining multiple images

### Bot Task Modes

The Instruct model supports three generation modes:

| Mode | Description | Speed |
|------|-------------|-------|
| `image` | Direct text-to-image, prompt used as-is | Fastest |
| `recaption` | Model rewrites prompt into detailed description, then generates | Medium |
| `think_recaption` | CoT reasoning → prompt enhancement → generation (best quality) | Slowest |

## Original Model

This is a quantized derivative of [Tencent's HunyuanImage-3.0 Instruct](https://huggingface.co/{info['base_model']}).

- **Architecture:** Diffusion Transformer with Mixture-of-Experts
- **Resolution:** Up to 2048x2048
- **Language Support:** English and Chinese prompts
- **License:** [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt)

## Limitations

- Requires high-end professional GPU ({info['total_vram']} VRAM)
- {info['quant_type']} quantization may introduce minor quality differences in edge cases
- Loading time adds ~1-2 minutes overhead to first generation
- CoT/recaption modes require additional time for text generation phase

## Credits

- **Original Model:** [Tencent Hunyuan Team](https://huggingface.co/tencent)
- **Quantization:** Eric Rollei
- **ComfyUI Integration:** [Comfy_HunyuanImage3](https://github.com/EricRollei/Comfy_HunyuanImage3)

## License

This model inherits the license from the original Hunyuan Image 3.0 model:
[Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanImage-3.0/blob/main/LICENSE.txt)

Please review the original license for commercial use restrictions and requirements.

## Citation

```bibtex
@misc{{hunyuan-image-3-{info['quant_type'].lower()}-instruct,
  author = {{Rollei, Eric}},
  title = {{{info['display_name']}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{DEFAULT_USERNAME}/{info['repo_name']}}}}}
}}
```
"""
    return card


# ── Upload Logic ──────────────────────────────────────────────────────────────

def upload_model(key: str, info: dict, api: HfApi, username: str, dry_run: bool = False):
    """Upload a single model to Hugging Face."""
    folder_path = MODEL_BASE_DIR / info["folder"]
    repo_id = f"{username}/{info['repo_name']}"

    print(f"\n{'='*60}")
    print(f"📦 {info['display_name']}")
    print(f"   Folder: {folder_path}")
    print(f"   Repo:   {repo_id}")
    print(f"   Size:   {info['disk_size']}")
    print(f"{'='*60}")

    if not folder_path.exists():
        print(f"   ❌ Folder not found: {folder_path}")
        return False

    # Create model card
    readme_path = folder_path / "README.md"
    card_content = make_model_card(info)

    if dry_run:
        print(f"\n   📝 Model card preview (first 40 lines):")
        for i, line in enumerate(card_content.split("\n")[:40]):
            print(f"      {line}")
        print(f"      ... ({len(card_content.split(chr(10)))} total lines)")
        print(f"\n   🔍 DRY RUN — no upload performed")
        return True

    # Write README.md
    print(f"   📝 Writing model card...")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    # Create repo
    print(f"   📁 Creating repository '{repo_id}'...")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"   ✅ Repository ready")
    except Exception as e:
        print(f"   ❌ Error creating repo: {e}")
        return False

    # Upload
    print(f"   ⬆️  Uploading files ({info['disk_size']})...")
    print(f"   ⏳ This will take a while depending on your internet speed.")
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[
                ".git", ".DS_Store", "__pycache__", "*.bak",
                "*.pyc", ".gitattributes",
            ],
        )
        print(f"   🎉 Upload complete!")
        print(f"   🔗 https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload quantized HunyuanImage-3.0 Instruct models to Hugging Face"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to upload (default: all)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview model cards without uploading"
    )
    parser.add_argument(
        "--username", "-u",
        default=None,
        help=f"HF username (default: auto-detect, fallback: {DEFAULT_USERNAME})"
    )
    args = parser.parse_args()

    print("🚀 Hugging Face Model Uploader — HunyuanImage-3.0 Instruct Quantizations")
    print("=" * 70)

    # Authenticate
    if not args.dry_run:
        print("\n🔑 Authenticating...")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            print("   Found HF_TOKEN in environment")
            login(token=token)
        else:
            print("   No HF_TOKEN found — attempting interactive login")
            login()

    api = HfApi()

    # Determine username
    username = args.username
    if not username and not args.dry_run:
        try:
            username = api.whoami()["name"]
            print(f"   Logged in as: {username}")
        except Exception:
            username = DEFAULT_USERNAME
            print(f"   Could not detect username, using: {username}")
    elif not username:
        username = DEFAULT_USERNAME

    # Select models to upload
    if args.model == "all":
        to_upload = MODELS
    else:
        to_upload = {args.model: MODELS[args.model]}

    print(f"\n📋 Models to {'preview' if args.dry_run else 'upload'}: {len(to_upload)}")
    for key, info in to_upload.items():
        print(f"   • {info['display_name']} ({info['disk_size']})")

    if not args.dry_run:
        total_size = sum(
            45.4 if "NF4" in info["quant_type"] else 81.1
            for info in to_upload.values()
        )
        print(f"\n   Total upload size: ~{total_size:.0f} GB")
        confirm = input("\n   Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("   Cancelled.")
            return

    # Upload each model
    results = {}
    for key, info in to_upload.items():
        results[key] = upload_model(key, info, api, username, args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    print("📊 Summary:")
    for key, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {MODELS[key]['display_name']}")

    if not args.dry_run and all(results.values()):
        print(f"\n🎉 All uploads complete! View at:")
        for key in results:
            print(f"   https://huggingface.co/{username}/{MODELS[key]['repo_name']}")


if __name__ == "__main__":
    main()
