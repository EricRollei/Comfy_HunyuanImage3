import os
from huggingface_hub import HfApi, login, create_repo
from pathlib import Path

# Configuration
DEFAULT_REPO_NAME = "HunyuanImage-3-NF4-ComfyUI" 

def upload_model():
    print("🚀 Hugging Face Model Uploader for HunyuanImage-3 NF4")
    print("----------------------------------------------------")

    # 1. Login
    print("\nStep 1: Authentication")
    print("If you haven't logged in, you'll need your HF Token (Write permission).")
    print("Get it here: https://huggingface.co/settings/tokens")
    try:
        login()
    except Exception as e:
        print(f"Login failed or cancelled: {e}")
        return

    api = HfApi()
    try:
        user = api.whoami()['name']
    except:
        print("❌ Could not determine username. Please ensure you are logged in.")
        return

    full_repo_id = f"{user}/{DEFAULT_REPO_NAME}"
    
    print(f"\nTarget Repository: {full_repo_id}")
    confirm = input("Is this correct? (y/n): ").lower()
    if confirm != 'y':
        full_repo_id = input("Enter the full Repo ID (e.g., username/model-name): ").strip()

    # 2. Resolve Path
    # We try to locate the model relative to this script or the ComfyUI structure
    # Script is likely in custom_nodes/Eric_Hunyuan3/quantization/
    
    # Try standard ComfyUI path relative to this script
    # .../ComfyUI/custom_nodes/Eric_Hunyuan3/quantization/upload_nf4_to_hf.py
    # -> .../ComfyUI/models/HunyuanImage-3-NF4
    
    script_dir = Path(__file__).parent.resolve()
    
    # Strategy 1: Relative to script in custom_nodes
    possible_path_1 = script_dir.parent.parent.parent / "models" / "HunyuanImage-3-NF4"
    
    # Strategy 2: Check if we are in the root of the repo
    possible_path_2 = script_dir / "HunyuanImage-3-NF4"
    
    base_path = None
    if possible_path_1.exists():
        base_path = possible_path_1
    elif possible_path_2.exists():
        base_path = possible_path_2
    
    if not base_path:
        print(f"\n⚠️ Could not automatically find the model folder.")
        print(f"Checked: {possible_path_1}")
        manual_path = input("Please enter the full path to your NF4 model folder: ").strip()
        base_path = Path(manual_path)
        if not base_path.exists():
            print("❌ Path does not exist. Aborting.")
            return

    print(f"\n✅ Found model at: {base_path}")

    # 3. Create Model Card (README.md)
    readme_path = base_path / "README.md"
    if not readme_path.exists():
        print("\n📝 Creating Model Card (README.md)...")
        model_card = f"""---
license: apache-2.0
base_model: tencent/HunyuanImage-3.0
tags:
- hunyuan
- comfyui
- quantization
- nf4
- text-to-image
---

# HunyuanImage-3.0 (NF4 Quantization)

This is a 4-bit (NF4) quantized version of the [Tencent HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) model, optimized for use with ComfyUI.

## License & Attribution

This model is based on the original work by the Tencent Hunyuan Team.
Original Repository: [https://github.com/Tencent-Hunyuan/HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
Original License: **Apache 2.0**

This quantized version is distributed under the same Apache 2.0 license.

## Usage

This model is designed to be used with the [Eric_Hunyuan3 ComfyUI Custom Nodes](https://github.com/ericRollei/Eric_Hunyuan3).

1. Install the custom nodes.
2. Place this model in your `ComfyUI/models/HunyuanImage-3-NF4` folder.
"""
        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(model_card)
            print("✅ Created README.md with attribution.")
        except Exception as e:
            print(f"⚠️ Could not write README.md: {e}")
            print("Continuing with upload...")

    # 4. Create Repo
    print(f"\nStep 2: Creating Repository '{full_repo_id}'...")
    try:
        create_repo(full_repo_id, repo_type="model", exist_ok=True)
        print("✅ Repository ready.")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return

    # 5. Upload
    print(f"\nStep 3: Uploading files from {base_path}...")
    print("This may take a while depending on your internet speed (approx 20-40GB).")
    
    try:
        api.upload_folder(
            folder_path=str(base_path),
            repo_id=full_repo_id,
            repo_type="model",
            ignore_patterns=[".git", ".DS_Store", "__pycache__", "*.bak"],
        )
        print("\n🎉 Upload Complete!")
        print(f"View your model here: https://huggingface.co/{full_repo_id}")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_model()
