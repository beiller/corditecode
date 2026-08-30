#!/usr/bin/env python3
"""
Script to download a GGUF model from Hugging Face.
Uses local_dir to get a clean copy instead of cache symlinks.
"""
import json
import os
from pathlib import Path
from dataclasses import asdict


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        from huggingface_hub import hf_hub_download
        return True
    except ImportError:
        print("❌ huggingface_hub not installed!")
        print("   Install it with: pip install huggingface_hub")
        return False



def search_model(search_query: str, limit: int = 10):
    """Search huggingface for a model"""
    from huggingface_hub import list_models

    models = [asdict(m) for m in list_models(search=search_query, limit=limit)]
    return json.dumps({"result": models}, default=str)




def download_model(
    repo_id: str, 
    filename: str
):
    """Download a GGUF model from Hugging Face. Provide the repo_id and filename. Feel free to search huggingface first (curl) to find the appropriate values. """
    from huggingface_hub import hf_hub_download
    
    # Default save directory
    save_dir = Path(os.getenv("LLAMA_MODELS_DIR"))
    
    # Create models directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / Path(filename)).exists():
        if os.environ.get("SILENT") == "1":
            print("⚠️ File already exists, skipping download (silent mode).")
            return
        print(f"\n⚠️ File Already Exists! Redownload [y/N]?")
        if input().lower() != "y":
            return
    
    print("=" * 60)
    print("📥 Downloading GGUF Model from Hugging Face")
    print("=" * 60)
    print(f"\n📦 Repository: {repo_id}")
    print(f"📄 Filename: {filename}")
    print(f"💾 Save to: {save_dir.absolute()}")
    
    # Download the model file with local_dir_copy=True for a clean copy
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=save_dir,
        force_download=True,
        # This ensures we get the actual file, not a symlink
        local_dir=str(save_dir)
    )
    
    print(f"\n✅ Download completed!")
    print(f"   File saved to: {local_path}")
    
    # Get actual file size
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024*1024)
        print(f"   File size: {size_mb:.2f} MB")
    else:
        print(f"\n❌ Error downloading model.")
        return None
    return local_path



if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv(".env.defaults", override=False)
    
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)

    MODEL_DOWNLOAD_REPO_ID = os.environ.get("MODEL_DOWNLOAD_REPO_ID") or "bartowski/Qwen3.8-27B-GGUF"
    MODEL_DOWNLOAD_FILENAME = os.environ.get("MODEL_DOWNLOAD_FILENAME") or "Qwen3.8-27B-Q4_K_M.gguf"

    download_model(MODEL_DOWNLOAD_REPO_ID, MODEL_DOWNLOAD_FILENAME)
