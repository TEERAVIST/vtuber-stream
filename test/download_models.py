#!/usr/bin/env python3
"""
Script to download required models for the VTuber chatbot
"""
import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, capture_output=True, text=True)
        print("[OK] Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False

def download_embedding_model():
    """Download the BGE embedding model"""
    model_dir = "models/embeds/bge-small-en-v1.5"
    os.makedirs(model_dir, exist_ok=True)
    
    # Try git clone first
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print("Downloading BGE embedding model...")
        if not run_command(f"git clone https://huggingface.co/BAAI/bge-small-en-v1.5 {model_dir}"):
            print("Failed to download with git clone. Trying alternative...")
            # Alternative: use huggingface_hub
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="BAAI/bge-small-en-v1.5", local_dir=model_dir)
                print("[OK] Downloaded with huggingface_hub!")
            except ImportError:
                print("Please install huggingface_hub: pip install huggingface_hub")
                return False
            except Exception as e:
                print(f"Failed to download: {e}")
                return False
    else:
        print(f"Embedding model already exists at {model_dir}")
    
    return True

def download_reranker_model():
    """Download the BGE reranker model"""
    model_dir = "models/reranker/bge-reranker-base"
    os.makedirs(model_dir, exist_ok=True)
    
    # Try git clone first
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print("Downloading BGE reranker model...")
        if not run_command(f"git clone https://huggingface.co/BAAI/bge-reranker-base {model_dir}"):
            print("Failed to download with git clone. Trying alternative...")
            # Alternative: use huggingface_hub
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="BAAI/bge-reranker-base", local_dir=model_dir)
                print("[OK] Downloaded with huggingface_hub!")
            except ImportError:
                print("Please install huggingface_hub: pip install huggingface_hub")
                return False
            except Exception as e:
                print(f"Failed to download: {e}")
                return False
    else:
        print(f"Reranker model already exists at {model_dir}")
    
    return True

def main():
    print("=== VTuber Chatbot Model Downloader ===")
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git is not installed. Please install git first.")
        return False
    
    # Download models
    success = True
    success &= download_embedding_model()
    success &= download_reranker_model()
    
    if success:
        print("\n[OK] All models downloaded successfully!")
        print("You can now run the chatbot with:")
        print("python run.py --model \"models/merged_model-20251017T181201Z-1-001/merged_model\" --ref assets/voice_ref.wav --no-thinking --device cuda:0 --base-speaker-key EN-US --lang EN --kb kb --embed-model \"models/embeds/bge-small-en-v1.5\" --reranker-model \"models/reranker/bge-reranker-base\" --reindex")
    else:
        print("\n[ERROR] Some models failed to download. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)