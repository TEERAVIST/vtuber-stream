#!/bin/bash

# Script to run run.py with the same parameters as used in Windows
# Adapted for Linux environment

# Set the working directory to the vtuber-stream directory
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Set environment variables
export NLTK_DATA="$(pwd)/nltk_data"
export HF_HOME="$(pwd)/hf_cache"
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# Model paths - you'll need to copy your merged model to this location
MODEL_PATH="models/merged_model-20251017T181201Z-1-001/merged_model"

# Check if model exists, if not use fallback
if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] Model not found at $MODEL_PATH"
    echo "Please copy your merged model from Windows to this location:"
    echo "  mkdir -p models/merged_model-20251017T181201Z-1-001"
    echo "  # Then copy the merged_model folder there"
    echo ""
    echo "For now, using fallback model: Qwen/Qwen2.5-0.5B-Instruct"
    export TRANSFORMERS_OFFLINE=0
    export HF_HUB_OFFLINE=0
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
fi

# Run the application
# Note: Audio will be saved to files in tmp_audio/ directory
# You can play these files manually or use the listen_to_audio.py script
python run.py \
  --model "$MODEL_PATH" \
  --ref assets/voice_ref.wav \
  --no-thinking \
  --device cuda:0 \
  --base-speaker-key EN-US \
  --lang EN \
  --kb kb \
  --embed-model "models/embeds/bge-small-en-v1.5" \
  --reranker-model "models/reranker/bge-reranker-base" \
  --reindex \
  --headless