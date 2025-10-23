#!/bin/bash

# Script to run run.py with Linux-compatible paths
# Make sure to copy your merged model to the specified path before running

# Set the working directory to the vtuber-stream directory
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Set environment variables
export NLTK_DATA="$(pwd)/nltk_data"
export HF_HOME="$(pwd)/hf_cache"
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# Run the application with Linux-compatible paths
# Note: Audio will be saved to files in tmp_audio/ directory
# You can play these files manually or use the listen_to_audio.py script
python run.py \
  --model "models/merged_model-20251017T181201Z-1-001/merged_model" \
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