#!/bin/bash

# Script to run the VTuber application with ngrok tunneling for audio streaming

# Set the working directory to the vtuber-stream directory
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Set environment variables
export NLTK_DATA="$(pwd)/nltk_data"
export HF_HOME="$(pwd)/hf_cache"
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# Model paths
MODEL_PATH="models/merged_model-20251017T181201Z-1-001/merged_model"

# Check if model exists, if not use fallback
if [ ! -d "$MODEL_PATH" ]; then
    echo "[WARN] Model not found at $MODEL_PATH"
    echo "Using fallback model: Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
fi

# Function to start ngrok tunnel
start_ngrok() {
    echo "[NGROK] Starting ngrok tunnel for port 8765..."
    
    # Use ngrok config from /workspace to persist across RunPod sessions
    export NGROK_CONFIG_PATH="/workspace/.config/ngrok/ngrok.yml"
    
    ./ngrok http 8765 --log=stdout --config=$NGROK_CONFIG_PATH > ngrok.log 2>&1 &
    NGROK_PID=$!
    echo "[NGROK] ngrok started with PID: $NGROK_PID"
    
    # Wait for ngrok to start and get the public URL
    sleep 5
    
    # Extract the public URL from ngrok
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)
    
    if [ -z "$NGROK_URL" ]; then
        echo "[ERROR] Could not get ngrok URL. Check ngrok.log for details."
        kill $NGROK_PID
        exit 1
    fi
    
    echo "[INFO] ngrok tunnel established at: $NGROK_URL"
    echo "[INFO] Use this URL in your client: ${NGROK_URL/http/ws}"
}

# Function to start the audio streaming server
start_streaming_server() {
    echo "[STREAMING] Starting audio streaming server..."
    python audio_streamer.py &
    STREAMING_PID=$!
    echo "[STREAMING] Audio streaming server started with PID: $STREAMING_PID"
    sleep 2  # Give the server time to start
}

# Function to start the VTuber application
start_vtuber() {
    echo "[VTUBER] Starting VTuber application..."
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
}

# Function to cleanup on exit
cleanup() {
    echo "[CLEANUP] Stopping processes..."
    if [ ! -z "$NGROK_PID" ]; then
        kill $NGROK_PID 2>/dev/null
        echo "[CLEANUP] ngrok stopped"
    fi
    if [ ! -z "$STREAMING_PID" ]; then
        kill $STREAMING_PID 2>/dev/null
        echo "[CLEANUP] Streaming server stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Install required packages for streaming
echo "[SETUP] Installing websockets for audio streaming..."
pip install websockets > /dev/null 2>&1

# Set up ngrok configuration for persistence
./setup_ngrok.sh

# Start ngrok first
start_ngrok

# Start the streaming server
start_streaming_server

# Start the VTuber application
start_vtuber