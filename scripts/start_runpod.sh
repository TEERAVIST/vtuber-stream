#!/usr/bin/env bash
set -euo pipefail

# Simple bootstrap for RunPod Pod environments.
# - Creates a venv
# - Installs requirements
# - Runs the app in headless mode (saves audio to tmp_audio/)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install --no-cache-dir -r requirements.txt

# Ensure git lfs for model pulls
if command -v git >/dev/null 2>&1; then
  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[SETUP] Installing git-lfs …"
    (sudo apt-get update && sudo apt-get install -y git-lfs) || true
  fi
  git lfs install || true
fi

# Environment (local-only HF cache + NLTK path)
export NLTK_DATA="$PROJECT_DIR/nltk_data"
export HF_HOME="$PROJECT_DIR/hf_cache"
mkdir -p "$HF_HOME" "$NLTK_DATA"

# We'll default to offline only if a local LLM is present; otherwise allow HF download.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Try to find a merged HF model under models/*/merged_model
MODEL_DIR=""
if [ -d "$PROJECT_DIR/models" ]; then
  while IFS= read -r -d '' p; do MODEL_DIR="$p"; break; done < <(find "$PROJECT_DIR/models" -type d -name merged_model -print0 | head -z -n1 || true)
fi

if [ -z "$MODEL_DIR" ]; then
  echo "[WARN] No local merged_model found under models/. Will use a small HF model."
  # Allow remote download for a small instruct model
  export TRANSFORMERS_OFFLINE=0
  export HF_HUB_OFFLINE=0
fi

EMBED_DIR="$PROJECT_DIR/models/embeds/bge-small-en-v1.5"
RERANK_DIR="$PROJECT_DIR/models/reranker/bge-reranker-base"
REF_WAV="$PROJECT_DIR/assets/voice_ref.wav"

# Auto-fetch OpenVoice V2 checkpoints if missing
if [ ! -f "$PROJECT_DIR/checkpoints_v2/checkpoints_v2/converter/checkpoint.pth" ]; then
  echo "[SETUP] Fetching OpenVoiceV2 converter + base speakers …"
  tmpdir=$(mktemp -d)
  git clone --depth 1 https://huggingface.co/myshell-ai/OpenVoiceV2 "$tmpdir/openvoicev2" || true
  mkdir -p "$PROJECT_DIR/checkpoints_v2/checkpoints_v2"
  cp -r "$tmpdir/openvoicev2/converter" "$PROJECT_DIR/checkpoints_v2/checkpoints_v2/" || true
  cp -r "$tmpdir/openvoicev2/base_speakers" "$PROJECT_DIR/checkpoints_v2/checkpoints_v2/" || true
  rm -rf "$tmpdir"
fi

# Auto-fetch embedding model
if [ ! -f "$EMBED_DIR/model.safetensors" ] && [ ! -f "$EMBED_DIR/pytorch_model.bin" ]; then
  echo "[SETUP] Fetching embedding model BAAI/bge-small-en-v1.5 …"
  mkdir -p "$PROJECT_DIR/models/embeds"
  git clone --depth 1 https://huggingface.co/BAAI/bge-small-en-v1.5 "$EMBED_DIR" || true
fi

# Auto-fetch reranker model (optional; used when present)
if [ ! -f "$RERANK_DIR/model.safetensors" ] && [ ! -f "$RERANK_DIR/pytorch_model.bin" ]; then
  echo "[SETUP] Fetching reranker model BAAI/bge-reranker-base …"
  mkdir -p "$PROJECT_DIR/models/reranker"
  git clone --depth 1 https://huggingface.co/BAAI/bge-reranker-base "$RERANK_DIR" || true
fi

# Ensure ref wav is optional — if missing we pass 'none' to run.py
if [ ! -f "$REF_WAV" ]; then
  echo "[INFO] No assets/voice_ref.wav found. Running without cloning (base voice only)."
  REF_WAV="none"
fi

DEVICE=${DEVICE:-"cuda:0"}
LANG=${LANG:-"EN"}
BASE_SPK=${BASE_SPK:-"EN-US"}
SR=${SR:-48000}
SPEED=${SPEED:-0.95}

set -x
python run.py \
  --ref "$REF_WAV" \
  --device "$DEVICE" \
  --lang "$LANG" \
  --base-speaker-key "$BASE_SPK" \
  --sr "$SR" \
  --speed "$SPEED" \
  --kb "kb" \
  --embed-model "$EMBED_DIR" \
  --reranker-model "$RERANK_DIR" \
  --top-k 6 \
  --rerank-k 4 \
  ${MODEL_DIR:+--model "$MODEL_DIR"} \
  ${MODEL_DIR:---model Qwen/Qwen2.5-0.5B-Instruct} \
  --headless
