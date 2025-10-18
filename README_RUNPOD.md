# RunPod Setup (Pod)

This project is prepared to run on a RunPod GPU Pod in a headless server environment (no speakers). Audio is written to `tmp_audio/` as WAV files while the assistant streams text in your SSH session.

## 1) Launch a Pod

- Use a CUDA PyTorch image (e.g., RunPod "PyTorch 2.x CUDA 12.x Ubuntu 22.04").
- Note your SSH command from the Pod Connect tab (either proxied SSH or Direct TCP over port forwarding).

## 2) Copy the project and models

- Clone or upload this repository to the pod.
The bootstrap script will auto-fetch most public models if they are missing:
- OpenVoice V2 converter + base speakers from `myshell-ai/OpenVoiceV2`.
- Embedding model `BAAI/bge-small-en-v1.5`.
- Reranker model `BAAI/bge-reranker-base` (optional).
- If no local merged LLM is found under `models/*/merged_model`, it will fall back to `Qwen/Qwen2.5-0.5B-Instruct` and allow Hugging Face downloads for that model only.

Notes:
- If you have a local merged model, place it under `models/<any>/merged_model/` to run fully offline.
- `assets/voice_ref.wav` is optional now. If missing, the app uses the base speaker voice without cloning.

## 3) Install and run

SSH into the pod, then run:

```
cd ~/vtuber-stream
bash scripts/start_runpod.sh
```

What the script does:
- Creates a Python venv and installs `requirements.txt` (including OpenVoice + MeloTTS from GitHub).
- Fetches public checkpoints if they’re missing.
- Sets `HF_HOME=./hf_cache`; runs offline when a local LLM exists, otherwise downloads `Qwen/Qwen2.5-0.5B-Instruct`.
- Auto-detects your merged model path under `models/*/merged_model`.
- Starts `run.py` in `--headless` mode (saves audio to `tmp_audio/`).

To customize:

```
DEVICE=cuda:0 LANG=EN BASE_SPK=EN-US SR=48000 SPEED=0.95 bash scripts/start_runpod.sh
```

Or run manually:

```
source .venv/bin/activate
python run.py \
  --ref none \
  --device cuda:0 \
  --lang EN --base-speaker-key EN-US \
  --sr 48000 --speed 0.95 \
  --kb kb \
  --embed-model models/embeds/bge-small-en-v1.5 \
  --reranker-model models/reranker/bge-reranker-base \
  --model models/merged_model-20251017T181201Z-1-001/merged_model \
  --headless
```

## 4) Using it

- Interact via the terminal (the assistant streams text).
- WAV segments are saved to `tmp_audio/` and the paths are printed inline.
- Use `/help` for commands; `/rag on` enables RAG if the index is present.

## Notes

- If you prefer audio playback, run on a machine with a sound device and omit `--headless`.
- If `faiss-gpu` is desired, replace `faiss-cpu` in `requirements.txt` with `faiss-gpu` (Linux/CUDA only).
- The default environment variables force offline HF loading. Clear them if you need to pull from the hub.
