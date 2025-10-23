# Audio Setup for VTuber Application in Headless Environment

## Issue
In a headless environment like RunPod, there's no audio output device to play sound directly. The application is working correctly and generating audio files, but they are saved to the `tmp_audio/` directory instead of being played.

## Solutions

### Option 1: Download Audio Files
1. The application saves audio files to `tmp_audio/` directory
2. You can download these files and play them on your local machine
3. Use the file transfer method available in your environment to download the WAV files

### Option 2: Use the Audio Monitor Script
We've created a Python script that can monitor the audio directory and play files locally:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run the audio monitor
python listen_to_audio.py
```

Note: This will only work if you have a local audio device connected to your environment.

### Option 3: Stream Audio
If you need to stream the audio output, you can:

1. Use a streaming solution like Icecast or Shoutcast
2. Set up a web server to serve the audio files
3. Use WebRTC to stream audio in real-time

### Option 4: Use a Virtual Audio Device
For Linux systems, you can create a virtual audio device:

```bash
# Install ALSA loopback device
sudo modprobe snd-aloop

# Set it as the default device
export ALSA_PCM_CARD=Loopback
export ALSA_PCM_DEVICE=0
```

## Current Status
- The application is running correctly
- Audio is being generated and saved to `tmp_audio/`
- The chat functionality is working
- The TTS (Text-to-Speech) is generating proper audio files

## Running the Application
Use one of these scripts:

```bash
# Run with automatic model detection
./run_with_params.sh

# Or run with explicit model path
./run_linux.sh
```

## Checking Generated Audio
To list the latest audio files:
```bash
ls -la tmp_audio/*.wav | tail -5
```

## Playing Audio Locally
To play the most recent audio file:
```bash
./play_audio.sh
```

Note: This will only work if you have an audio device connected to the environment.