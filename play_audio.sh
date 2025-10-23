#!/bin/bash

# Simple script to play the most recent audio file
# Usage: ./play_audio.sh

cd "$(dirname "$0")"

# Find the most recent wav file
LATEST_FILE=$(ls -t tmp_audio/*.wav 2>/dev/null | head -n 1)

if [ -z "$LATEST_FILE" ]; then
    echo "No audio files found in tmp_audio directory."
    exit 1
fi

echo "Playing: $LATEST_FILE"

# Try different methods to play the audio
if command -v aplay >/dev/null 2>&1; then
    aplay "$LATEST_FILE"
elif command -v paplay >/dev/null 2>&1; then
    paplay "$LATEST_FILE"
elif command -v ffplay >/dev/null 2>&1; then
    ffplay -nodisp -autoexit "$LATEST_FILE"
else
    echo "No audio player found. Please install one of: aplay, paplay, or ffplay"
    echo "You can download the file and play it locally:"
    echo "File path: $(pwd)/$LATEST_FILE"
    exit 1
fi