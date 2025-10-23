#!/usr/bin/env python3
"""
Script to listen to generated audio files in a headless environment.
This will monitor the tmp_audio directory and play new audio files as they are generated.
"""

import os
import sys
import time
import pygame
from pathlib import Path

def main():
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Directory to monitor
    audio_dir = Path("tmp_audio")
    
    if not audio_dir.exists():
        print(f"Audio directory {audio_dir} does not exist.")
        return
    
    print(f"Monitoring {audio_dir} for new audio files...")
    print("Press Ctrl+C to stop.")
    
    # Track already played files
    played_files = set()
    
    try:
        while True:
            # Get all wav files
            wav_files = list(audio_dir.glob("*.wav"))
            
            # Sort by modification time (newest first)
            wav_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Play the newest file that hasn't been played yet
            for wav_file in wav_files:
                if wav_file.name not in played_files:
                    print(f"\nPlaying: {wav_file.name}")
                    try:
                        pygame.mixer.music.load(str(wav_file))
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        played_files.add(wav_file.name)
                        print("Playback completed.")
                    except Exception as e:
                        print(f"Error playing {wav_file.name}: {e}")
            
            # Check every second for new files
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping audio monitor.")

if __name__ == "__main__":
    main()