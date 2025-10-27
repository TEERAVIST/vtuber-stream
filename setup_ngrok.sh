#!/bin/bash

# Script to set up ngrok configuration in /workspace for persistence across RunPod sessions

echo "[SETUP] Setting up ngrok configuration for persistence..."

# Create the config directory if it doesn't exist
mkdir -p /workspace/.config/ngrok

# Check if ngrok config already exists in /workspace
if [ -f "/workspace/.config/ngrok/ngrok.yml" ]; then
    echo "[INFO] ngrok configuration already exists in /workspace/.config/ngrok/ngrok.yml"
    
    # Copy to root location for current session
    mkdir -p /root/.config/ngrok
    cp /workspace/.config/ngrok/ngrok.yml /root/.config/ngrok/
    echo "[INFO] Copied existing configuration to /root/.config/ngrok/ngrok.yml"
else
    # Check if ngrok config exists in root (from previous setup)
    if [ -f "/root/.config/ngrok/ngrok.yml" ]; then
        echo "[INFO] Found existing ngrok configuration in /root/.config/ngrok/ngrok.yml"
        cp /root/.config/ngrok/ngrok.yml /workspace/.config/ngrok/
        echo "[INFO] Backed up configuration to /workspace/.config/ngrok/ngrok.yml"
    else
        echo "[WARN] No ngrok configuration found. Please run:"
        echo "  ./ngrok config add-authtoken YOUR_AUTHTOKEN"
        echo "Then run this script again to backup the configuration."
    fi
fi

# Set environment variable for the current session
export NGROK_CONFIG_PATH="/workspace/.config/ngrok/ngrok.yml"

echo "[DONE] ngrok configuration setup complete."
echo "[INFO] The configuration will persist in /workspace/.config/ngrok/ngrok.yml"