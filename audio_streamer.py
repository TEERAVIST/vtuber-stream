#!/usr/bin/env python3
"""
Audio streaming server for VTuber application.
This creates a WebSocket server that streams audio to connected clients.
"""

import asyncio
import websockets
import json
import os
import time
from pathlib import Path
import threading
import base64

class AudioStreamer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.audio_dir = Path("tmp_audio")
        self.latest_file = None
        self.running = False
        self.broadcasted_files = set()  # Track files that have already been broadcasted
        
    async def handle_client(self, websocket, path):
        """Handle a client connection."""
        self.clients.add(websocket)
        print(f"Client connected from {websocket.remote_address}")
        
        try:
            # Send the latest audio file if available
            if self.latest_file:
                await self.send_latest_audio(websocket)
            
            # Keep the connection alive and handle messages
            async for message in websocket:
                # We don't expect messages from clients, but we'll handle them if they come
                try:
                    data = json.loads(message)
                    print(f"Received message from client: {data}")
                except json.JSONDecodeError:
                    print(f"Received non-JSON message from client")
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected from {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)
            print(f"Client removed from {websocket.remote_address}")
    
    async def send_latest_audio(self, websocket):
        """Send the latest audio file to a client."""
        if self.latest_file and os.path.exists(self.latest_file):
            with open(self.latest_file, 'rb') as f:
                audio_data = f.read()
            
            # Send audio data as base64 encoded
            message = {
                'type': 'audio',
                'filename': os.path.basename(self.latest_file),
                'data': base64.b64encode(audio_data).decode('utf-8')
            }
            await websocket.send(json.dumps(message))
    
    async def broadcast_new_audio(self, file_path):
        """Broadcast new audio to all connected clients."""
        if not self.clients:
            return
        
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        message = {
            'type': 'audio',
            'filename': os.path.basename(file_path),
            'data': base64.b64encode(audio_data).decode('utf-8')
        }
        
        # Send to all clients
        if self.clients:
            # Create a list of tasks to send to all clients
            tasks = []
            for client in self.clients.copy():  # Use copy to avoid modification during iteration
                if not client.closed:
                    tasks.append(client.send(json.dumps(message)))
                else:
                    self.clients.discard(client)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                print(f"Broadcasted {file_path} to {len(tasks)} clients")
    
    def monitor_audio_files(self):
        """Monitor the audio directory for new files."""
        print(f"Monitoring {self.audio_dir} for new audio files...")
        
        # Get initial state
        files = list(self.audio_dir.glob("*.wav"))
        if files:
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            self.latest_file = str(files[0])
        
        # Mark all existing files as already broadcasted to prevent duplicate playback
        for f in files:
            self.broadcasted_files.add(str(f))
            print(f"[INIT] Marked existing file as already broadcasted: {f}")
        
        # Start monitoring
        last_mtime = {}
        for f in files:
            last_mtime[str(f)] = f.stat().st_mtime
        
        while self.running:
            try:
                current_files = list(self.audio_dir.glob("*.wav"))
                current_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Check for new files
                for f in current_files:
                    f_path = str(f)
                    if f_path not in last_mtime or f.stat().st_mtime > last_mtime[f_path]:
                        # New file found
                        self.latest_file = f_path
                        last_mtime[f_path] = f.stat().st_mtime
                        
                        # Only broadcast if we haven't already sent this file
                        if f_path not in self.broadcasted_files:
                            self.broadcasted_files.add(f_path)
                            # Broadcast to clients
                            asyncio.run_coroutine_threadsafe(
                                self.broadcast_new_audio(f_path),
                                self.loop
                            )
                        else:
                            print(f"[DEBUG] Skipping already broadcasted file: {f_path}")
                
                time.sleep(1)  # Check every second
            except Exception as e:
                print(f"Error monitoring audio files: {e}")
                time.sleep(5)
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.running = True
        self.loop = asyncio.get_event_loop()
        
        # Start the file monitor in a separate thread
        monitor_thread = threading.Thread(target=self.monitor_audio_files)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start the WebSocket server
        server = await websockets.serve(self.handle_client, self.host, self.port)
        print(f"Audio streaming server started on ws://{self.host}:{self.port}")
        
        await server.wait_closed()
    
    def stop(self):
        """Stop the server."""
        self.running = False

if __name__ == "__main__":
    import sys
    
    # Allow command line arguments for host and port
    host = "0.0.0.0"
    port = 8765
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    streamer = AudioStreamer(host, port)
    
    try:
        asyncio.run(streamer.start_server())
    except KeyboardInterrupt:
        print("\nShutting down audio streaming server...")
        streamer.stop()