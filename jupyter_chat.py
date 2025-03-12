#!/usr/bin/env python3
"""
Jupyter Chat Interface for QwQ-32B

This script starts a chat interface for QwQ-32B in a Jupyter environment.
It can be imported and run directly in a Jupyter notebook cell.

Usage in Jupyter:
    from jupyter_chat import start_chat
    start_chat()
"""

import os
import subprocess
import socket
import time
import json
from IPython.display import display, HTML, Javascript

def get_ip_address():
    """Get the IP address of the machine."""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def update_script_js(ip_address):
    """Update the script.js file with the correct API URL."""
    try:
        with open('script.js', 'r') as f:
            script_content = f.read()
        
        # Replace localhost with the actual IP address
        updated_script = script_content.replace(
            "const API_URL = 'http://localhost:8000/v1/chat/completions';", 
            f"const API_URL = 'http://{ip_address}:8000/v1/chat/completions';"
        )
        
        # Write the updated script back to the file
        with open('script.js', 'w') as f:
            f.write(updated_script)
        
        return True
    except Exception as e:
        print(f"Error updating script.js: {e}")
        return False

def start_vllm_server():
    """Start the vLLM server."""
    print("Starting vLLM server...")
    vllm_process = subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "half",
        "--quantization", "awq",
        "--max-model-len", "32768",
        "--model", "Qwen/QwQ-32B-AWQ"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit for the server to start
    print("Waiting for vLLM server to initialize...")
    time.sleep(10)
    
    return vllm_process

def start_web_server():
    """Start the web server in a separate thread."""
    from server import jupyter_server
    return jupyter_server()

def start_chat():
    """Start the chat interface in a Jupyter notebook."""
    # Get the IP address
    ip_address = get_ip_address()
    print(f"IP Address: {ip_address}")
    
    # Update the script.js file
    if update_script_js(ip_address):
        print(f"Updated script.js to use API at http://{ip_address}:8000")
    
    # Start the vLLM server
    vllm_process = start_vllm_server()
    print(f"vLLM API running at: http://{ip_address}:8000")
    
    # Start the web server
    server_url = start_web_server()
    print(f"Chat interface running at: {server_url}")
    
    # Return the vLLM process so it can be terminated later
    return vllm_process

def stop_chat(vllm_process):
    """Stop the chat interface."""
    if vllm_process:
        print("Stopping vLLM server...")
        vllm_process.terminate()
        vllm_process.wait()
        print("vLLM server stopped.")
    else:
        print("No vLLM server process found.")

if __name__ == "__main__":
    # This is for running directly, not in Jupyter
    vllm_process = start_chat()
    
    try:
        # Keep the script running until Ctrl+C
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_chat(vllm_process)
        print("Chat interface stopped.") 