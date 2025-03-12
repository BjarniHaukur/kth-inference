#!/bin/bash

# Simple script to start the vLLM server and chat interface

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python3 -c "import requests" 2>/dev/null || { echo "Installing requests package..."; pip install requests; }

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "vLLM is not installed. Would you like to install it? (y/n)"
    read -r install_vllm
    if [[ $install_vllm == "y" ]]; then
        echo "Installing vLLM..."
        pip install vllm
    else
        echo "vLLM is required to run the model server."
        exit 1
    fi
fi

# Make chat.py executable
chmod +x chat.py

# Start the vLLM server
echo "Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --quantization awq \
    --max-model-len 32768 \
    --model Qwen/QwQ-32B-AWQ &

# Store the PID of the vLLM server
VLLM_PID=$!

# Wait for the server to start
echo "Waiting for vLLM server to initialize..."
sleep 10

# Start the chat interface
echo "Starting chat interface..."
./chat.py

# When the chat interface is closed, stop the vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 