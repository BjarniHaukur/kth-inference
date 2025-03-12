#!/bin/bash

# Function to get the IP address
get_ip_address() {
    hostname=$(hostname)
    ip_address=$(hostname -I | awk '{print $1}')
    echo $ip_address
}

# Check if venv exists
if [ ! -d "vllm" ]; then
    echo "Creating virtual environment..."
    uv venv vllm --python 3.12 --seed
    uv pip install vllm
fi

# Get the IP address
IP_ADDRESS=$(get_ip_address)

# Update the script.js file with the correct API URL
echo "Updating script.js with the correct API URL..."
sed -i.bak "s|const API_URL = 'http://localhost:8000/v1/chat/completions';|const API_URL = 'http://${IP_ADDRESS}:8000/v1/chat/completions';|g" script.js

# Start the model server in the background
echo "Starting vLLM server..."
echo "API will be available at: http://${IP_ADDRESS}:8000"
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --quantization awq \
    --max-model-len 32768 \
    --model Qwen/QwQ-32B-AWQ &

# Store the PID of the model server
MODEL_PID=$!

# Wait a bit for the model server to start
echo "Waiting for vLLM server to initialize..."
sleep 10

# Start the web server
echo "Starting web server..."
echo "Chat interface will be available at: http://${IP_ADDRESS}:3000"
python3 server.py

# When the web server is stopped, also stop the model server
echo "Stopping model server..."
kill $MODEL_PID 