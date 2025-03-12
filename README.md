# Simple Chat Interface for QwQ-32B

A minimal chat interface that connects to a locally running vLLM server with an OpenAI-compatible API.

## Setup

1. Create a Python virtual environment:
```bash
uv venv --python 3.12 --seed
```

2. Install vLLM:
```bash
uv pip install vllm
```

## Running the Application

### Option 1: Using the start script (recommended)

1. Make the start script executable:
```bash
chmod +x start.sh
```

2. Run the start script:
```bash
./start.sh
```

This will start both the model server and the web server. The script will display URLs for both local and remote access.

### Option 2: Manual startup

1. Start the vLLM server:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --quantization awq \
    --max-model-len 32768 \
    --model Qwen/QwQ-32B-AWQ
```

2. In a separate terminal, start the web server:
```bash
python3 server.py
```

3. Open your browser and navigate to the URL displayed in the terminal.

## Running in Jupyter Notebook (Cloud Environment)

If you're running this in a Jupyter environment (like Google Colab or a cloud VM), you have two options:

### Option 1: Using the Jupyter notebook

1. Open the `chat_interface.ipynb` notebook
2. Run all cells in sequence
3. The chat interface will be displayed directly in the notebook
4. A remote access URL will also be provided if you want to access it from another device

### Option 2: Using the Python module in any notebook

1. Copy and paste the following code into a Jupyter notebook cell:

```python
# Import the chat interface module
from jupyter_chat import start_chat, stop_chat

# Start the chat interface
vllm_process = start_chat()

# To stop the chat interface later, run:
# stop_chat(vllm_process)
```

2. Run the cell to start the chat interface
3. The interface will be displayed directly in the notebook
4. A remote access URL will also be provided for external access

## Features

- Simple back-and-forth conversation with the model
- Real-time token generation speed display
- Streaming responses for a more interactive experience
- Remote access capability for cloud environments
- Jupyter notebook integration for cloud environments
