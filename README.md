# Simple CLI Chat Interface for QwQ-32B

A minimal command-line chat interface for interacting with the QwQ-32B model using vLLM.

## Requirements

- Python 3.8+
- All other dependencies will be automatically installed by the start script

## Quick Start

### Step 1: Start the vLLM Server

1. Make the start script executable:
```bash
chmod +x start.sh
```

2. Run the start script to launch the vLLM server:
```bash
# Use default model (Qwen/QwQ-32B-AWQ)
./start.sh

# Or specify a different model from Hugging Face
./start.sh mistralai/Mistral-7B-Instruct-v0.2
```

This will:
- Install uv package manager if not present
- Install required packages (requests, rich) if not present
- Start the vLLM server with the specified model
- The server will run in the foreground (press Ctrl+C to stop)

### Step 2: Start the Chat Interface

In a separate terminal, run the chat interface:

```bash
# Make the chat script executable if needed
chmod +x chat.py

# Run with default settings
./chat.py

# Or specify a different model
./chat.py --model mistralai/Mistral-7B-Instruct-v0.2

# Or specify a custom system prompt
./chat.py --system "You are a helpful AI assistant that specializes in Python programming."
```

The chat interface will:
- Check if the vLLM server is running
- Display available models
- Start a chat session with the specified model

## Manual Usage

If you want more control, you can run the components with additional options:

1. Start the vLLM server with custom parameters:
```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --quantization awq \
    --max-model-len 32768 \
    --model Qwen/QwQ-32B-AWQ  # or your preferred model
```

2. Start the chat interface with custom parameters:
```bash
python chat.py --api http://localhost:8000/v1/chat/completions --model Qwen/QwQ-32B-AWQ
```

## Features

- Beautiful command-line interface using the Rich library
- Colorful output for better readability
- Real-time token generation speed display
- Streaming responses for a more interactive experience
- Conversation history maintained during the session
- Automatic detection of available models
