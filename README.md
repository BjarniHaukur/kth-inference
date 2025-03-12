# Simple CLI Chat Interface for QwQ-32B

A minimal command-line chat interface for interacting with the QwQ-32B model using vLLM.

## Requirements

- Python 3.8+
- vLLM (will be installed by the start script if not present)
- requests (will be installed by the start script if not present)

## Quick Start

1. Make the start script executable:
```bash
chmod +x start.sh
```

2. Run the start script:
```bash
./start.sh
```

This will:
- Check for required packages and install them if needed
- Start the vLLM server with the QwQ-32B model
- Launch the chat interface

## Manual Usage

If you want more control, you can run the components separately:

1. Start the vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --quantization awq \
    --max-model-len 32768 \
    --model Qwen/QwQ-32B-AWQ
```

2. In a separate terminal, start the chat interface:
```bash
python chat.py
```

## Features

- Simple command-line interface
- Colorful output for better readability
- Real-time token generation speed display
- Streaming responses for a more interactive experience
- Conversation history maintained during the session
