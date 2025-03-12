# Simple CLI Chat Interface

A minimal command-line chat interface for interacting with LLMs using vLLM.

## Quick Start

1. Start the vLLM server:
```bash
sh serve.sh
```

2. In a separate terminal, start the chat interface:
```bash
uv run chat.py
```

## Configuration

You can customize the chat interface with these parameters:

```bash
# Use a different model
uv run chat.py --model mistralai/Mistral-7B-Instruct-v0.2

# Set a custom system prompt
uv run chat.py --system "You are a Python programming expert"
```

## Features

- **Beautiful tokens/s display** with visual indicators and progress bar
- Dynamic speed indicators (🚀 for very fast, ⚡ for fast, etc.)
- Multi-line input support with keyboard shortcuts to send
- Scrollable conversation history
- Real-time streaming responses

## Usage

- Type your message and press Enter for new lines
- Press Ctrl+J or Ctrl+M to send the message
- Supports pasting multi-line text
- Use chat commands (see below) for additional functionality

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~76.8 tokens/s (recommended)
- --enforce-eager: 

## Chat Commands

- `help` or `?`: Show available commands
- `clear` or `reset`: Clear conversation history
- `exit` or `quit`: End the chat session
- `scroll up/down`: Navigate through message history
