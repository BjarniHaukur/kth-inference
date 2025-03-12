# Simple CLI for fast inference on a single H100

Perfect for use on KTH's DGX H100 since it mostly sits idle ¯\_(ツ)_/¯

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

## Usage

- Type your message and press Enter for new lines
- Press Ctrl+J or ESC+Enter to send the message
- Supports pasting multi-line text

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
  