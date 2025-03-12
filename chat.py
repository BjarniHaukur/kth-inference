#!/usr/bin/env python3
"""
Simple CLI chat interface for QwQ-32B using vLLM's OpenAI-compatible API.
"""

import argparse
import json
import time
import requests
import sys
from datetime import datetime

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "gray": "\033[90m",
}

def clear_line():
    """Clear the current line in the terminal."""
    sys.stdout.write("\033[K")
    sys.stdout.flush()

def print_header():
    """Print a nice header for the chat interface."""
    print(f"\n{COLORS['bold']}{COLORS['cyan']}=== QwQ-32B Chat Interface ==={COLORS['reset']}")
    print(f"{COLORS['gray']}Type 'exit' or 'quit' to end the conversation{COLORS['reset']}\n")

def chat_with_model(api_url, system_prompt=None):
    """
    Start a chat session with the model.
    
    Args:
        api_url: URL of the OpenAI-compatible API
        system_prompt: Optional system prompt to set the behavior of the assistant
    """
    # Initialize conversation history
    conversation = []
    
    # Add system message if provided
    if system_prompt:
        conversation.append({
            "role": "system",
            "content": system_prompt
        })
    
    print_header()
    
    # Main chat loop
    while True:
        # Get user input
        user_input = input(f"{COLORS['bold']}{COLORS['blue']}You: {COLORS['reset']}")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print(f"\n{COLORS['yellow']}Goodbye!{COLORS['reset']}\n")
            break
        
        # Add user message to conversation
        conversation.append({
            "role": "user",
            "content": user_input
        })
        
        # Print assistant indicator
        print(f"{COLORS['bold']}{COLORS['green']}Assistant: {COLORS['reset']}", end="", flush=True)
        
        try:
            # Prepare the API request
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": "Qwen/QwQ-32B-AWQ",
                "messages": conversation,
                "stream": True,
                "max_tokens": 1000
            }
            
            # Make the API request
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                stream=True
            )
            
            if response.status_code != 200:
                print(f"{COLORS['red']}Error: API request failed with status {response.status_code}{COLORS['reset']}")
                continue
            
            # Variables for tracking tokens per second
            start_time = time.time()
            token_count = 0
            full_response = ""
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        try:
                            json_data = json.loads(line[6:])
                            if 'choices' in json_data and json_data['choices'] and 'delta' in json_data['choices'][0]:
                                delta = json_data['choices'][0]['delta']
                                if 'content' in delta and delta['content']:
                                    content = delta['content']
                                    full_response += content
                                    print(content, end="", flush=True)
                                    
                                    # Update token count and calculate tokens per second
                                    token_count += 1
                                    elapsed_time = time.time() - start_time
                                    tokens_per_second = int(token_count / elapsed_time) if elapsed_time > 0 else 0
                                    
                                    # Update the tokens/s display (on the same line)
                                    sys.stdout.write(f"\r{COLORS['gray']}[{tokens_per_second} tokens/s]{COLORS['reset']}")
                                    sys.stdout.flush()
                        except json.JSONDecodeError:
                            pass
            
            # Add a newline after the response
            print("\n")
            
            # Add assistant message to conversation
            conversation.append({
                "role": "assistant",
                "content": full_response
            })
            
            # Final tokens per second calculation
            elapsed_time = time.time() - start_time
            tokens_per_second = int(token_count / elapsed_time) if elapsed_time > 0 else 0
            print(f"{COLORS['gray']}Generated {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second} tokens/s){COLORS['reset']}\n")
            
        except Exception as e:
            print(f"\n{COLORS['red']}Error: {str(e)}{COLORS['reset']}\n")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple CLI chat interface for QwQ-32B")
    parser.add_argument("--api", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="URL of the OpenAI-compatible API (default: http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--system", type=str, 
                        default="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.",
                        help="System prompt to set the behavior of the assistant")
    
    args = parser.parse_args()
    
    try:
        chat_with_model(args.api, args.system)
    except KeyboardInterrupt:
        print(f"\n\n{COLORS['yellow']}Chat session ended by user.{COLORS['reset']}\n")

if __name__ == "__main__":
    main() 