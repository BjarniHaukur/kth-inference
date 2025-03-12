#!/usr/bin/env python3
"""
Simple CLI chat interface for LLMs using vLLM's OpenAI-compatible API.
"""

import argparse
import json
import time
import requests
import sys
import os
import shutil
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

# ANSI cursor movement codes
CURSOR = {
    "up": "\033[A",
    "down": "\033[B",
    "right": "\033[C",
    "left": "\033[D",
    "save": "\033[s",
    "restore": "\033[u",
    "clear_line": "\033[K",
    "clear_screen": "\033[2J",
    "home": "\033[H",
}

def clear_line():
    """Clear the current line in the terminal."""
    sys.stdout.write(CURSOR["clear_line"])
    sys.stdout.flush()

def print_header(model_name):
    """Print a nice header for the chat interface."""
    # Extract just the model name without the organization
    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
    print(f"\n{COLORS['bold']}{COLORS['cyan']}=== {display_name} Chat Interface ==={COLORS['reset']}")
    print(f"{COLORS['gray']}Type 'exit' or 'quit' to end the conversation{COLORS['reset']}\n")

def get_available_models(api_base_url):
    """
    Get the list of available models from the API.
    
    Args:
        api_base_url: Base URL of the OpenAI-compatible API
    
    Returns:
        List of available model IDs, or None if the request failed
    """
    try:
        # Extract the base URL (without the /v1/chat/completions part)
        base_url = api_base_url.split('/v1/')[0]
        models_url = f"{base_url}/v1/models"
        
        response = requests.get(models_url)
        if response.status_code == 200:
            models_data = response.json()
            if 'data' in models_data:
                return [model['id'] for model in models_data['data']]
        return None
    except Exception as e:
        print(f"{COLORS['red']}Error fetching available models: {str(e)}{COLORS['reset']}")
        return None

def wait_for_server(api_base_url, max_retries=10, retry_delay=2):
    """
    Wait for the server to become available.
    
    Args:
        api_base_url: Base URL of the OpenAI-compatible API
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
    
    Returns:
        True if the server is available, False otherwise
    """
    print(f"{COLORS['yellow']}Checking if vLLM server is running...{COLORS['reset']}")
    
    for i in range(max_retries):
        try:
            # Extract the base URL (without the /v1/chat/completions part)
            base_url = api_base_url.split('/v1/')[0]
            models_url = f"{base_url}/v1/models"
            
            response = requests.get(models_url)
            if response.status_code == 200:
                print(f"{COLORS['green']}Server is running!{COLORS['reset']}")
                return True
            
            print(f"{COLORS['yellow']}Waiting for server to start (attempt {i+1}/{max_retries})...{COLORS['reset']}")
            time.sleep(retry_delay)
        except requests.exceptions.ConnectionError:
            print(f"{COLORS['yellow']}Server not ready yet (attempt {i+1}/{max_retries})...{COLORS['reset']}")
            time.sleep(retry_delay)
    
    print(f"{COLORS['red']}Server did not become available after {max_retries} attempts.{COLORS['reset']}")
    print(f"{COLORS['yellow']}Make sure the vLLM server is running with: ./start.sh{COLORS['reset']}")
    return False

def get_terminal_width():
    """Get the width of the terminal."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80  # Default width if we can't determine it

def chat_with_model(api_url, model_name, system_prompt=None):
    """
    Start a chat session with the model.
    
    Args:
        api_url: URL of the OpenAI-compatible API
        model_name: Name of the model to use
        system_prompt: Optional system prompt to set the behavior of the assistant
    """
    # Check if the server is running
    if not wait_for_server(api_url):
        return
    
    # Get available models
    available_models = get_available_models(api_url)
    if available_models:
        print(f"{COLORS['cyan']}Available models:{COLORS['reset']}")
        for model in available_models:
            print(f"  - {model}")
        
        # Check if the requested model is available
        if model_name not in available_models:
            print(f"{COLORS['yellow']}Warning: Model '{model_name}' not found in available models.{COLORS['reset']}")
            print(f"{COLORS['yellow']}Will try to use it anyway, but it might not work.{COLORS['reset']}")
    
    # Initialize conversation history
    conversation = []
    
    # Add system message if provided
    if system_prompt:
        conversation.append({
            "role": "system",
            "content": system_prompt
        })
    
    print_header(model_name)
    
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
        
        try:
            # Prepare the API request
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "messages": conversation,
                "stream": True,
                "max_tokens": 1000
            }
            
            # Variables for tracking tokens per second
            start_time = time.time()
            token_count = 0
            full_response = ""
            
            # Print the tokens/s display line
            tokens_per_second = 0
            tokens_display = f"{COLORS['gray']}[{tokens_per_second} tokens/s]{COLORS['reset']}"
            print(tokens_display)
            
            # Print assistant indicator
            print(f"{COLORS['bold']}{COLORS['green']}Assistant: {COLORS['reset']}", end="", flush=True)
            
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
            
            # Process the streaming response
            last_update_time = start_time
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
                                    
                                    # Print the new content
                                    print(content, end="", flush=True)
                                    
                                    # Update token count
                                    token_count += 1
                                    
                                    # Update tokens/s display every 0.5 seconds to avoid too much flickering
                                    current_time = time.time()
                                    if current_time - last_update_time >= 0.5:
                                        elapsed_time = current_time - start_time
                                        tokens_per_second = int(token_count / elapsed_time) if elapsed_time > 0 else 0
                                        
                                        # Move cursor up to the tokens/s line
                                        sys.stdout.write(f"\r\033[2A")  # Move up 2 lines and to beginning of line
                                        
                                        # Clear the line and update tokens/s
                                        term_width = get_terminal_width()
                                        tokens_display = f"{COLORS['gray']}[{tokens_per_second} tokens/s]{COLORS['reset']}"
                                        padding = term_width - len(tokens_display) + len(COLORS['gray']) + len(COLORS['reset'])
                                        padding = max(0, padding)  # Ensure padding is not negative
                                        
                                        sys.stdout.write(f"{' ' * padding}{tokens_display}\n\n")  # Add newlines to go back down
                                        
                                        # Move back to the end of the current response
                                        sys.stdout.write(f"\033[1A")  # Move up 1 line
                                        sys.stdout.write(f"\r{COLORS['bold']}{COLORS['green']}Assistant: {COLORS['reset']}{full_response}")
                                        sys.stdout.flush()
                                        
                                        last_update_time = current_time
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
            
        except requests.exceptions.ConnectionError:
            print(f"\n{COLORS['red']}Error: Could not connect to the API server. Make sure it's running.{COLORS['reset']}\n")
        except Exception as e:
            print(f"\n{COLORS['red']}Error: {str(e)}{COLORS['reset']}\n")

def main():
    """Main function."""
    # Get default model from environment or use QwQ-32B
    default_model = os.environ.get("VLLM_MODEL", "Qwen/QwQ-32B-AWQ")
    
    parser = argparse.ArgumentParser(description="Simple CLI chat interface for LLMs")
    parser.add_argument("--api", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="URL of the OpenAI-compatible API (default: http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--model", type=str, default=default_model,
                        help=f"Model name to use (default: {default_model})")
    parser.add_argument("--system", type=str, 
                        default="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.",
                        help="System prompt to set the behavior of the assistant")
    
    args = parser.parse_args()
    
    try:
        chat_with_model(args.api, args.model, args.system)
    except KeyboardInterrupt:
        print(f"\n\n{COLORS['yellow']}Chat session ended by user.{COLORS['reset']}\n")

if __name__ == "__main__":
    main() 