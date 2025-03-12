#!/usr/bin/env python3
"""
Simple CLI chat interface for LLMs using vLLM's OpenAI-compatible API.
"""

import os
import json
import time
import requests
import argparse

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich.table import Table
from rich.prompt import Prompt


# Create console
console = Console()

def print_header(model_name):
    """Print a nice header for the chat interface."""
    # Extract just the model name without the organization
    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    console.print(f"\n[bold cyan]=== {display_name} Chat Interface ===[/bold cyan]")
    console.print("[dim]Type 'exit' or 'quit' to end the conversation[/dim]\n")

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
        console.print(f"[bold red]Error fetching available models: {str(e)}[/bold red]")
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
    console.print("[yellow]Checking if vLLM server is running...[/yellow]")
    
    with console.status("[yellow]Connecting to server...[/yellow]") as status:
        for i in range(max_retries):
            try:
                # Extract the base URL (without the /v1/chat/completions part)
                base_url = api_base_url.split('/v1/')[0]
                models_url = f"{base_url}/v1/models"
                
                response = requests.get(models_url)
                if response.status_code == 200:
                    console.print("[bold green]Server is running![/bold green]")
                    return True
                
                status.update(f"[yellow]Waiting for server to start (attempt {i+1}/{max_retries})...[/yellow]")
                time.sleep(retry_delay)
            except requests.exceptions.ConnectionError:
                status.update(f"[yellow]Server not ready yet (attempt {i+1}/{max_retries})...[/yellow]")
                time.sleep(retry_delay)
    
    console.print("[bold red]Server did not become available after {max_retries} attempts.[/bold red]")
    console.print("[yellow]Make sure the vLLM server is running with: ./start.sh[/yellow]")
    return False

def create_chat_layout():
    """Create the layout for the chat interface."""
    layout = Layout()
    
    # Create the tokens/s display
    tokens_table = Table.grid(padding=0)
    tokens_table.add_row(Text("0 tokens/s", style="dim"))
    
    # Create the main layout
    layout.split(
        Layout(tokens_table, name="tokens", size=1),
        Layout(name="chat")
    )
    
    return layout

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
        console.print("[cyan]Available models:[/cyan]")
        for model in available_models:
            console.print(f"  - {model}")
        
        # Check if the requested model is available
        if model_name not in available_models:
            console.print(f"[yellow]Warning: Model '{model_name}' not found in available models.[/yellow]")
            console.print(f"[yellow]Will try to use it anyway, but it might not work.[/yellow]")
    
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
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            console.print("\n[yellow]Goodbye![/yellow]\n")
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
                "max_tokens": 32768
            }
            
            # Variables for tracking tokens per second
            start_time = time.time()
            token_count = 0
            full_response = ""
            
            # Create the layout
            layout = create_chat_layout()
            
            # Start the live display
            with Live(layout, refresh_per_second=4, console=console) as live:
                # Print assistant indicator
                console.print("[bold green]Assistant:[/bold green] ", end="")
                
                # Make the API request
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    stream=True
                )
                
                if response.status_code != 200:
                    console.print(f"[bold red]Error: API request failed with status {response.status_code}[/bold red]")
                    continue
                
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
                                        
                                        # Print the new content
                                        console.print(content, end="", highlight=False)
                                        
                                        # Update token count
                                        token_count += 1
                                        
                                        # Update tokens/s display
                                        elapsed_time = time.time() - start_time
                                        tokens_per_second = int(token_count / elapsed_time) if elapsed_time > 0 else 0
                                        
                                        # Update the tokens/s display
                                        tokens_table = Table.grid(padding=0)
                                        tokens_table.add_row(Text(f"{tokens_per_second} tokens/s", style="dim"))
                                        layout["tokens"].update(tokens_table)
                            except json.JSONDecodeError:
                                pass
            
            # Add a newline after the response
            console.print()
            
            # Add assistant message to conversation
            conversation.append({
                "role": "assistant",
                "content": full_response
            })
            
            # Final tokens per second calculation
            elapsed_time = time.time() - start_time
            tokens_per_second = int(token_count / elapsed_time) if elapsed_time > 0 else 0
            console.print(f"[dim]Generated {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second} tokens/s)[/dim]\n")
            
        except requests.exceptions.ConnectionError:
            console.print("\n[bold red]Error: Could not connect to the API server. Make sure it's running.[/bold red]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")

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
        console.print(f"\n\n[yellow]Chat session ended by user.[/yellow]\n")

if __name__ == "__main__":
    main() 