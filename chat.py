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
from rich.panel import Panel
from rich.align import Align
from rich.layout import Layout
from rich.prompt import Prompt
from rich.box import ROUNDED
from rich.progress import Progress, SpinnerColumn, TextColumn

# Create console
console = Console()

def print_header(model_name):
    """Print a nice header for the chat interface."""
    # Extract just the model name without the organization
    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    console.print()
    console.print(Panel(
        f"[bold cyan]{display_name} Chat Interface[/bold cyan]",
        subtitle="[dim]Type 'exit' or 'quit' to end the conversation[/dim]",
        box=ROUNDED,
        expand=False,
        border_style="cyan"
    ))
    console.print()

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
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[yellow]Connecting to server..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Connecting", total=max_retries)
        
        for i in range(max_retries):
            try:
                # Extract the base URL (without the /v1/chat/completions part)
                base_url = api_base_url.split('/v1/')[0]
                models_url = f"{base_url}/v1/models"
                
                response = requests.get(models_url)
                if response.status_code == 200:
                    console.print("[bold green]Server is running![/bold green]")
                    return True
                
                progress.update(task, advance=1, description=f"[yellow]Waiting for server (attempt {i+1}/{max_retries})...")
                time.sleep(retry_delay)
            except requests.exceptions.ConnectionError:
                progress.update(task, advance=1, description=f"[yellow]Server not ready (attempt {i+1}/{max_retries})...")
                time.sleep(retry_delay)
    
    console.print("[bold red]Server did not become available after {max_retries} attempts.[/bold red]")
    console.print("[yellow]Make sure the vLLM server is running with: ./start.sh[/yellow]")
    return False

class ChatInterface:
    """Chat interface with a status bar for tokens/s display."""
    
    def __init__(self, api_url, model_name, system_prompt=None):
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.conversation = []
        self.token_count = 0
        self.tokens_per_second = 0
        self.start_time = 0
        self.is_generating = False
        self.full_response = ""
        
        # Add system message if provided
        if system_prompt:
            self.conversation.append({
                "role": "system",
                "content": system_prompt
            })
    
    def create_layout(self):
        """Create the layout for the chat interface."""
        layout = Layout()
        
        # Create the main layout with chat area and status bar
        layout.split_column(
            Layout(name="chat", ratio=1),
            Layout(name="status_bar", size=1)
        )
        
        # Initialize the status bar
        layout["status_bar"].update(
            Text("Ready", style="dim")
        )
        
        return layout
    
    def update_status_bar(self, layout, message=None):
        """Update the status bar with tokens/s information."""
        if self.is_generating:
            elapsed_time = max(0.1, time.time() - self.start_time)
            self.tokens_per_second = int(self.token_count / elapsed_time)
            status_text = f"Generating: {self.token_count} tokens | {self.tokens_per_second} tokens/s"
        elif message:
            status_text = message
        else:
            status_text = "Ready"
        
        # Right-align the status text
        status = Text(status_text, style="dim")
        layout["status_bar"].update(Align.right(status))
    
    def run(self):
        """Run the chat interface."""
        # Check if the server is running
        if not wait_for_server(self.api_url):
            return
        
        # Get available models
        available_models = get_available_models(self.api_url)
        if available_models:
            console.print("[cyan]Available models:[/cyan]")
            for model in available_models:
                console.print(f"  - {model}")
            
            # Check if the requested model is available
            if self.model_name not in available_models:
                console.print(f"[yellow]Warning: Model '{self.model_name}' not found in available models.[/yellow]")
                console.print(f"[yellow]Will try to use it anyway, but it might not work.[/yellow]")
        
        print_header(self.model_name)
        
        # Create the layout
        layout = self.create_layout()
        
        # Main chat loop
        with Live(layout, refresh_per_second=10, console=console, screen=True) as live:
            while True:
                # Update status bar
                self.update_status_bar(layout)
                live.refresh()
                
                # Get user input (outside of Live context to allow input)
                live.stop()
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                live.start()
                
                # Check if user wants to exit
                if user_input.lower() in ["exit", "quit"]:
                    self.update_status_bar(layout, "Goodbye!")
                    live.refresh()
                    time.sleep(1)  # Show goodbye message briefly
                    break
                
                # Add user message to conversation
                self.conversation.append({
                    "role": "user",
                    "content": user_input
                })
                
                try:
                    # Reset counters
                    self.token_count = 0
                    self.tokens_per_second = 0
                    self.start_time = time.time()
                    self.is_generating = True
                    self.full_response = ""
                    
                    # Update status bar to show we're generating
                    self.update_status_bar(layout)
                    live.refresh()
                    
                    # Temporarily stop the live display to print the assistant prompt
                    live.stop()
                    console.print("[bold green]Assistant:[/bold green] ", end="")
                    live.start()
                    
                    # Prepare the API request
                    headers = {
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": self.model_name,
                        "messages": self.conversation,
                        "stream": True,
                        "max_tokens": 32768
                    }
                    
                    # Make the API request
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=data,
                        stream=True
                    )
                    
                    if response.status_code != 200:
                        live.stop()
                        console.print(f"[bold red]Error: API request failed with status {response.status_code}[/bold red]")
                        live.start()
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
                                            self.full_response += content
                                            
                                            # Print the new content
                                            live.stop()
                                            console.print(content, end="", highlight=False)
                                            live.start()
                                            
                                            # Update token count
                                            self.token_count += 1
                                            
                                            # Update status bar
                                            self.update_status_bar(layout)
                                except json.JSONDecodeError:
                                    pass
                    
                    # Add a newline after the response
                    live.stop()
                    console.print()
                    live.start()
                    
                    # Add assistant message to conversation
                    self.conversation.append({
                        "role": "assistant",
                        "content": self.full_response
                    })
                    
                    # Final tokens per second calculation
                    self.is_generating = False
                    elapsed_time = time.time() - self.start_time
                    self.tokens_per_second = int(self.token_count / elapsed_time) if elapsed_time > 0 else 0
                    
                    # Show final generation stats
                    live.stop()
                    console.print(f"[dim]Generated {self.token_count} tokens in {elapsed_time:.2f}s ({self.tokens_per_second} tokens/s)[/dim]\n")
                    live.start()
                    
                except requests.exceptions.ConnectionError:
                    live.stop()
                    console.print("\n[bold red]Error: Could not connect to the API server. Make sure it's running.[/bold red]\n")
                    live.start()
                except Exception as e:
                    live.stop()
                    console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")
                    live.start()

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
        # Create and run the chat interface
        chat_interface = ChatInterface(args.api, args.model, args.system)
        chat_interface.run()
    except KeyboardInterrupt:
        console.print(f"\n\n[yellow]Chat session ended by user.[/yellow]\n")

if __name__ == "__main__":
    main() 