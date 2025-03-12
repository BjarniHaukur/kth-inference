#!/usr/bin/env python3
import http.server
import socketserver
import os
import socket
import json
from IPython.display import display, HTML

PORT = 3000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def get_ip_address():
    # Get the hostname
    hostname = socket.gethostname()
    # Get the IP address
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def run_server():
    ip_address = get_ip_address()
    
    # Print access information
    print(f"Starting server at http://{ip_address}:{PORT}")
    print(f"Local access: http://localhost:{PORT}")
    print(f"Remote access: http://{ip_address}:{PORT}")
    print("Press Ctrl+C to stop")
    
    # Create a server that allows connections from any IP
    with socketserver.TCPServer(("0.0.0.0", PORT), CORSHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
            httpd.server_close()

# For Jupyter notebook integration
def jupyter_server():
    ip_address = get_ip_address()
    
    # Create an iframe that points to the server
    iframe_html = f"""
    <div style="width:100%; height:600px; margin-top:20px; position:relative;">
        <div style="position:absolute; top:0; right:0; padding:5px; background:#f0f0f0; z-index:1000; font-family:monospace;">
            Remote URL: <a href="http://{ip_address}:{PORT}" target="_blank">http://{ip_address}:{PORT}</a>
        </div>
        <iframe src="http://localhost:{PORT}" width="100%" height="100%" frameborder="0"></iframe>
    </div>
    """
    
    # Start the server in a separate thread
    import threading
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Display the iframe
    display(HTML(iframe_html))
    
    return f"Server running at http://{ip_address}:{PORT}"

if __name__ == "__main__":
    run_server() 