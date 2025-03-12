document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const tokensPerSecondElement = document.getElementById('tokens-per-second');
    
    // Get the API URL - this will be dynamically replaced when running in Jupyter
    const API_URL = 'http://localhost:8000/v1/chat/completions';
    
    let conversation = [
        {
            role: 'system',
            content: 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'
        }
    ];
    let streamingInProgress = false;
    let tokensPerSecond = 0;
    let tokenCount = 0;
    let startTime = null;
    
    // Add event listeners
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    // Add a system message at the start
    addMessage('Hello! How can I help you today?', 'system');
    
    function handleSendMessage() {
        const message = userInput.value.trim();
        if (message === '' || streamingInProgress) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        userInput.value = '';
        
        // Add user message to conversation history
        conversation.push({
            role: 'user',
            content: message
        });
        
        // Disable input during streaming
        streamingInProgress = true;
        sendButton.disabled = true;
        
        // Create a new message element for the system response
        const systemMessageElement = document.createElement('div');
        systemMessageElement.className = 'message system-message';
        chatContainer.appendChild(systemMessageElement);
        
        // Reset token counting
        tokenCount = 0;
        startTime = Date.now();
        
        // Stream the response
        streamResponse(systemMessageElement);
    }
    
    function addMessage(content, sender) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}-message`;
        messageElement.textContent = content;
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    async function streamResponse(messageElement) {
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': window.location.origin
                },
                body: JSON.stringify({
                    model: 'Qwen/QwQ-32B-AWQ',
                    messages: conversation,
                    stream: true,
                    max_tokens: 1000
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed with status ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let systemMessage = '';
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                        try {
                            const data = JSON.parse(line.substring(6));
                            if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
                                const content = data.choices[0].delta.content;
                                systemMessage += content;
                                messageElement.textContent = systemMessage;
                                
                                // Update token count and tokens per second
                                tokenCount++;
                                const elapsedSeconds = (Date.now() - startTime) / 1000;
                                tokensPerSecond = Math.round(tokenCount / elapsedSeconds);
                                tokensPerSecondElement.textContent = `${tokensPerSecond} tokens/s`;
                                
                                // Scroll to bottom
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        } catch (e) {
                            console.error('Error parsing JSON:', e);
                        }
                    }
                }
            }
            
            // Add the complete system message to conversation history
            conversation.push({
                role: 'assistant',
                content: systemMessage
            });
            
        } catch (error) {
            console.error('Error:', error);
            messageElement.textContent = 'Error: Could not connect to the API. Make sure the server is running.';
            messageElement.style.color = 'red';
        } finally {
            streamingInProgress = false;
            sendButton.disabled = false;
        }
    }
}); 