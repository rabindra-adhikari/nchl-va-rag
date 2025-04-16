// Initialize variables to track conversation
let messageId = 0;
const messages = [];

// Add this at an appropriate location in your script.js file
// This should be loaded by your EMI form page

document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the EMI form page
    const emiForm = document.getElementById('emiForm');
    if (emiForm) {
        emiForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator in the form if you have one
            const submitBtn = document.querySelector('#emiForm button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = 'Calculating...';
            }
            
            // Get form data
            const formData = {
                loan_type: document.getElementById('loan_type').value,
                amount: document.getElementById('amount').value,
                tenure: document.getElementById('tenure').value
            };
            
            // Send to backend
            fetch('/calculate_emi', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                // Reset button state
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = 'Calculate';
                }
                
                // Send result back to parent window (chat interface)
                window.parent.postMessage({
                    type: 'emi_result',
                    data: data.message
                }, '*');
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Reset button state
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = 'Calculate';
                }
                
                // Send error to parent
                window.parent.postMessage({
                    type: 'emi_result',
                    data: 'Sorry, an error occurred while calculating the EMI. Please try again.'
                }, '*');
            });
        });
    }
});

// Send user message to the bot
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    
    if (message) {
        // Add user message to chat
        addMessage('user', message);
        
        // Clear input field
        userInput.value = '';
        
        // Send message to bot
        sendToBot(message);
    }
}

// Send message to the bot API
function sendToBot(message, isInitial = false) {
    // Show typing indicator
    if (!isInitial) {
        showTypingIndicator();
    }
    
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove typing indicator
        removeTypingIndicator();
        
        console.log('Bot response:', data); // Debug: Log bot response
        
        // Process each response from the bot
        data.forEach(response => {
            if (response.text) {
                addMessage('bot', response.text);
            }
            
            // Handle form payloads
            const formPayload = response.json_message || response.custom;
            if (formPayload && formPayload.type === 'form') {
                console.log('Form payload detected:', formPayload.payload); // Debug: Log form payload
                addForm(formPayload.payload);
            }
        });
        
        // Scroll to the bottom of the chat
        scrollToBottom();
    })
    .catch(error => {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage('bot', 'Sorry, I encountered an error. Please try again.');
    });
}

// Handle EMI calculation results
function handleEmiResults(responses) {
    console.log('Handling EMI results:', responses); // Debug: Log responses
    // Process each response from the bot
    responses.forEach(response => {
        if (response.text) {
            addMessage('bot', response.text);
        }
    });
    
    // Scroll to the bottom of the chat
    scrollToBottom();
}

// Add message to the chat
function addMessage(sender, content) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
    
    const iconSpan = document.createElement('span');
    iconSpan.classList.add('message-icon');
    iconSpan.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = content;
    
    messageDiv.appendChild(iconSpan);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Store message
    messages.push({
        id: messageId++,
        sender: sender,
        content: content
    });
}

// Add form to the chat
function addForm(formPayload) {
    console.log('Adding form to chat:', formPayload); // Debug: Log form addition
    const chatMessages = document.getElementById('chat-messages');
    const formContainerDiv = document.createElement('div');
    formContainerDiv.classList.add('form-message');
    
    // Create an iframe to load the form
    const iframe = document.createElement('iframe');
    iframe.style.width = '100%';
    iframe.style.height = '470px'; // Adjust height as needed
    iframe.style.border = 'none';
    iframe.style.borderRadius = '10px';
    iframe.style.overflow = 'hidden';
    
    // Add the iframe to the chat
    formContainerDiv.appendChild(iframe);
    chatMessages.appendChild(formContainerDiv);
    
    // Set iframe source to load the form
    if (formPayload.template === 'emi_form.html') {
        iframe.src = '/emi_form';
    } else {
        console.error('Invalid form template:', formPayload.template);
        addMessage('bot', 'Error loading the form. Please try again.');
    }
    
    // Scroll to the bottom
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    
    typingDiv.id = 'typing-indicator';
    typingDiv.classList.add('message', 'bot-message');
    
    const iconSpan = document.createElement('span');
    iconSpan.classList.add('message-icon');
    iconSpan.innerHTML = '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = '<div class="typing-dots"><span>.</span><span>.</span><span>.</span></div>';
    
    typingDiv.appendChild(iconSpan);
    typingDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Scroll to the bottom of the chat
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}