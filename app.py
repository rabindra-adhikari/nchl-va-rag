import streamlit as st
import requests
import json
import time
import os
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(
    page_title="Banking Virtual Assistant",
    page_icon="🏦",
    layout="centered"
)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = f"conversation_{int(time.time())}"

# Function to send message to Rasa
def send_message_to_rasa(message_text):
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {
        "sender": st.session_state.conversation_id,
        "message": message_text
    }
    try:
        response = requests.post(rasa_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Rasa: {e}")
        return [{"text": "Sorry, I'm having trouble connecting to my backend right now."}]

# Function to handle example question clicks
def handle_example_click(question):
    # Set as the new input and trigger processing
    process_user_input(question)
    # Force streamlit to rerun to show the new messages
    st.rerun()

# Function to process user input - extracted to be reusable
def process_user_input(user_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response from Rasa
    responses = send_message_to_rasa(user_input)
    
    # Process all responses (Rasa might return multiple messages)
    bot_message = ""
    for response in responses:
        if "text" in response:
            bot_message += response["text"] + "\n\n"
    
    if not bot_message:
        bot_message = "I'm sorry, I didn't get a proper response."
    
    # Remove trailing newlines
    bot_message = bot_message.strip()
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_message})

# Sidebar with example questions
with st.sidebar:
    st.header("🏦 Siddhartha Bank")
    
    # Add clear conversation button
    if st.button("Clear Conversation", key="clear_button"):
        st.session_state.messages = []
        st.session_state.conversation_id = f"conversation_{int(time.time())}"
        st.rerun()
    
    # Example questions section
    st.subheader("Example Questions")
    st.markdown("Click on any example to ask:")
    
    # List of example questions
    example_questions = [
        "What are the various features of mobile banking / Siddhartha BankSmart XP?",
        "How can I register/activate the mobile banking / Siddhartha BankSmart XP service?",
        "Can I subscribe mobile banking / Siddhartha BankSmart XP without visiting branch?",
        "Can a non-account holder subscribe mobile banking / Siddhartha BankSmart XP?",
        "From which channels can mobile banking / Siddhartha BankSmart XP be used?"
    ]
    
    # Create buttons for each example question
    for i, question in enumerate(example_questions):
        question_button = st.button(f"📱 {question}", key=f"q{i}", use_container_width=True)
        if question_button:
            handle_example_click(question)

# Main chat interface
st.title("🏦 Banking Virtual Assistant")
st.markdown("Ask me questions about mobile banking services.")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Handle user input
user_input = st.chat_input("Type your message here...")
if user_input:
    process_user_input(user_input)
    st.rerun()