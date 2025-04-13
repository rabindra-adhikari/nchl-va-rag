from flask import Flask, render_template, request, jsonify, session
import requests
import uuid
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key for production

# Rasa endpoint
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

# Load location data (from locator system)
json_path = os.path.join(app.root_path, 'data/knowledge_base/locations.json')
try:
    with open(json_path, 'r') as f:
        location_data = json.load(f)
    branches = location_data.get('branches', [])
    atms = location_data.get('atms', [])
except Exception as e:
    print(f"Error loading location data: {e}")
    branches, atms = [], []

def get_conversation_id():
    """Generate and return a unique conversation ID stored in session."""
    if 'conversation_id' not in session:
        session['conversation_id'] = f"conversation_{uuid.uuid4().hex}"
    return session['conversation_id']

@app.route('/')
def index():
    """Render the main chat page along with any previous messages and example questions."""
    messages = session.get('messages', [])
    example_questions = [
        "features of Siddhartha BankSmart XP",
        "activate Siddhartha BankSmart XP",
        "channels for Siddhartha BankSmart XP",
        "Where’s the nearest ATM?",
        "Find a branch near me"
    ]
    return render_template("index.html", messages=messages, example_questions=example_questions)

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history and conversation ID from the session."""
    session.pop('messages', None)
    session.pop('conversation_id', None)
    return jsonify(success=True)

@app.route('/send_message', methods=['POST'])
def send_message():
    """Receive a user message, pass it to Rasa with location data, and return the assistant’s reply."""
    data = request.get_json()
    user_message = data.get('message')
    user_location = data.get('location')  # Geolocation from frontend (e.g., {"lat": 27.7172, "lon": 85.3240})
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get or create conversation id
    conversation_id = get_conversation_id()
    messages = session.get('messages', [])
    messages.append({"role": "user", "content": user_message})
    session['messages'] = messages

    # Prepare payload for Rasa, including user location as metadata
    payload = {
        "sender": conversation_id,
        "message": user_message,
        "metadata": {"location": user_location} if user_location else {}
    }

    try:
        response = requests.post(RASA_URL, json=payload)
        response.raise_for_status()
        responses = response.json()
    except Exception as e:
        error_message = f"Error communicating with Rasa: {e}"
        messages.append({"role": "assistant", "content": error_message})
        session['messages'] = messages
        return jsonify({"reply": error_message})

    # Process Rasa responses
    bot_reply = {"text": "", "custom": None}
    for res in responses:
        if "text" in res:
            bot_reply["text"] += res["text"] + "\n\n"
        if "custom" in res:
            bot_reply["custom"] = res["custom"]

    bot_reply["text"] = bot_reply["text"].strip() or "I'm sorry, I didn't get a proper response."

    # Store the assistant's response in session
    messages.append({"role": "assistant", "content": bot_reply["text"], "custom": bot_reply["custom"]})
    session['messages'] = messages

    # Return the full response (text + custom payload) to the frontend
    return jsonify({"reply": bot_reply["text"], "custom": bot_reply["custom"]})

if __name__ == '__main__':
    app.run(debug=True, port=8501)