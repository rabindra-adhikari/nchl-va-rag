from flask import Flask, render_template, request, jsonify, session
import requests
import uuid
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Rasa endpoint (for your REST channel)
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

# Load location data (for mobile banking)
json_path = os.path.join(app.root_path, 'data/knowledge_base/locations.json')
try:
    with open(json_path, 'r') as f:
        location_data = json.load(f)
    branches = location_data.get('branches', [])
    atms = location_data.get('atms', [])
except Exception as e:
    print(f"Error loading location data: {e}")
    branches, atms = [], []

# --- Helper for conversation ID ---
def get_conversation_id():
    if 'conversation_id' not in session:
        session['conversation_id'] = f"conversation_{uuid.uuid4().hex}"
    return session['conversation_id']

@app.route('/')
def index():
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
    session.pop('messages', None)
    session.pop('conversation_id', None)
    return jsonify(success=True)

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message')
    user_location = data.get('location')  # Optional location data
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    conversation_id = get_conversation_id()
    messages = session.get('messages', [])
    messages.append({"role": "user", "content": user_message})
    session['messages'] = messages

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

    bot_reply = {"text": "", "custom": None}
    for res in responses:
        if "text" in res:
            bot_reply["text"] += res["text"] + "\n\n"
        if "custom" in res:
            bot_reply["custom"] = res["custom"]

    bot_reply["text"] = bot_reply["text"].strip() or "I'm sorry, I didn't get a proper response."
    messages.append({"role": "assistant", "content": bot_reply["text"], "custom": bot_reply["custom"]})
    session['messages'] = messages

    return jsonify({"reply": bot_reply["text"], "custom": bot_reply["custom"]})

# ----- EMI Endpoints Integration -----
# Note: These endpoints are added so that the EMI form (loaded via an iframe) works seamlessly within the main UI.

# Import EMI helper functions (ensure that actions/emi_calculator.py is in your project)
try:
    from actions.emi_calculator import loan_types, calculate_emi
except ImportError:
    loan_types = {}
    def calculate_emi(principal, annual_rate, tenure_years):
        return {"monthly_emi": 0, "total_payment": 0, "total_interest": 0, "yearly_emi": 0}

@app.route('/emi_form')
def emi_form():
    # Render the EMI form template; it will use the EMI module’s HTML and load its own static assets if needed.
    return render_template('emi_form.html', loan_types=loan_types.keys())

@app.route('/submit_emi', methods=['POST'])
def submit_emi():
    data = request.form
    try:
        loan_type = data.get('loan_type')
        amount = float(data.get('amount', 0))
        tenure = data.get('tenure')
        try:
            tenure_years = float(tenure.split()[0]) if "year" in tenure.lower() else float(tenure.split()[0]) / 12
        except Exception as e:
            print(f"Invalid tenure format: {e}")
            return jsonify([{"text": "Invalid tenure format."}])
        interest_rates = loan_types.get(loan_type, {}).get("interest_rates", {})
        selected_rate = None
        for key, rate in interest_rates.items():
            if key in tenure.lower() or "default" in key:
                selected_rate = rate
                break
        if not selected_rate:
            selected_rate = list(interest_rates.values())[0] if interest_rates else 0
        if not selected_rate:
            print("Invalid loan type or tenure")
            return jsonify([{"text": "Invalid loan type or tenure."}])
        result = calculate_emi(amount, selected_rate, tenure_years)
        message = (
            f"For {loan_type} of NPR {amount:,.2f}:\n"
            f"- Monthly EMI: NPR {result['monthly_emi']:,.2f}\n"
            f"- Yearly EMI: NPR {result['yearly_emi']:,.2f}\n"
            f"- Interest Rate: {selected_rate}% per annum\n"
            f"- Total Interest: NPR {result['total_interest']:,.2f}\n"
            f"- Total Payment: NPR {result['total_payment']:,.2f}"
        )
        print(f"EMI calculation result: {message}")
        responses = [{"text": message}]
    except Exception as e:
        print(f"Error in submit_emi: {e}")
        responses = [{"text": f"Error calculating EMI: {str(e)}. Please try again."}]
    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True, port=8501)