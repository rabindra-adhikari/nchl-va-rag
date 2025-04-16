import json
import os
import logging
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# ----- Logging Setup -----
logger = logging.getLogger(__name__)

# ----- Bank-Specific Knowledge Base Setup -----
try:
    with open('./data/knowledge_base/bank_knowledge_base.json', 'r', encoding='utf-8') as f:
        bank_faq_data = json.load(f)
    if isinstance(bank_faq_data, dict) and "faqs" in bank_faq_data:
        bank_faq_data = bank_faq_data["faqs"]
except Exception as e:
    logger.error(f"Error loading bank FAQ data: {e}")
    bank_faq_data = []

try:
    bank_index = faiss.read_index('./data/index/bank_faiss.index')
except Exception as e:
    logger.error(f"Error loading bank FAISS index: {e}")
    bank_index = None

# ----- General Knowledge Base Setup -----
try:
    with open('./data/knowledge_base/general_knowledge_base.json', 'r', encoding='utf-8') as f:
        general_faq_data = json.load(f)
    if isinstance(general_faq_data, dict) and "faqs" in general_faq_data:
        general_faq_data = general_faq_data["faqs"]
except Exception as e:
    logger.error(f"Error loading general FAQ data: {e}")
    general_faq_data = []

try:
    general_index = faiss.read_index('./data/index/general_faiss.index')
except Exception as e:
    logger.error(f"Error loading general FAISS index: {e}")
    general_index = None

# ----- Location Data Setup -----
import os
json_path = os.path.join(os.path.dirname(__file__), '../data/knowledge_base/locations.json')
try:
    with open(json_path, 'r') as f:
        location_data = json.load(f)
    branches = location_data.get('branches', [])
    atms = location_data.get('atms', [])
except Exception as e:
    logger.error(f"Error loading location data: {e}")
    branches, atms = [], []

# ----- Models Initialization -----
load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    logger.error("HF_TOKEN not found in environment variables. Please set it.")

try:
    retrieval_model = SentenceTransformer("./fine_tuned_model")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    logger.error(f"Error loading retrieval or cross-encoder model: {e}")

try:
    tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_auth_token=token)
    model_llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_auth_token=token)
except Exception as e:
    logger.error(f"Error loading Llama‑3.2‑1B model: {e}")
    tokenizer_llama, model_llama = None, None

# ----- Helper Function: FAQ Retrieval with Deduplication -----
def retrieve_faq_custom(query: str, kb_data: List[Dict[Text, Any]], kb_index,
                        initial_k: int = 15, final_k: int = 3, ce_threshold: float = 0.0) -> List[Dict[Text, Any]]:
    if kb_index is None:
        logger.error("FAISS index is not loaded for the given knowledge base.")
        return None

    try:
        query_embedding = retrieval_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = kb_index.search(query_embedding, initial_k)
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}")
        return None

    candidate_pairs = []
    candidate_results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(kb_data):
            continue
        faq_entry = kb_data[idx].copy()
        combined_text = f"{faq_entry.get('question', '')} {faq_entry.get('answer', '')}"
        candidate_pairs.append((query, combined_text))
        faq_entry["retrieval_distance"] = float(distances[0][i])
        candidate_results.append(faq_entry)

    try:
        ce_scores = cross_encoder.predict(candidate_pairs)
    except Exception as e:
        logger.error(f"Error during cross-encoder prediction: {e}")
        return None

    for i, score in enumerate(ce_scores):
        candidate_results[i]["ce_score"] = score

    distinct_results = {}
    for entry in candidate_results:
        answer = entry.get("answer", "").strip()
        if answer in distinct_results:
            if entry["ce_score"] > distinct_results[answer]["ce_score"]:
                distinct_results[answer] = entry
        else:
            distinct_results[answer] = entry

    sorted_results = sorted(distinct_results.values(), key=lambda x: x["ce_score"], reverse=True)
    if ce_threshold is not None:
        sorted_results = [res for res in sorted_results if res["ce_score"] >= ce_threshold]

    return sorted_results[:final_k] if sorted_results else None

# ----- Llama‑3.2 Final Answer Generation -----
def generate_final_answer_llama2(query: str, retrieved_faqs: List[Dict[Text, Any]]) -> str:
    if not retrieved_faqs:
        return (
            "I am the virtual assistant for this bank. "
            "I can help you with queries related to our banking services. "
            "Please ask me something specific about this bank, or contact your branch for further assistance."
        )
    else:
        template = (
            "System: {instructions}\n"
            "Context: {context}\n"
            "User: {query}\n"
            "Assistant: Final Answer: "
        )
        prompt = template.format(
            instructions="You are an expert mobile banking assistant synthesizing info from our FAQ. "
                         "Please analyze the context and provide the best answer.",
            context="\n".join([faq.get('answer', '') for faq in retrieved_faqs]),
            query=query
        )

        if not tokenizer_llama or not model_llama:
            logger.error("Llama‑3.2‑1B model or tokenizer is not loaded.")
            return "There was an error generating the final answer. Please try again later."

        try:
            input_ids = tokenizer_llama.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model_llama.generate(
                input_ids,
                max_new_tokens=512,
                num_beams=4,
                temperature=0.5,
                repetition_penalty=1.2,
                early_stopping=True
            )
            generated_tokens = outputs[0][input_ids.shape[1]:]
            final_answer = tokenizer_llama.decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error during Llama‑3.2‑1B generation: {e}")
            final_answer = "There was an error generating the final answer. Please try again later."
        return final_answer

# ----- Haversine Distance Calculation -----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dLat = (lat2 - lat1) * 3.14159 / 180
    dLon = (lon2 - lon1) * 3.14159 / 180
    a = (dLat / 2) * (dLat / 2) + (lat1 * 3.14159 / 180) * (lat2 * 3.14159 / 180) * (dLon / 2) * (dLon / 2)
    c = 2 * ((a ** 0.5) * ((1 - a) ** 0.5))
    return R * c

# ----- Custom Action for Mobile Banking -----
class ActionMobileBankingResponse(Action):
    def name(self) -> Text:
        return "action_mobile_banking_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            user_query = tracker.latest_message.get("text")
            if not user_query:
                dispatcher.utter_message(text="Sorry, I didn't catch that. Could you please rephrase?")
                return []
            
            logger.info(f"Received user query: {user_query}")
            retrieved_bank_faqs = retrieve_faq_custom(user_query, bank_faq_data, bank_index,
                                                      initial_k=15, final_k=3, ce_threshold=0.0)
            if retrieved_bank_faqs:
                final_answer = generate_final_answer_llama2(user_query, retrieved_bank_faqs)
                dispatcher.utter_message(text=final_answer)
                return []
            
            retrieved_general_faqs = retrieve_faq_custom(user_query, general_faq_data, general_index,
                                                         initial_k=15, final_k=3, ce_threshold=0.0)
            if retrieved_general_faqs:
                final_answer = generate_final_answer_llama2(user_query, retrieved_general_faqs)
                dispatcher.utter_message(text=final_answer)
                return []
            
            bank_name = os.environ.get("BANK_NAME", "this bank")
            fallback_message = (
                f"I'm sorry, I cannot provide information regarding your question. "
                f"I am the virtual assistant for {bank_name}. "
                f"Please ask me something specific about {bank_name}'s services, or contact your branch for further assistance."
            )
            dispatcher.utter_message(text=fallback_message)
        except Exception as e:
            logger.error(f"Error in ActionMobileBankingResponse: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request.")
        return []

# ----- Custom Action for Nearest Location with Map -----
class ActionFindNearestLocation(Action):
    def name(self) -> Text:
        return "action_find_nearest_location"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message.get("intent", {}).get("name")
        user_message = tracker.latest_message.get("text", "").lower()
        
        location_keywords = ["near", "nearest", "where", "location", "find", "show", "close", "closest", "atm", "branch"]
        if not any(keyword in user_message for keyword in location_keywords):
            dispatcher.utter_message(text="It seems like you're not asking for a location. Could you clarify your request?")
            return []

        metadata = tracker.latest_message.get("metadata", {})
        user_location = metadata.get("location", {"lat": 27.7172, "lon": 85.3240})
        user_lat, user_lon = user_location["lat"], user_location["lon"]

        if "atm" in intent or "atm" in user_message:
            data = atms
            location_type = "ATMs"
        else:
            data = branches
            location_type = "Branches"

        if not data:
            dispatcher.utter_message(text=f"Sorry, I don’t have data for {location_type} at the moment.")
            return []

        for item in data:
            item["distance"] = haversine(user_lat, user_lon, item["latitude"], item["longitude"])
        data.sort(key=lambda x: x["distance"])

        locations = [
            {
                "name": item["name"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "distance": item["distance"]
            } for item in data
        ]

        payload = {
            "text": f"Here are the nearest {location_type} to your current location. Please select a location option below to proceed with finding the closest one for you:",
            "custom": {
                "type": "location_map",
                "user_location": {"lat": user_lat, "lon": user_lon},
                "locations": locations,
                "location_type": location_type.lower()
            }
        }
        dispatcher.utter_message(**payload)
        return []

# ----- EMI Module Actions -----
# Import EMI helper functions and data from a separate module.
try:
    from .emi_calculator import loan_types, calculate_emi
except Exception as e:
    logger.error(f"Error importing EMI calculator module: {e}")
    loan_types = {}
    def calculate_emi(principal, annual_rate, tenure_years):
        return {"monthly_emi": 0, "total_payment": 0, "total_interest": 0, "yearly_emi": 0}
    

class ActionShowEmiForm(Action):
    def name(self) -> Text:
        return "action_show_emi_form"  # Exact match with domain.yml

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logger.debug("Executing action_show_emi_form")
        
        # Send a form payload to be rendered in the frontend
        payload = {
            "text": "Please provide your loan details below to calculate your monthly EMI, total interest, and overall payment.",
            "custom": {
                "type": "form"
            }
        }
        dispatcher.utter_message(**payload)
        return []

class ActionCalculateEMI(Action):
    def name(self) -> Text:
        return "action_calculate_emi"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logger.debug("Executing action_calculate_emi")
        
        # Check if this is the initial request for the form
        # If slots are not filled, we should show the form
        if not all([tracker.get_slot("loan_type"), tracker.get_slot("amount"), tracker.get_slot("tenure")]):
            # Send a form payload to be rendered in the frontend
            payload = {
                "text": "Please provide your loan details below to calculate your monthly EMI, total interest, and overall payment.",
                "custom": {
                    "type": "form"
                }
            }
            dispatcher.utter_message(**payload)
            return []
            
        # The rest of your existing code for when the form is submitted
        logger.debug(f"Slots: loan_type={tracker.get_slot('loan_type')}, amount={tracker.get_slot('amount')}, tenure={tracker.get_slot('tenure')}")
        
        loan_type = tracker.get_slot("loan_type")
        try:
            amount = float(tracker.get_slot("amount") or 0)
        except Exception as e:
            dispatcher.utter_message(text="Invalid amount provided.")
            return []
        tenure = tracker.get_slot("tenure")

        if not all([loan_type, amount, tenure]):
            logger.warning("Missing required information")
            dispatcher.utter_message(text="Please provide all required information.")
            return []

        try:
            tenure_years = float(tenure.split()[0]) if "year" in tenure.lower() else float(tenure.split()[0]) / 12
        except Exception as e:
            logger.error(f"Invalid tenure format: {e}")
            dispatcher.utter_message(text="Invalid tenure format.")
            return []

        interest_rates = loan_types.get(loan_type, {}).get("interest_rates", {})
        selected_rate = None
        for key, rate in interest_rates.items():
            if key in tenure.lower() or "default" in key:
                selected_rate = rate
                break
        if not selected_rate:
            selected_rate = list(interest_rates.values())[0] if interest_rates else 0

        if not selected_rate:
            logger.error("Invalid loan type or tenure")
            dispatcher.utter_message(text="Invalid loan type or tenure.")
            return []

        result = calculate_emi(amount, selected_rate, tenure_years)

        message = (
            f"For {loan_type} of NPR {amount:,.2f}:\n"
            f"- Monthly EMI: NPR {result['monthly_emi']:,.2f}\n"
            f"- Yearly EMI: NPR {result['yearly_emi']:,.2f}\n"
            f"- Interest Rate: {selected_rate}% per annum\n"
            f"- Total Interest: NPR {result['total_interest']:,.2f}\n"
            f"- Total Payment: NPR {result['total_payment']:,.2f}"
        )

        logger.debug(f"Sending EMI calculation: {message}")
        dispatcher.utter_message(text=message)
        return []

class ActionShowInterestRates(Action):
    def name(self) -> Text:
        return "action_show_interest_rates"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logger.debug("Executing action_show_interest_rates")
        rates_message = "Interest Rates:\n"
        for loan, details in loan_types.items():
            rates_message += f"\n{loan}:\n"
            for tenure, rate in details["interest_rates"].items():
                rates_message += f"- {tenure.replace('_', ' ')}: {rate}%\n"
        dispatcher.utter_message(text=rates_message)
        return []
