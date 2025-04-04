import json
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

# Set up logging
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

# ----- Models Initialization -----
load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    logger.error("HF_TOKEN not found in environment variables. Please set it.")

try:
    retrieval_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    #retrieval_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    #cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
except Exception as e:
    logger.error(f"Error loading retrieval or cross-encoder model: {e}")

try:
    tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=token)
    model_llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=token)
except Exception as e:
    logger.error(f"Error loading Llama‑3.2-1B model: {e}")
    tokenizer_llama, model_llama = None, None

# ----- Helper: Parameterized Retrieval Function -----
def retrieve_faq_custom(query: str, kb_data: List[Dict[Text, Any]], kb_index, k: int = 3, ce_threshold: float = 0.0) -> List[Dict[Text, Any]]:
    """
    Retrieve FAQ entries from a given knowledge base using dense retrieval and cross-encoder re-ranking.
    """
    if kb_index is None:
        logger.error("FAISS index is not loaded for the given knowledge base.")
        return None

    try:
        query_embedding = retrieval_model.encode([query], convert_to_numpy=True)
        distances, indices = kb_index.search(query_embedding, k)
    except Exception as e:
        logger.error(f"Error during FAISS search: {e}")
        return None

    candidate_pairs = []
    candidate_indices = []
    for idx in indices[0]:
        candidate_question = kb_data[idx]['question']
        candidate_pairs.append((query, candidate_question))
        candidate_indices.append(idx)

    try:
        ce_scores = cross_encoder.predict(candidate_pairs)
    except Exception as e:
        logger.error(f"Error during cross-encoder prediction: {e}")
        return None

    results = []
    for distance, idx, ce_score in zip(distances[0], candidate_indices, ce_scores):
        faq_entry = kb_data[idx].copy()
        faq_entry["retrieval_distance"] = float(distance)
        faq_entry["ce_score"] = ce_score
        results.append(faq_entry)

    results = sorted(results, key=lambda x: x["ce_score"], reverse=True)
    if ce_threshold is not None:
        results = [res for res in results if res["ce_score"] >= ce_threshold]

    return results if results else None

# ----- Llama‑2 Final Answer Generation -----
def generate_final_answer_llama2(query: str, retrieved_faqs: List[Dict[Text, Any]]) -> str:
    if not retrieved_faqs:
        return ("I am the virtual assistant for this bank. "
                "I can help you with queries related to our banking services. "
                "Please ask me something specific about this bank, or contact your branch for further assistance.")
    else:
        context = "\n".join([faq['answer'] for faq in retrieved_faqs])
        prompt = (
            "System: You are a knowledgeable assistant that synthesizes relevant information from a specialized knowledge base to answer mobile banking queries.\n"
            "User: " + query + "\n"
            "System: Here is the relevant information extracted from our knowledge base:\n" + context + "\n"
            "Assistant: Final Answer: "
        )

        if not tokenizer_llama or not model_llama:
            logger.error("Llama‑3.2-1B model or tokenizer is not loaded.")
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
            logger.error(f"Error during Llama‑3.1-8B generation: {e}")
            final_answer = "There was an error generating the final answer. Please try again later."
        return final_answer

# ----- Custom Action for Rasa -----
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

            # 1. Try bank-specific knowledge base retrieval
            retrieved_bank_faqs = retrieve_faq_custom(user_query, bank_faq_data, bank_index, k=3, ce_threshold=0.0)
            if retrieved_bank_faqs:
                final_answer = generate_final_answer_llama2(user_query, retrieved_bank_faqs)
                dispatcher.utter_message(text=final_answer)
                return []
            
            # 2. If not found, try general knowledge base retrieval
            retrieved_general_faqs = retrieve_faq_custom(user_query, general_faq_data, general_index, k=3, ce_threshold=0.0)
            if retrieved_general_faqs:
                final_answer = generate_final_answer_llama2(user_query, retrieved_general_faqs)
                dispatcher.utter_message(text=final_answer)
                return []
            
            # 3. Fallback response using a dynamic message with bank name
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
