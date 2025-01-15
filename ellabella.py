import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import requests
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='ellabella.log'
)

@dataclass
class Message:
    content: str
    timestamp: datetime
    sender: str

class KnowledgeBase:
    """Knowledge base for common topics"""
    def __init__(self):
        self.knowledge = {
            "usa": """
                The United States of America (USA) is a federal republic consisting of 50 states, 
                a federal district, and various territories. It is the world's third-largest country 
                by total area and population. The capital is Washington, D.C., and the largest city 
                is New York City. The USA is known for its diverse geography, multicultural population, 
                strong economy, and significant global influence in politics, culture, and technology.
                """,
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know about?",
                "Greetings! I'm here to assist you.",
            ],
            "about_me": """
                I'm EllaBella, an AI assistant designed to help answer questions and provide information 
                on various topics. I aim to be helpful while being clear about my capabilities and limitations.
                """
        }

    def get_knowledge(self, topic: str) -> Optional[str]:
        """Get information about a specific topic"""
        # Clean and standardize the topic key
        topic_key = topic.lower().strip()
        
        # Check for exact matches
        if topic_key in self.knowledge:
            return self.knowledge[topic_key]
            
        # Check for partial matches
        for key in self.knowledge:
            if key in topic_key or topic_key in key:
                return self.knowledge[key]
        
        return None

class EllaBella:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.conversation_history: List[Message] = []
        self.user_preferences: Dict = self._load_preferences()
        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        self.knowledge_base = KnowledgeBase()
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU
            )
        except Exception as e:
            logging.error(f"Error loading sentiment model: {e}")
            self.sentiment_analyzer = None

    def _load_preferences(self) -> Dict:
        default_preferences = {
            "name": "User",
            "preferences": {
                "theme": "light",
                "notification_enabled": True,
                "language": "en"
            }
        }
        
        try:
            with open('user_preferences.json', 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    self._save_default_preferences(default_preferences)
                    return default_preferences
        except FileNotFoundError:
            self._save_default_preferences(default_preferences)
            return default_preferences

    def _identify_intent(self, user_input: str) -> Tuple[str, float]:
        """Identify the user's intent from their input"""
        input_lower = user_input.lower()
        
        # Define intent patterns
        intents = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"],
            "about_bot": ["who are you", "what are you", "tell me about yourself"],
            "farewell": ["goodbye", "bye", "see you", "quit", "exit"],
            "about_topic": ["tell me about", "what is", "who is", "explain"],
            "personal_info": ["i am", "i'm", "my name", "about me"],
            "affirmation": ["yes", "yeah", "sure", "okay", "ok"],
            "negation": ["no", "nope", "not"],
            "gratitude": ["thank", "thanks", "appreciate"]
        }
        
        # Check each intent
        for intent, patterns in intents.items():
            for pattern in patterns:
                if pattern in input_lower:
                    return intent, 0.8
        
        return "general_query", 0.5

    def generate_response(self, user_input: str) -> str:
        """Generate a response based on user input"""
        # Add user message to history
        self.add_message(user_input, "user")
        
        try:
            # Identify intent
            intent, confidence = self._identify_intent(user_input)
            
            # Handle different intents
            if intent == "greeting":
                response = "Hello! How can I help you today?"
            elif intent == "about_bot":
                response = self.knowledge_base.get_knowledge("about_me")
            elif intent == "farewell":
                response = "Goodbye! Have a great day!"
            elif intent == "gratitude":
                response = "You're welcome! Is there anything else you'd like to know?"
            elif "tell me about" in user_input.lower():
                # Extract topic after "tell me about"
                topic = user_input.lower().split("tell me about")[-1].strip()
                knowledge = self.knowledge_base.get_knowledge(topic)
                if knowledge:
                    response = knowledge
                else:
                    # Use API for unknown topics
                    response = self._get_api_response(user_input)
            else:
                response = self._get_api_response(user_input)
            
            # Clean and format the response
            response = self._format_response(response)
            
        except Exception as e:
            response = "I apologize, but I encountered an error while processing your request."
            logging.error(f"Error generating response: {e}")
        
        # Add bot response to history
        self.add_message(response, "EllaBella")
        return response

    def _get_api_response(self, user_input: str) -> str:
        """Get response from API for unknown topics"""
        API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        
        try:
            response = requests.post(
                API_URL,
                headers=self.headers,
                json={
                    "inputs": user_input,
                    "parameters": {
                        "max_length": 100,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                }
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if isinstance(response_data, list) and response_data:
                    return response_data[0].get('generated_text', '')
            
            return "I apologize, but I need more context to provide a helpful response. Could you please elaborate?"
            
        except Exception as e:
            logging.error(f"API error: {e}")
            return "I apologize, but I'm having trouble accessing my knowledge base right now."

    def _format_response(self, text: str) -> str:
        """Clean and format the response"""
        if not text:
            return "I apologize, but I'm having trouble generating a response right now."
            
        # Remove any prefixes
        text = text.replace("EllaBella:", "").replace("Assistant:", "").strip()
        
        # Clean up the text
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        
        # Ensure response isn't too long
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:497] + "..."
            
        return cleaned_text

    def add_message(self, content: str, sender: str):
        message = Message(
            content=content,
            timestamp=datetime.now(),
            sender=sender
        )
        self.conversation_history.append(message)
        logging.info(f"New message from {sender}: {content[:50]}...")

    def analyze_sentiment(self, text: str) -> Optional[Dict]:
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)
                return result[0]
            except Exception as e:
                logging.error(f"Error analyzing sentiment: {e}")
                return None
        return None

    def get_conversation_summary(self) -> List[Dict]:
        return [
            {
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "sender": msg.sender
            }
            for msg in self.conversation_history[-10:]  # Last 10 messages
        ]

def main():
    # Get API token from environment variable
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        raise ValueError("Please set HUGGINGFACE_API_TOKEN environment variable")

    try:
        # Initialize EllaBella
        chatbot = EllaBella(api_token)
        
        print("EllaBella: Hello! I'm EllaBella, your personal AI assistant. How can I help you today?")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("EllaBella: Goodbye! Have a great day!")
                break
                
            response = chatbot.generate_response(user_input)
            print(f"EllaBella: {response}")

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()