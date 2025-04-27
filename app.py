import os
import json
import pandas as pd
import requests
import time
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK resources
nltk.download('punkt', force=True)
nltk.download('punkt_tab', force=True)
nltk.download('stopwords', force=True)

# Initialize Flask app
app = Flask(__name__)

# Configure Google Generative AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
available_models = genai.list_models()
print("Available models:", [model.name for model in available_models])
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Data paths
SESSION_DATA_PATH = "data/session_details.json"
JOB_LISTINGS_PATH = "data/job_listing_data.csv"
KNOWLEDGE_BASE_PATH = "data/knowledge_base.json"
BIAS_PATTERNS_PATH = "data/bias_patterns.json"
WOMEN_EMPOWERMENT_SOURCES = [
    "https://api.example.com/women-empowerment/updates", 
    "https://api.example.com/leadership-programs",
]

# Class to handle context management
class ContextManager:
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id):
        """Create a new session with timestamp"""
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "conversation_history": [],
            "current_context": {"topic": None, "intent": None}
        }
        return self.sessions[session_id]
    
    def update_session(self, session_id, user_message, bot_response):
        """Update session with new messages"""
        if session_id not in self.sessions:
            self.create_session(session_id)
            
        self.sessions[session_id]["last_active"] = datetime.now()
        self.sessions[session_id]["conversation_history"].append({
            "user": user_message,
            "bot": bot_response,
            "timestamp": datetime.now()
        })
        
    def get_session(self, session_id):
        """Get session data or create if not exists"""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]
    
    def update_context(self, session_id, topic=None, intent=None):
        """Update the context for a specific session"""
        if session_id in self.sessions:
            if topic:
                self.sessions[session_id]["current_context"]["topic"] = topic
            if intent:
                self.sessions[session_id]["current_context"]["intent"] = intent
    
    def cleanup_old_sessions(self, max_age_minutes=30):
        """Remove sessions older than specified time"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.sessions.items():
            last_active = session_data["last_active"]
            age_minutes = (current_time - last_active).total_seconds() / 60
            
            if age_minutes > max_age_minutes:
                sessions_to_remove.append(session_id)
                
        for session_id in sessions_to_remove:
            del self.sessions[session_id]

# Class to handle data integration
class DataIntegration:
    def __init__(self):
        self.job_data = None
        self.session_data = None
        self.knowledge_base = None
        self.bias_patterns = None
        self.last_update = None
        self.update_interval = 3600 
        
    def load_data(self):
        """Load all required data sources"""
        # Load job listings
        try:
            self.job_data = pd.read_csv(JOB_LISTINGS_PATH)
        except Exception as e:
            print(f"Error loading job data: {e}")
            self.job_data = pd.DataFrame()
            
        # Load session details
        try:
            with open(SESSION_DATA_PATH, 'r') as f:
                self.session_data = json.load(f)
        except Exception as e:
            print(f"Error loading session data: {e}")
            self.session_data = {"sessions": []}
            
        # Load knowledge base
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                self.knowledge_base = json.load(f)
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.knowledge_base = {"faqs": [], "events": [], "mentorship": [], "empowerment": []}
            
        # Load bias patterns
        try:
            with open(BIAS_PATTERNS_PATH, 'r') as f:
                self.bias_patterns = json.load(f)
        except Exception as e:
            print(f"Error loading bias patterns: {e}")
            self.bias_patterns = {"patterns": [], "redirects": {}}
            
        self.last_update = datetime.now()
        
    def check_update_needed(self):
        """Check if data update is needed based on time interval"""
        if not self.last_update:
            return True
            
        current_time = datetime.now()
        elapsed_seconds = (current_time - self.last_update).total_seconds()
        return elapsed_seconds > self.update_interval
        
    def update_knowledge_base(self):
        """Update knowledge base with latest information"""
        if not self.check_update_needed():
            return
            
        # Update women empowerment content from external APIs
        try:
            for source_url in WOMEN_EMPOWERMENT_SOURCES:
                response = requests.get(source_url)
                if response.status_code == 200:
                    new_data = response.json()
                    # Merge with existing data (implementation depends on data structure)
                    if "empowerment" in self.knowledge_base:
                        self.knowledge_base["empowerment"].extend(new_data.get("items", []))
        except Exception as e:
            print(f"Error updating knowledge base from external sources: {e}")
            
        # Save updated knowledge base
        try:
            with open(KNOWLEDGE_BASE_PATH, 'w') as f:
                json.dump(self.knowledge_base, f)
        except Exception as e:
            print(f"Error saving updated knowledge base: {e}")
            
        self.last_update = datetime.now()
    
    def search_job_listings(self, query, limit=5):
        """Search job listings based on keywords"""
        if self.job_data.empty:
            return []
            
        # Simple keyword matching (could be enhanced with more sophisticated search)
        query_tokens = set(word_tokenize(query.lower()))
        stop_words = set(stopwords.words('english'))
        query_tokens = [token for token in query_tokens if token not in stop_words]
        
        results = []
        for _, job in self.job_data.iterrows():
            job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('skills', '')}".lower()
            
            # Check if any query tokens match job text
            if any(token in job_text for token in query_tokens):
                results.append({
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "url": job.get("url", ""),
                    "description": job.get("description", "")[:150] + "..." if len(job.get("description", "")) > 150 else job.get("description", "")
                })
                
                if len(results) >= limit:
                    break
                    
        return results
    
    def get_upcoming_events(self, limit=3):
        """Get upcoming community events"""
        if not self.session_data or "sessions" not in self.session_data:
            return []
            
        current_date = datetime.now()
        upcoming = []
        
        for session in self.session_data.get("sessions", []):
            session_date_str = session.get("date", "")
            try:
                session_date = datetime.strptime(session_date_str, "%Y-%m-%d %H:%M")
                if session_date > current_date:
                    upcoming.append(session)
                    if len(upcoming) >= limit:
                        break
            except Exception:
                continue
                
        return upcoming
    
    def search_knowledge_base(self, query, category=None):
        """Search knowledge base for relevant information"""
        if not self.knowledge_base:
            return []
            
        query_tokens = set(word_tokenize(query.lower()))
        stop_words = set(stopwords.words('english'))
        query_tokens = [token for token in query_tokens if token not in stop_words]
        
        categories = [category] if category else ["faqs", "events", "mentorship", "empowerment"]
        results = []
        
        for cat in categories:
            if cat not in self.knowledge_base:
                continue
                
            for item in self.knowledge_base.get(cat, []):
                item_text = f"{item.get('question', '')} {item.get('answer', '')} {item.get('title', '')} {item.get('description', '')}".lower()
                
                # Check if any query tokens match item text
                if any(token in item_text for token in query_tokens):
                    results.append({
                        "category": cat,
                        "content": item
                    })
                    
        return results

# Class to handle NLP processing
class NLPProcessor:
    def __init__(self, data_integration, gemini_model):
        self.data_integration = data_integration
        self.model = gemini_model
        
    def detect_intent(self, message):
        """Detect the user's intent from their message"""
        # Simplified intent detection using keywords
        message_lower = message.lower()
        
        if any(kw in message_lower for kw in ["job", "work", "career", "position", "employment", "hiring"]):
            return "job_search"
        elif any(kw in message_lower for kw in ["event", "webinar", "session", "workshop", "meetup"]):
            return "event_info"
        elif any(kw in message_lower for kw in ["mentor", "guidance", "advise", "coach"]):
            return "mentorship"
        elif any(kw in message_lower for kw in ["women", "gender", "equality", "empower", "leadership"]):
            return "women_empowerment"
        else:
            return "general_inquiry"
    
    def detect_bias(self, message):
        """Detect if the message contains gender bias"""
        message_lower = message.lower()
        
        if not self.data_integration.bias_patterns:
            return False, None
            
        for pattern in self.data_integration.bias_patterns.get("patterns", []):
            if re.search(pattern, message_lower):
                redirect_key = next((k for k in self.data_integration.bias_patterns.get("redirects", {}) 
                                  if re.search(k, message_lower)), None)
                return True, self.data_integration.bias_patterns["redirects"].get(redirect_key)
                
        return False, None
        
    def generate_response(self, message, session_context):
        """Generate response using Gemini API with context"""
        
        is_biased, redirect_message = self.detect_bias(message)
        if is_biased and redirect_message:
            return redirect_message
         
        intent = self.detect_intent(message)
        
        prompt = self._create_context_prompt(message, intent, session_context)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing your request right now. Could you try asking again differently?"
    
    def _create_context_prompt(self, message, intent, session_context):
        """Create a context-rich prompt for the AI model"""
        
        # Base system instruction
        system_instruction = """
        You are AWEbot, an AI assistant for the JobsForHer Foundation. Your role is to help women with their
        career journeys by providing information about job opportunities, community events, mentorship
        programs, and women empowerment initiatives. Follow these guidelines:
        
        1. Be supportive, encouraging, and empowering in your responses.
        2. Provide factual information based on the available data.
        3. If you don't know something, admit it and offer alternative help.
        4. Focus on empowering women in their careers without making gender comparisons.
        5. Keep responses concise, helpful, and focused on the user's needs.
        """
        
        # Add context data based on intent
        context_data = ""
        if intent == "job_search":
            job_results = self.data_integration.search_job_listings(message)
            if job_results:
                context_data += "Here are some relevant job listings:\n"
                for i, job in enumerate(job_results):
                    context_data += f"{i+1}. {job['title']} at {job['company']} ({job['location']})\n"
            else:
                context_data += "No specific job listings were found for this query.\n"
                
        elif intent == "event_info":
            events = self.data_integration.get_upcoming_events()
            if events:
                context_data += "Here are upcoming community events:\n"
                for i, event in enumerate(events):
                    context_data += f"{i+1}. {event.get('title', 'Event')} on {event.get('date', 'TBD')}: {event.get('description', '')[:100]}...\n"
            else:
                context_data += "No upcoming events were found.\n"
                
        elif intent in ["mentorship", "women_empowerment", "general_inquiry"]:
            kb_results = self.data_integration.search_knowledge_base(message)
            if kb_results:
                context_data += "Relevant information from our knowledge base:\n"
                for i, result in enumerate(kb_results[:3]):
                    content = result["content"]
                    if "question" in content:
                        context_data += f"Q: {content['question']}\nA: {content['answer']}\n\n"
                    else:
                        context_data += f"{content.get('title', 'Information')}: {content.get('description', '')[:150]}...\n\n"
        #changed
        else:
            search_result = call_google_search_api(message)
            if search_result:
                context_data += "\nAs an alternative, here's what I found from the web:\n"
                context_data += f"{search_result}\n"
        #/changed
        
        conversation_context = ""
        if session_context and "conversation_history" in session_context:
            recent_history = session_context["conversation_history"][-3:] if session_context["conversation_history"] else []
            if recent_history:
                conversation_context += "Recent conversation:\n"
                for exchange in recent_history:
                    conversation_context += f"User: {exchange.get('user', '')}\nAWE: {exchange.get('bot', '')}\n\n"
        
        # Combine all contexts
        full_prompt = f"{system_instruction}\n\n{conversation_context}\n{context_data}\nUser: {message}\nAWE:"
        return full_prompt

#changed
def call_google_search_api(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("SEARCH_ENGINE_ID")
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        results = data.get("items", [])
        
        if not results:
            return "No web results found."

        summary = ""
        for item in results[:3]: 
            title = item.get("title")
            snippet = item.get("snippet")
            link = item.get("link")
            summary += f"{title}:\n{snippet}\n{link}\n\n"

        return summary.strip()

    except Exception as e:
        return f"Error retrieving web results: {str(e)}"
#/changed

class AshaBot:
    def __init__(self):
        self.context_manager = ContextManager()
        self.data_integration = DataIntegration()
        self.data_integration.load_data()
        self.nlp_processor = NLPProcessor(self.data_integration, model)
        
        # Schedule regular data updates
        self._schedule_data_updates()
        
    def _schedule_data_updates(self):
        """Set up regular data updates"""
        import threading
        
        def update_routine():
            while True:
                try:
                    print("Updating knowledge base...")
                    self.data_integration.update_knowledge_base()
                    print("Knowledge base updated successfully")
                    
                    self.context_manager.cleanup_old_sessions()
                    
                    time.sleep(self.data_integration.update_interval)
                except Exception as e:
                    print(f"Error in update routine: {e}")
                    time.sleep(300) 
        
        # Start update thread
        update_thread = threading.Thread(target=update_routine)
        update_thread.daemon = True
        update_thread.start()
        
    def process_message(self, session_id, message):
        """Process incoming user message and generate response"""
        # Get or create session
        session = self.context_manager.get_session(session_id)
        
        # Generate response
        response = self.nlp_processor.generate_response(message, session)
        
        # Update session with new interaction
        self.context_manager.update_session(session_id, message, response)
        
        return response

# API endpoints
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
        
    session_id = data.get('session_id', str(time.time()))
    message = data['message']
    
    bot = AshaBot()
    response = bot.process_message(session_id, message)
    
    return jsonify({
        'session_id': session_id,
        'response': response
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 10))
    
    data_integration = DataIntegration()
    data_integration.load_data()
    jobs = data_integration.search_job_listings(query, limit)
    
    return jsonify({'jobs': jobs})

@app.route('/api/events', methods=['GET'])
def get_events():
    limit = int(request.args.get('limit', 5))
    
    data_integration = DataIntegration()
    data_integration.load_data()
    events = data_integration.get_upcoming_events(limit)
    
    return jsonify({'events': events})

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if required data files exist, create empty ones if not
    if not os.path.exists(JOB_LISTINGS_PATH):
        pd.DataFrame(columns=['title', 'company', 'location', 'description', 'skills', 'url']).to_csv(JOB_LISTINGS_PATH, index=False)
        
    if not os.path.exists(SESSION_DATA_PATH):
        with open(SESSION_DATA_PATH, 'w') as f:
            json.dump({"sessions": []}, f)
            
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, 'w') as f:
            json.dump({
                "faqs": [],
                "events": [],
                "mentorship": [],
                "empowerment": []
            }, f)
            
    if not os.path.exists(BIAS_PATTERNS_PATH):
        with open(BIAS_PATTERNS_PATH, 'w') as f:
            json.dump({
                "patterns": [
                    "women (are|should be) in the kitchen",
                    "women can't (handle|do|perform) technical",
                    "women are too emotional for",
                    "women are not suited for",
                    "women lack the (ability|capability|skill)"
                ],
                "redirects": {
                    "women (are|should be) in the kitchen": "I'd like to share that women have excelled across all professional domains. Would you like to learn about women leaders in various industries?",
                    "women can't (handle|do|perform) technical": "Women have made significant contributions to technical fields. Would you like to explore success stories of women in technology and STEM?",
                    "women are too emotional": "Research shows diverse leadership styles enhance organizational performance. Would you like to learn about effective leadership approaches?",
                    "women are not suited": "Women have demonstrated excellence in all professional fields. Would you like information on specific career paths?",
                    "women lack": "Women have consistently demonstrated exceptional capabilities across all domains. Would you like to learn about women's achievements in specific fields?"
                }
            }, f)
    
    
    app.run(host='0.0.0.0', port=5000, debug=True)