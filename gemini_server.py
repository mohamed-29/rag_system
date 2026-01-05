import asyncio
import nest_asyncio
import edge_tts
import google.generativeai as genai
import os
import time
import re
import uuid
import glob
import requests
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from dotenv import load_dotenv
load_dotenv()

# Patch asyncio for Flask+EdgeTTS compatibility
nest_asyncio.apply()

app = Flask(__name__)
app.secret_key = 'operation-kayan-secret-key-2026'  # Required for sessions

# --- User Authentication ---
USERS = {
    'admin': '123',
    'kayan': 'mobica2026'
}

# --- Configuration ---
# SECURITY WARNING: Never share your API key publicly.
API_KEY = os.getenv("GEMINI_API_KEY", "") 

# Point to a FOLDER containing all knowledge images
KNOWLEDGE_FOLDER = r"C:\Users\moham\Desktop\Ivend\rag_system\knowledge_base"

# Allowed image extensions
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.webp']

# Using 1.5 Flash for best speed/cost/multimodality balance
MODEL_NAME = "gemini-3-flash-preview" 

AUDIO_FOLDER = 'static/audio'
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# --- Voice Settings ---
# Options: "Male" or "Female"
PREFERRED_GENDER = "Female" 

VOICE_MAP = {
    "English": {
        "Female": "en-US-AriaNeural",      # Clear, professional
        "Male": "en-US-ChristopherNeural"  # Deep, calm, assertive
    },
    "Arabic": {
        "Female": "ar-EG-SalmaNeural",     # Egyptian Standard
        "Male": "ar-EG-ShakirNeural"       # Egyptian Male
    }
}

# --- Global State ---
uploaded_files = [] 
model = None
chat_sessions = {}

def initialize_gemini():
    """Initializes the Gemini model and uploads ALL images in the knowledge folder."""
    global model, uploaded_files
    
    print(f"[{time.strftime('%X')}] Configuring Gemini API...")
    genai.configure(api_key=API_KEY)

    uploaded_files = []
    
    # 1. Find all images in the folder
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        search_path = os.path.join(KNOWLEDGE_FOLDER, ext)
        image_paths.extend(glob.glob(search_path))

    print(f"[{time.strftime('%X')}] Found {len(image_paths)} images in knowledge base.")

    # 2. Upload Loop
    for file_path in image_paths:
        try:
            print(f"Ingesting: {os.path.basename(file_path)}...")
            g_file = genai.upload_file(path=file_path)
            
            # Wait for processing
            while g_file.state.name == "PROCESSING":
                time.sleep(1)
                g_file = genai.get_file(g_file.name)
            
            if g_file.state.name == "FAILED":
                print(f"Skipping failed file: {file_path}")
                continue
                
            uploaded_files.append(g_file)
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

    print(f"[{time.strftime('%X')}] Successfully loaded {len(uploaded_files)} images into memory.")

    # 3. System Prompt (Updated for multiple files)
    system_instruction = """
You are the Lead Strategic Analyst at iVend by Mobica. Your role is to serve as a trusted, expert partner to the leadership team.

CORE IDENTITY & BEHAVIOR:
1.  **Ownership & Confidence:** You are part of the core team. Speak with "we," "our," and "us." 
    * *Bad:* "The image shows sales will be..."
    * *Good:* "We are projecting sales to reach..."
    * Never reference "the file," "the image," or "the chart" as external objects. This data is your own internalized knowledge.

2.  **Absolute Accuracy (Zero Guessing Policy):** Your credibility is paramount.
    * **Strict Adherence:** Base every number, date, and projection *strictly* on the internal visual data provided (Sales Forecasts, Growth Projections, Production Schedules).
    * **Uncertainty Handling:** If the user asks for a number NOT in your data (e.g., "What is the marketing budget?"), DO NOT GUESS. 
    * **Standard Response for Missing Data:** "I don't have the specific figures for that in our current performance reports, but I can walk you through the sales and production data we do have."

3.  **Synthesized Intelligence:** Don't just read numbers; connect them.
    * *Example:* Instead of just saying "Q1 production is 80," say "We are ramping up production to 80 units in Q1 to prepare for the sales demand we expect later in the year." (Connecting production charts to sales charts).

4.  **Conversational & Professional Voice (TTS Optimized):**
    * Speak clearly and concisely.
    * **NO Markdown:** Do not use asterisks (*), hashes (#), or bullet points.
    * **Tone:** Professional, encouraging, and forward-looking.

5.  **Multilingual Adaptability:** Detect the language of the user's query (English or Arabic) and respond fluently in that same language, maintaining the same professional persona.

SUMMARY OF KEY FACTS (Do not contradict these):
* **Growth:** Scaling to 214 machines by Q4 2025.
* **Efficiency:** Driving failure rates down to 5% by 2026.
* **Sales:** 2025 forecast is 699 units; Combo and Coffee machines are our leaders.
* **Revenue:** 14 Million EGP total (12M Supplies, 2M Renting).
* **Key Client:** "Gish" is critical, accounting for 200 Buffet Stations.
"""

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction
        )
        print(f"[{time.strftime('%X')}] Agent Ready with {len(uploaded_files)} knowledge sources.")
        
    except Exception as e:
        print(f"Initialization Error: {e}")

# Initialize on Import
initialize_gemini()

async def generate_audio(text, output_file, voice):
    """
    Generates audio using ElevenLabs for 'GPT-like' human quality.
    Falls back to EdgeTTS if API key is missing or quota exceeded.
    """
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
    print(f"Generating Audio (Voice: {voice})...")
    
    # Try ElevenLabs first (Only if API Key is set)
    if ELEVENLABS_API_KEY and ELEVENLABS_API_KEY != "sk_...":
        try:
            # Use Multilingual Model for best results in Ent/Ar
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2", # Supports English & Arabic
                "voice_settings": {
                    "stability": 0.5, 
                    "similarity_boost": 0.75
                }
            }

            print("Calling ElevenLabs API...")
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print("ElevenLabs Success!")
                return # Success!
            else:
                print(f"ElevenLabs Error {response.status_code}: {response.text}")
                # Fall through to EdgeTTS
        except Exception as e:
            print(f"ElevenLabs Failed: {e}")

    # Fallback to EdgeTTS (The free/unlimited backup)
    print("Falling back to EdgeTTS...")
    # Use the 'voice' argument passed from chat_endpoint (handles Ar/En selection)
    communicate = edge_tts.Communicate(text, voice, rate="-5%")
    await communicate.save(output_file)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('gemini_chat.html')

@app.route('/chat', methods=['POST'])
async def chat_endpoint():
    global model, uploaded_files
    
    if not model:
        return jsonify({"error": "Model not initialized."}), 500

    # Get data (Form or JSON)
    session_id = request.form.get('session_id')
    json_data = request.get_json(silent=True)
    if not session_id and json_data:
        session_id = json_data.get('session_id')
    
    # Initialize Session
    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        # We implicitly pass the files into history so the model "knows" them immediately
        # Pass the entire list of uploaded_files (all images loaded at once)
        history_parts = uploaded_files + ["Internalize this library of data as your core knowledge."] if uploaded_files else []
        initial_history = [{"role": "user", "parts": history_parts}] if history_parts else []
        chat_sessions[session_id] = model.start_chat(history=initial_history)
    
    chat_session = chat_sessions[session_id]
    
    try:
        # --- INPUT HANDLING ---
        if 'audio' in request.files:
            audio_file = request.files['audio']
            temp_path = os.path.join(AUDIO_FOLDER, f"temp_in_{uuid.uuid4()}.webm")
            audio_file.save(temp_path)
            
            gemini_audio = genai.upload_file(temp_path, mime_type="audio/webm")
            
            # Send audio to Gemini
            response = chat_session.send_message(
                ["Listen to this user query and respond naturally.", gemini_audio]
            )
            
            # Cleanup
            try: os.remove(temp_path) 
            except: pass

        else:
            if not json_data:
                return jsonify({"error": "Invalid content."}), 400
            user_message = json_data.get('message')
            response = chat_session.send_message(user_message)

        # --- RESPONSE HANDLING ---
        response_text = response.text
        
        # Strip markdown (asterisks, hashes) just in case, for cleaner TTS
        clean_text = response_text.replace("*", "").replace("#", "").replace("- ", "")
        
        audio_url = None
        if clean_text:
            # 1. Detect Language
            is_arabic = bool(re.search(r'[\u0600-\u06FF]', clean_text))
            
            # 2. Select Voice based on Preference
            lang_key = "Arabic" if is_arabic else "English"
            selected_voice = VOICE_MAP[lang_key][PREFERRED_GENDER]
            
            # 3. Generate Audio
            audio_filename = f"response_{uuid.uuid4()}.mp3"
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
            
            await generate_audio(clean_text, audio_path, voice=selected_voice)
            audio_url = f"/static/audio/{audio_filename}"
        
        return jsonify({
            "response": clean_text,
            "session_id": session_id,
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False)