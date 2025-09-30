import os
import json
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types


API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    if API_KEY:
        client = genai.Client(api_key=API_KEY)
    else:
        raise ValueError("GEMINI_API_KEY is not set.")
        
except Exception as e:
    print(f"ERROR: Could not initialize Gemini Client. Details: {e}")
    client = None

app = Flask(__name__)
chat_history = []
SYSTEM_INSTRUCTION = "You are a friendly and helpful AI assistant named PyBot, powered by Google Gemini. Keep your answers concise, engaging, and use markdown formatting."


@app.route('/')
def index():
    """Renders the main chat interface template."""
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    """
    Handles the POST request from the frontend, communicates with Gemini, 
    and returns the AI's response.
    """
    if not client:
        return jsonify({"response": "AI service is not configured. Please ensure the GEMINI_API_KEY is set in your terminal."}), 500
    
    try:
        user_message = request.json.get('message')
    except Exception:
        return jsonify({"response": "Invalid request format."}), 400

    if not user_message:
        return jsonify({'response': 'Please provide a message.'}), 400

    global chat_history
    chat_history.append({"role": "user", "parts": [{"text": user_message}]})

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=chat_history,
            config=config,
        )

        model_response_text = response.text

        chat_history.append({"role": "model", "parts": [{"text": model_response_text}]})
        
        return jsonify({'response': model_response_text})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        chat_history.pop() 
        return jsonify({"response": "Sorry, I ran into an error communicating with the AI model."}), 500

if __name__ == '__main__':
    print("="*60)
    print("Starting Flask Chat API.")
    print(f"API Key Status: {'SET' if API_KEY else 'MISSING'}")
    print("Go to: http://127.0.0.1:5000/")
    print("="*60)
    app.run(debug=True)
