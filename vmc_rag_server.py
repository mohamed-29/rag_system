
from flask import Flask, request, jsonify, render_template
from vmc_rag_app import initialize_rag_system
import os

app = Flask(__name__)

# Initialize RAG System at startup
print("Starting Flask Server & Initializing RAG...")
rag_chain = initialize_rag_system()
print("RAG System Ready.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_rag():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id', 'default_user_session') # Default if not provided
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Pass session_id to RAG chain for memory
        response = rag_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        return jsonify({"answer": response, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
