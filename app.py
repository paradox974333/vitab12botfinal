from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
import time
import asyncio
import threading

app = Flask(__name__)

# Initialize the InferenceClient with your model and token
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    token="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",  # Replace with your actual token
)

# Function to handle streaming in a separate thread
def generate_response(user_message, response_queue):
    response = ""
    for message in client.chat_completion(
        messages=[{"role": "user", "content": user_message}],
        max_tokens=5000,
        stream=True  # Use streaming
    ):
        if 'choices' in message and message['choices']:
            response += message['choices'][0]['delta']['content']
    response_queue.append(response)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    try:
        start_time = time.time()
        
        # Use a list as a thread-safe queue to collect the response
        response_queue = []
        # Run response generation in a separate thread to avoid blocking
        thread = threading.Thread(target=generate_response, args=(user_message, response_queue))
        thread.start()
        thread.join()  # Wait for the thread to complete
        
        response = response_queue[0] if response_queue else "No response received"

        # Log response time for performance monitoring
        elapsed_time = time.time() - start_time
        print(f"Response Time: {elapsed_time:.2f} seconds")

        return jsonify({'reply': response})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader for production
