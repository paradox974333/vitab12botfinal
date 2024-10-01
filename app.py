from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import threading
import time

app = Flask(__name__)  # Fixed __name__

# Initialize the InferenceClient with your model and token directly
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    token="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",  # Replace with your actual token
)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    # Simplified response handling for this example
    response = generate_response(user_message)
    return jsonify({'reply': response})

def generate_response(user_message):
    response = ""
    prompt = f"Tell me about exoplanets: {user_message}"
    
    # Using the client to generate a response from the model
    for message in client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5000,
        stream=True
    ):
        if 'choices' in message and message['choices']:
            response += message['choices'][0]['delta']['content']
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
