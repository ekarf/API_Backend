from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello from Flask!'

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    # Check if the message is "hi"
    if user_input.lower() == "hi":
        return jsonify({'response': 'Hello!'})
    else:
        return jsonify({'response': 'I don\'t understand.'})

if __name__ == "__main__":
    app.run(debug=True)
