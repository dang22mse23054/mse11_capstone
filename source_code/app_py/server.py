import os
import json
import time

from flask import Flask, request, jsonify, make_response, g

app = Flask(__name__)
print('Server is running')


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})



if __name__ == "__main__":
    host = os.environ.get('APP_HOST', '0.0.0.0')
    port = os.environ.get('APP_PORT', 5001)
    app.run(host=host, port=port)