# app.py
import numpy as np
import cv2
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('mnist_cnn_model.keras')

def preprocess_base64_image(data):
    # Decode base64 image
    content = data.split(',')[1]
    decoded = base64.b64decode(content)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert image (critical fix!)
    gray = 255 - gray

    # Resize to 28x28
    gray = cv2.resize(gray, (28, 28))

    # Normalize and reshape
    gray = gray.astype("float32") / 255.0
    gray = gray.reshape(1, 28, 28, 1)
    return gray


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    img = preprocess_base64_image(image_data)
    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    confidence = float(prediction[0][digit])
    return jsonify({'digit': digit, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)
