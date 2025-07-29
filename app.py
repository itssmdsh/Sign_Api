from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle

# Load model and preprocessors once at startup
model = tf.keras.models.load_model('final_sign_model_no_hello.keras')
with open('scaler_no_hello.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_no_hello.pkl', 'rb') as f:
    le = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Expects JSON: { "landmarks": [126 floats] }
    data = request.json
    landmarks = np.array(data["landmarks"], dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(landmarks)
    probs = model.predict(X_scaled, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    confidence = float(probs[idx])
    return jsonify({
        "label": label,
        "confidence": confidence,
        "probs": [float(p) for p in probs]  # for debugging if needed
    })

# If running locally (ignore if deploying with gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
