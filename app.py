"""
Handwritten Digit Recognition with Drawing Pad using CNN
Flask Backend - app.py
"""

import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable GPU checks
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import base64
import io
import threading
import time
import logging

from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load

# ─────────────────────────────────────────────
# TensorFlow optimization for Render
# ─────────────────────────────────────────────
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder='.')

# Thread lock for TensorFlow
model_lock = threading.Lock()

# ─────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────
MODEL_PATH = "digit_model.h5"


def load_model():
    """
    Load trained model
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found."
        )

    logger.info(f"Loading model from {MODEL_PATH}")

    model = keras_load(MODEL_PATH)

    return model


# ─────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────
logger.info("Initializing model...")

model = load_model()

# Warmup prediction (VERY IMPORTANT)
dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)

with model_lock:
    model.predict(dummy, verbose=0)

logger.info("Model ready.")

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_data: str) -> np.ndarray:

    # Remove base64 header
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Decode image
    img_bytes = base64.b64decode(image_data)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # White background
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))

    background.paste(img, mask=img.split()[3])

    # Convert grayscale
    img = background.convert("L")

    # Invert colors
    img = ImageOps.invert(img)

    # Resize to MNIST size
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy
    img_array = np.array(img, dtype=np.float32)

    # Normalize
    img_array /= 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    start = time.time()

    try:

        logger.info("Prediction request received")

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image received"
            }), 400

        image_data = data["image"]

        # Preprocess image
        img_array = preprocess_image(image_data)

        # Predict safely
        with model_lock:
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

        predicted_digit = int(np.argmax(predictions))

        confidence = float(np.max(predictions)) * 100

        probabilities = [
            round(float(p) * 100, 2)
            for p in predictions
        ]

        logger.info(
            f"Predicted: {predicted_digit} | "
            f"Confidence: {confidence:.2f}%"
        )

        logger.info(
            f"Prediction took "
            f"{time.time() - start:.2f} sec"
        )

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:

        logger.exception("Prediction failed")

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
