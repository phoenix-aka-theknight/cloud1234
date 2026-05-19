"""
Handwritten Digit Recognition with Drawing Pad using CNN
Flask Backend - app.py
"""

import os
import numpy as np
import base64
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import logging
from tensorflow.keras.models import load_model as keras_load

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.')

# ─────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────
MODEL_PATH = "digit_model.h5"


def load_model():
    """
    Load the saved Keras model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            f"Please upload digit_model.h5 to the project root."
        )

    logger.info(f"Loading model from {MODEL_PATH}")
    model = keras_load(MODEL_PATH)

    return model


# ─────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────
logger.info("Initializing model...")
model = load_model()
logger.info("Model ready.")


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_data: str) -> np.ndarray:
    """
    Convert a base64-encoded PNG from the canvas into a
    (1, 28, 28, 1) numpy array suitable for prediction.
    """

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

    # Resize
    img = img.resize((28, 28), Image.LANCZOS)

    # Normalize
    img_array = np.array(img).astype("float32") / 255.0

    # Reshape
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
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        image_data = data["image"]

        # Preprocess image
        img_array = preprocess_image(image_data)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        probabilities = [
            round(float(p) * 100, 2)
            for p in predictions
        ]

        logger.info(
            f"Predicted: {predicted_digit} | Confidence: {confidence:.2f}%"
        )

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
