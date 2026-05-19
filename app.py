"""
Handwritten Digit Recognition with Drawing Pad using CNN
Flask Backend - app.py

This file handles:
- CNN model training (if no saved model found)
- Flask API endpoint for digit prediction
- Image preprocessing pipeline
"""

import os
import numpy as np
import base64
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.')

# ─────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "digit_model.h5")


def build_and_train_model():
    """
    Build a CNN model and train it on the MNIST dataset.
    This function runs automatically if no saved model is found.
    Returns the trained Keras model.
    """
    # Import TensorFlow here to avoid slow startup if model already exists
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    logger.info("No saved model found. Training CNN on MNIST dataset...")

    # ── Load MNIST ──────────────────────────────
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # ── Preprocess ──────────────────────────────
    # Reshape to (samples, 28, 28, 1) and normalize to [0, 1]
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test,  10)

    # ── Build CNN ────────────────────────────────
    model = Sequential([
        # Block 1 – extract low-level features
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2 – extract higher-level features
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classifier head
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation="softmax")   # 10 classes (digits 0-9)
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ── Train ────────────────────────────────────
    early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

    model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ── Evaluate ─────────────────────────────────
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    # ── Save model ───────────────────────────────
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    return model


def load_model():
    """
    Load the saved Keras model. If it doesn't exist, train and save it first.
    Returns the loaded model.
    """
    from tensorflow.keras.models import load_model as keras_load

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}")
        model = keras_load(MODEL_PATH)
    else:
        model = build_and_train_model()

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
    Convert a base64-encoded PNG from the canvas into a (1, 28, 28, 1) numpy array
    suitable for model prediction.

    Steps:
      1. Decode base64 → PIL Image (RGBA)
      2. Convert to grayscale
      3. Invert colors  (canvas: white digit on black → MNIST: white digit on black)
         The canvas sends black strokes on white background, so we invert to match MNIST.
      4. Resize to 28×28 with LANCZOS resampling
      5. Normalize pixel values to [0, 1]
      6. Reshape to (1, 28, 28, 1)
    """
    # Strip data-URL header  e.g. "data:image/png;base64,..."
    if "," in image_data:
        image_data = image_data.split(",")[1]

    # Decode base64 → bytes → PIL Image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Composite onto white background (canvas may have transparent areas)
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])          # use alpha as mask
    img = background.convert("L")                       # convert to grayscale

    # Invert: user draws dark on light; MNIST has light digit on dark background
    img = ImageOps.invert(img)

    # Resize to 28×28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img).astype("float32") / 255.0

    # Reshape to (1, 28, 28, 1)  – batch_size=1, channels=1
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main drawing pad page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Expects JSON body: { "image": "<base64 data-URL>" }
    Returns JSON:      { "digit": int, "confidence": float, "probabilities": list }
    """
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        image_data = data["image"]

        # Preprocess the canvas image
        img_array = preprocess_image(image_data)

        # Run prediction
        predictions = model.predict(img_array, verbose=0)[0]   # shape: (10,)

        predicted_digit = int(np.argmax(predictions))
        confidence      = float(np.max(predictions)) * 100

        # Round all probabilities for display
        probabilities = [round(float(p) * 100, 2) for p in predictions]

        logger.info(f"Predicted: {predicted_digit}  Confidence: {confidence:.2f}%")

        return jsonify({
            "digit":         predicted_digit,
            "confidence":    round(confidence, 2),
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
    app.run(debug=False, host="0.0.0.0", port=port)