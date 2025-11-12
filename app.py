from flask import Flask, render_template, request
from joblib import load
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load trained model
model_path = "models/ripeness_model.joblib"
model = load(model_path)
print("Model loaded from", model_path)

def extract_avg_color(image_path):
    """Reads image, resizes, and computes average RGB color."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    avg_color = img.mean(axis=(0, 1))  # Average color in RGB
    return avg_color.reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="No file uploaded.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result="No file selected.")

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Extract color features
    features = extract_avg_color(filepath)

    # Predict ripeness
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max() * 100

    return render_template(
        "index.html",
        result=f"{prediction.upper()} ({probability:.2f}% confidence)",
        uploaded_image=file.filename
    )

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
