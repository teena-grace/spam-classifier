"""Small Flask app for the trained email spam classifier."""
from pathlib import Path
import pickle
import re

import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
with (ROOT / "model.pkl").open("rb") as f:
    model = pickle.load(f)
with (ROOT / "features.pkl").open("rb") as f:
    feature_columns = pickle.load(f)

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify(error="Paste an email message to analyze."), 400
    if len(message) > 20000:
        return jsonify(error="Please keep the message under 20,000 characters."), 400

    known = set(feature_columns)
    counts = {}
    for word in re.findall(r"[a-zA-Z]+", message.lower()):
        if word in known:
            counts[word] = counts.get(word, 0) + 1
    row = pd.DataFrame([[counts.get(word, 0) for word in feature_columns]], columns=feature_columns)
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    spam_index = classes.index(1) if 1 in classes else len(classes) - 1
    spam_probability = float(probabilities[spam_index])
    label = "spam" if spam_probability >= 0.5 else "safe"
    confidence = spam_probability if label == "spam" else 1 - spam_probability
    return jsonify(
        label=label,
        confidence=round(confidence * 100, 1),
        spam_probability=round(spam_probability * 100, 1),
        matched_words=len(counts),
        word_count=len(message.split()),
        message=message,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
