
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "models/genz_multilabel_model.joblib"
MLB_PATH   = "models/genz_mlb.joblib"

model = joblib.load(MODEL_PATH)   # ideal: Pipeline completo (preprocess + clf)
mlb   = joblib.load(MLB_PATH)     # MultiLabelBinarizer

TOPK = 3

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "GenZ API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({"error": "JSON inválido o vacío"}), 400

    # Formato recomendado: {"data": {...}}
    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    if not isinstance(data, dict) or len(data) == 0:
        return jsonify({"error": "Formato inválido. Envía {'data': {...}} con features."}), 400

    X = pd.DataFrame([data])

    try:
        proba = model.predict_proba(X)  # (1, n_labels)
        proba = np.asarray(proba)[0]
    except Exception as e:
        return jsonify({
            "error": "Fallo al predecir. Revisa que las columnas del request coincidan con las del entrenamiento.",
            "details": str(e)
        }), 400

    idx = np.argsort(proba)[::-1][:TOPK]
    return jsonify({
        "top3": [
            {"career": str(mlb.classes_[i]), "prob": float(proba[i])}
            for i in idx
        ]
    })
