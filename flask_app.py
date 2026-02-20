import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

TOPK = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "models", "genz_bundle.pkl")  # <-- raíz/models

def load_bundle(path: str):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("genz_bundle.pkl debe ser un dict")

    model = bundle.get("model") or bundle.get("model_tuned")
    mlb = bundle.get("mlb")
    features = bundle.get("features")

    if model is None:
        raise KeyError("No encontré modelo: esperé 'model' o 'model_tuned'")
    if mlb is None:
        raise KeyError("No encontré 'mlb'")
    if features is None:
        raise KeyError("No encontré 'features'")

    return bundle, model, mlb, features

try:
    BUNDLE, MODEL, MLB, FEATURES = load_bundle(BUNDLE_PATH)
    LOAD_ERROR = None
except Exception as e:
    BUNDLE, MODEL, MLB, FEATURES = None, None, None, None
    LOAD_ERROR = str(e)

app = Flask(__name__)

@app.get("/")
def health():
    if LOAD_ERROR:
        return jsonify({"status": "error", "details": LOAD_ERROR}), 500
    return jsonify({"status": "ok", "message": "GenZ API running", "topk": TOPK})

@app.post("/predict")
def predict():
    if LOAD_ERROR:
        return jsonify({"error": "Bundle no cargado", "details": LOAD_ERROR}), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "JSON inválido o vacío"}), 400

    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Envía {'data': {...}}"}), 400

    X = pd.DataFrame([data])

    # Alinear features
    for c in FEATURES:
        if c not in X.columns:
            X[c] = None
    X = X[FEATURES]

    try:
        proba = np.asarray(MODEL.predict_proba(X))[0].reshape(-1)
    except Exception as e:
        return jsonify({"error": "Fallo al predecir", "details": str(e)}), 400

    idx = np.argsort(proba)[::-1][:TOPK]
    top = [{"career": str(MLB.classes_[i]), "prob": float(proba[i])} for i in idx]
    return jsonify({"top3": top})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)