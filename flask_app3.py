# api/app.py
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

TOPK = 3

# Ruta robusta al bundle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "..", "models", "genz_bundle.pkl")

def load_bundle(path: str):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("genz_bundle.pkl debe ser un dict")

    # Compatibilidad: tuned-only (model) o formato anterior (model_tuned)
    model = bundle.get("model", None)
    if model is None:
        model = bundle.get("model_tuned", None)

    mlb = bundle.get("mlb", None)
    features = bundle.get("features", None)

    if model is None:
        raise KeyError("No encontré modelo en el bundle: esperé 'model' o 'model_tuned'")
    if mlb is None:
        raise KeyError("No encontré 'mlb' en el bundle")
    if features is None:
        raise KeyError("No encontré 'features' en el bundle")

    mode = bundle.get("mode", "tuned")
    target = bundle.get("target", None)

    return bundle, model, mlb, features, mode, target

try:
    BUNDLE, MODEL, MLB, FEATURES, MODE, TARGET = load_bundle(BUNDLE_PATH)
    LOAD_ERROR = None
except Exception as e:
    BUNDLE, MODEL, MLB, FEATURES, MODE, TARGET = None, None, None, None, None, None
    LOAD_ERROR = str(e)

app = Flask(__name__)

@app.get("/")
def health():
    if LOAD_ERROR:
        return jsonify({
            "status": "error",
            "message": "API no lista. Falló carga del bundle.",
            "details": LOAD_ERROR
        }), 500

    return jsonify({
        "status": "ok",
        "message": "GenZ API is running",
        "mode": MODE,
        "topk": TOPK,
        "n_features": len(FEATURES),
        "target": TARGET
    })

@app.post("/predict")
def predict():
    if LOAD_ERROR or MODEL is None or MLB is None or FEATURES is None:
        return jsonify({"error": "API no lista. Bundle no cargado.", "details": LOAD_ERROR}), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "JSON inválido o vacío"}), 400

    # Formato recomendado: {"data": {...}}
    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, dict) or len(data) == 0:
        return jsonify({"error": "Formato inválido. Envía {'data': {...}} con features."}), 400

    X = pd.DataFrame([data])

    # Alinear features: agrega faltantes con None y reordena
    for c in FEATURES:
        if c not in X.columns:
            X[c] = None
    X = X[FEATURES]

    try:
        proba = np.asarray(MODEL.predict_proba(X))[0]  # (n_labels,)
        proba = np.asarray(proba).reshape(-1)
    except Exception as e:
        return jsonify({
            "error": "Fallo al predecir. Revisa columnas vs entrenamiento.",
            "details": str(e)
        }), 400

    idx = np.argsort(proba)[::-1][:TOPK]
    top = [{"career": str(MLB.classes_[i]), "prob": float(proba[i])} for i in idx]

    return jsonify({"top3": top})

if __name__ == "__main__":
    # Ejecutar directo: python api/app.py
    app.run(host="0.0.0.0", port=8000, debug=False)
