"""
inference.py  -  PHASE 11 (integration)

Connects everything:
   image  -> preprocess -> DL model -> class probabilities (+ Grad-CAM)
   probabilities + symptoms -> ML risk model -> low / moderate / high

It loads the REAL trained models from the models/ folder. Nothing is hard-coded.
"""
import json

import joblib
import pandas as pd
from tensorflow import keras

import config
from evaluation.gradcam import make_gradcam_heatmap, overlay_heatmap
from preprocessing.preprocess import preprocess_for_model, to_model_input

_cache = {}


def dl_model_ready():
    return config.BEST_MODEL_PATH.exists() and config.MODEL_INFO_PATH.exists()


def load_dl_model():
    """Loads the model once and keeps it in memory."""
    if "dl" not in _cache:
        if not dl_model_ready():
            raise FileNotFoundError("No trained image model found. Train it first: python -m training.train_dl")
        info = json.loads(config.MODEL_INFO_PATH.read_text())
        _cache["dl"] = (keras.models.load_model(config.BEST_MODEL_PATH), info)
    return _cache["dl"]


def predict_condition(rgb):
    """rgb: validated RGB image (any size). Returns a dict with the result."""
    model, info = load_dl_model()
    class_names = info["class_names"]
    processed = preprocess_for_model(rgb)
    batch = to_model_input(processed)
    probs = model.predict(batch, verbose=0)[0]
    top = int(probs.argmax())
    heatmap = make_gradcam_heatmap(model, batch, top)
    return {
        "class_name": class_names[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
        "uncertain": float(probs[top]) < config.CONFIDENCE_THRESHOLD,
        "processed_image": processed,
        "gradcam_overlay": overlay_heatmap(processed, heatmap),
        "model_name": info["best_model"],
    }


# ----------------------------------------------------------------------------
# Risk
# ----------------------------------------------------------------------------
def risk_model_info():
    if config.RISK_MODEL_PATH.exists() and config.RISK_INFO_PATH.exists():
        return json.loads(config.RISK_INFO_PATH.read_text())
    return None


def _rule_based_risk(probs, s):
    """Simple, transparent fallback used when no trained risk model exists."""
    disease = 1 - probs.get("healthy", 0.0)
    score = (s.get("pain_level", 0) / 10 * 3 + s.get("sensitivity", 0) + s.get("gum_bleeding", 0)
             + s.get("swelling", 0) * 3 + (s.get("symptom_days", 0) > 14) + disease * 2)
    return "low" if score < 3 else ("moderate" if score < 6 else "high")


def predict_risk(class_probs, symptoms):
    """Returns {'category', 'method', 'warning'}."""
    info = risk_model_info()
    if info is None:
        return {"category": _rule_based_risk(class_probs, symptoms),
                "method": "Simple starter rules (no trained ML risk model found)",
                "warning": "Train the ML risk model to use Machine Learning here (Step 6 in the guide)."}
    if set(info["classes"]) != set(class_probs):
        return {"category": _rule_based_risk(class_probs, symptoms),
                "method": "Simple starter rules (risk model classes differ from image model classes)",
                "warning": "Re-train the risk model so its prob_ columns match the image model classes."}

    if "risk" not in _cache:
        _cache["risk"] = joblib.load(config.RISK_MODEL_PATH)
    bundle = _cache["risk"]
    row = {f"prob_{c}": p for c, p in class_probs.items()}
    row.update({k: symptoms.get(k, 0) for k in config.SYMPTOM_FEATURES})
    X = pd.DataFrame([row])[bundle["feature_columns"]]
    pred = bundle["model"].predict(X)[0]
    category = bundle["label_encoder"].inverse_transform([int(pred)])[0]
    return {"category": category,
            "method": f"Machine Learning ({info['best_model'].replace('_', ' ').title()})",
            "warning": info["data_source"] if info.get("is_demo") else None}
