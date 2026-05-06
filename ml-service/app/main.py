from __future__ import annotations

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.model import load_all_bundles, preprocess_image, probs_from_output

app = FastAPI(title="MedAI ML Service", version="1.0.0")

# 🔥 Lazy loading cache
BUNDLES = {}
ALLOWED_MODALITIES = {"xray", "ct", "ultrasound", "mri"}


def get_bundle(modality: str):
    if modality not in BUNDLES:
        all_bundles = load_all_bundles()
        BUNDLES[modality] = all_bundles[modality]
    return BUNDLES[modality]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_modalities": list(BUNDLES.keys()),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), modality: str = Form("xray")):
    m = (modality or "xray").strip().lower()

    if m not in ALLOWED_MODALITIES:
        raise HTTPException(400, f"Unsupported modality: {modality}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    bundle = get_bundle(m)
    labels = bundle.labels
    batch = preprocess_image(raw)

    if bundle.model is None:
        probs = np.full((len(labels),), 1.0 / max(1, len(labels)), dtype=np.float32)
    else:
        pred = bundle.model.predict(batch, verbose=0)[0]
        probs = probs_from_output(pred, labels)

    idx = int(np.argmax(probs))
    label = labels[idx]
    confidence = float(probs[idx])

    class_scores = {
        labels[i]: float(probs[i]) for i in range(len(labels))
    }

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "class_scores": class_scores,
        "modality": m,
        "model_source": "trained" if bundle.model is not None else "fallback",
    }