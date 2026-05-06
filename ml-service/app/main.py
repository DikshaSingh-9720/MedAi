from __future__ import annotations

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.model import load_all_bundles, preprocess_image, probs_from_output

app = FastAPI(title="MedAI ML Service", version="1.0.0")

BUNDLES = load_all_bundles()
ALLOWED_MODALITIES = {"xray", "ct", "ultrasound", "mri"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modalities": {
            k: {
                "model_loaded": v.model is not None,
                "labels": v.labels,
                "source_path": v.source_path,
            }
            for k, v in BUNDLES.items()
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), modality: str = Form("xray")):
    m = (modality or "xray").strip().lower()
    if m not in ALLOWED_MODALITIES:
        raise HTTPException(400, f"Unsupported modality: {modality}")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file (PNG, JPEG, etc.)")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    bundle = BUNDLES[m]
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
    class_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "class_scores": class_scores,
        "modality": m,
        "model_source": "trained" if bundle.model is not None else "fallback_no_model",
    }
