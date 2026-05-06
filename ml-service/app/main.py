"""FastAPI inference service for MedAI CNN."""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from app.model import (
    MODALITY_SPECS,
    ModalityBundle,
    get_bundle,
    heuristic_scores,
    load_all_modalities,
    modality_health_snapshot,
    preprocess_image,
)

# Must match registry keys in app.model.MODALITY_SPECS (single source of truth).
ALLOWED_MODALITIES = frozenset(MODALITY_SPECS.keys())


def _ct_probs_canonical(class_scores: dict[str, float]) -> tuple[float, float, float] | None:
    """Return (normal, benign, malignant) from flexible CT labels (handles typos/suffixes)."""
    got: dict[str, float] = {}
    for k, v in class_scores.items():
        key = str(k).lower().replace("_", " ").strip()
        score = float(v)
        if "normal" in key:
            got["normal"] = score
            continue
        if "malig" in key:
            got["malignant"] = score
            continue
        # Handle benign typos from public datasets: "bengin", "benign cases", etc.
        if "benign" in key or "bengin" in key:
            got["benign"] = score
            continue
        got[key] = score
    if not all(x in got for x in ("normal", "benign", "malignant")):
        return None
    return got["normal"], got["benign"], got["malignant"]


def lung_ct_rule_based_decision(class_scores: dict[str, float]) -> tuple[str, dict[str, float]] | None:
    """
    Threshold-first decision (default 0.6), then argmax-style fallback — always returns one of the
    user-facing strings (no \"unclear\" from the API).
    """
    triple = _ct_probs_canonical(class_scores)
    if triple is None:
        return None
    normal, benign, malignant = triple
    scores = {"normal": round(normal, 6), "benign": round(benign, 6), "malignant": round(malignant, 6)}
    t = float(os.environ.get("MEDAI_CT_RULE_THRESHOLD", "0.75"))

    if malignant > t:
        return "Lung Cancer Detected (Malignant Pattern)", scores
    if benign > t:
        return "Lung Abnormality — Benign Pattern (lung cancer unlikely)", scores
    if normal > t:
        return "No Lung Cancer Detected (Normal)", scores

    # Fallback: follow argmax so the decision always matches the highest bar (Normal / Benign / Malignant).
    i_max = int(np.argmax(np.array([normal, benign, malignant], dtype=np.float64)))
    if i_max == 2:
        return "Lung Cancer Likely (Malignant Pattern)", scores
    if i_max == 1:
        return "Lung Cancer Unlikely — Benign Pattern More Likely", scores
    return "No Lung Cancer Likely (Normal)", scores


def _us_is_cancer_positive_label(lab: str) -> bool:
    s = str(lab).lower().replace("_", " ")
    if "not cancer" in s or "no cancer" in s:
        return False
    if "negative" in s and "cancer" not in s:
        return False
    return "cancer" in s or "malignant" in s


def ultrasound_rule_based_decision(
    class_scores: dict[str, float], class_names: list[str]
) -> tuple[str, dict[str, float]] | None:
    """Binary BUSI-style ultrasound: threshold then argmax; `scores` uses not_cancer / cancer keys."""
    if len(class_names) != 2:
        return None

    def get_score(label: str) -> float:
        for k, v in class_scores.items():
            if str(k).lower() == str(label).lower():
                return float(v)
        return -1.0

    p0 = get_score(class_names[0])
    p1 = get_score(class_names[1])
    if p0 < 0 or p1 < 0:
        return None

    pos0 = _us_is_cancer_positive_label(class_names[0])
    pos1 = _us_is_cancer_positive_label(class_names[1])
    if pos0 and not pos1:
        p_can, p_not = p0, p1
    elif pos1 and not pos0:
        p_can, p_not = p1, p0
    else:
        # Default training order: index 0 = negative, 1 = positive (cancer)
        p_not, p_can = p0, p1

    scores = {"not_cancer": round(p_not, 6), "cancer": round(p_can, 6)}
    # Keep this conservative to reduce false-positive "cancer" calls.
    t = float(os.environ.get("MEDAI_US_RULE_THRESHOLD", "0.9999"))
    margin = float(os.environ.get("MEDAI_US_RULE_MARGIN", "0.20"))

    if p_can >= t and (p_can - p_not) >= margin:
        return "Breast Cancer Detected (Ultrasound)", scores
    if p_not >= t and (p_not - p_can) >= margin:
        return "No Breast Cancer Detected (Ultrasound)", scores
    return "Ultrasound Result Inconclusive - Please retest/verify clinically", scores


def _mri_get_score(class_scores: dict[str, float], label: str) -> float:
    for k, v in class_scores.items():
        if str(k).lower() == label.lower():
            return float(v)
    return -1.0


def xray_fracture_rule_based_decision(
    class_scores: dict[str, float], class_names: list[str]
) -> tuple[str, dict[str, float]] | None:
    """Binary X-ray decision tuned to reduce false fracture positives on healthy hands."""
    if len(class_names) != 2:
        return None

    p0 = float(class_scores.get(class_names[0], -1.0))
    p1 = float(class_scores.get(class_names[1], -1.0))
    if p0 < 0 or p1 < 0:
        return None

    c0 = str(class_names[0]).lower().replace("_", " ").strip()
    c1 = str(class_names[1]).lower().replace("_", " ").strip()
    frac_tokens = ("fracture", "fractured", "broken")
    not_tokens = ("not fractured", "no fracture", "normal", "healthy")

    c0_is_fracture = any(t in c0 for t in frac_tokens) and not any(t in c0 for t in not_tokens)
    c1_is_fracture = any(t in c1 for t in frac_tokens) and not any(t in c1 for t in not_tokens)

    if c0_is_fracture and not c1_is_fracture:
        p_frac, p_not = p0, p1
    elif c1_is_fracture and not c0_is_fracture:
        p_frac, p_not = p1, p0
    else:
        # Fallback to common training order in this project: index 0 fractured, index 1 not fractured.
        p_frac, p_not = p0, p1

    scores = {"fractured": round(p_frac, 6), "not_fractured": round(p_not, 6)}
    t = float(os.environ.get("MEDAI_XRAY_FRACTURE_THRESHOLD", "0.85"))
    if p_frac >= t:
        return "Bone Fracture Detected (X-ray)", scores
    if p_not >= 0.5:
        return "No Bone Fracture Detected (X-ray)", scores
    return "Fracture Uncertain (X-ray) — Please verify clinically", scores


def _mri_subtype_title(name: str) -> str:
    n = str(name).lower()
    if n == "glioma":
        return "Glioma"
    if n == "meningioma":
        return "Meningioma"
    if n == "pituitary":
        return "Pituitary"
    if n == "notumor":
        return "No tumor"
    return str(name).replace("_", " ").title()


def mri_brain_tumor_rule_based_decision(
    class_scores: dict[str, float], class_names: list[str]
) -> tuple[str, dict[str, float]] | None:
    """
    4-class brain MRI (Kaggle / Colab CNN): glioma, meningioma, notumor, pituitary.
    Collapses to Colab-style Tumor vs No Tumor for the headline; ``scores`` keeps all four probabilities.
    """
    lowered = [str(c).lower() for c in class_names]
    need = frozenset({"glioma", "meningioma", "notumor", "pituitary"})
    if len(class_names) != 4 or frozenset(lowered) != need:
        return None

    p_g = _mri_get_score(class_scores, "glioma")
    p_m = _mri_get_score(class_scores, "meningioma")
    p_n = _mri_get_score(class_scores, "notumor")
    p_p = _mri_get_score(class_scores, "pituitary")
    if min(p_g, p_m, p_n, p_p) < 0:
        return None

    scores = {str(c): round(float(_mri_get_score(class_scores, c)), 6) for c in class_names}
    t = float(os.environ.get("MEDAI_MRI_RULE_THRESHOLD", "0.7"))
    probs = np.array([_mri_get_score(class_scores, c) for c in class_names], dtype=np.float64)
    i_max = int(np.argmax(probs))
    top = str(class_names[i_max]).lower()

    if p_n > t:
        return "No Brain Tumor Detected (MRI)", scores
    if top != "notumor" and float(probs[i_max]) > t:
        return f"Brain Tumor Detected (MRI) — {_mri_subtype_title(class_names[i_max])}", scores
    if top == "notumor":
        return "No Brain Tumor Likely (MRI)", scores
    return f"Brain Tumor Likely (MRI) — {_mri_subtype_title(class_names[i_max])}", scores


def _normalize_modality(raw: str) -> str:
    m = raw.lower().strip()
    if m in ("ctscan", "ct_scan"):
        m = "ct"
    if m in ("brain_mri", "brainmri", "brain", "fmri"):
        m = "mri"
    return m


def _prediction_probs(raw: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Map model output to a probability vector aligned with class_names."""
    v = np.ravel(np.asarray(raw, dtype=np.float32))
    n = len(class_names)
    if v.size == 1 and n == 2:
        # Keras binary sigmoid: one logit = P(class index 1).
        p1 = float(np.clip(v[0], 0.0, 1.0))
        return np.array([1.0 - p1, p1], dtype=np.float32)
    if v.size != n:
        raise HTTPException(
            500,
            f"Model returned {v.size} outputs but {n} classes are configured",
        )
    s = float(np.sum(v))
    if s > 0:
        v = v / s
    return v


def _predict_batch(bundle: ModalityBundle, pil: Image.Image) -> tuple[np.ndarray, list[str]]:
    names = bundle.class_names
    sz = bundle.img_size
    mode = bundle.preprocess

    def _forward(p: Image.Image) -> np.ndarray:
        batch = preprocess_image(p, sz, mode=mode)
        raw = bundle.model.predict(batch, verbose=0)[0]
        return _prediction_probs(raw, names)

    if bundle.trained:
        probs = _forward(pil)
        if bundle.use_tta:
            flipped = pil.transpose(Image.FLIP_LEFT_RIGHT)
            probs = (probs + _forward(flipped)) / 2.0
        s = float(np.sum(probs))
        if s > 0:
            probs = probs / s
        return probs, names

    batch_model = preprocess_image(pil, sz, mode=mode)
    batch_heur = preprocess_image(pil, sz, mode="default")
    cnn = bundle.model.predict(batch_model, verbose=0)[0]
    heur = heuristic_scores(batch_heur, names)
    cnn = np.ravel(cnn)
    if cnn.size == 1 and len(names) == 2:
        p1 = float(np.clip(cnn[0], 0.0, 1.0))
        cnn = np.array([1.0 - p1, p1], dtype=np.float32)
    probs = 0.35 * cnn + 0.65 * heur
    s = float(np.sum(probs))
    if s > 0:
        probs = probs / s
    return probs, names


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_modalities()
    yield


app = FastAPI(title="MedAI ML Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    class_scores: dict[str, float]
    modality: str
    model_source: str
    result: str | None = None
    scores: dict[str, float] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "modalities": modality_health_snapshot()}


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
async def predict(
    file: UploadFile = File(...),
    modality: str = Form("xray"),
):
    m = _normalize_modality(modality)
    if m not in ALLOWED_MODALITIES:
        raise HTTPException(
            400,
            f"modality must be one of: {', '.join(sorted(ALLOWED_MODALITIES))}",
        )
    try:
        bundle = get_bundle(m)
    except KeyError:
        raise HTTPException(400, "Unknown modality") from None
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file (PNG, JPEG, etc.)")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 15MB)")
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except Exception:
        raise HTTPException(400, "Could not read image") from None

    probs, class_names = _predict_batch(bundle, pil)
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = float(probs[idx])
    class_scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    rule_result: str | None = None
    rule_scores: dict[str, float] | None = None
    if m == "ct":
        dec = lung_ct_rule_based_decision(class_scores)
        if dec:
            rule_result, rule_scores = dec
    elif m == "xray":
        dec = xray_fracture_rule_based_decision(class_scores, class_names)
        if dec:
            rule_result, rule_scores = dec
    elif m == "ultrasound":
        dec = ultrasound_rule_based_decision(class_scores, class_names)
        if dec:
            rule_result, rule_scores = dec
    elif m == "mri":
        dec = mri_brain_tumor_rule_based_decision(class_scores, class_names)
        if dec:
            rule_result, rule_scores = dec

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        class_scores=class_scores,
        modality=m,
        model_source="trained" if bundle.trained else "demo_blend_untrained_cnn",
        result=rule_result,
        scores=rule_scores,
    )
