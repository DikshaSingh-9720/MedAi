from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = 224


@dataclass
class ModelBundle:
    key: str
    model: keras.Model | None
    labels: list[str]
    preprocess: str = "unit_range"
    source_path: str = ""


MODALITY_SPECS = {
    "xray": {
        "model_candidates": ["fracture_model.keras", "your_model.keras"],
        "labels_candidates": ["fracture_model.labels.json", "labels.json"],
        "defaults": ["Fractured", "Not Fractured"],
    },
    "ct": {
        "model_candidates": ["best_cnn_lung_model ctscan.keras", "best_cnn_lung_model_ctscan.keras", "your_model.keras"],
        "labels_candidates": ["ctscan_model.labels.json"],
        "defaults": ["Bengin cases", "Malignant cases", "Normal cases"],
    },
    "ultrasound": {
        "model_candidates": ["best_busi_mobilenet_model ultrasound.keras", "best_busi_mobilenet_model.keras", "your_model.keras"],
        "labels_candidates": ["ultrasound_model.labels.json"],
        "defaults": ["Not Cancer", "Cancer"],
    },
    "mri": {
        "model_candidates": ["best_brain_tumor_cnn_mri.keras", "best_brain_tumor_cnn.keras", "your_model.keras"],
        "labels_candidates": ["mri_model.labels.json"],
        "defaults": ["glioma", "meningioma", "notumor", "pituitary"],
    },
}


def _read_labels(path: Path, defaults: list[str]) -> list[str]:
    if not path.is_file():
        return defaults
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
        return [str(x) for x in data["classes"]]
    if isinstance(data, dict):
        numeric = [(int(k), str(v)) for k, v in data.items() if str(k).isdigit()]
        if numeric and len(numeric) == len(data):
            numeric.sort(key=lambda x: x[0])
            return [v for _, v in numeric]
    if isinstance(data, list):
        return [str(x) for x in data]
    return defaults


def _strip_keras_noise(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _strip_keras_noise(v)
    elif isinstance(obj, list):
        for x in obj:
            _strip_keras_noise(x)


def _load_model_path(path: Path):
    if path.is_file():
        try:
            return keras.models.load_model(path)
        except Exception:
            import zipfile
            import tempfile

            with zipfile.ZipFile(path, "r") as zin, tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                    for info in zin.infolist():
                        raw = zin.read(info.filename)
                        if info.filename.endswith("config.json"):
                            try:
                                cfg = json.loads(raw.decode("utf-8"))
                                _strip_keras_noise(cfg)
                                raw = json.dumps(cfg).encode("utf-8")
                            except Exception:
                                pass
                        zout.writestr(info, raw)
                patched = tmp.name
            try:
                return keras.models.load_model(patched)
            finally:
                Path(patched).unlink(missing_ok=True)
    if path.is_dir() and (path / "config.json").is_file() and (path / "model.weights.h5").is_file():
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        _strip_keras_noise(cfg)
        model = keras.models.model_from_json(json.dumps(cfg))
        model.load_weights(path / "model.weights.h5")
        return model
    return None


def _load_one(modality: str) -> ModelBundle:
    spec = MODALITY_SPECS[modality]
    labels = list(spec["defaults"])
    for lp in spec["labels_candidates"]:
        cand = MODELS_DIR / lp
        if cand.is_file():
            labels = _read_labels(cand, labels)
            break

    model = None
    source_path = ""
    for mp in spec["model_candidates"]:
        cand = MODELS_DIR / mp
        if cand.exists():
            try:
                loaded = _load_model_path(cand)
                if loaded is not None:
                    model = loaded
                    source_path = str(cand)
                    break
            except Exception:
                continue
    return ModelBundle(key=modality, model=model, labels=labels, preprocess="unit_range", source_path=source_path)


def load_all_bundles() -> dict[str, ModelBundle]:
    return {m: _load_one(m) for m in MODALITY_SPECS}


def preprocess_image(file_bytes: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def probs_from_output(raw: np.ndarray, labels: list[str]) -> np.ndarray:
    pred = np.asarray(raw, dtype=np.float32).ravel()
    n = len(labels)
    if pred.size == 1 and n == 2:
        p1 = float(np.clip(pred[0], 0.0, 1.0))
        return np.array([1.0 - p1, p1], dtype=np.float32)
    if pred.size != n:
        probs = np.full((n,), 1.0 / max(1, n), dtype=np.float32)
        return probs
    s = float(np.sum(pred))
    return pred / s if s > 0 else np.full((n,), 1.0 / max(1, n), dtype=np.float32)
