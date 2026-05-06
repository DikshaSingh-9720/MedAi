"""CNN for medical image screening; multi-modality registry (X-ray, CT, ultrasound, MRI)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_CLASS_NAMES = [
    "Normal",
    "Fracture_detected",
    "Pneumonia_pattern",
    "Other_abnormality",
]

# Mutated in place when loading sidecar labels so imports keep a stable reference.
CLASS_NAMES: list[str] = list(DEFAULT_CLASS_NAMES)

IMG_SIZE = 224
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# Default to bundled fracture weights when present; override with MEDAI_MODEL_FILE.
MODEL_FILE = os.environ.get("MEDAI_MODEL_FILE", "fracture_model.keras")
MODEL_PATH = MODEL_DIR / MODEL_FILE
LABELS_PATH = MODEL_PATH.with_suffix(".labels.json")

MODALITY_SPECS: dict[str, dict[str, str]] = {
    "xray": {"model": "fracture_model.keras", "labels": "fracture_model.labels.json"},
    # CT export is commonly a SavedModel directory named `best_cnn_lung_model_ctscan`.
    "ct": {"model": "best_cnn_lung_model_ctscan", "labels": "ctscan_model.labels.json"},
    "ultrasound": {"model": "best_busi_mobilenet_model_ultrasound", "labels": "ultrasound_model.labels.json"},
    # SavedModel directory (not named *.keras — Keras 3 treats *.keras paths as zip archives).
    # Colab saves ``best_brain_tumor_cnn.keras``; local trainer saves ``mri_brain_model.keras`` (both resolved via glob).
    "mri": {"model": "best_brain_tumor_cnn_mri", "labels": "mri_model.labels.json"},
}


@dataclass
class ModalityBundle:
    key: str
    model: keras.Model
    class_names: list[str]
    img_size: int
    model_path: Path
    trained: bool
    load_error: str | None = None
    preprocess: str = "default"  # "default" | "mobilenet_v2" | "unit_range" (÷255, Colab-style CNN)
    use_tta: bool = True  # average with horizontal flip when trained


_BUNDLES: dict[str, ModalityBundle] = {}


def _class_list_from_labels_meta(meta: dict) -> list[str]:
    """Accepts our training format {"classes": [...]} or Colab/Keras index map {"0": "A", "1": "B"}."""
    if "classes" in meta:
        return [str(x) for x in meta["classes"]]
    pairs: list[tuple[int, str]] = []
    for k, v in meta.items():
        if str(k).isdigit():
            pairs.append((int(k), str(v)))
    if pairs and len(pairs) == len(meta):
        pairs.sort(key=lambda x: x[0])
        return [label for _, label in pairs]
    raise ValueError("labels.json needs 'classes' or numeric keys like '0','1' with label strings")


def apply_metadata() -> None:
    """Load class names and image size from *.labels.json when present (e.g. fracture binary training)."""
    global IMG_SIZE
    if LABELS_PATH.is_file():
        meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        CLASS_NAMES.clear()
        CLASS_NAMES.extend(_class_list_from_labels_meta(meta))
        IMG_SIZE = int(meta.get("img_size", 224))
        return
    CLASS_NAMES.clear()
    CLASS_NAMES.extend(DEFAULT_CLASS_NAMES)
    IMG_SIZE = 224


def _strip_keras_serialization_noise(obj: object) -> None:
    """Drop keys from saved configs that older bundled Keras rejects (e.g. Colab Keras 3 `quantization_config`)."""
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _strip_keras_serialization_noise(v)
    elif isinstance(obj, list):
        for x in obj:
            _strip_keras_serialization_noise(x)


def load_model_from_disk(path: Path) -> keras.Model:
    """Load a `.keras` zip bundle or a Keras 3 saved-model directory; patch configs if needed."""
    # Directory models must not use the `.keras` suffix: Keras opens `.keras` as a zip file even when the path is a folder.
    if path.is_dir():
        try:
            return keras.models.load_model(path)
        except (TypeError, ValueError) as e:
            if "quantization_config" not in str(e).lower():
                raise
            cfg_path = path / "config.json"
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                _strip_keras_serialization_noise(cfg)
                cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            return keras.models.load_model(path)

    try:
        return keras.models.load_model(path)
    except (TypeError, ValueError) as e:
        if "quantization_config" not in str(e).lower():
            raise

    buf = io.BytesIO()
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            raw = zin.read(info.filename)
            if info.filename.endswith("config.json"):
                try:
                    cfg = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    zout.writestr(info, raw)
                else:
                    _strip_keras_serialization_noise(cfg)
                    zout.writestr(info, json.dumps(cfg).encode("utf-8"))
            else:
                zout.writestr(info, raw)

    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(buf.read())
        patched = tmp.name
    try:
        return keras.models.load_model(patched)
    finally:
        Path(patched).unlink(missing_ok=True)


def build_cnn(num_classes: int = 4, img_size: int | None = None) -> keras.Model:
    sz = int(img_size if img_size is not None else IMG_SIZE)
    inputs = keras.Input(shape=(sz, sz, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="medai_cnn")


def load_or_build_model() -> keras.Model:
    apply_metadata()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return load_model_from_disk(MODEL_PATH)
    return build_cnn(len(CLASS_NAMES), IMG_SIZE)


def preprocess_image(
    pil_image: Image.Image,
    img_size: int | None = None,
    *,
    mode: str = "default",
) -> np.ndarray:
    """RGB, resize. ``default``: 0–255 float32. ``mobilenet_v2``: Keras MobileNetV2 preprocessing."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    sz = int(img_size if img_size is not None else IMG_SIZE)
    img = pil_image.resize((sz, sz), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape != (sz, sz, 3):
        raise ValueError("Invalid image array shape")
    batch = np.expand_dims(arr, axis=0)
    if mode == "mobilenet_v2":
        batch = keras.applications.mobilenet_v2.preprocess_input(batch)
    elif mode == "unit_range":
        # Match ImageDataGenerator(rescale=1./255) used in Colab CNN training.
        batch = batch / 255.0
    return batch


def _read_labels_file(labels_path: Path) -> tuple[list[str], int]:
    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    names = _class_list_from_labels_meta(meta)
    img_sz = int(meta.get("img_size", 224))
    return names, img_sz


def _resolve_model_path(key: str, configured: Path) -> Path:
    """Pick the best weights path: prefer a real ``.keras`` zip over an incomplete SavedModel folder."""
    if configured.is_file():
        return configured
    if configured.is_dir() and (configured / "config.json").is_file() and (configured / "model.weights.h5").is_file():
        return configured

    patterns_by_modality = {
        "mri": [
            "best_brain_tumor_cnn_mri",
            "mri_brain_model.keras",
            "best_brain_tumor_cnn.keras",
            "*brain*tumor*.keras",
            "*mri*.keras",
        ],
        # Avoid generic `*ct*.keras` because it can accidentally match `fracture_model.keras`.
        "ct": ["best_cnn_lung_model_ctscan", "best_cnn_lung_model_ctscan.keras", "*ct*scan*.keras", "*lung*ctscan*.keras", "*lung*ct*model*.keras"],
        "ultrasound": ["best_busi_mobilenet_model_ultrasound", "best_busi_mobilenet_model.keras", "*busi*.keras", "*ultrasound*.keras"],
        "xray": ["fracture_model.keras", "medai_fracture_binary.keras", "*fracture*.keras", "*xray*.keras"],
    }
    patterns = patterns_by_modality.get(key, [])
    # Search only in models/ to avoid accidentally picking stray exports from project root.
    search_roots = [MODEL_DIR]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for pat in patterns:
            for p in root.glob(pat):
                if p.is_file() and p.suffix.lower() == ".keras":
                    candidates.append(p)
                    continue
                if p.is_dir() and (p / "config.json").is_file() and (p / "model.weights.h5").is_file():
                    candidates.append(p)
    if candidates:
        uniq = {p.resolve(): p for p in candidates}
        return max(uniq.values(), key=lambda p: p.stat().st_mtime)

    # Explicit fallbacks when primary filename is missing (common Colab / trainer names).
    fallbacks: dict[str, list[Path]] = {
        "mri": [
            MODEL_DIR / "best_brain_tumor_cnn_mri",
            MODEL_DIR / "best_brain_tumor_cnn.keras",
            MODEL_DIR / "mri_brain_model.keras",
            MODEL_DIR.parent / "best_brain_tumor_cnn_mri",
            MODEL_DIR.parent / "best_brain_tumor_cnn.keras",
            MODEL_DIR.parent / "mri_brain_model.keras",
        ],
        "ultrasound": [
            MODEL_DIR / "best_busi_mobilenet_model_ultrasound",
            MODEL_DIR / "best_busi_mobilenet_model.keras",
            MODEL_DIR.parent / "best_busi_mobilenet_model_ultrasound",
            MODEL_DIR.parent / "best_busi_mobilenet_model.keras",
        ],
    }
    for fp in fallbacks.get(key, []):
        if fp.is_file() or (fp.is_dir() and (fp / "config.json").is_file() and (fp / "model.weights.h5").is_file()):
            return fp

    if configured.exists():
        return configured
    return configured


def _load_one_modality(key: str) -> ModalityBundle:
    spec = MODALITY_SPECS[key]
    labels_path = MODEL_DIR / spec["labels"]
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels for {key}: {labels_path}")
    class_names, img_size = _read_labels_file(labels_path)
    path = _resolve_model_path(key, MODEL_DIR / spec["model"])
    trained = False
    load_error: str | None = None
    model: keras.Model | None = None
    if path.exists():
        if path.is_dir() and not (path / "model.weights.h5").is_file():
            load_error = (
                f"Incomplete model export for {key}: missing {path / 'model.weights.h5'}. "
                "Export/upload the full Keras model (including weights)."
            )
        try:
            if load_error is None:
                model = load_model_from_disk(path)
                trained = True
        except Exception as e:  # noqa: BLE001 — log and fall back for dev
            load_error = f"Could not load weights for {key} ({path}): {e}"
            print(f"[medai] {load_error}")
    else:
        load_error = f"Missing model file/folder for {key}: {path}"
    if model is None:
        model = build_cnn(len(class_names), img_size)
    if key == "ultrasound":
        # BUSI trainer in this repo uses ImageDataGenerator(rescale=1./255),
        # so inference must use unit-range pixels (not MobileNet preprocess_input).
        preprocess = "unit_range"
    elif key == "xray":
        # Colab fracture models usually train with ImageDataGenerator(rescale=1./255).
        # Keep configurable in case a different export expects raw 0-255 input.
        preprocess = os.environ.get("MEDAI_XRAY_PREPROCESS", "unit_range").strip().lower() or "unit_range"
    elif key == "mri":
        preprocess = "unit_range"
    elif key == "ct":
        # CT trainers in this project use ImageDataGenerator(rescale=1./255).
        preprocess = "unit_range"
    else:
        preprocess = "default"
    # Disable TTA by default: horizontal flips can degrade medical-image predictions.
    tta_raw = os.environ.get("MEDAI_INFER_TTA", "0").lower()
    use_tta = tta_raw not in ("0", "false", "no", "off")
    # Brain MRI is asymmetric in pathology; horizontal-flip TTA often hurts vs helps (training rarely augments flips).
    if key == "mri":
        mri_tta = os.environ.get("MEDAI_MRI_TTA", "0").lower()
        use_tta = mri_tta in ("1", "true", "yes", "on")
    return ModalityBundle(
        key=key,
        model=model,
        class_names=list(class_names),
        img_size=img_size,
        model_path=path,
        trained=trained,
        load_error=load_error,
        preprocess=preprocess,
        use_tta=use_tta,
    )


def load_all_modalities() -> dict[str, ModalityBundle]:
    """Load every modality into the registry (call once at startup)."""
    global _BUNDLES
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _BUNDLES = {k: _load_one_modality(k) for k in MODALITY_SPECS}
    return _BUNDLES


def get_bundle(modality: str) -> ModalityBundle:
    m = modality.lower().strip()
    if m in ("ctscan", "ct_scan"):
        m = "ct"
    if m not in _BUNDLES:
        raise KeyError(modality)
    return _BUNDLES[m]


def modality_health_snapshot() -> dict[str, dict]:
    """Serializable status for GET /health (registry must be loaded)."""
    return {
        k: {
            "trained": v.trained,
            "classes": v.class_names,
            "img_size": v.img_size,
            "weights_path": str(v.model_path),
            "load_error": v.load_error,
            "preprocess": v.preprocess,
            "tta": v.use_tta,
        }
        for k, v in _BUNDLES.items()
    }


def heuristic_scores(arr: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Deterministic demo scores when weights are missing (not for clinical use)."""
    n = len(class_names)
    x = arr[0] / 255.0
    m = float(np.mean(x))
    s = float(np.std(x))
    g = float(np.mean(np.abs(np.diff(x, axis=0)))) + float(np.mean(np.abs(np.diff(x, axis=1))))

    if n == 2:
        # Rough proxy: higher edge activity → more likely "positive" (index 1).
        p1 = float(np.clip(0.35 + g * 2.5 + abs(m - 0.48) * 1.5 - s * 0.8, 0.05, 0.95))
        raw = np.array([1.0 - p1, p1], dtype=np.float32)
        e = np.exp(raw - np.max(raw))
        return e / np.sum(e)

    if n == 3:
        # Template order: [Normal, Benign, Malignant] logits before permuting to class_names.
        raw_nbm = np.array(
            [
                1.1 - g * 1.5 - abs(m - 0.42) * 2.0,
                0.6 + g * 1.2 + s * 1.5,
                g * 2.5 + abs(m - 0.52) * 2.2,
            ],
            dtype=np.float32,
        )
        lowered = [str(c).lower() for c in class_names]
        if lowered == ["normal", "malignant", "benign"]:
            raw = np.array([raw_nbm[0], raw_nbm[2], raw_nbm[1]], dtype=np.float32)
        else:
            raw = raw_nbm
        e = np.exp(raw - np.max(raw))
        return e / np.sum(e)

    if n == 4:
        raw = np.array(
            [
                1.2 - abs(m - 0.45) * 3.0 - s * 2.0,
                g * 4.0 + abs(m - 0.5) * 2.0,
                (1.0 - m) * 2.0 + s * 3.0,
                s * 2.5 + abs(m - 0.55) * 2.0,
            ],
            dtype=np.float32,
        )
        e = np.exp(raw - np.max(raw))
        return e / np.sum(e)

    raw = np.ones(n, dtype=np.float32)
    e = np.exp(raw - np.max(raw))
    return e / np.sum(e)
