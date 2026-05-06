"""
Improved 3-class lung CT training (IQ-OTH/NCCD style):
Normal, Benign, Malignant.

Expected split layout:
  DATA_ROOT/train/<Normal|Benign|Malignant>/*
  DATA_ROOT/val/...
  DATA_ROOT/test/... (optional for final evaluation)

Writes:
  models/best_cnn_lung_model_ctscan.keras
  models/ctscan_model.labels.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def parse_args():
    p = argparse.ArgumentParser(description="Train improved lung CT model")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30, help="Phase 1 frozen-backbone epochs")
    p.add_argument("--finetune-epochs", type=int, default=14, help="Phase 2 fine-tune epochs")
    p.add_argument("--finetune-layers", type=int, default=45, help="Top MobileNetV2 layers to unfreeze")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.06)
    return p.parse_args()


def _canon_name(raw_name: str) -> str:
    x = str(raw_name).lower().replace("_", " ").strip()
    if "normal" in x:
        return "Normal"
    if "malig" in x:
        return "Malignant"
    if "benign" in x or "bengin" in x:
        return "Benign"
    return str(raw_name).strip().replace("_", " ").title()


def build_model(img_size: int, num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.45)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="ct_lung_mobilenet")
    return model, base


def _best_thresholds_ovr(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(np.int32)
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.35, 0.91, 0.02):
            pred_bin = (probs[:, i] >= t).astype(np.int32)
            f1 = f1_score(y_bin, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        out[cls] = round(best_t, 2)
    return out


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    test_dir = args.data_root / "test"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Need {train_dir} and {val_dir}.")

    train_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=12,
        zoom_range=0.15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.06,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_aug = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=args.seed,
    )
    val_gen = eval_aug.flow_from_directory(
        val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = None
    if test_dir.is_dir():
        test_gen = eval_aug.flow_from_directory(
            test_dir,
            target_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    class_names_raw = [k for k, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]
    class_names = [_canon_name(c) for c in class_names_raw]
    n_classes = len(class_names)

    y = train_gen.classes
    uniq = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=uniq, y=y)
    class_weight = {int(c): float(w) for c, w in zip(uniq, cw)}
    print("[ct] class_weight:", class_weight)
    print("[ct] class map:", dict(enumerate(class_names)))

    ls = max(0.0, min(0.3, float(args.label_smoothing)))
    model, base = build_model(args.img_size, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(2e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy")],
    )

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_model = model_dir / "best_cnn_lung_model_ctscan.keras"
    out_labels = model_dir / "ctscan_model.labels.json"
    ckpt1 = str(out_model).replace(".keras", "_phase1_best.keras")

    cb1 = [
        keras.callbacks.ModelCheckpoint(ckpt1, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cb1,
        class_weight=class_weight,
        verbose=1,
    )

    base.trainable = True
    freeze_before = len(base.layers) - max(1, args.finetune_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= freeze_before

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy")],
    )
    cb2 = [
        keras.callbacks.ModelCheckpoint(str(out_model), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-8, verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.finetune_epochs,
        callbacks=cb2,
        class_weight=class_weight,
        verbose=1,
    )

    # Validation diagnostics + per-class thresholds for API tuning.
    val_gen.reset()
    val_probs = model.predict(val_gen, verbose=0)
    val_pred = np.argmax(val_probs, axis=1)
    val_true = val_gen.classes
    print("\n[ct] Validation classification report:")
    print(classification_report(val_true, val_pred, target_names=class_names, digits=4))
    print("[ct] Validation confusion matrix:")
    print(confusion_matrix(val_true, val_pred))
    print("[ct] Suggested one-vs-rest thresholds:", _best_thresholds_ovr(val_true, val_probs, class_names))

    if test_gen is not None:
        print("\n[ct] Final test evaluation:")
        test_metrics = model.evaluate(test_gen, verbose=1)
        print(dict(zip(model.metrics_names, [float(x) for x in test_metrics])))

    model.save(out_model)
    meta = {"classes": class_names, "img_size": int(args.img_size)}
    out_labels.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", out_model)
    print("Labels:", out_labels)


if __name__ == "__main__":
    main()
