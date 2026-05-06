"""
Improved binary X-ray fracture training (matches Colab rescale=1/255 + transfer learning).

Phases:
  1) Frozen EfficientNetB0 head training
  2) Fine-tune last backbone layers (large accuracy gain on medical X-rays)

Expected:
  DATA_ROOT/train/<class>/*.{jpg,png}
  DATA_ROOT/val/<class>/*

Writes:
  models/fracture_model.keras
  models/fracture_model.labels.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras


def parse_args():
    p = argparse.ArgumentParser(description="Train improved binary fracture X-ray model")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25, help="Phase 1 epochs (frozen backbone)")
    p.add_argument("--finetune-epochs", type=int, default=15, help="Phase 2 epochs (unfrozen layers)")
    p.add_argument("--finetune-layers", type=int, default=40, help="Unfreeze this many top EfficientNet layers")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(img_size: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="xray_fracture_efficientnet")
    return model, base


def best_threshold(val_gen, model) -> float:
    val_gen.reset()
    probs = model.predict(val_gen, verbose=0).ravel()
    y_true = val_gen.classes
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.35, 0.92, 0.02):
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"[xray] Validation best F1 threshold ≈ {best_t:.2f} (F1={best_f1:.4f}); set MEDAI_XRAY_FRACTURE_THRESHOLD to match if desired.")
    return best_t


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Need {train_dir} and {val_dir}.")

    train_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.12,
        shear_range=0.08,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_aug = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=True,
        seed=args.seed,
    )
    val_gen = eval_aug.flow_from_directory(
        val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="binary",
        shuffle=False,
    )
    if len(train_gen.class_indices) != 2:
        raise SystemExit(f"Expected 2 classes, got {train_gen.class_indices}")

    y = train_gen.classes
    counts = np.bincount(y, minlength=2).astype(np.float32)
    total = float(np.sum(counts))
    class_weight = {i: total / (2.0 * float(max(c, 1.0))) for i, c in enumerate(counts)}

    model, base = build_model(args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(), keras.metrics.Recall()],
    )

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_model = model_dir / "fracture_model.keras"
    out_labels = model_dir / "fracture_model.labels.json"
    ckpt = str(out_model).replace(".keras", "_phase1_best.keras")

    cb1 = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cb1,
        class_weight=class_weight,
        verbose=1,
    )

    # Phase 2: fine-tune top of backbone
    base.trainable = True
    freeze_before = len(base.layers) - max(1, args.finetune_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= freeze_before
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(), keras.metrics.Recall()],
    )
    cb2 = [
        keras.callbacks.ModelCheckpoint(str(out_model), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.finetune_epochs,
        callbacks=cb2,
        class_weight=class_weight,
        verbose=1,
    )

    best_threshold(val_gen, model)
    model.save(out_model)
    class_map = {str(v): str(k) for k, v in train_gen.class_indices.items()}
    meta = dict(class_map)
    meta["img_size"] = int(args.img_size)
    out_labels.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", out_model)
    print("Labels:", out_labels)


if __name__ == "__main__":
    main()
