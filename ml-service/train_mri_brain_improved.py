"""
Improved 4-class brain MRI training (MobileNetV2 transfer learning + class weights + fine-tune).

Expected:
  DATA_ROOT/train/<glioma|meningioma|notumor|pituitary>/*
  DATA_ROOT/val/...

Writes:
  models/mri_brain_model.keras
  models/mri_model.labels.json  (classes in folder index order)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def parse_args():
    p = argparse.ArgumentParser(description="Train improved brain MRI tumor model")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--finetune-epochs", type=int, default=12)
    p.add_argument("--finetune-layers", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.05, help="Softens softmax targets; reduces overconfidence")
    return p.parse_args()


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
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="mri_brain_mobilenet")
    return model, base


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Need {train_dir} and {val_dir}.")

    train_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
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

    class_names = [k for k, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)
    y = train_gen.classes
    uniq = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=uniq, y=y)
    class_weight = {int(c): float(w) for c, w in zip(uniq, cw)}
    print("[mri] class_weight:", class_weight)

    model, base = build_model(args.img_size, n_classes)
    ls = max(0.0, min(0.3, args.label_smoothing))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_model = model_dir / "mri_brain_model.keras"
    out_labels = model_dir / "mri_model.labels.json"
    ckpt1 = str(out_model).replace(".keras", "_phase1_best.keras")

    cb1 = [
        keras.callbacks.ModelCheckpoint(ckpt1, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max", verbose=1),
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
        optimizer=keras.optimizers.Adam(5e-6),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=ls),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy")],
    )
    cb2 = [
        keras.callbacks.ModelCheckpoint(str(out_model), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max", verbose=1),
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

    model.save(out_model)
    meta = {"classes": class_names, "img_size": int(args.img_size)}
    out_labels.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", out_model)
    print("Labels:", out_labels)


if __name__ == "__main__":
    main()
