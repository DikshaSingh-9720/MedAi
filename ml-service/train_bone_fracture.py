"""
Train a binary fracture classifier from folder-organized X-rays (Colab-style workflow).

Expected layout (same as typical Kaggle unzip):
  DATA_ROOT/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg   # optional; used only for final evaluation if present

Example class folders: Fractured, Not_fractured (any two folder names; order is preserved).

Usage:
  set PYTHONPATH=medai/ml-service   # or run from ml-service directory
  python train_bone_fracture.py --data-root ./data/bone-fracture

Requires: tensorflow, pillow (see requirements.txt)

Security: Do NOT commit Kaggle API keys. Use ~/.kaggle/kaggle.json locally or env vars.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras


def parse_args():
    p = argparse.ArgumentParser(description="Train binary bone-fracture CNN")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root with train/ and val/ subdirs (each containing class subfolders)",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable light augmentation on training set",
    )
    return p.parse_args()


def build_binary_cnn(img_size: int) -> keras.Model:
    """Similar depth to Colab baseline; 2-class softmax (clearer than single sigmoid)."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="bone_fracture_binary")


def make_datasets(train_dir: Path, val_dir: Path, img_size: int, batch: int, seed: int, augment: bool):
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        interpolation="bilinear",
        batch_size=batch,
        shuffle=True,
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        interpolation="bilinear",
        batch_size=batch,
        shuffle=False,
        seed=seed,
    )
    class_names = train_ds.class_names
    if tuple(class_names) != tuple(val_ds.class_names):
        raise ValueError(
            f"Train classes {train_ds.class_names} != val classes {val_ds.class_names}"
        )
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly 2 class folders for binary task; got {class_names}")

    if augment:

        def aug(img, y):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.06)
            return tf.clip_by_value(img, 0.0, 255.0), y

        train_ds = train_ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Need {train_dir} and {val_dir} under --data-root")

    train_ds, val_ds, class_names = make_datasets(
        train_dir, val_dir, args.img_size, args.batch_size, args.seed, args.augment
    )

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_model = model_dir / "medai_fracture_binary.keras"
    out_labels = model_dir / "medai_fracture_binary.labels.json"

    model = build_binary_cnn(args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(out_model),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    reduce = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, early, reduce],
        verbose=1,
    )

    # Persist best weights from checkpoint path (may already be best)
    best = keras.models.load_model(out_model)
    best.save(out_model)

    meta = {
        "classes": class_names,
        "img_size": args.img_size,
        "history_final_epochs": len(history.history.get("loss", [])),
    }
    out_labels.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model: {out_model}")
    print(f"Saved labels: {out_labels}")
    print("Classes (softmax index order):", class_names)
    print()
    print("Point the API at this model:")
    print('  set MEDAI_MODEL_FILE=medai_fracture_binary.keras')


if __name__ == "__main__":
    main()
