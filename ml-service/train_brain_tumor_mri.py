"""
Train the 4-class brain-tumor MRI CNN (same architecture as the Colab notebook).

Expected folder layout (after your 70/15/15 split):
  DATA_ROOT/
    train/glioma|meningioma|notumor|pituitary/*.jpg
    val/...
    test/...   # optional

Or download Kaggle `masoudnickparvar/brain-tumor-mri-dataset`, build splits in Colab,
then copy `brain_tumor_70_15_15` here and point --data-root at it.

Saves:
  models/mri_brain_model.keras  (standard zip bundle — preferred filename for deployment)
  models/mri_model.labels.json   (overwrites with class order matching training)

After training, either copy the bundle to ``models/best_brain_tumor_cnn_mri`` (SavedModel directory)
or symlink/copy ``mri_brain_model.keras`` and register it in ``app.model.MODALITY_SPECS``.
If exporting from Colab as a SavedModel **folder**, do not name the folder ``*.keras``
(Keras 3 opens ``*.keras`` as a zip file even when the path is a directory).

Inference uses preprocess ``unit_range`` (pixels ÷ 255), matching ``ImageDataGenerator(rescale=1./255)``.

Usage (from ``ml-service`` directory):
  python train_brain_tumor_mri.py --data-root ./data/brain_tumor_70_15_15 --epochs 30

You can instead upload ``best_brain_tumor_cnn.keras`` from Colab into ``models/mri_brain_model.keras``
if class order in the file matches labels: glioma, meningioma, notumor, pituitary.
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
    p = argparse.ArgumentParser(description="Train 4-class brain MRI CNN (Colab architecture)")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root with train/ and val/ (class subfolders: glioma, meningioma, notumor, pituitary)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_colab_cnn(img_size: int, num_classes: int = 4) -> keras.Model:
    """Sequential CNN from Colab (rescaling done in pipeline, not in layers)."""
    return keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(img_size, img_size, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="brain_tumor_mri_cnn",
    )


def make_datasets(train_dir: Path, val_dir: Path, img_size: int, batch: int, seed: int):
    train_raw = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch,
        shuffle=True,
        seed=seed,
    )
    val_raw = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch,
        shuffle=False,
    )

    norm_layer = keras.layers.Rescaling(1.0 / 255.0)

    def norm_xy(x, y):
        return norm_layer(x), y

    train_ds = train_raw.map(norm_xy, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.map(norm_xy, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    class_names = list(train_raw.class_names)
    return train_ds, val_ds, class_names


def main():
    args = parse_args()
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Need {train_dir} and {val_dir} with class subfolders.")

    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, class_names = make_datasets(train_dir, val_dir, args.img_size, args.batch_size, args.seed)
    medai_order = ["glioma", "meningioma", "notumor", "pituitary"]
    if [c.lower() for c in class_names] != medai_order:
        print("Warning: folder class names/order differ from MedAI rule engine:", class_names)
        print("For API rules + UI bars, use subfolders exactly:", medai_order)

    model = build_colab_cnn(args.img_size, num_classes=len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_keras = model_dir / "mri_brain_model.keras"
    out_labels = model_dir / "mri_model.labels.json"

    ckpt = str(out_keras).replace(".keras", "_best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(out_keras)
    meta = {"classes": [str(c) for c in class_names], "img_size": int(args.img_size)}
    out_labels.write_text(json.dumps(meta), encoding="utf-8")
    print("Saved:", out_keras)
    print("Labels:", out_labels)


if __name__ == "__main__":
    main()
