"""
Train a small CNN on synthetic grayscale-like images so softmax is learned (still not clinically valid).
Run once: python train_synthetic.py
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.model import MODEL_DIR, MODEL_PATH, build_cnn, IMG_SIZE

N_PER_CLASS = 80
BATCH = 16
EPOCHS = 8
NUM_CLASSES = 4
SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)


def synthetic_batch(n: int, class_idx: int) -> np.ndarray:
    """Class-dependent noise patterns (demo only)."""
    base_means = [0.42, 0.58, 0.35, 0.50]
    base_std = [0.12, 0.18, 0.14, 0.16]
    m = base_means[class_idx]
    s = base_std[class_idx]
    x = rng.normal(m, s, (n, IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)
    x = np.clip(x, 0, 1)
    rgb = np.repeat(x, 3, axis=-1)
    rgb *= 255.0
    return rgb


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_list = []
    y_list = []
    for c in range(NUM_CLASSES):
        X_list.append(synthetic_batch(N_PER_CLASS, c))
        y_list.append(np.full(N_PER_CLASS, c, dtype=np.int32))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    p = rng.permutation(len(X))
    X, y = X[p], y[p]

    y_one = keras.utils.to_categorical(y, NUM_CLASSES)
    model = build_cnn(NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X, y_one, batch_size=BATCH, epochs=EPOCHS, validation_split=0.15, verbose=1)
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
