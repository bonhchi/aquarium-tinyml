"""
train_core.py - Module 5 của pipeline TinyML
--------------------------------------------
Chức năng:
    - Đọc dữ liệu đã qua bước tạo đặc trưng
    - Chia tập train/validation/test
    - Xây dựng model MLP nhỏ gọn phù hợp TinyML
    - Huấn luyện, đánh giá và lưu model_fp32.keras + metrics.json (kèm precision/recall/AUC)

Dùng:
    python -m src.ml.train.train_core [--input-file ... --model-out ...]
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

# =============================================================
# Cấu hình cơ bản
# =============================================================
INPUT_FILE = "dataset/interim/aquarium_tinyml_features.csv"
MODEL_OUT = "ml/artifacts/model_fp32.keras"
METRICS_OUT = "ml/artifacts/metrics.json"
LABEL_ENCODER_OUT = "ml/artifacts/label_encoder.joblib"

EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42


@dataclass
class TrainingConfig:
    """Tập hợp các tham số để job training có thể tuỳ biến."""

    input_file: str = INPUT_FILE
    model_out: str = MODEL_OUT
    metrics_out: str = METRICS_OUT
    label_encoder_out: str = LABEL_ENCODER_OUT
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    test_size: float = TEST_SIZE
    val_size: float = VAL_SIZE
    random_state: int = RANDOM_STATE


# =============================================================
# Hàm hỗ trợ
# =============================================================
def set_global_seed(seed: int) -> None:
    """Cố định seed cho Python/NumPy/TensorFlow để dễ tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(file_path: str = INPUT_FILE):
    """Đọc và tách dữ liệu thành X, y kèm danh sách cột đặc trưng."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")

    df = pd.read_csv(file_path)
    if "label" not in df.columns:
        raise ValueError("Thiếu cột 'label' trong dataset")

    feature_cols = [c for c in df.columns if c not in ["timestamp", "label", "source_file"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)  # mặc định GOOD -> 1, BAD -> 0

    print(f"Tổng dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
    return X, y, feature_cols, encoder


def build_tiny_mlp(input_dim: int, n_classes: int = 2):
    """Khởi tạo MLP nhỏ gọn cho TinyML."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,), name="input"),
            keras.layers.Dense(32, activation="relu", name="dense_1"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu", name="dense_2"),
            keras.layers.Dense(n_classes, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Tính class weight cân bằng để xử lý dữ liệu lệch GOOD/BAD."""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def save_metrics(
    history: keras.callbacks.History,
    model: keras.Model,
    X_test,
    y_test,
    class_weights: Dict[int, float],
    metrics_path: str,
):
    """Lưu lại metric (loss/acc/precision/recall/auc + confusion matrix) ra file JSON."""
    results = {key: [float(v) for v in values] for key, values in history.history.items()}
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results["test_accuracy"] = float(test_acc)
    results["test_loss"] = float(test_loss)

    proba = model.predict(X_test, verbose=0)
    preds = np.argmax(proba, axis=1)
    report = classification_report(
        y_test,
        preds,
        labels=[0, 1],
        target_names=["BAD", "GOOD"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, preds).tolist()
    results["classification_report"] = report
    results["confusion_matrix"] = matrix
    results["class_weights"] = class_weights

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Đã lưu metrics tại {metrics_path}")


# =============================================================
# Hàm chính
# =============================================================
def train_model(config: Optional[TrainingConfig] = None):
    """Huấn luyện model MLP và lưu lại model + metrics."""
    cfg = config or TrainingConfig()
    if cfg.val_size <= 0 or cfg.test_size <= 0:
        raise ValueError("val_size/test_size phải > 0")
    if (cfg.val_size + cfg.test_size) >= 0.9:
        raise ValueError("val_size + test_size nên < 0.9 để còn dữ liệu train")

    set_global_seed(cfg.random_state)

    X, y, feature_cols, encoder = load_dataset(cfg.input_file)
    print(f"Sử dụng {len(feature_cols)} đặc trưng: {feature_cols}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=(cfg.val_size + cfg.test_size),
        stratify=y,
        random_state=cfg.random_state,
    )
    rel_test_size = cfg.test_size / (cfg.val_size + cfg.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=rel_test_size,
        stratify=y_tmp,
        random_state=cfg.random_state,
    )

    print(f"Tỉ lệ chia: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    class_weights = build_class_weights(y_train)
    print(f"Class weights (train): {class_weights}")

    model = build_tiny_mlp(X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Độ chính xác tập test: {test_acc:.3f} (loss: {test_loss:.3f})")

    os.makedirs(os.path.dirname(cfg.model_out), exist_ok=True)
    model.save(cfg.model_out)
    print(f"Đã lưu model tại {cfg.model_out}")

    joblib.dump({"classes": encoder.classes_, "features": feature_cols}, cfg.label_encoder_out)
    print(f"Đã lưu mapping label tại {cfg.label_encoder_out}")

    save_metrics(history, model, X_test, y_test, class_weights, cfg.metrics_out)

    return {
        "modelPath": cfg.model_out,
        "metricsPath": cfg.metrics_out,
        "labelEncoderPath": cfg.label_encoder_out,
        "trainSamples": len(X_train),
        "valSamples": len(X_val),
        "testSamples": len(X_test),
        "features": feature_cols,
    }


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Huấn luyện TinyML model.")
    parser.add_argument("--input-file", default=INPUT_FILE, help="Đường dẫn CSV đặc trưng")
    parser.add_argument("--model-out", default=MODEL_OUT, help="Đường dẫn lưu model .keras")
    parser.add_argument("--metrics-out", default=METRICS_OUT, help="Đường dẫn lưu metrics.json")
    parser.add_argument(
        "--label-encoder-out",
        default=LABEL_ENCODER_OUT,
        help="Đường dẫn lưu file label_encoder.joblib",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    cfg = TrainingConfig(
        input_file=cli_args.input_file,
        model_out=cli_args.model_out,
        metrics_out=cli_args.metrics_out,
        label_encoder_out=cli_args.label_encoder_out,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        test_size=cli_args.test_size,
        val_size=cli_args.val_size,
        random_state=cli_args.random_state,
    )
    train_model(cfg)
