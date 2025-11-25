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
import csv
import json
import os
import random
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")
try:
    tf.keras.utils.disable_interactive_logging()
except AttributeError:
    pass

# =============================================================
# Cấu hình cơ bản
# =============================================================
INPUT_FILE = "dataset/interim/aquarium_tinyml_features.csv"
MODEL_OUT = "ml/artifacts/model_fp32.keras"
METRICS_OUT = "ml/artifacts/metrics.json"
LABEL_ENCODER_OUT = "ml/artifacts/label_encoder.joblib"
RUN_ARCHIVE_ROOT = Path("ml/artifacts/runs")

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


def build_tiny_mlp(input_dim: int):
    """Khởi tạo MLP nhỏ gọn cho TinyML (dạng binary sigmoid)."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,), name="input"),
            keras.layers.Dense(32, activation="relu", name="dense_1"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu", name="dense_2"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
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
    test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    results["test_accuracy"] = float(
        test_metrics.get("accuracy", test_metrics.get("binary_accuracy", 0.0))
    )
    results["test_loss"] = float(test_metrics.get("loss", 0.0))

    proba = model.predict(X_test, verbose=0).ravel()
    preds = (proba >= 0.5).astype(int)
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
    return results


def _write_history_csv(history: keras.callbacks.History, csv_path: Path) -> None:
    """Xuất lịch sử huấn luyện dạng CSV giống ví dụ tham khảo."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch"] + list(history.history.keys())
    rows = []
    num_epochs = len(history.epoch)
    for idx in range(num_epochs):
        row = [idx + 1]
        for key in history.history.keys():
            values = history.history[key]
            row.append(float(values[idx]) if idx < len(values) else "")
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _write_epoch_history_json(history: keras.callbacks.History, json_path: Path) -> None:
    """Lưu thông số từng epoch ra file JSON để dễ trực quan hóa."""
    if not json_path:
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"epochs": []}
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    max_len = max((len(values) for values in history_dict.values()), default=0)
    for idx in range(max_len):
        entry: Dict[str, Any] = {"epoch": idx + 1}
        for key, values in history_dict.items():
            if idx < len(values):
                entry[key] = values[idx]
        payload["epochs"].append(entry)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def archive_training_run(
    cfg: TrainingConfig,
    history: keras.callbacks.History,
    summary: Dict[str, Any],
    log_path: Optional[str] = None,
    epoch_history_path: Optional[str] = None,
) -> Path:
    """Lưu trữ kết quả mỗi lần chạy vào thư mục timestamp."""
    RUN_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = RUN_ARCHIVE_ROOT / f"{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(exist_ok=False)

    history_payload = {key: [float(v) for v in values] for key, values in history.history.items()}

    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(run_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history_payload, fh, indent=2)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    _write_history_csv(history, run_dir / "history_epoch.csv")

    if log_path and os.path.exists(log_path):
        shutil.copy2(log_path, run_dir / "training.log")

    if epoch_history_path and os.path.exists(epoch_history_path):
        shutil.copy2(epoch_history_path, run_dir / "epoch_history.json")

    for artifact_path in (cfg.model_out, cfg.metrics_out, cfg.label_encoder_out):
        if artifact_path and os.path.exists(artifact_path):
            dest = run_dir / Path(artifact_path).name
            shutil.copy2(artifact_path, dest)

    print(f"Đã lưu snapshot run tại {run_dir}")
    return run_dir


# =============================================================
# Hàm chính
# =============================================================
def train_model(
    config: Optional[TrainingConfig] = None,
    log_path: Optional[str] = None,
    epoch_history_path: Optional[str] = None,
):
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
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=2,
        class_weight=class_weights,
    )
    if epoch_history_path:
        _write_epoch_history_json(history, Path(epoch_history_path))

    eval_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    test_loss = float(eval_metrics.get("loss", 0.0))
    test_acc = float(eval_metrics.get("accuracy", eval_metrics.get("binary_accuracy", 0.0)))
    print(f"Độ chính xác tập test: {test_acc:.3f} (loss: {test_loss:.3f})")

    os.makedirs(os.path.dirname(cfg.model_out), exist_ok=True)
    model.save(cfg.model_out)
    print(f"Đã lưu model tại {cfg.model_out}")

    joblib.dump({"classes": encoder.classes_, "features": feature_cols}, cfg.label_encoder_out)
    print(f"Đã lưu mapping label tại {cfg.label_encoder_out}")

    metrics_payload = save_metrics(
        history, model, X_test, y_test, class_weights, cfg.metrics_out
    )

    summary = {
        "modelPath": cfg.model_out,
        "metricsPath": cfg.metrics_out,
        "labelEncoderPath": cfg.label_encoder_out,
        "trainSamples": len(X_train),
        "valSamples": len(X_val),
        "testSamples": len(X_test),
        "features": feature_cols,
    }
    summary_with_metrics = {**summary, "metrics": metrics_payload}
    run_dir = archive_training_run(
        cfg,
        history,
        summary_with_metrics,
        log_path=log_path,
        epoch_history_path=epoch_history_path,
    )
    summary["runDir"] = str(run_dir)
    return summary


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
