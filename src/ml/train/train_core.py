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
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

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

EPOCHS = 125  # theo mẫu trong RUN_PROJECT.txt
BATCH_SIZE = 64
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
USE_EARLY_STOPPING = False
EARLY_STOP_PATIENCE = 40
REDUCE_LR_PATIENCE = 10


@dataclass
class TrainingConfig:
    """Tập hợp các tham số để job training có thể tuỳ biến."""

    input_file: str = INPUT_FILE
    extra_datasets: List[str] = field(default_factory=list)
    model_out: str = MODEL_OUT
    metrics_out: str = METRICS_OUT
    label_encoder_out: str = LABEL_ENCODER_OUT
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    test_size: float = TEST_SIZE
    val_size: float = VAL_SIZE
    random_state: int = RANDOM_STATE
    use_early_stopping: bool = USE_EARLY_STOPPING
    early_stop_patience: int = EARLY_STOP_PATIENCE
    reduce_lr_patience: int = REDUCE_LR_PATIENCE
    push_adjustment_url: Optional[str] = None
    pond_id: Optional[str] = None
    model_id: Optional[str] = None


# =============================================================
# Hàm hỗ trợ
# =============================================================
def set_global_seed(seed: int) -> None:
    """Cố định seed cho Python/NumPy/TensorFlow để dễ tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(file_path: str = INPUT_FILE, extra_paths: Optional[List[str]] = None):
    """Đọc và ghép dataset chính + danh sách extra, trả về X, y, feature_cols, encoder.

    - extra_paths: danh sách CSV bổ sung (vd. dataset/gateway/gateway_samples.csv).
    - Các cột không có trong file chính sẽ bị bỏ, cột thiếu sẽ được fill NA.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")

    df = pd.read_csv(file_path)
    base_columns = list(df.columns)
    if "label" not in df.columns:
        raise ValueError("Thiếu cột 'label' trong dataset chính")

    frames = [df]
    extra_paths = extra_paths or []
    for path in extra_paths:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"[warn] Bỏ qua extra dataset (không tồn tại): {path}")
            continue
        extra_df = pd.read_csv(path)
        # điền cột thiếu để khớp schema file chính
        for col in base_columns:
            if col not in extra_df.columns:
                extra_df[col] = np.nan
        extra_df = extra_df[base_columns]
        frames.append(extra_df)
        print(f"[debug] Nối thêm {len(extra_df)} dòng từ {path}")

    df_all = pd.concat(frames, ignore_index=True)
    # Bỏ các dòng chưa có label (gateway thu thập nhưng chưa gán nhãn)
    before_drop = len(df_all)
    df_all = df_all[df_all["label"].notna() & (df_all["label"].astype(str).str.len() > 0)]
    dropped = before_drop - len(df_all)
    if dropped:
        print(f"[debug] Đã bỏ {dropped} dòng chưa có label trước khi train")

    # Danh sách cột đặc trưng: bỏ timestamp/label/source/meta
    exclude_cols = {"timestamp", "label", "source_file", "dataset_type", "meta_json"}
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]
    # Ép numeric an toàn cho các cột feature
    for col in feature_cols:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all[feature_cols] = df_all[feature_cols].fillna(0.0)

    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)  # mặc định GOOD -> 1, BAD -> 0

    print(f"Tổng dữ liệu sau ghép: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
    return X, y, feature_cols, encoder


def build_tiny_mlp(input_dim: int, normalization_data: Optional[np.ndarray] = None):
    """Khởi tạo MLP nhỏ gọn có chuẩn hoá đầu vào để học ổn định hơn."""
    inputs = keras.layers.Input(shape=(input_dim,), name="input")
    x = inputs

    if normalization_data is not None:
        norm_layer = keras.layers.Normalization(name="feature_norm")
        norm_layer.adapt(normalization_data)
        x = norm_layer(x)

    x = keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="dropout_1")(x)

    x = keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_2",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_2")(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)

    x = keras.layers.Dense(16, activation="relu", name="dense_3")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="aquarium_tinyml_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
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


def _normalize_report_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Đổi key dict về dạng snake_case (thay space/hyphen bằng underscore)."""

    def convert_key(key: Any) -> Any:
        if isinstance(key, str):
            return key.replace(" ", "_").replace("-", "_")
        return key

    def recurse(value: Any) -> Any:
        if isinstance(value, dict):
            return {convert_key(k): recurse(v) for k, v in value.items()}
        if isinstance(value, list):
            return [recurse(v) for v in value]
        return value

    return recurse(payload)


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
    report = _normalize_report_keys(report)
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


class EpochTimingCallback(keras.callbacks.Callback):
    """Theo dõi thời gian huấn luyện mỗi epoch để lưu vào JSON."""

    def __init__(self) -> None:
        super().__init__()
        self._epoch_start: Optional[float] = None
        self.durations: List[float] = []

    def on_epoch_begin(self, epoch, logs=None):  # type: ignore[override]
        del logs
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
        del logs
        if self._epoch_start is not None:
            self.durations.append(time.perf_counter() - self._epoch_start)
            self._epoch_start = None


def _write_epoch_history_json(
    history: keras.callbacks.History,
    json_path: Path,
    epoch_durations: Optional[List[float]] = None,
) -> None:
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
        if epoch_durations and idx < len(epoch_durations):
            entry["epoch_duration"] = float(epoch_durations[idx])
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


def _push_adjustment(cfg: TrainingConfig, metrics_payload: Dict[str, Any]) -> None:
    """Đẩy kết quả train sang Gateway Main nếu có cấu hình URL."""
    if not cfg.push_adjustment_url:
        return
    try:
        import requests  # type: ignore
    except Exception:
        print("[warn] Không tìm thấy thư viện requests, bỏ qua push adjustment")
        return

    body = {
        "pond_id": cfg.pond_id or "pond-unknown",
        "model_id": cfg.model_id or Path(cfg.model_out).stem,
        "recommendation": {
            "status": "trained",
            "metrics_file": cfg.metrics_out,
            "model_file": cfg.model_out,
        },
        "metrics": metrics_payload,
        "note": "auto-push from train_core",
    }
    try:
        resp = requests.post(cfg.push_adjustment_url, json=body, timeout=5)
        resp.raise_for_status()
        print(f"[debug] Đã gửi kết quả train tới {cfg.push_adjustment_url}: {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Push adjustment thất bại: {exc}")


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

    X, y, feature_cols, encoder = load_dataset(cfg.input_file, cfg.extra_datasets)
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

    model = build_tiny_mlp(X_train.shape[1], normalization_data=X_train)
    timing_callback = EpochTimingCallback()
    callbacks: List[keras.callbacks.Callback] = [timing_callback]

    if cfg.reduce_lr_patience and cfg.reduce_lr_patience > 0:
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.reduce_lr_patience,
            min_lr=1e-5,
            verbose=1,
        )
        callbacks.append(lr_scheduler)

    if cfg.use_early_stopping:
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=max(1, cfg.early_stop_patience),
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-3,
        )
        callbacks.append(early_stop)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=2,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    if epoch_history_path:
        _write_epoch_history_json(
            history,
            Path(epoch_history_path),
            epoch_durations=timing_callback.durations,
        )

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
    _push_adjustment(cfg, metrics_payload)
    return summary


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Huấn luyện TinyML model.")
    parser.add_argument("--input-file", default=INPUT_FILE, help="Đường dẫn CSV đặc trưng")
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        help="CSV bổ sung (vd. dataset/gateway/gateway_samples.csv), có thể truyền nhiều lần",
    )
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
    parser.add_argument(
        "--enable-early-stopping",
        action="store_true",
        help="Bật early stopping (mặc định tắt để train đủ epoch).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="Số epoch không cải thiện trước khi early stopping kích hoạt.",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=REDUCE_LR_PATIENCE,
        help="Số epoch không cải thiện trước khi giảm learning rate (0 để tắt).",
    )
    parser.add_argument(
        "--push-adjustment-url",
        default=os.getenv("PUSH_ADJUSTMENT_URL", ""),
        help="Nếu set, sau khi train sẽ POST metrics về Gateway Main (vd. http://127.0.0.1:5001/api/model/adjustment)",
    )
    parser.add_argument("--pond-id", default=os.getenv("POND_ID", ""), help="Pond ID cho payload adjustment")
    parser.add_argument(
        "--model-id",
        default=os.getenv("MODEL_ID", ""),
        help="Model ID cho payload adjustment (mặc định lấy tên file model_out)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    cfg = TrainingConfig(
        input_file=cli_args.input_file,
        extra_datasets=cli_args.extra_dataset,
        model_out=cli_args.model_out,
        metrics_out=cli_args.metrics_out,
        label_encoder_out=cli_args.label_encoder_out,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        test_size=cli_args.test_size,
        val_size=cli_args.val_size,
        random_state=cli_args.random_state,
        use_early_stopping=cli_args.enable_early_stopping,
        early_stop_patience=cli_args.early_stop_patience,
        reduce_lr_patience=cli_args.reduce_lr_patience,
        push_adjustment_url=cli_args.push_adjustment_url or None,
        pond_id=cli_args.pond_id or None,
        model_id=cli_args.model_id or None,
    )
    train_model(cfg)
