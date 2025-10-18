"""
train.py - Module 5 của pipeline TinyML
---------------------------------------
Chức năng:
    - Đọc dữ liệu đã qua bước tạo đặc trưng
    - Chia tập train/validation/test
    - Xây dựng model MLP nhỏ gọn phù hợp TinyML
    - Huấn luyện, đánh giá và lưu model_fp32.keras + metrics.json

Dùng:
    python ml/src/train.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


# =============================================================
# Hàm hỗ trợ
# =============================================================
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
        metrics=["accuracy"],
    )
    return model


def save_metrics(history: keras.callbacks.History, model: keras.Model, X_test, y_test):
    """Lưu lại metric (loss, acc, val_acc, test_acc) ra file JSON."""
    results = {key: [float(v) for v in values] for key, values in history.history.items()}
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results["test_accuracy"] = float(test_acc)
    results["test_loss"] = float(test_loss)

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Đã lưu metrics tại {METRICS_OUT}")


# =============================================================
# Hàm chính
# =============================================================
def train_model():
    """Huấn luyện model MLP và lưu lại model + metrics."""
    X, y, feature_cols, encoder = load_dataset(INPUT_FILE)
    print(f"Sử dụng {len(feature_cols)} đặc trưng: {feature_cols}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(VAL_SIZE + TEST_SIZE), stratify=y, random_state=RANDOM_STATE
    )
    rel_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test_size, stratify=y_tmp, random_state=RANDOM_STATE
    )

    print(f"Tỉ lệ chia: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    model = build_tiny_mlp(X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Độ chính xác tập test: {test_acc:.3f} (loss: {test_loss:.3f})")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"Đã lưu model tại {MODEL_OUT}")

    joblib.dump({"classes": encoder.classes_, "features": feature_cols}, LABEL_ENCODER_OUT)
    print(f"Đã lưu mapping label tại {LABEL_ENCODER_OUT}")

    save_metrics(history, model, X_test, y_test)


if __name__ == "__main__":
    train_model()
