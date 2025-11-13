"""
preprocess.py - Module 3 của pipeline TinyML
--------------------------------------------
Chức năng:
    - Đọc dữ liệu đã gán nhãn từ dataset/labeled/aquarium_tinyml_dataset_labeled.csv
    - Làm sạch: loại bỏ NaN và các giá trị ngoài khoảng hợp lý
    - Chuẩn hoá các cột numeric (MinMax)
    - Lưu lại tham số scaler để dùng khi inference
    - Xuất dữ liệu sang dataset/interim/aquarium_tinyml_interim.csv

Dùng:
    python ml/src/preprocess.py
"""

import os
import joblib
import pandas as pd
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

# === Đường dẫn đầu vào/đầu ra ===
INPUT_FILE = "dataset/labeled/aquarium_tinyml_dataset_labeled.csv"
OUTPUT_FILE = "dataset/interim/aquarium_tinyml_interim.csv"
SCALER_PATH = "ml/artifacts/scaler.joblib"

# === Các cột numeric cần chuẩn hoá ===
NUMERIC_COLS = ["temp_c", "ph", "tds_ppm", "turbidity_ntu", "orp_mV", "lux"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dataset: giữ lại dòng hợp lệ và sắp xếp theo timestamp."""
    # Loại bỏ dòng thiếu timestamp hoặc label vì không thể suy ra chất lượng
    df = df.dropna(subset=["timestamp", "label"])

    # Lọc theo ngưỡng cơ bản để tránh dữ liệu ngoại lệ quá lớn
    df = df[
        (df["temp_c"].between(10, 40))
        & (df["ph"].between(4, 10))
        & (df["tds_ppm"].between(0, 5000))
        & (df["turbidity_ntu"].between(0, 100))
        & (df["orp_mV"].between(-500, 500))
        & (df["lux"] >= 0)
    ]

    # Chuẩn hoá timestamp về datetime và loại bỏ dòng lỗi
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Loại bỏ các dòng trùng timestamp để tránh nhân đôi quan sát
    df = df.drop_duplicates(subset=["timestamp"])

    return df.reset_index(drop=True)


def normalize_data(df: pd.DataFrame, scaler: Optional[MinMaxScaler] = None):
    """Chuẩn hoá các cột numeric và trả về scaler đã fit (nếu có)."""
    if scaler is None:
        scaler = MinMaxScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df, scaler


def preprocess():
    """Thực hiện toàn bộ bước làm sạch và chuẩn hoá dữ liệu."""
    if not os.path.exists(INPUT_FILE):
        print(f"[LỖI] Không tìm thấy file đầu vào: {INPUT_FILE}")
        return

    print(f"Đọc dữ liệu từ {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    print("Làm sạch dữ liệu ...")
    df = clean_data(df)
    print(f"   Còn lại {len(df):,} dòng sau khi làm sạch.")

    print("Chuẩn hoá các cột numeric bằng MinMaxScaler ...")
    df, scaler = normalize_data(df)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"   Đã lưu scaler tại {SCALER_PATH}")

    # Bổ sung các đặc trưng thời gian nếu chưa có
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "is_daylight" not in df.columns:
        df["is_daylight"] = df["hour"].between(6, 18).astype(int)

    # Ghi file đầu ra
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Đã lưu dữ liệu chuẩn hoá tại {OUTPUT_FILE}")
    print(f"   Tổng số cột: {len(df.columns)}, số dòng: {len(df):,}")


if __name__ == "__main__":
    preprocess()
