"""
weak_label.py - Module 2 của pipeline TinyML
--------------------------------------------
Chức năng:
    - Đọc dữ liệu hợp nhất từ dataset/interim/raw_merged.csv
    - Gán nhãn GOOD/BAD dựa theo ngưỡng tham chiếu
    - Xuất kết quả sang dataset/labeled/aquarium_tinyml_dataset_labeled.csv

Dùng:
    python ml/src/weak_label.py
"""

import os
import numpy as np
import pandas as pd

INPUT_FILE = "dataset/interim/raw_merged.csv"
OUTPUT_FILE = "dataset/labeled/aquarium_tinyml_dataset_labeled.csv"

# ==========================================================
# Ngưỡng tham chiếu cơ bản (tham khảo mô hình nuôi thuỷ sản)
# ==========================================================
THRESHOLDS = {
    "temp_c": (26, 32),       # do C
    "ph": (6.5, 8.5),
    "tds_ppm": (300, 1200),   # ppm
    "turbidity_ntu": (0, 50), # NTU
    "orp_mV": (200, 450),     # mV
    "lux": (300, 20000),      # lux
}


def evaluate_quality(row: pd.Series) -> str:
    """Kiểm tra từng dòng dữ liệu và trả về 'GOOD' hoặc 'BAD'."""
    bad_conditions = []

    for col, (low, high) in THRESHOLDS.items():
        val = row.get(col, np.nan)
        if np.isnan(val) or val < low or val > high:
            bad_conditions.append(col)

    # Tăng thêm quy tắc cho trường hợp dễ nhận biết nước xấu
    if row.get("is_daylight", 1) == 0 and row.get("aerator_on", 1) == 0:
        bad_conditions.append("low_oxygen_night")

    if row.get("turbidity_ntu", 0) > 70 and row.get("orp_mV", 999) < 150:
        bad_conditions.append("polluted_water")

    return "BAD" if bad_conditions else "GOOD"


def generate_labels(input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE):
    """Gán nhãn cho toàn bộ dataset và lưu ra file CSV."""
    if not os.path.exists(input_path):
        print(f"[LỖI] Không tìm thấy file: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Đọc {len(df):,} dòng từ {input_path}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour
        df["is_daylight"] = df["hour"].between(6, 18).astype(int)

    print("Đang gán nhãn GOOD/BAD ...")
    df["label"] = df.apply(evaluate_quality, axis=1)

    counts = df["label"].value_counts().to_dict()
    print(f"Thống kê nhãn: {counts}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Đã lưu labeled dataset tại {output_path}")


if __name__ == "__main__":
    generate_labels()
