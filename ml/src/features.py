"""
features.py - Module 4 của pipeline TinyML
------------------------------------------
Chức năng:
    - Đọc dữ liệu từ dataset/interim/aquarium_tinyml_interim.csv
    - Tạo các đặc trưng rolling (mean, std) và delta
    - Bổ sung đặc trưng thời gian (hour_sin/cos, time_since_aerator_on)
    - Xuất file dataset/interim/aquarium_tinyml_features.csv

Dùng:
    python ml/src/features.py
"""

import os
from typing import List
import numpy as np
import pandas as pd

INPUT_FILE = "dataset/interim/aquarium_tinyml_interim.csv"
OUTPUT_FILE = "dataset/interim/aquarium_tinyml_features.csv"

# Cửa sổ rolling tính theo số mẫu (mặc định lấy mẫu mỗi 2 giây)
WINDOW = 15  # ~30 giây nếu tần số lấy mẫu là 2s


def add_rolling_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Tạo rolling mean và std cho các cột được chọn."""
    for col in cols:
        if col in df.columns:
            df[f"{col}_roll_mean"] = df[col].rolling(WINDOW, min_periods=1).mean()
            df[f"{col}_roll_std"] = df[col].rolling(WINDOW, min_periods=1).std().fillna(0)
    return df


def add_delta_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Tính sai phân bậc 1 (delta) để bắt các biến động nhanh."""
    for col in cols:
        if col in df.columns:
            df[f"{col}_delta"] = df[col].diff().fillna(0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung các đặc trưng liên quan đến chu kỳ ngày đêm và trạng thái aerator."""
    if "hour" in df.columns:
        # Mã hoá chu kỳ 24h bằng sin/cos
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "aerator_on" in df.columns:
        # Tính thời gian trong trạng thái "aerator_off" để phát hiện thời điểm thiếu oxy
        time_since = []
        count = 0
        for state in df["aerator_on"]:
            if state == 1:
                count = 0
            else:
                count += 2  # mỗi bước lấy mẫu cách nhau 2 giây
            time_since.append(count)
        df["time_since_aerator_on"] = time_since
        max_val = df["time_since_aerator_on"].max()
        if max_val > 0:
            df["time_since_aerator_on"] = df["time_since_aerator_on"] / max_val

    return df


def generate_features():
    """Thực thi toàn bộ bước tạo đặc trưng."""
    if not os.path.exists(INPUT_FILE):
        print(f"[LỖI] Không tìm thấy file đầu vào: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Đọc {len(df):,} dòng từ {INPUT_FILE}")

    numeric_cols = ["ph", "turbidity_ntu", "orp_mV", "temp_c", "tds_ppm"]

    df = add_rolling_features(df, numeric_cols)
    df = add_delta_features(df, numeric_cols)
    df = add_time_features(df)

    df = df.fillna(0)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Đã tạo xong đặc trưng tại {OUTPUT_FILE}")
    print(f"   Tổng số cột: {len(df.columns)}")


if __name__ == "__main__":
    generate_features()
