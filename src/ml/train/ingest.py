"""
ingest.py - Module 1 của pipeline TinyML
----------------------------------------
Chức năng:
    - Đọc và hợp nhất tất cả file CSV trong dataset/raw/
    - Kiểm tra schema và chuẩn hoá cột timestamp
    - Xuất file dataset/interim/raw_merged.csv
    - Ghi log số dòng, tên file và cảnh báo khi gặp lỗi

Dùng:
    python ml/src/ingest.py
"""

import glob
import os
import pandas as pd

# === Đường dẫn cơ bản ===
RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/interim"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_merged.csv")

# === Schema bắt buộc phải có ===
REQUIRED_COLUMNS = [
    "timestamp",
    "temp_c",
    "ph",
    "tds_ppm",
    "turbidity_ntu",
    "orp_mV",
    "lux",
    "aerator_on",
    "pump_on",
    "hour",
    "is_daylight",
]


def validate_columns(df: pd.DataFrame, file: str) -> bool:
    """Kiểm tra file CSV có đủ các cột bắt buộc hay không."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[CẢNH BÁO] File {file} thiếu cột: {missing}")
        return False
    return True


def read_and_clean_csv(file_path: str) -> pd.DataFrame:
    """Đọc file CSV, chuẩn hoá timestamp và ép kiểu dữ liệu trạng thái."""
    try:
        df = pd.read_csv(file_path)
        if not validate_columns(df, file_path):
            return pd.DataFrame()

        # Chuẩn hoá timestamp về datetime, loại bỏ giá trị không hợp lệ
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        # Chuyển các cột trạng thái về int để đồng nhất
        for col in ["aerator_on", "pump_on", "is_daylight"]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Sắp xếp theo thời gian và bổ sung tên file nguồn
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["source_file"] = os.path.basename(file_path)
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[LỖI] Không thể đọc file {file_path}: {exc}")
        return pd.DataFrame()


def merge_raw_files():
    """Hợp nhất tất cả file CSV trong dataset/raw/ thành một bảng duy nhất."""
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        print("[CẢNH BÁO] Không tìm thấy file CSV nào trong dataset/raw/.")
        return

    print(f"Đang xử lý {len(files)} file trong {RAW_DIR} ...")
    all_frames: list[pd.DataFrame] = []
    for file_path in files:
        df = read_and_clean_csv(file_path)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("[LỖI] Không có file hợp lệ để hợp nhất.")
        return

    merged = pd.concat(all_frames, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Đã hợp nhất {len(files)} file vào {OUTPUT_FILE}")
    print(f"   Tổng số dòng: {len(merged):,}")


if __name__ == "__main__":
    merge_raw_files()
