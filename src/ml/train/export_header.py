"""
export_header.py - Module 7 của pipeline TinyML
-----------------------------------------------
Chức năng:
    - Đọc model TFLite sau quantization
    - Chuyển thành mảng byte C (uint8[]) và lưu ra model.h
    - Đọc scaler.joblib để xuất tham số min/max hoặc mean/scale
    - Lưu file header tại firmware/inference/

Dùng:
    python ml/src/export_header.py
"""

import os
import joblib
import numpy as np

TFLITE_MODEL_PATH = "ml/artifacts/model_int8.tflite"
SCALER_PATH = "ml/artifacts/scaler.joblib"
OUTPUT_DIR = "firmware/inference"
MODEL_HEADER = os.path.join(OUTPUT_DIR, "model.h")
SCALER_HEADER = os.path.join(OUTPUT_DIR, "scaler_params.h")


def convert_tflite_to_header(tflite_path: str, header_path: str):
    """Chuyển file .tflite sang mảng byte C và ghi ra header."""
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Không tìm thấy model TFLite: {tflite_path}")

    with open(tflite_path, "rb") as fh:
        data = fh.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    arr_len = len(arr)

    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    with open(header_path, "w", encoding="utf-8") as fh:
        fh.write("// Auto-generated TinyML model header\n")
        fh.write("// Model: aquarium-tinyml INT8\n\n")
        fh.write("#ifndef AQUARIUM_TINYML_MODEL_H\n#define AQUARIUM_TINYML_MODEL_H\n\n")
        fh.write(f"const unsigned int model_len = {arr_len};\n")
        fh.write("const unsigned char model_data[] = {\n")

        for idx in range(0, arr_len, 12):
            chunk = ", ".join(str(byte) for byte in arr[idx : idx + 12])
            fh.write(f"  {chunk},\n")

        fh.write("};\n\n#endif // AQUARIUM_TINYML_MODEL_H\n")

    print(f"Đã tạo header model tại {header_path} ({arr_len:,} bytes)")


def export_scaler_params(scaler_path: str, header_path: str):
    """Xuất tham số scaler (MinMax hoặc Standard) ra file header."""
    if not os.path.exists(scaler_path):
        print(f"[CẢNH BÁO] Không tìm thấy scaler: {scaler_path}")
        return

    scaler = joblib.load(scaler_path)

    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        min_vals = scaler.data_min_.tolist()
        max_vals = scaler.data_max_.tolist()
        n_features = len(min_vals)

        with open(header_path, "w", encoding="utf-8") as fh:
            fh.write("// Auto-generated scaler parameters\n")
            fh.write("#ifndef AQUARIUM_TINYML_SCALER_H\n#define AQUARIUM_TINYML_SCALER_H\n\n")
            fh.write(f"#define N_FEATURES {n_features}\n\n")
            fh.write(
                "const float data_min[N_FEATURES] = { "
                + ", ".join(f"{val:.6f}" for val in min_vals)
                + " };\n"
            )
            fh.write(
                "const float data_max[N_FEATURES] = { "
                + ", ".join(f"{val:.6f}" for val in max_vals)
                + " };\n"
            )
            fh.write("\n#endif // AQUARIUM_TINYML_SCALER_H\n")

        print(f"Đã tạo scaler_params.h tại {header_path}")
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        means = scaler.mean_.tolist()
        scales = scaler.scale_.tolist()
        n_features = len(means)

        with open(header_path, "w", encoding="utf-8") as fh:
            fh.write("// Auto-generated scaler parameters\n")
            fh.write("#ifndef AQUARIUM_TINYML_SCALER_H\n#define AQUARIUM_TINYML_SCALER_H\n\n")
            fh.write(f"#define N_FEATURES {n_features}\n\n")
            fh.write(
                "const float scaler_mean[N_FEATURES] = { "
                + ", ".join(f"{val:.6f}" for val in means)
                + " };\n"
            )
            fh.write(
                "const float scaler_scale[N_FEATURES] = { "
                + ", ".join(f"{val:.6f}" for val in scales)
                + " };\n"
            )
            fh.write("\n#endif // AQUARIUM_TINYML_SCALER_H\n")

        print(f"Đã tạo scaler_params.h tại {header_path}")
    else:
        print("[CẢNH BÁO] Không nhận diện được kiểu scaler hỗ trợ (MinMax hoặc Standard).")


if __name__ == "__main__":
    convert_tflite_to_header(TFLITE_MODEL_PATH, MODEL_HEADER)
    export_scaler_params(SCALER_PATH, SCALER_HEADER)
