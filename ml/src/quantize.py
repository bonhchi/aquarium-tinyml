"""
quantize.py - Module 6 của pipeline TinyML
------------------------------------------
Chức năng:
    - Load model Keras đã huấn luyện
    - Load tập dữ liệu đại diện (representative dataset) cho quantization
    - Chuyển model sang định dạng TFLite (INT8 hoặc Float16)
    - Lưu model TFLite và kiểm tra nhanh việc suy diễn

Dùng:
    python ml/src/quantize.py
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

MODEL_PATH = "ml/artifacts/model_fp32.keras"
REP_DATA_PATH = "dataset/rep_data.npy"
OUTPUT_TFLITE = "ml/artifacts/model_int8.tflite"


def load_representative_dataset():
    """Load tập dữ liệu đại diện dùng cho bước quantization INT8."""
    if os.path.exists(REP_DATA_PATH):
        print(f"Loading representative dataset từ {REP_DATA_PATH}")
        data = np.load(REP_DATA_PATH)
    else:
        csv_path = "dataset/splits/train.csv"
        print(f"[CẢNH BÁO] Không tìm thấy {REP_DATA_PATH}, chuyển sang {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Không có dữ liệu đại diện. Vui lòng tạo dataset/splits/train.csv hoặc rep_data.npy"
            )

        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ["timestamp", "label"]]
        data = df[feature_cols].values.astype(np.float32)
        np.save(REP_DATA_PATH, data[:500])
        print(f"Đã lưu tập đại diện mới tại {REP_DATA_PATH}")
        data = data[:500]

    def representative_dataset():
        for i in range(0, len(data)):
            yield [np.expand_dims(data[i], axis=0)]

    return representative_dataset


def quantize_model(model_path: str = MODEL_PATH, output_path: str = OUTPUT_TFLITE, mode: str = "int8"):
    """Chuyển model Keras sang định dạng .tflite theo mode được chọn."""
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy model: {model_path}")
        return

    print(f"Loading model từ {model_path}")
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == "int8":
        print("Quantization INT8 full integer")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = load_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif mode == "float16":
        print("Quantization Float16")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        print("Không quantize, giữ model float32 gốc")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Đã lưu model TFLite ({mode}) tại {output_path}")
    print(f"   Kích thước: {size_kb:.2f} KB")


def test_tflite_inference(tflite_path: str = OUTPUT_TFLITE):
    """Chạy thử inference trên model TFLite để đảm bảo file hợp lệ."""
    if not os.path.exists(tflite_path):
        print(f"[LỖI] Không tìm thấy model: {tflite_path}")
        return

    print(f"Test nhanh TFLite tại {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dtype = input_details[0]["dtype"]
    dummy_input = np.zeros(input_details[0]["shape"], dtype=dtype)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print(f"Inference thành công, output shape: {output_data.shape}")


if __name__ == "__main__":
    quantize_model(mode="int8")
    test_tflite_inference()
