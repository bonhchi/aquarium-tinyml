# Dataset: Aquarium TinyML Water Monitoring

## Structure

- `raw/`: dữ liệu gốc thu từ ESP32 (logger) — chưa gán nhãn.
- `labeled/`: dữ liệu đã gán nhãn theo luật yếu (GOOD/BAD).
- `interim/`: dữ liệu làm sạch, chuẩn hóa.
- `splits/`: dữ liệu chia train/validation/test.
- `rep_data.npy`: mẫu dữ liệu đại diện để quantize mô hình.

## Schema

| Column        | Type    | Description                 |
| ------------- | ------- | --------------------------- |
| timestamp     | ISO8601 | thời điểm đo                |
| temp_c        | float   | nhiệt độ nước (°C)          |
| ph            | float   | độ pH                       |
| tds_ppm       | float   | tổng chất rắn hòa tan (ppm) |
| turbidity_ntu | float   | độ đục                      |
| orp_mV        | float   | thế oxy hóa khử (mV)        |
| lux           | float   | cường độ sáng               |
| aerator_on    | int     | trạng thái máy sục khí      |
| pump_on       | int     | trạng thái bơm nước         |
| hour          | int     | giờ trong ngày              |
| is_daylight   | int     | 1 = ban ngày, 0 = ban đêm   |
| label         | string  | GOOD hoặc BAD               |

## Notes

- Dataset được tổng hợp giả lập 6 kịch bản thực tế.
- Tổng số dòng: ~1500, GOOD ≈ 1329, BAD ≈ 171.
- Dữ liệu `rep_data.npy` dùng cho quantization INT8.
