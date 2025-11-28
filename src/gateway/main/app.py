import csv
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pytz
from flask import Flask, jsonify, render_template_string, request, url_for

from database import (
    Prediction,
    PredictionSessionLocal,
    SessionLocal,
    TemperatureHumidity,
    Turbidity,
    Water,
    get_vietnam_time,
)

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway.main")

# Thư mục lưu export / log cho bundle và kết quả điều chỉnh
BASE_DIR = Path(__file__).resolve().parents[3]
EXPORT_DIR = BASE_DIR / "dataset" / "live_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_JSONL = EXPORT_DIR / "bundle_telemetry.jsonl"
ADJUST_JSONL = EXPORT_DIR / "adjustment_results.jsonl"
LOG_RESULT_DIR = Path(__file__).resolve().parent / "log"
LOG_RESULT_DIR.mkdir(parents=True, exist_ok=True)
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")


def append_jsonl(path: Path, row: dict) -> None:
    """Ghi thêm một dòng JSONL (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ts(value: str | None) -> datetime:
    """Chuyển đổi timestamp ISO nếu có, fallback về VN time hiện tại."""
    if not value:
        return get_vietnam_time()
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return get_vietnam_time()


def _get_latest_bundle() -> dict:
    """Đọc bundle telemetry mới nhất (nếu có)."""
    if not BUNDLE_JSONL.exists():
        return {}
    last_line = ""
    with BUNDLE_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            last_line = line
    try:
        return json.loads(last_line) if last_line else {}
    except Exception:
        return {}


def _parse_request_timestamp(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=VN_TZ)
    if isinstance(value, str):
        text_value = value.strip()
        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text_value)
        except ValueError:
            try:
                dt = datetime.fromtimestamp(float(text_value), tz=VN_TZ)
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=VN_TZ)
        else:
            dt = dt.astimezone(VN_TZ)
        return dt
    return None


def _fetch_snapshot_by_timestamp(ts_value) -> Tuple[Optional[dict], Optional[str]]:
    target_dt = _parse_request_timestamp(ts_value)
    if not target_dt:
        return None, "invalid_timestamp"

    session = PredictionSessionLocal()
    row = None
    try:
        row = (
            session.query(Prediction)
            .filter(Prediction.timestamp <= target_dt)
            .order_by(Prediction.timestamp.desc())
            .first()
        )
        if not row:
            row = (
                session.query(Prediction)
                .filter(Prediction.timestamp >= target_dt)
                .order_by(Prediction.timestamp.asc())
                .first()
            )
    finally:
        session.close()

    if not row:
        return None, "not_found"

    snapshot = row.raw_payload or {}
    snapshot.setdefault("pond_id", row.pond_id)
    snapshot.setdefault("timestamp", row.timestamp.isoformat())
    snapshot.setdefault("model_id", snapshot.get("model_id", "unknown"))
    if "prediction" not in snapshot:
        snapshot["prediction"] = {}
    return snapshot, None


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_sensor_snapshot(bundle: dict | None) -> dict:
    bundle = bundle or {}
    meta = bundle.get("meta") or {}
    return {
        "temperature_c": _safe_float(bundle.get("temperature")),
        "humidity_percent": _safe_float(bundle.get("humidity")),
        "turbidity_ntu": _safe_float(bundle.get("turbidity_ntu", bundle.get("turbidity_raw"))),
        "water_level_cm": _safe_float(bundle.get("water_value", bundle.get("water"))),
        "ph": _safe_float(bundle.get("ph", meta.get("ph"))),
    }


def derive_predicted_adjustments(metrics: dict | None, latest_bundle: dict | None = None) -> dict:
    """Suy luận thông số điều chỉnh gợi ý từ payload metrics + giá trị cảm biến mới nhất."""
    if not metrics:
        metrics = {}
    latest_bundle = latest_bundle or _get_latest_bundle()

    classification = metrics.get("classification_report") or {}
    bad_stats = classification.get("BAD", {})
    good_stats = classification.get("GOOD", {})
    weighted_stats = classification.get("weighted_avg", {})

    accuracy = float(metrics.get("test_accuracy") or classification.get("accuracy") or 0.0)
    bad_recall = float(bad_stats.get("recall") or 0.0)
    bad_precision = float(bad_stats.get("precision") or 0.0)
    good_recall = float(good_stats.get("recall") or 0.0)
    good_precision = float(good_stats.get("precision") or 0.0)
    weighted_f1 = float(weighted_stats.get("f1_score") or 0.0)

    if bad_recall < 0.75:
        pump_mode = "boost"
        pump_targets = [14, 20]
    elif bad_recall > 0.9 and bad_precision > 0.45:
        pump_mode = "relax"
        pump_targets = [10, 16]
    else:
        pump_mode = "maintain"
        pump_targets = [12, 18]

    if good_recall < 0.8:
        aeration_mode = "boost"
        oxygen_target = 6.2
    elif good_recall > 0.9 and good_precision > 0.95:
        aeration_mode = "eco"
        oxygen_target = 5.0
    else:
        aeration_mode = "stabilize"
        oxygen_target = 5.6

    if accuracy >= 0.9:
        sampling_interval = 10
    elif accuracy >= 0.85:
        sampling_interval = 7
    else:
        sampling_interval = 5

    alert_threshold = max(0.1, round(1 - accuracy, 2))

    ph_target = 7.1
    if weighted_f1 < 0.8:
        ph_target = 7.25
    elif weighted_f1 > 0.88:
        ph_target = 7.0

    notes = []
    if pump_mode == "boost":
        notes.append("Bad-class recall thấp, tăng độ nhạy bơm để xử lý mực nước bất thường.")
    if aeration_mode == "boost":
        notes.append("AER nhu cầu cao do recall GOOD thấp.")
    if not notes:
        notes.append("Giữ các tham số vận hành hiện tại, chỉ tinh chỉnh ngưỡng cảnh báo.")

    sensor_targets = {
        "temperature_c": {"range": [24, 30], "note": "Duy trì trong 24-30°C"},
        "humidity_percent": {"range": [60, 85], "note": "Không khí ổn định cho thiết bị"},
        "turbidity_ntu": {"range": [0, 15], "note": "Giữ nước trong, kiểm tra nếu >20 NTU"},
        "water_level_cm": {"range": pump_targets, "note": "Khoảng mực nước mục tiêu bơm tự động"},
        "ph": {"range": [6.8, 7.4], "note": "Giữ pH trung tính; lệch nhiều cần hiệu chỉnh"},
    }

    # Ước đoán cảm biến cần ưu tiên dựa trên giá trị mới nhất (nếu có)
    latest_temp = latest_bundle.get("temperature")
    latest_hum = latest_bundle.get("humidity")
    latest_turbidity = latest_bundle.get("turbidity_ntu") or latest_bundle.get("turbidity_raw")
    latest_water = latest_bundle.get("water_value") or latest_bundle.get("water")
    latest_ph = latest_bundle.get("ph") or (latest_bundle.get("meta") or {}).get("ph")

    def _out_of_range(val, low, high):
        return val is not None and (val < low or val > high)

    sensor_alerts = []
    if _out_of_range(latest_temp, 24, 30):
        sensor_alerts.append({"sensor": "temperature", "current": latest_temp, "action": "kiểm tra sưởi/làm mát"})
    if _out_of_range(latest_hum, 60, 85):
        sensor_alerts.append({"sensor": "humidity", "current": latest_hum, "action": "kiểm tra thông gió"})
    if _out_of_range(latest_turbidity, 0, 20):
        sensor_alerts.append({"sensor": "turbidity", "current": latest_turbidity, "action": "lọc nước/tăng lưu thông"})
    if _out_of_range(latest_water, pump_targets[0], pump_targets[1]):
        sensor_alerts.append({"sensor": "water_level", "current": latest_water, "action": "điều chỉnh bơm"})
    if _out_of_range(latest_ph, 6.8, 7.4):
        sensor_alerts.append({"sensor": "ph", "current": latest_ph, "action": "hiệu chỉnh pH (đệm/bổ sung nước)"})

    # Ước tính thiếu oxy khi nước đục cao hoặc pH lệch, dù chưa có cảm biến DO
    oxygen_risk = "normal"
    oxygen_reason = "Không có cảm biến oxy; suy luận từ độ đục/pH."
    if latest_turbidity is not None and latest_turbidity > 30:
        oxygen_risk = "high"
        oxygen_reason = "Độ đục cao → nguy cơ DO thấp, cần tăng sục khí."
    elif latest_ph is not None and (latest_ph < 6.5 or latest_ph > 8.0):
        oxygen_risk = "medium"
        oxygen_reason = "pH lệch → có thể ảnh hưởng hấp thụ oxy, theo dõi và sục khí."

    return {
        "model_confidence": round(accuracy, 3),
        "pump_adjustment": {
            "mode": pump_mode,
            "target_water_cm": pump_targets,
            "failure_recall": round(bad_recall, 3),
        },
        "aeration_adjustment": {
            "mode": aeration_mode,
            "dissolved_oxygen_target_mg_per_l": oxygen_target,
            "healthy_recall": round(good_recall, 3),
        },
        "sensor_sampling": {
            "interval_minutes": sampling_interval,
            "adaptive": accuracy >= 0.9,
        },
        "alerting": {
            "anomaly_threshold": alert_threshold,
            "expected_bad_events_per_day": round(max(0.5, (1 - good_precision) * 10), 2),
        },
        "ph_target": round(ph_target, 2),
        "sensor_targets": sensor_targets,
        "sensor_alerts": sensor_alerts,
        "oxygen_assessment": {"risk": oxygen_risk, "reason": oxygen_reason},
        "notes": " ".join(notes),
    }


def build_adjustment_snapshot(
    payload: dict,
    predicted: dict,
    sensor_values: dict,
) -> dict:
    """Chuẩn hóa JSON trả về cho gateway/ESP theo mẫu yêu cầu."""
    confidence = predicted.get("model_confidence", 0.0)
    status = "GOOD" if confidence >= 0.85 else "BAD"

    oxygen_risk = (predicted.get("oxygen_assessment") or {}).get("risk", "normal")
    risk_map = {"high": "critical", "medium": "warning", "normal": "normal"}
    risk_level = risk_map.get(oxygen_risk, "normal")
    if predicted.get("sensor_alerts"):
        risk_level = "warning" if risk_level == "normal" else risk_level

    alerts = []
    for alert in predicted.get("sensor_alerts", []):
        sensor = alert.get("sensor", "sensor")
        current = alert.get("current")
        message = alert.get("action", "kiểm tra")
        severity = "warning"
        if sensor == "water_level" and abs((sensor_values.get("water_level_cm") or 0) - 15) > 5:
            severity = "critical"
        alerts.append(
            {
                "type": sensor,
                "message": f"{message} (giá trị hiện tại: {current})",
                "severity": severity,
            }
        )

    snapshot = {
        "pond_id": payload.get("pond_id"),
        "model_id": payload.get("model_id"),
        "timestamp": get_vietnam_time().isoformat(),
        "prediction": {
            "status": status,
            "score": round(confidence, 3),
            "risk_level": risk_level,
        },
        "adjustments": {
            "pump": predicted.get("pump_adjustment", {}),
            "aeration": predicted.get("aeration_adjustment", {}),
            "sampling": predicted.get("sensor_sampling", {}),
            "ph_target": predicted.get("ph_target"),
        },
        "telemetry": sensor_values,
        "alerts": alerts,
    }
    return snapshot


def _read_adjustment_logs(limit: int | None = None) -> list[dict[str, Any]]:
    """Đọc adjustment log từ JSONL, mới nhất đứng trước."""
    if not ADJUST_JSONL.exists():
        return []

    rows: list[dict[str, Any]] = []
    with ADJUST_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows = list(reversed(rows))
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_swagger_spec():
    """Return OpenAPI schema describing the available sensor endpoints."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Aquarium Gateway API",
            "version": "1.0.0",
            "description": "REST endpoints that collect raw telemetry from ESP32 sensors.",
        },
        "servers": [{"url": "/"}],
        "components": {
            "schemas": {
                "TurbidityPayload": {
                    "type": "object",
                    "required": ["device_id", "turbidity_raw", "turbidity_ntu"],
                    "properties": {
                        "device_id": {"type": "string", "example": "esp32-aqua-01"},
                        "turbidity_raw": {"type": "number", "example": 512},
                        "turbidity_ntu": {"type": "number", "example": 4.3},
                    },
                },
                "TemperatureHumidityPayload": {
                    "type": "object",
                    "required": ["device_id", "temperature", "humidity"],
                    "properties": {
                        "device_id": {"type": "string"},
                        "temperature": {"type": "number", "example": 28.5},
                        "humidity": {"type": "number", "example": 76.2},
                    },
                },
                "WaterPayload": {
                    "type": "object",
                    "required": ["device_id", "value"],
                    "properties": {
                        "device_id": {"type": "string"},
                        "value": {"type": "number", "example": 1},
                    },
                },
                "TurbidityRecord": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "device_id": {"type": "string"},
                        "raw": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "TemperatureHumidityRecord": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "device_id": {"type": "string"},
                        "temperature": {"type": "number"},
                        "humidity": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "WaterRecord": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "device_id": {"type": "string"},
                        "value": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "AcknowledgeResponse": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "example": "ok"}},
                },
                "TelemetryBundlePayload": {
                    "type": "object",
                    "required": ["device_id"],
                    "properties": {
                        "device_id": {"type": "string", "example": "gateway-total-01"},
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "example": "2025-01-01T12:00:00",
                        },
                        "temperature": {"type": "number", "example": 28.5},
                        "humidity": {"type": "number", "example": 76.2},
                        "turbidity_raw": {"type": "number", "example": 512},
                        "turbidity_ntu": {"type": "number", "example": 4.3},
                        "water_value": {"type": "number", "example": 1234},
                        "meta": {
                            "type": "object",
                            "additionalProperties": True,
                            "example": {"site_id": "pond-01", "note": "live"},
                        },
                    },
                },
                "TelemetryBundleResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "ok"},
                        "storedRows": {"type": "integer", "example": 3},
                        "bundleLogged": {"type": "boolean", "example": True},
                    },
                },
                "TrainingExportResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "count": {"type": "integer", "example": 120},
                        "csv": {"type": "string", "example": "dataset/live_exports/bundle_export_1700000000.csv"},
                        "minutes": {"type": "integer", "example": 10},
                        "dataset_type": {"type": "string", "example": "train_live"},
                    },
                },
                "AdjustmentPayload": {
                    "type": "object",
                    "required": ["timestamp"],
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "description": "ISO timestamp cần truy vấn snapshot",
                            "example": "2025-11-28T21:52:25.750943+07:00",
                        }
                    },
                },
                "AdjustmentResponse": {
                    "type": "object",
                    "properties": {
                        "pond_id": {"type": "string", "example": "pond-01"},
                        "model_id": {"type": "string", "example": "aquarium_v1"},
                        "timestamp": {"type": "string", "example": "2025-11-28T21:52:25.750943+07:00"},
                        "prediction": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "example": "GOOD"},
                                "score": {"type": "number", "example": 0.893},
                                "risk_level": {"type": "string", "example": "normal"},
                            },
                        },
                        "adjustments": {
                            "type": "object",
                            "properties": {
                                "pump": {
                                    "type": "object",
                                    "properties": {
                                        "mode": {"type": "string", "example": "maintain"},
                                        "target_water_cm": {"type": "array", "items": {"type": "number"}},
                                    },
                                },
                                "aeration": {
                                    "type": "object",
                                    "properties": {
                                        "mode": {"type": "string", "example": "stabilize"},
                                        "dissolved_oxygen_target_mg_per_l": {"type": "number", "example": 5.6},
                                    },
                                },
                                "sampling": {
                                    "type": "object",
                                    "properties": {"interval_minutes": {"type": "number", "example": 7}},
                                },
                                "ph_target": {"type": "number", "example": 7.0},
                            },
                        },
                        "alerts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "example": "water_level"},
                                    "message": {"type": "string", "example": "Mực nước lệch khỏi 12-18 cm"},
                                    "severity": {"type": "string", "example": "warning"},
                                },
                            },
                        },
                    },
                },
                "AdjustmentStoredResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "ok"},
                        "stored": {"type": "boolean", "example": True},
                    },
                },
            }
        },
        "paths": {
            "/api/telemetry/bundle": {
                "post": {
                    "summary": "Submit bundled telemetry (tổng hợp tất cả cảm biến)",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TelemetryBundlePayload"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Accepted & stored",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/TelemetryBundleResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/telemetry/export": {
                "get": {
                    "summary": "Xuất CSV 10 phút gần nhất từ bundle telemetry",
                    "parameters": [
                        {
                            "name": "minutes",
                            "in": "query",
                            "schema": {"type": "integer", "default": 10},
                            "required": False,
                            "description": "Khoảng thời gian (phút) cần lấy dữ liệu",
                        },
                        {
                            "name": "dataset_type",
                            "in": "query",
                            "schema": {"type": "string", "default": "train_live"},
                            "required": False,
                            "description": "Nhãn phân biệt dữ liệu train thực tế / còn lại",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "CSV path và số dòng",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/TrainingExportResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/model/adjustment": {
                "post": {
                    "summary": "Trả snapshot điều chỉnh theo timestamp đã lưu",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AdjustmentPayload"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Snapshot theo timestamp",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AdjustmentResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/turbidity": {
                "post": {
                    "summary": "Submit turbidity reading",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TurbidityPayload"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Accepted",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AcknowledgeResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/temperature_humidity": {
                "post": {
                    "summary": "Submit temperature & humidity reading",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/TemperatureHumidityPayload"
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Accepted",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AcknowledgeResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/water": {
                "post": {
                    "summary": "Submit water level reading",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/WaterPayload"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Accepted",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AcknowledgeResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/turbidity/all": {
                "get": {
                    "summary": "List stored turbidity readings",
                    "responses": {
                        "200": {
                            "description": "All turbidity rows",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/TurbidityRecord"
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/temperature_humidity/all": {
                "get": {
                    "summary": "List stored temperature & humidity readings",
                    "responses": {
                        "200": {
                            "description": "All temperature/humidity rows",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/TemperatureHumidityRecord"
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/water/all": {
                "get": {
                    "summary": "List stored water level readings",
                    "responses": {
                        "200": {
                            "description": "All water rows",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/WaterRecord"},
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
    }


SWAGGER_SPEC = build_swagger_spec()

SWAGGER_UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Aquarium Gateway API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>body { margin: 0; }</style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.onload = () => {
        SwaggerUIBundle({
          url: "{{ spec_url }}",
          dom_id: '#swagger-ui',
        });
      };
    </script>
  </body>
</html>
"""


@app.route("/api/telemetry/bundle", methods=["POST"])
def receive_telemetry_bundle():
    """
    Nhận gói telemetry tổng (temp/hum/turbidity/water) và lưu vào từng bảng + log JSONL.
    """
    payload = request.get_json(force=True)
    ts = parse_ts(payload.get("timestamp"))
    device_id = payload.get("device_id") or "gateway-total"
    dataset_type = payload.get("dataset_type", "live_bundle")

    session = PredictionSessionLocal()
    created_rows = 0
    try:
        if payload.get("turbidity_raw") is not None or payload.get("turbidity_ntu") is not None:
            session.add(
                Turbidity(
                    device_id=device_id,
                    raw=payload.get("turbidity_raw"),
                    ntu=payload.get("turbidity_ntu"),
                    timestamp=ts,
                )
            )
            created_rows += 1

        if payload.get("temperature") is not None or payload.get("humidity") is not None:
            session.add(
                TemperatureHumidity(
                    device_id=device_id,
                    temperature=payload.get("temperature"),
                    humidity=payload.get("humidity"),
                    timestamp=ts,
                )
            )
            created_rows += 1

        water_value = payload.get("water_value")
        if water_value is None:
            water_value = payload.get("water")
        if water_value is not None:
            session.add(
                Water(
                    device_id=device_id,
                    value=water_value,
                    timestamp=ts,
                )
            )
            created_rows += 1

        if created_rows:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    bundle_row = {
        "timestamp": ts.isoformat(),
        "device_id": device_id,
        "temperature": payload.get("temperature"),
        "humidity": payload.get("humidity"),
        "turbidity_raw": payload.get("turbidity_raw"),
        "turbidity_ntu": payload.get("turbidity_ntu"),
        "water_value": water_value,
        "dataset_type": dataset_type,
        "meta": payload.get("meta") or {},
    }
    append_jsonl(BUNDLE_JSONL, bundle_row)

    return jsonify({"status": "ok", "storedRows": created_rows, "bundleLogged": True}), 200


@app.route("/api/telemetry/export")
def export_bundle_csv():
    """
    Xuất CSV từ bundle telemetry trong khoảng N phút (mặc định 10).
    """
    minutes = int(request.args.get("minutes", 10))
    dataset_type = request.args.get("dataset_type", "train_live")
    cutoff = get_vietnam_time() - timedelta(minutes=minutes)

    rows: list[dict] = []
    if BUNDLE_JSONL.exists():
        with BUNDLE_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ts = parse_ts(obj.get("timestamp"))
                    if ts >= cutoff:
                        obj["dataset_type"] = dataset_type
                        rows.append(obj)
                except Exception:
                    continue

    if not rows:
        return jsonify({"status": "empty", "count": 0, "csv": None, "minutes": minutes})

    csv_path = EXPORT_DIR / f"bundle_export_{int(get_vietnam_time().timestamp())}.csv"
    fieldnames = [
        "timestamp",
        "device_id",
        "temperature",
        "humidity",
        "turbidity_raw",
        "turbidity_ntu",
        "water_value",
        "dataset_type",
        "meta",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return jsonify(
        {
            "status": "ok",
            "count": len(rows),
            "csv": str(csv_path.relative_to(BASE_DIR)),
            "minutes": minutes,
            "dataset_type": dataset_type,
        }
    )


@app.route("/api/model/adjustment", methods=["POST"])
def receive_model_adjustment():
    """
    - Nếu payload chỉ gồm timestamp: trả snapshot đã lưu tương ứng.
    - Ngược lại: lưu log điều chỉnh giống trước đây.
    """
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "payload phải là JSON object"}), 400

    if set(payload.keys()) == {"timestamp"}:
        snapshot, error = _fetch_snapshot_by_timestamp(payload.get("timestamp"))
        if error == "invalid_timestamp":
            return jsonify({"status": "error", "message": "timestamp không hợp lệ"}), 400
        if error == "not_found" or snapshot is None:
            return jsonify({"status": "not_found", "message": "Không tìm thấy snapshot cho timestamp đã cho"}), 404
        return jsonify(snapshot), 200

    metrics_payload = payload.get("metrics")
    latest_bundle = _get_latest_bundle()
    sensor_values = _extract_sensor_snapshot(latest_bundle)
    predicted = derive_predicted_adjustments(metrics_payload, latest_bundle=latest_bundle)
    snapshot = build_adjustment_snapshot(payload, predicted, sensor_values)
    logger.info(
        "Model adjustment received for pond=%s model=%s score=%.3f",
        snapshot.get("pond_id"),
        snapshot.get("model_id"),
        (predicted or {}).get("model_confidence", 0.0),
    )
    if not any(sensor_values.values()):
        logger.warning("No recent telemetry found for pond %s; snapshot telemetry empty.", snapshot.get("pond_id"))
    record = {
        "received_at": get_vietnam_time().isoformat(),
        "pond_id": payload.get("pond_id"),
        "model_id": payload.get("model_id"),
        "recommendation": payload.get("recommendation"),
        "metrics": metrics_payload,
        "predicted_adjustments": predicted,
        "snapshot": snapshot,
        "note": payload.get("note"),
    }
    append_jsonl(ADJUST_JSONL, record)
    timestamp = get_vietnam_time().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_RESULT_DIR / f"log_result_{timestamp}.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    # Lưu snapshot xuống bảng predictions
    session = PredictionSessionLocal()
    try:
        prediction_row = Prediction(
            pond_id=snapshot.get("pond_id"),
            timestamp=get_vietnam_time(),
            temperature_c=sensor_values.get("temperature_c"),
            ph=sensor_values.get("ph"),
            turbidity_ntu=sensor_values.get("turbidity_ntu"),
            water_level_cm=sensor_values.get("water_level_cm"),
            humidity_percent=sensor_values.get("humidity_percent"),
            raw_payload=snapshot,
        )
        session.add(prediction_row)
        session.commit()
        logger.info(
            "Saved prediction snapshot id=%s pond=%s temp=%.2f ph=%.2f water=%.2f",
            prediction_row.id,
            prediction_row.pond_id,
            prediction_row.temperature_c or float("nan"),
            prediction_row.ph or float("nan"),
            prediction_row.water_level_cm or float("nan"),
        )
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.warning("Không thể lưu snapshot vào bảng predictions: %s", exc)
    finally:
        session.close()
    return jsonify(snapshot), 200


@app.route("/api/model/adjustment/logs", methods=["GET"])
def list_adjustment_logs():
    """
    Trả về danh sách log adjustment (mặc định 50 dòng mới nhất).
    """
    limit = request.args.get("limit")
    limit_int = int(limit) if limit else 50
    rows = _read_adjustment_logs(limit_int)
    return jsonify({"count": len(rows), "items": rows})


@app.route("/api/model/adjustment/latest", methods=["GET"])
def latest_adjustment_log():
    """
    Trả về bản ghi adjustment mới nhất (nếu có).
    """
    rows = _read_adjustment_logs(1)
    if not rows:
        return jsonify({"status": "empty"}), 200
    return jsonify({"status": "ok", "latest": rows[0]})


@app.route("/api/turbidity", methods=["POST"])
def receive_turbidity():
    json_data = request.get_json(force=True)

    session = SessionLocal()
    turbine = Turbidity(
        device_id=json_data.get("device_id"),
        raw=json_data.get("turbidity_raw"),
        ntu=json_data.get("turbidity_ntu"),
        timestamp=get_vietnam_time()
    )
    session.add(turbine)
    session.commit()
    session.close()

    return jsonify({"status": "ok"}), 200


@app.route("/api/temperature_humidity", methods=["POST"])
def receive_temperature_humidity():
    json_data = request.get_json(force=True)

    session = SessionLocal()
    temperature_humidity = TemperatureHumidity(
        device_id=json_data.get("device_id"),
        temperature=json_data.get("temperature"),
        humidity=json_data.get("humidity"),
        timestamp=get_vietnam_time()
    )
    session.add(temperature_humidity)
    session.commit()
    session.close()

    return jsonify({"status": "ok"}), 200


@app.route("/api/water", methods=["POST"])
def receive_water():
    json_data = request.get_json(force=True)

    session = SessionLocal()
    water = Water(
        device_id=json_data.get("device_id"),
        value=json_data.get("value"),
        timestamp=get_vietnam_time()
    )
    session.add(water)
    session.commit()
    session.close()

    return jsonify({"status": "ok"}), 200


@app.route("/api/turbidity/all")
def get_all_turbidity():
    session = SessionLocal()
    rows = session.query(Turbidity).all()
    session.close()
    return jsonify([
        {
            "id": r.id,
            "device_id": r.device_id,
            "raw": r.raw,
            # "ntu": r.ntu,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in rows
    ])


@app.route("/api/temperature_humidity/all")
def get_all_temperature_humidity():
    session = SessionLocal()
    rows = session.query(TemperatureHumidity).all()
    session.close()
    return jsonify([
        {
            "id": r.id,
            "device_id": r.device_id,
            "temperature": r.temperature,
            "humidity": r.humidity,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in rows
    ])

@app.route("/api/water/all")
def get_all_water():
    session = SessionLocal()
    rows = session.query(Water).all()
    session.close()
    return jsonify([
        {
            "id": r.id,
            "device_id": r.device_id,
            "value": r.value,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in rows
    ])


@app.route("/")
def home():
    return "Server is running!"


@app.route("/swagger.json")
def swagger_json():
    return jsonify(SWAGGER_SPEC)


@app.route("/docs")
def swagger_ui():
    spec_url = url_for("swagger_json", _external=True)
    return render_template_string(SWAGGER_UI_TEMPLATE, spec_url=spec_url)


if __name__ == "__main__":
    port = int(os.getenv("GATEWAY_MAIN_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
    
