import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, url_for

from database import SessionLocal, TemperatureHumidity, Turbidity, Water, get_vietnam_time

app = Flask(__name__)

# Thư mục lưu export / log cho bundle và kết quả điều chỉnh
BASE_DIR = Path(__file__).resolve().parents[3]
EXPORT_DIR = BASE_DIR / "dataset" / "live_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_JSONL = EXPORT_DIR / "bundle_telemetry.jsonl"
ADJUST_JSONL = EXPORT_DIR / "adjustment_results.jsonl"
LOG_RESULT_DIR = Path(__file__).resolve().parent / "log"
LOG_RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
                    "required": ["pond_id", "model_id", "recommendation"],
                    "properties": {
                        "pond_id": {"type": "string", "example": "pond-01"},
                        "model_id": {"type": "string", "example": "aquarium_v1"},
                        "recommendation": {
                            "type": "object",
                            "description": "Kết quả model sau xử lý, ví dụ hành động bơm/chiếu sáng.",
                            "example": {"pump": "ON", "aeration": "LOW", "confidence": 0.91},
                        },
                        "metrics": {
                            "type": "object",
                            "description": "Thông tin train/validation hoặc threshold đi kèm.",
                            "example": {"f1": 0.88, "loss": 0.12},
                        },
                        "note": {"type": "string", "example": "auto-adjust from ML gateway"},
                    },
                },
                "AdjustmentResponse": {
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
                    "summary": "Nhận kết quả điều chỉnh từ model/ML gateway",
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
                            "description": "Lưu log kết quả điều chỉnh",
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

    session = SessionLocal()
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
    Nhận kết quả/khuyến nghị điều chỉnh từ model sau train hoặc từ ML gateway.
    """
    payload = request.get_json(force=True)
    record = {
        "received_at": get_vietnam_time().isoformat(),
        "pond_id": payload.get("pond_id"),
        "model_id": payload.get("model_id"),
        "recommendation": payload.get("recommendation"),
        "metrics": payload.get("metrics"),
        "note": payload.get("note"),
    }
    append_jsonl(ADJUST_JSONL, record)
    timestamp = get_vietnam_time().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_RESULT_DIR / f"log_result_{timestamp}.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok", "stored": True, "log_path": str(log_path)}), 200


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
    
