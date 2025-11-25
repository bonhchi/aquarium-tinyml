import os

from flask import Flask, request, jsonify, render_template_string, url_for

from database import SessionLocal, Turbidity, TemperatureHumidity, Water, get_vietnam_time

app = Flask(__name__)


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
            }
        },
        "paths": {
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
    
