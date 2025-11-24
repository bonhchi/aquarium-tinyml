from flask import Flask, request, jsonify
from database import SessionLocal, Turbidity, TemperatureHumidity, Water, get_vietnam_time

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    