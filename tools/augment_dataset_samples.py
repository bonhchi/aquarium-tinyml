"""Generate and append synthetic sensor samples to dataset CSV files.

This helper simulates 3000 additional readings that follow the same schema
as the existing raw/labeled/interim sample datasets so experiments can run
with a larger corpus.  Newly generated rows are appended to:

* dataset/raw/aquarium_tinyml_dataset_raw_sample.csv
* dataset/labeled/aquarium_tinyml_dataset_labeled_sample.csv
* dataset/interim/aquarium_tinyml_interim_sample.csv

Each synthetic sample shares the same timestamp/value pair across the raw
and labeled files.  The interim dataset stores normalized versions of those
values plus rolling averages for selected features (ph/turbidity/orp).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "dataset/raw/aquarium_tinyml_dataset_raw_sample.csv"
LABELED_FILE = ROOT / "dataset/labeled/aquarium_tinyml_dataset_labeled_sample.csv"
INTERIM_FILE = ROOT / "dataset/interim/aquarium_tinyml_interim_sample.csv"
NUM_NEW_ROWS = 3000
TIMESTAMP_STEP = timedelta(seconds=2)


@dataclass
class SensorSample:
    timestamp: datetime
    temp_c: float
    ph: float
    tds_ppm: float
    turbidity_ntu: float
    orp_mV: float
    lux: float
    aerator_on: int
    pump_on: int
    hour: int
    is_daylight: int
    label: str

    def to_row(self) -> Dict[str, str]:
        return {
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "temp_c": f"{self.temp_c:.2f}",
            "ph": f"{self.ph:.2f}",
            "tds_ppm": f"{self.tds_ppm:.1f}",
            "turbidity_ntu": f"{self.turbidity_ntu:.2f}",
            "orp_mV": f"{self.orp_mV:.1f}",
            "lux": f"{self.lux:.1f}",
            "aerator_on": str(self.aerator_on),
            "pump_on": str(self.pump_on),
            "hour": str(self.hour),
            "is_daylight": str(self.is_daylight),
            "label": self.label,
        }


def _load_last_timestamp(path: Path) -> datetime:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        last_row = None
        for row in reader:
            last_row = row
    if not last_row:
        raise RuntimeError(f"{path} has no data rows")
    return datetime.fromisoformat(last_row["timestamp"])


def _risk_label(temp: float, ph: float, tds: float, turbidity: float, orp: float) -> str:
    risk_points = 0
    if not (26.0 <= temp <= 30.5):
        risk_points += 1
    if not (7.2 <= ph <= 8.1):
        risk_points += 1
    if not (450 <= tds <= 750):
        risk_points += 1
    if turbidity >= 18:
        risk_points += 1
    if not (220 <= orp <= 420):
        risk_points += 1
    return "BAD" if risk_points >= 2 else "GOOD"


def _simulate_sample(ts: datetime) -> SensorSample:
    hour = ts.hour
    is_daylight = 1 if 6 <= hour < 18 else 0
    day_bias = 0.8 if is_daylight else -0.4

    temp = random.gauss(28 + day_bias, 0.9)
    ph = random.gauss(7.7 + day_bias * 0.1, 0.2)
    tds = random.gauss(640 + day_bias * 30, 70)
    turbidity = max(2.5, random.gauss(9 + (1 - is_daylight) * 6, 4))
    orp = random.gauss(320 - (1 - is_daylight) * 40, 55)
    lux = max(0.0, random.gauss(7000 if is_daylight else 120, 1400))
    aerator_on = 1 if random.random() < (0.7 if is_daylight else 0.4) else 0
    pump_on = 1 if random.random() < 0.55 else 0

    label = _risk_label(temp, ph, tds, turbidity, orp)
    return SensorSample(
        timestamp=ts,
        temp_c=temp,
        ph=ph,
        tds_ppm=tds,
        turbidity_ntu=turbidity,
        orp_mV=orp,
        lux=lux,
        aerator_on=aerator_on,
        pump_on=pump_on,
        hour=hour,
        is_daylight=is_daylight,
        label=label,
    )


def _normalize(value: float, min_val: float, max_val: float) -> float:
    norm = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, norm))


def _append_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writerows(rows)


def _append_raw_and_labeled(samples: List[SensorSample]) -> None:
    _, labeled_fields = _read_header(LABELED_FILE)
    _, raw_fields = _read_header(RAW_FILE)
    raw_rows = []
    labeled_rows = []
    for sample in samples:
        row = sample.to_row()
        labeled_rows.append({field: row[field] for field in labeled_fields})
        raw_row = {**row, "label": ""}
        raw_rows.append({field: raw_row.get(field, "") for field in raw_fields})
    _append_rows(LABELED_FILE, labeled_rows)
    _append_rows(RAW_FILE, raw_rows)


def _load_interim_state() -> Dict[str, List[float]]:
    history = {"ph": [], "turbidity_ntu": [], "orp_mV": []}
    if not INTERIM_FILE.exists():
        return history
    with INTERIM_FILE.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                history["ph"].append(float(row["ph"]))
                history["turbidity_ntu"].append(float(row["turbidity_ntu"]))
                history["orp_mV"].append(float(row["orp_mV"]))
            except (ValueError, KeyError):
                continue
    return history


def _moving_average(values: List[float], new_value: float, window: int = 5) -> float:
    buffer = values[-(window - 1) :] if window > 1 else []
    buffer.append(new_value)
    return sum(buffer) / len(buffer)


def _append_interim(samples: List[SensorSample]) -> None:
    ranges = {
        "temp_c": (22.0, 34.0),
        "ph": (6.5, 8.5),
        "tds_ppm": (300.0, 900.0),
        "turbidity_ntu": (1.0, 30.0),
        "orp_mV": (120.0, 500.0),
        "lux": (0.0, 10000.0),
    }
    history = _load_interim_state()
    _, fieldnames = _read_header(INTERIM_FILE)

    rows: List[Dict[str, str]] = []
    for sample in samples:
        temp_norm = _normalize(sample.temp_c, *ranges["temp_c"])
        ph_norm = _normalize(sample.ph, *ranges["ph"])
        tds_norm = _normalize(sample.tds_ppm, *ranges["tds_ppm"])
        ntu_norm = _normalize(sample.turbidity_ntu, *ranges["turbidity_ntu"])
        orp_norm = _normalize(sample.orp_mV, *ranges["orp_mV"])
        lux_norm = _normalize(sample.lux, *ranges["lux"])

        ph_roll = _moving_average(history["ph"], ph_norm)
        ntu_roll = _moving_average(history["turbidity_ntu"], ntu_norm)
        orp_roll = _moving_average(history["orp_mV"], orp_norm)
        history["ph"].append(ph_norm)
        history["turbidity_ntu"].append(ntu_norm)
        history["orp_mV"].append(orp_norm)

        row = {
            "timestamp": sample.timestamp.isoformat(timespec="seconds"),
            "temp_c": f"{temp_norm:.4f}",
            "ph": f"{ph_norm:.4f}",
            "tds_ppm": f"{tds_norm:.4f}",
            "turbidity_ntu": f"{ntu_norm:.4f}",
            "orp_mV": f"{orp_norm:.4f}",
            "lux": f"{lux_norm:.4f}",
            "aerator_on": str(sample.aerator_on),
            "pump_on": str(sample.pump_on),
            "hour": str(sample.hour),
            "is_daylight": str(sample.is_daylight),
            "ph_roll_mean": f"{ph_roll:.4f}",
            "ntu_roll_mean": f"{ntu_roll:.4f}",
            "orp_roll_mean": f"{orp_roll:.4f}",
            "label": sample.label,
        }
        rows.append({key: row[key] for key in fieldnames})
    _append_rows(INTERIM_FILE, rows)


def _read_header(path: Path) -> tuple[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().strip()
    fieldnames = header_line.split(",")
    return header_line, fieldnames


def main() -> None:
    random.seed(42)
    last_ts = _load_last_timestamp(LABELED_FILE)
    samples = []
    for idx in range(1, NUM_NEW_ROWS + 1):
        ts = last_ts + TIMESTAMP_STEP * idx
        samples.append(_simulate_sample(ts))

    _append_raw_and_labeled(samples)
    _append_interim(samples)
    print(f"Appended {NUM_NEW_ROWS} synthetic rows to the dataset files.")


if __name__ == "__main__":
    main()
