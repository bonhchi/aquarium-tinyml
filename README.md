# Aquarium TinyML

Pipeline and gateway to collect pond telemetry, train a TinyML model, and serve it to edge devices.

## Setup
- Python 3.11+ recommended.
- Install dependencies (gateway + ML pipeline): `pip install -r requirements.txt`.
- Optional: copy `.env.example` to `.env` if you need to override gateway paths or ports.

## Training Pipeline (src/ml/train)
1. `python src/ml/train/ingest.py` – merge raw CSVs into `dataset/interim/raw_merged.csv`.
2. `python src/ml/train/weak_label.py` – assign GOOD/BAD labels via thresholding rules.
3. `python src/ml/train/preprocess.py` – clean + normalize + save `ml/artifacts/scaler.joblib`.
4. `python src/ml/train/features.py` – add rolling/time features → `dataset/interim/aquarium_tinyml_features.csv`.
5. `python src/ml/train/train_core.py` – train the Keras model, export metrics/label encoder.
6. `python src/ml/train/quantize.py` – convert to INT8 `.tflite` and test inference.
7. `python src/ml/train/export_header.py` – emit firmware headers from the quantized model/scaler.

Notes:
- `train_core.py` now enforces deterministic seeding and applies class weights to combat the GOOD/BAD imbalance.
- If `dataset/interim/aquarium_tinyml_features.csv` is missing, generate it via steps 1-4 above (a smaller sample also exists at `dataset/interim/aquarium_tinyml_interim_sample.csv` for quick experiments).

## Gateway API (src/gateway)
- Run with: `uvicorn src.gateway.app:app --reload --port 9000`.
- Key endpoints:
  - `POST /ingest/telemetry` – store telemetry JSON (validated via Pydantic).
  - `GET /telemetry` – retrieve raw JSONL rows for analytics/training.
  - `POST /model/publish` / `GET /model/current` – push & fetch model metadata per pond.
  - `GET /dataset/training[?format=csv]` – export the feature CSV slice that fed training.
  - `POST /training/start` – kick off a TinyML training job (accepts dataset/model overrides).
  - `GET /training/jobs` / `/training/jobs/{id}` – follow the progress + fetch log/artifact info.

`src/client/client.py` contains a minimal helper to publish new model artifacts to the gateway.
