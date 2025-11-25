# Aquarium TinyML

Pipeline and gateway to collect pond telemetry, train a TinyML model, and serve it to edge devices.

## Setup

### Shared prerequisites
- Python 3.11+ (3.12 works too).
- Optional: copy `.env.example` to `.env` if you need to override gateway paths or ports.
- The helper script `python install_dependencies.py --file <requirements-file>` works on both OSes.
- `requirements.txt` uses environment markers so one command works everywhere, but `requirements-macos.txt` and `requirements-windows.txt` remain available if you prefer explicit files per OS.

### macOS (zsh/bash)
1. Create a virtualenv: `python3.11 -m venv .venv`.
2. Activate it: `source .venv/bin/activate`.
3. Upgrade pip and install deps: `python -m pip install --upgrade pip` then `pip install -r requirements-macos.txt` (or `requirements.txt` if you want one file for every OS).
4. Run the gateway: `uvicorn src.gateway.app:app --reload --port 9000`.
5. Open Swagger UI at `http://127.0.0.1:9000/docs` (Redoc at `/redoc`).

### Windows (PowerShell)
1. Create a virtualenv: `py -3.11 -m venv .venv`.
2. Activate it: PowerShell → `.\.venv\Scripts\Activate.ps1` (run `Set-ExecutionPolicy -Scope Process RemoteSigned` if blocked); CMD → `.\.venv\Scripts\activate.bat`.
3. Upgrade pip + install deps: `python -m pip install --upgrade pip` then `pip install -r requirements-windows.txt` (or stick with the unified `requirements.txt`).
4. Run uvicorn (module form avoids PATH issues): `python -m uvicorn src.gateway.app:app --reload --port 9000`.
5. Browse to `http://127.0.0.1:9000/docs` for Swagger (root `/` returns `{"detail":"Not Found"}` by design).

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

### Flask Telemetry API (src/gateway/flask_api)
- Lightweight Flask endpoints (`app.py`) push turbidity, temperature/humidity, and water depth rows into MySQL via SQLAlchemy.
- Shares the same dependency list as the FastAPI gateway: `python -m pip install -r src/gateway/requirements.txt` then `python src/gateway/flask_api/app.py`.
- Docker/Compose helpers now live alongside the service: `cd src/gateway/flask_api` then `docker-compose up --build` (build context points back to the repo root so the shared requirements file is copied in automatically).
