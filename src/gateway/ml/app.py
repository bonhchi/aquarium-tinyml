# ai_gateway/app.py
from __future__ import annotations

import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse

if __package__:
    from .config import settings
    from .model_store import load_model_meta, save_model_meta
    from .schemas import (
        GatewaySamplePayload,
        ModelPublishRequest,
        TelemetryPayload,
        TelemetryQuery,
        TrainingDataQuery,
        TrainingJobConfigPayload,
        TrainingJobInfo,
    )
    from .telemetry_store import append_telemetry, query_telemetry
    from .training_data_store import training_data_store
    from .training_jobs import get_job as get_training_job
    from .training_jobs import list_jobs as list_training_jobs
    from .training_jobs import start_training_job
else:
    # Allow running via `python src/gateway/ml/app.py` by appending repo root.
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from src.gateway.ml.config import settings
    from src.gateway.ml.model_store import load_model_meta, save_model_meta
    from src.gateway.ml.schemas import (
        GatewaySamplePayload,
        ModelPublishRequest,
        TelemetryPayload,
        TelemetryQuery,
        TrainingDataQuery,
        TrainingJobConfigPayload,
        TrainingJobInfo,
    )
    from src.gateway.ml.telemetry_store import append_telemetry, query_telemetry
    from src.gateway.ml.training_data_store import training_data_store
    from src.gateway.ml.training_jobs import get_job as get_training_job
    from src.gateway.ml.training_jobs import list_jobs as list_training_jobs
    from src.gateway.ml.training_jobs import start_training_job

tags_metadata = [
    {
        "name": "telemetry",
        "description": "Ingest và truy vấn dữ liệu cảm biến thô được gửi từ các thiết bị/ESP32.",
    },
    {
        "name": "model",
        "description": "Quản lý metadata model TinyML (publish, fetch phiên bản hiện tại).",
    },
    {
        "name": "dataset",
        "description": "Cung cấp dữ liệu đã qua bước tiền xử lý phục vụ training/QA.",
    },
    {
        "name": "system",
        "description": "Các endpoint hỗ trợ vận hành như healthcheck.",
    },
    {
        "name": "training",
        "description": "Khởi động và theo dõi job training TinyML.",
    },
]

app = FastAPI(
    title="Aquarium AI Gateway",
    version="0.2.0",
    description="Gateway phục vụ thu thập telemetry và phân phối TinyML model cho hệ thống nuôi trồng.",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"defaultModelsExpandDepth": 0},
)

GATEWAY_DATA_DIR = settings.GATEWAY_DATA_DIR
GATEWAY_CSV = settings.GATEWAY_CSV
GATEWAY_JSONL = GATEWAY_DATA_DIR / "gateway_samples.jsonl"
GATEWAY_FIELDNAMES = [
    "timestamp",
    "device_id",
    "temperature",
    "humidity",
    "turbidity_raw",
    "turbidity_ntu",
    "water_value",
    "ph",
    "label",
    "dataset_type",
    "source_file",
    "meta_json",
]
GATEWAY_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _parse_timestamp(ts: str | int | None) -> str:
    """Trả về timestamp ISO; dùng now nếu trống hoặc lỗi."""
    if ts is None:
        return datetime.utcnow().isoformat()
    try:
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(float(ts)).isoformat()
        return datetime.fromisoformat(str(ts)).isoformat()
    except Exception:
        return datetime.utcnow().isoformat()


def _append_gateway_csv(row: dict) -> None:
    """Ghi một dòng vào gateway_samples.csv, tự tạo header nếu chưa có."""
    GATEWAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = GATEWAY_CSV.exists()
    with GATEWAY_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GATEWAY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in GATEWAY_FIELDNAMES})


def _append_gateway_jsonl(row: dict) -> None:
    """Log JSONL để debug pipeline ingest."""
    GATEWAY_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with GATEWAY_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.post("/ingest/telemetry", tags=["telemetry"], summary="Nhận bản ghi telemetry")
async def ingest_telemetry(payload: TelemetryPayload):
    """
    Gateway tổng gửi dữ liệu nước vào đây.
    """
    path = append_telemetry(payload.model_dump())
    return {"ok": True, "storedIn": str(path)}


@app.get("/telemetry", tags=["telemetry"], summary="Truy vấn telemetry thô")
async def get_telemetry(filters: TelemetryQuery = Depends()):
    """
    API export telemetry raw để phục vụ dashboard hoặc training.
    """
    rows = query_telemetry(
        site_id=filters.siteId,
        pond_id=filters.pondId,
        start_ts=filters.startTs,
        end_ts=filters.endTs,
        limit=filters.limit,
    )
    return {"rows": rows, "count": len(rows)}


@app.post(
    "/ingest/gateway-sample",
    tags=["telemetry"],
    summary="Nhận gói telemetry tổng từ Gateway Main và lưu CSV cho training",
)
async def ingest_gateway_sample(payload: GatewaySamplePayload = Body(...)):
    """
    Gateway ML nhận bundle 10 phút từ Gateway Main, log JSONL + append CSV `dataset/gateway/gateway_samples.csv`.
    """
    ts_iso = _parse_timestamp(payload.timestamp)
    meta_json = payload.meta or {}
    row = {
        "timestamp": ts_iso,
        "device_id": payload.device_id,
        "temperature": payload.temperature,
        "humidity": payload.humidity,
        "turbidity_raw": payload.turbidity_raw,
        "turbidity_ntu": payload.turbidity_ntu,
        "water_value": payload.water_value,
        "ph": payload.ph,
        "label": payload.label or "",
        "dataset_type": payload.dataset_type or "gateway",
        "source_file": GATEWAY_CSV.name,
        "meta_json": json.dumps(meta_json, ensure_ascii=False),
    }
    _append_gateway_csv(row)
    _append_gateway_jsonl(row)
    return {"ok": True, "csv": str(GATEWAY_CSV), "timestamp": ts_iso}


@app.post("/model/publish", tags=["model"], summary="Publish model mới cho một hồ")
async def publish_model(body: ModelPublishRequest):
    """
    Training đăng model mới cho 1 hồ cụ thể.
    """
    meta = {
        "modelId": body.modelId,
        "pondId": body.pondId,
        "downloadUrl": body.downloadUrl,
        "sha256": body.sha256,
        "input": body.input,
        "preprocess": body.preprocess,
        "thresholds": body.thresholds,
        "notes": body.notes or "",
    }
    save_model_meta(body.pondId, meta)
    return {"ok": True, "pondId": body.pondId, "modelId": body.modelId}


@app.get("/model/current", tags=["model"], summary="Lấy metadata model hiện hành của hồ")
async def get_current_model(pondId: str):
    """
    Gateway tổng gọi để lấy model mới nhất cho 1 hồ.
    """
    meta = load_model_meta(pondId)
    if not meta:
        raise HTTPException(404, f"No model for pond {pondId}")
    return meta


@app.get("/dataset/training", tags=["dataset"], summary="Xuất dữ liệu training features")
async def get_training_dataset(query: TrainingDataQuery = Depends()):
    """
    Xuất dữ liệu feature đã dùng để train (đọc từ CSV).
    Có thể trả JSON hoặc CSV tuỳ tham số format.
    """
    try:
        rows, meta = training_data_store.fetch_rows(
            label=query.label, limit=query.limit, offset=query.offset
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Training data not found: {exc}") from exc

    if query.format == "csv":
        csv_buffer = io.StringIO()
        fieldnames = meta.get("columns") or (list(rows[0].keys()) if rows else [])
        if fieldnames:
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        csv_buffer.seek(0)
        filename = training_data_store.data_file.stem
        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}_slice.csv"'
            },
        )

    return {"rows": rows, "meta": meta}


@app.get("/dataset/training/stats", tags=["dataset"], summary="Thống kê file training đang phục vụ")
async def get_training_dataset_stats():
    """
    Lấy thông tin nhanh về file training đang được gateway phục vụ.
    """
    stats = training_data_store.stats()
    return stats


@app.post(
    "/training/start",
    tags=["training"],
    summary="Kích hoạt job training TinyML mới",
    response_model=TrainingJobInfo,
)
async def start_training_job_endpoint(body: TrainingJobConfigPayload):
    """
    Tạo job training chạy nền, trả về thông tin job để theo dõi qua API.
    """
    job = await start_training_job(body)
    return job


@app.get(
    "/training/jobs",
    tags=["training"],
    summary="Liệt kê các job training gần đây",
    response_model=list[TrainingJobInfo],
)
async def list_training_jobs_endpoint():
    """
    Trả về danh sách các job đã chạy kể từ khi gateway khởi động.
    """
    return await list_training_jobs()


@app.get(
    "/training/jobs/{job_id}",
    tags=["training"],
    summary="Xem chi tiết job training",
    response_model=TrainingJobInfo,
)
async def get_training_job_endpoint(job_id: str):
    """
    Chi tiết job cụ thể gồm trạng thái, log file và đường dẫn artifacts.
    """
    try:
        return await get_training_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"Training job {job_id} not found") from exc


@app.get("/health", tags=["system"], summary="Healthcheck chuẩn")
async def health():
    """
    Healthcheck đơn giản.
    """
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index():
    """
    Redirect root to Swagger docs for convenience.
    """
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    port = int(os.getenv("ML_API_PORT", "5000"))
    import uvicorn

    uvicorn.run("src.gateway.ml.app:app", host="0.0.0.0", port=port, reload=True)
