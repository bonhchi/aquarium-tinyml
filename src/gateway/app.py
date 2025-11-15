# ai_gateway/app.py
from __future__ import annotations

import csv
import io
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .model_store import load_model_meta, save_model_meta
from .schemas import (
    ModelPublishRequest,
    TelemetryPayload,
    TelemetryQuery,
    TrainingDataQuery,
)
from .telemetry_store import append_telemetry, query_telemetry
from .training_data_store import training_data_store

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
]

app = FastAPI(
    title="Aquarium AI Gateway",
    version="0.2.0",
    description="Gateway phục vụ thu thập telemetry và phân phối TinyML model cho hệ thống nuôi trồng.",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"defaultModelsExpandDepth": 0},
)


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


@app.get("/health", tags=["system"], summary="Healthcheck chuẩn")
async def health():
    """
    Healthcheck đơn giản.
    """
    return {"status": "ok"}
