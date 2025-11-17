from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from src.ml.train.train_core import TrainingConfig, train_model

from .config import settings
from .schemas import TrainingJobConfigPayload, TrainingJobInfo


LOG_DIR = Path("ml/training_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingJobRecord:
    job_id: str
    config: TrainingJobConfigPayload
    status: str = "pending"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    log_path: Path = field(default_factory=lambda: LOG_DIR / f"{uuid.uuid4().hex}.log")
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


_JOBS: Dict[str, TrainingJobRecord] = {}
_LOCK = asyncio.Lock()


def _merge_training_config(body: TrainingJobConfigPayload) -> TrainingConfig:
    cfg = TrainingConfig()
    data = body.model_dump(exclude_unset=True)

    cfg.input_file = data.get("datasetFile") or str(settings.TRAINING_DATA_FILE)
    if data.get("modelOut"):
        cfg.model_out = data["modelOut"]
    if data.get("metricsOut"):
        cfg.metrics_out = data["metricsOut"]
    if data.get("labelEncoderOut"):
        cfg.label_encoder_out = data["labelEncoderOut"]
    if data.get("epochs") is not None:
        cfg.epochs = data["epochs"]
    if data.get("batchSize") is not None:
        cfg.batch_size = data["batchSize"]
    if data.get("valSize") is not None:
        cfg.val_size = data["valSize"]
    if data.get("testSize") is not None:
        cfg.test_size = data["testSize"]
    if data.get("randomState") is not None:
        cfg.random_state = data["randomState"]
    return cfg


async def start_training_job(config_payload: TrainingJobConfigPayload) -> TrainingJobInfo:
    job_id = uuid.uuid4().hex
    record = TrainingJobRecord(
        job_id=job_id,
        config=config_payload,
        log_path=LOG_DIR / f"{job_id}.log",
        notes=config_payload.notes,
    )
    async with _LOCK:
        _JOBS[job_id] = record

    asyncio.create_task(_run_job(record))
    return _record_to_schema(record)


async def _run_job(record: TrainingJobRecord) -> None:
    record.status = "running"
    record.started_at = datetime.utcnow()
    cfg = _merge_training_config(record.config)

    def _runner():
        with open(record.log_path, "w", encoding="utf-8") as log_file, redirect_stdout(
            log_file
        ), redirect_stderr(log_file):
            return train_model(cfg)

    try:
        record.result = await asyncio.to_thread(_runner)
    except Exception as exc:  # pylint: disable=broad-except
        record.error = str(exc)
        record.status = "failed"
    else:
        record.status = "succeeded"
    finally:
        record.finished_at = datetime.utcnow()


def _record_to_schema(record: TrainingJobRecord) -> TrainingJobInfo:
    return TrainingJobInfo(
        jobId=record.job_id,
        status=record.status,
        config=record.config,
        startedAt=record.started_at,
        finishedAt=record.finished_at,
        logPath=str(record.log_path),
        result=record.result,
        error=record.error,
    )


async def list_jobs() -> list[TrainingJobInfo]:
    async with _LOCK:
        return [_record_to_schema(job) for job in _JOBS.values()]


async def get_job(job_id: str) -> TrainingJobInfo:
    async with _LOCK:
        record = _JOBS.get(job_id)
    if not record:
        raise KeyError(job_id)
    return _record_to_schema(record)
