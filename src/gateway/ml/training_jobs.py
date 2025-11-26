from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json
import uuid
import traceback

from src.ml.train.train_core import TrainingConfig, train_model

from .config import settings
from .schemas import TrainingJobConfigPayload, TrainingJobInfo


LOG_DIR = Path("ml/training_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PREFIX = "train_aquarium"


@dataclass
class TrainingJobRecord:
    job_id: str
    config: TrainingJobConfigPayload
    status: str = "pending"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    log_path: Path = field(default_factory=lambda: LOG_DIR / f"{uuid.uuid4().hex}.log")
    result_path: Path = field(default_factory=lambda: LOG_DIR / f"{uuid.uuid4().hex}.json")
    epoch_history_path: Path = field(default_factory=lambda: LOG_DIR / f"{uuid.uuid4().hex}_epochs.json")
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


_JOBS: Dict[str, TrainingJobRecord] = {}
_LOCK = asyncio.Lock()


def _merge_training_config(body: TrainingJobConfigPayload) -> TrainingConfig:
    cfg = TrainingConfig()
    data = body.model_dump(exclude_unset=True)

    cfg.input_file = data.get("datasetFile") or str(settings.TRAINING_DATA_FILE)
    if data.get("extraDatasets"):
        cfg.extra_datasets = data["extraDatasets"]
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
    if data.get("pushAdjustmentUrl"):
        cfg.push_adjustment_url = data["pushAdjustmentUrl"]
    if data.get("pondId"):
        cfg.pond_id = data["pondId"]
    if data.get("modelId"):
        cfg.model_id = data["modelId"]
    return cfg


def _allocate_log_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_name = f"{LOG_FILE_PREFIX}_{stamp}"
    log_path = LOG_DIR / f"{base_name}.log"
    result_path = LOG_DIR / f"{base_name}.json"
    epoch_path = LOG_DIR / f"{base_name}_epochs.json"
    counter = 1
    while log_path.exists() or result_path.exists() or epoch_path.exists():
        suffix = f"{base_name}_{counter}"
        log_path = LOG_DIR / f"{suffix}.log"
        result_path = LOG_DIR / f"{suffix}.json"
        epoch_path = LOG_DIR / f"{suffix}_epochs.json"
        counter += 1
    return log_path, result_path, epoch_path


async def start_training_job(config_payload: TrainingJobConfigPayload) -> TrainingJobInfo:
    job_id = uuid.uuid4().hex
    log_path, result_path, epoch_path = _allocate_log_paths()
    record = TrainingJobRecord(
        job_id=job_id,
        config=config_payload,
        log_path=log_path,
        result_path=result_path,
        epoch_history_path=epoch_path,
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
        with open(record.log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                f"# Training job {record.job_id}\nStarted: {record.started_at.isoformat()}Z\n"
            )
            log_file.write(f"Config: {cfg.__dict__}\n\n")
            log_file.flush()
            try:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    result = train_model(
                        cfg,
                        log_path=str(record.log_path),
                        epoch_history_path=str(record.epoch_history_path),
                    )
            except Exception:  # pylint: disable=broad-except
                log_file.write("\n[ERROR] Training crashed:\n")
                traceback.print_exc(file=log_file)
                log_file.flush()
                raise
            else:
                log_file.write("\n[INFO] Training finished successfully.\n")
                log_file.flush()
                return result

    try:
        record.result = await asyncio.to_thread(_runner)
    except Exception as exc:  # pylint: disable=broad-except
        record.error = str(exc)
        record.status = "failed"
    else:
        record.status = "succeeded"
    finally:
        record.finished_at = datetime.utcnow()
        _persist_result_json(record)


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return f"{value.isoformat()}Z"


def _persist_result_json(record: TrainingJobRecord) -> None:
    if not record.result_path:
        return
    payload = {
        "jobId": record.job_id,
        "status": record.status,
        "startedAt": _serialize_datetime(record.started_at),
        "finishedAt": _serialize_datetime(record.finished_at),
        "logPath": str(record.log_path),
        "epochHistoryPath": str(record.epoch_history_path),
        "result": record.result,
        "error": record.error,
        "notes": record.notes,
        "config": record.config.model_dump(exclude_none=True),
    }
    try:
        with open(record.result_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:  # pragma: no cover - unlikely path
        print(f"[WARN] Không thể ghi file kết quả training {record.result_path}: {exc}")


def _record_to_schema(record: TrainingJobRecord) -> TrainingJobInfo:
    return TrainingJobInfo(
        jobId=record.job_id,
        status=record.status,
        config=record.config,
        startedAt=record.started_at,
        finishedAt=record.finished_at,
        logPath=str(record.log_path),
        epochHistoryPath=str(record.epoch_history_path),
        resultJsonPath=str(record.result_path),
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
