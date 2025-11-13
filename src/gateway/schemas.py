# ai_gateway/schemas.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator

_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


class TelemetryPayload(BaseModel):
    """
    Envelope cho bản ghi telemetry.
    Cho phép kèm thêm trường động (sensor readings) qua extra fields.
    """

    siteId: str = Field(..., min_length=1, description="Mã farm hoặc địa điểm")
    pondId: str = Field(..., min_length=1, description="Mã hồ nuôi")
    timestamp: Optional[int] = Field(
        None, ge=0, description="Unix time (UTC). Nếu bỏ trống sẽ tự động gán"
    )

    class Config:
        anystr_strip_whitespace = True
        extra = "allow"


class TelemetryQuery(BaseModel):
    siteId: Optional[str] = Field(None, min_length=1)
    pondId: Optional[str] = Field(None, min_length=1)
    startTs: Optional[int] = Field(None, ge=0, description="Lọc từ timestamp (>=)")
    endTs: Optional[int] = Field(None, ge=0, description="Lọc đến timestamp (<=)")
    limit: int = Field(500, ge=1, le=5000, description="Số record tối đa trả về")

    @validator("endTs")
    def _validate_range(cls, end_ts: Optional[int], values: Dict[str, Any]):  # type: ignore[override]
        start_ts = values.get("startTs")
        if end_ts is not None and start_ts is not None and end_ts < start_ts:
            raise ValueError("endTs must be >= startTs")
        return end_ts


class ModelPublishRequest(BaseModel):
    modelId: str = Field(..., min_length=1)
    pondId: str = Field(..., min_length=1)
    downloadUrl: str = Field(..., min_length=1)
    sha256: str = Field(..., description="Checksum 64 hex chars")
    input: Dict[str, Any]
    preprocess: Dict[str, Any]
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = ""

    @validator("sha256")
    def _validate_sha(cls, value: str):  # type: ignore[override]
        if not _SHA256_RE.match(value):
            raise ValueError("sha256 must be 64 hex characters")
        return value


class TrainingDataQuery(BaseModel):
    label: Optional[str] = Field(None, description="Lọc label (GOOD/BAD)")
    limit: int = Field(500, ge=1, le=5000)
    offset: int = Field(0, ge=0)
    format: str = Field("json", regex=r"^(json|csv)$")

