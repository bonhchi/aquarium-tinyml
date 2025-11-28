# ai_gateway/schemas.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    field_validator,
)

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

    model_config = ConfigDict(str_strip_whitespace=True, extra="allow")


class GatewaySamplePayload(BaseModel):
    """Payload gói telemetry tổng hợp do Gateway Main gửi sang ML."""

    device_id: str = Field(..., min_length=1, description="ID gateway tổng hợp")
    timestamp: Optional[str] = Field(
        None, description="ISO datetime hoặc unix epoch; nếu trống sẽ dùng now"
    )
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    turbidity_raw: Optional[float] = None
    turbidity_ntu: Optional[float] = None
    water_value: Optional[float] = None
    ph: Optional[float] = None
    label: Optional[str] = Field(None, description="Label nếu có (GOOD/BAD/...)")
    dataset_type: str = Field(
        "gateway",
        description="Ghi chú loại dữ liệu (vd. gateway/live/weak_label) để debug/đối chiếu.",
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Trường tự do thêm info.")

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")


class TelemetryQuery(BaseModel):
    siteId: Optional[str] = Field(None, min_length=1)
    pondId: Optional[str] = Field(None, min_length=1)
    startTs: Optional[int] = Field(None, ge=0, description="Lọc từ timestamp (>=)")
    endTs: Optional[int] = Field(None, ge=0, description="Lọc đến timestamp (<=)")
    limit: int = Field(500, ge=1, le=5000, description="Số record tối đa trả về")

    @field_validator("endTs")
    @classmethod
    def _validate_range(
        cls, end_ts: Optional[int], info: FieldValidationInfo
    ):
        start_ts = info.data.get("startTs")
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

    @field_validator("sha256")
    @classmethod
    def _validate_sha(cls, value: str):
        if not _SHA256_RE.match(value):
            raise ValueError("sha256 must be 64 hex characters")
        return value


class TrainingDataQuery(BaseModel):
    label: Optional[str] = Field(None, description="Lọc label (GOOD/BAD)")
    limit: int = Field(500, ge=1, le=5000)
    offset: int = Field(0, ge=0)
    format: str = Field("json", pattern=r"^(json|csv)$")


class TrainingJobConfigPayload(BaseModel):
    datasetFile: Optional[str] = Field(
        None, description="Đường dẫn CSV features. Mặc định dùng settings.TRAINING_DATA_FILE."
    )
    extraDatasets: Optional[list[str]] = Field(
        None,
        description="Danh sách CSV bổ sung (vd. dataset/gateway/gateway_samples.csv)",
    )
    modelOut: Optional[str] = Field(None, description="Đường dẫn lưu model .keras")
    metricsOut: Optional[str] = Field(None, description="Đường dẫn lưu metrics.json")
    labelEncoderOut: Optional[str] = Field(None, description="Đường dẫn lưu label_encoder.joblib")
    epochs: Optional[int] = Field(None, ge=1, le=500, description="Số epoch train")
    batchSize: Optional[int] = Field(None, ge=1, le=1024)
    valSize: Optional[float] = Field(None, gt=0, lt=1)
    testSize: Optional[float] = Field(None, gt=0, lt=1)
    randomState: Optional[int] = Field(None, description="Seed tái lập kết quả")
    useEarlyStopping: Optional[bool] = Field(
        None, description="Bật/tắt early stopping (mặc định tắt để chạy đủ epoch)"
    )
    earlyStopPatience: Optional[int] = Field(
        None, ge=1, le=300, description="Patience của early stopping nếu bật"
    )
    reduceLrPatience: Optional[int] = Field(
        None, ge=0, le=300, description="Số epoch không cải thiện trước khi giảm LR (0 = tắt)"
    )
    pushAdjustmentUrl: Optional[str] = Field(
        None,
        description="Nếu set, sau khi train sẽ POST metrics về Gateway Main (vd. http://127.0.0.1:5001/api/model/adjustment)",
    )
    pondId: Optional[str] = Field(None, description="Pond ID gửi kèm khi push adjustment")
    modelId: Optional[str] = Field(None, description="Model ID gửi kèm khi push adjustment")
    notes: Optional[str] = Field(None, max_length=500, description="Ghi chú cho job training")

    model_config = ConfigDict(str_strip_whitespace=True)


class TrainingJobInfo(BaseModel):
    jobId: str
    status: Literal["pending", "running", "succeeded", "failed"]
    config: TrainingJobConfigPayload
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None
    logPath: str
    epochHistoryPath: Optional[str] = None
    resultJsonPath: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
