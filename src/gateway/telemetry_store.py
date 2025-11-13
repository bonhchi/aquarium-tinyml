# ai_gateway/telemetry_store.py
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import settings

RAW_DIR: Path = settings.DATA_ROOT / "raw"


class TelemetryStore:
    """
    Đảm nhiệm lưu & đọc raw telemetry dưới dạng JSONL để phục vụ ingest + training.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def append(self, payload: Dict[str, Any]) -> Path:
        """
        Ghi 1 bản ghi telemetry vào file JSONL theo ngày:
        dataset/raw/{siteId}_{pondId}_{YYYYMMDD}.jsonl
        """
        site = payload.get("siteId", "unknown")
        pond = payload.get("pondId", "unknown")

        ts = payload.get("timestamp")
        if ts is None:
            ts = int(datetime.utcnow().timestamp())
            payload["timestamp"] = ts

        day = datetime.utcfromtimestamp(ts).strftime("%Y%m%d")
        file_path = self.raw_dir / f"{site}_{pond}_{day}.jsonl"

        line = json.dumps(payload, separators=(",", ":"))
        with file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return file_path

    def query(
        self,
        *,
        site_id: Optional[str] = None,
        pond_id: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Đọc telemetry theo bộ lọc cơ bản để phục vụ việc phân tích/training.
        """
        candidates = self._candidate_files(site_id, pond_id, start_ts, end_ts)
        records: List[Dict[str, Any]] = []
        for file_path in candidates:
            records.extend(
                self._read_file(
                    file_path,
                    site_id=site_id,
                    pond_id=pond_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )
        records.sort(key=lambda rec: rec.get("timestamp", 0), reverse=True)
        return records[:limit]

    def _candidate_files(
        self,
        site_id: Optional[str],
        pond_id: Optional[str],
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> List[Path]:
        site_part = site_id or "*"
        pond_part = pond_id or "*"

        if start_ts is None and end_ts is None:
            pattern = f"{site_part}_{pond_part}_*.jsonl"
            return sorted(self.raw_dir.glob(pattern), reverse=True)

        window_days = self._build_day_window(start_ts, end_ts)
        files: List[Path] = []
        for day in window_days:
            pattern = f"{site_part}_{pond_part}_{day}.jsonl"
            files.extend(self.raw_dir.glob(pattern))
        return sorted(files, reverse=True)

    @staticmethod
    def _build_day_window(start_ts: Optional[int], end_ts: Optional[int]) -> Iterable[str]:
        if start_ts is None and end_ts is None:
            return ()

        if start_ts is None:
            start_ts = end_ts
        if end_ts is None:
            end_ts = int(datetime.utcnow().timestamp())

        start_day = datetime.utcfromtimestamp(start_ts).date()
        end_day = datetime.utcfromtimestamp(end_ts).date()
        if end_day < start_day:
            start_day, end_day = end_day, start_day

        cursor = start_day
        while cursor <= end_day:
            yield cursor.strftime("%Y%m%d")
            cursor += timedelta(days=1)

    @staticmethod
    def _read_file(
        file_path: Path,
        *,
        site_id: Optional[str],
        pond_id: Optional[str],
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if site_id and record.get("siteId") != site_id:
                    continue
                if pond_id and record.get("pondId") != pond_id:
                    continue

                ts = record.get("timestamp")
                if ts is not None:
                    try:
                        ts_int = int(ts)
                    except (TypeError, ValueError):
                        ts_int = None
                else:
                    ts_int = None

                if start_ts is not None and ts_int is not None and ts_int < start_ts:
                    continue
                if end_ts is not None and ts_int is not None and ts_int > end_ts:
                    continue

                results.append(record)
        return results


telemetry_store = TelemetryStore(RAW_DIR)
append_telemetry = telemetry_store.append
query_telemetry = telemetry_store.query
