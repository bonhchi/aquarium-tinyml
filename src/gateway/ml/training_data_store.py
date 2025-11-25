# ai_gateway/training_data_store.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import settings


class TrainingDataStore:
    """
    Đọc dữ liệu feature đã chuẩn hóa để phục vụ train/ phục vụ API export.
    """

    def __init__(self, data_file: Path):
        self.data_file = data_file

    def fetch_rows(
        self, *, label: Optional[str], limit: int, offset: int
    ) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
        if not self.data_file.exists():
            raise FileNotFoundError(str(self.data_file))

        rows: List[Dict[str, str]] = []
        matched = 0
        columns: List[str] = []
        with self.data_file.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames or []
            for row in reader:
                if label and row.get("label") != label:
                    continue
                if matched >= offset and len(rows) < limit:
                    rows.append(row)
                matched += 1

        has_more = matched > offset + len(rows)
        meta = {
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "label": label,
            "hasMore": has_more,
            "columns": columns,
            "path": str(self.data_file),
        }
        return rows, meta

    def stats(self) -> Dict[str, object]:
        if not self.data_file.exists():
            return {
                "available": False,
                "path": str(self.data_file),
                "rows": 0,
                "columns": [],
                "labels": {},
            }

        label_counts: Dict[str, int] = {}
        total = 0
        columns: List[str] = []
        with self.data_file.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames or []
            for row in reader:
                label = row.get("label", "UNKNOWN")
                label_counts[label] = label_counts.get(label, 0) + 1
                total += 1

        mtime = self.data_file.stat().st_mtime
        return {
            "available": True,
            "path": str(self.data_file),
            "rows": total,
            "labels": label_counts,
            "columns": columns,
            "lastModified": int(mtime),
        }


training_data_store = TrainingDataStore(settings.TRAINING_DATA_FILE)
