# ai_gateway/model_store.py
from pathlib import Path
from typing import Dict, Any, Optional
import json
from .config import settings

META_DIR: Path = settings.META_DIR

def _meta_file_for_pond(pond_id: str) -> Path:
    return META_DIR / f"{pond_id}.json"

def save_model_meta(pond_id: str, meta: Dict[str, Any]) -> None:
    f = _meta_file_for_pond(pond_id)
    with f.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

def load_model_meta(pond_id: str) -> Optional[Dict[str, Any]]:
    f = _meta_file_for_pond(pond_id)
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))
