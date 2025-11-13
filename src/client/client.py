# gateway_client/client.py
import os, hashlib, requests
from typing import Dict, Any

AI_GATEWAY_URL = os.getenv("AI_GATEWAY_URL", "http://localhost:9000")

def sha256_file(path: str) -> str:
    h = hashlib.file_digest(open(path, "rb"), "sha256")
    return h.hexdigest()

def publish_model(model_path: str, pond_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    meta:
      - model_id
      - input
      - preprocess
      - thresholds (optional)
    """
    body = {
        "modelId": meta["model_id"],
        "pondId": pond_id,
        "downloadUrl": meta.get("download_url", model_path),  # tạm: dùng path cục bộ
        "sha256": sha256_file(model_path),
        "input": meta["input"],
        "preprocess": meta["preprocess"],
        "thresholds": meta.get("thresholds", {}),
        "notes": meta.get("notes", ""),
    }
    resp = requests.post(f"{AI_GATEWAY_URL}/model/publish", json=body, timeout=20)
    resp.raise_for_status()
    return resp.json()
