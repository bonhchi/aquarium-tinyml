"""
Helper script để cài dependency mà không cần nhớ lệnh pip dài.

Usage:
    python install_dependencies.py              # cài toàn bộ requirements.txt (bao gồm cả gateway)
    python install_dependencies.py --gateway    # chỉ cài deps của src/gateway

Script giữ nguyên file requirements để bạn tham khảo thủ công khi cần.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ROOT_REQUIREMENTS = REPO_ROOT / "requirements.txt"
GATEWAY_REQUIREMENTS = REPO_ROOT / "src" / "gateway" / "requirements.txt"


def run(cmd: list[str]) -> None:
    print(f"[install] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def install(requirements_file: Path) -> None:
    if not requirements_file.exists():
        raise FileNotFoundError(f"Không thấy file {requirements_file}")

    python = sys.executable
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", str(requirements_file)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Cài dependency cho Aquarium TinyML.")
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Chỉ cài deps của gateway (src/gateway/requirements.txt).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Chỉ định file requirements tuỳ ý, ưu tiên cao hơn các flag khác.",
    )

    args = parser.parse_args()
    if args.file:
        install(args.file.resolve())
    elif args.gateway:
        install(GATEWAY_REQUIREMENTS)
    else:
        install(ROOT_REQUIREMENTS)


if __name__ == "__main__":
    main()
