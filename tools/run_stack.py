#!/usr/bin/env python3
"""Chạy đồng thời Dashboard, Gateway thu thập và Gateway ML."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_env(port_name: str, port_value: int) -> dict[str, str]:
    env = os.environ.copy()
    env[port_name] = str(port_value)
    return env


def run_services(dashboard_port: int, gateway_port: int, ml_port: int) -> int:
    commands: List[tuple[str, List[str], dict[str, str]]] = [
        (
            "dashboard",
            [sys.executable, "dashboard/app.py"],
            build_env("DASHBOARD_PORT", dashboard_port),
        ),
        (
            "gateway-main",
            [sys.executable, "src/gateway/main/app.py"],
            build_env("GATEWAY_MAIN_PORT", gateway_port),
        ),
        (
            "gateway-ml",
            [sys.executable, "src/gateway/ml/app.py"],
            build_env("ML_API_PORT", ml_port),
        ),
    ]

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for name, cmd, env in commands:
            print(f"[run_stack] start {name}: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
            processes.append((name, proc))

        while True:
            all_done = True
            for name, proc in processes:
                retcode = proc.poll()
                if retcode is None:
                    all_done = False
                else:
                    print(f"[run_stack] {name} exited with code {retcode}")
            if all_done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[run_stack] Ctrl+C detected, stopping services…")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                print(f"[run_stack] force killed {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chạy Dashboard, Gateway main và Gateway ML cùng lúc."
    )
    parser.add_argument("--dashboard-port", type=int, default=3000)
    parser.add_argument("--gateway-port", type=int, default=5001)
    parser.add_argument("--ml-port", type=int, default=5000)
    args = parser.parse_args()
    return run_services(args.dashboard_port, args.gateway_port, args.ml_port)


if __name__ == "__main__":
    raise SystemExit(main())
