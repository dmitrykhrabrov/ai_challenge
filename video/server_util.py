"""Старт/стоп uvicorn для шага restart_app в демо-видео."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent


def project_python() -> str:
    """Всегда предпочитать .venv проекта (не system Python без httpx)."""
    if os.name == "nt":
        candidates = [
            ROOT / ".venv" / "Scripts" / "python.exe",
            ROOT / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            ROOT / ".venv" / "bin" / "python",
            ROOT / "venv" / "bin" / "python",
        ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return sys.executable


def parse_host_port(base_url: str) -> tuple[str, int]:
    u = urlparse(base_url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    return host, int(port)


def wait_health(base_url: str, timeout_s: float = 45.0) -> None:
    health = base_url.rstrip("/") + "/health"
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urlopen(health, timeout=2) as resp:
                if getattr(resp, "status", 200) < 400:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(0.4)
    raise RuntimeError(f"Сервер не поднялся: {health} ({last_err})")


def _pids_on_port_windows(port: int) -> list[int]:
    # netstat -ano | findstr :8000
    out = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    needle = f":{port}"
    for line in (out.stdout or "").splitlines():
        if needle not in line:
            continue
        # только LISTENING
        if "LISTENING" not in line.upper():
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return sorted(pids)


def stop_server(base_url: str) -> None:
    _, port = parse_host_port(base_url)
    if os.name == "nt":
        for pid in _pids_on_port_windows(port):
            if pid <= 0:
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True,
                check=False,
            )
    else:
        subprocess.run(
            ["bash", "-lc", f"fuser -k {port}/tcp || true"],
            capture_output=True,
            check=False,
        )
    # дать сокету освободиться
    time.sleep(1.0)


def start_server(base_url: str, *, log_path: Path | None = None) -> subprocess.Popen:
    host, port = parse_host_port(base_url)
    py = project_python()
    print(f"restart_app: python={py}", flush=True)
    cmd = [
        py,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_f = None
    kwargs: dict = {
        "cwd": str(ROOT),
        "env": os.environ.copy(),
    }
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        kwargs["stdout"] = log_f
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(cmd, **kwargs)
    # держим файл открытым на время жизни процесса — привяжем к proc
    if log_f is not None:
        proc._video_log_file = log_f  # type: ignore[attr-defined]
    wait_health(base_url, timeout_s=60)
    return proc


def restart_server(base_url: str, *, log_path: Path | None = None) -> subprocess.Popen:
    print(f"restart_app: stop {base_url}", flush=True)
    stop_server(base_url)
    print(f"restart_app: start {base_url}", flush=True)
    return start_server(base_url, log_path=log_path)
