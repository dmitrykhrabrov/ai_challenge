"""Найти ffmpeg.exe (PATH или типичные пути WinGet/Gyan)."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import which


def find_ffmpeg() -> str:
    exe = which("ffmpeg")
    if exe:
        return exe

    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA", "")
    prog = os.environ.get("ProgramFiles", r"C:\Program Files")
    home = Path.home()

    if local:
        winget = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if winget.is_dir():
            candidates.extend(winget.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
            candidates.extend(winget.glob("Gyan.FFmpeg*/**/ffmpeg.exe"))
        # иногда winget кладёт shim сюда
        candidates.append(Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")

    candidates.append(Path(prog) / "ffmpeg" / "bin" / "ffmpeg.exe")
    candidates.append(home / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe")
    candidates.append(home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")

    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    raise SystemExit(
        "ffmpeg не найден. Закройте и откройте новый терминал после winget install ffmpeg,\n"
        "или добавьте bin ffmpeg в PATH. Проверка: ffmpeg -version"
    )
