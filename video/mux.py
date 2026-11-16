"""Склеить screen.webm + voiceover.mp3 → final.mp4

  .\\.venv\\Scripts\\python.exe -u video\\mux.py --run data\\video\\YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ffmpeg_util import find_ffmpeg


def _duration(ffmpeg: str, path: Path) -> float:
    probe = subprocess.run(
        [ffmpeg, "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    for line in (probe.stderr or "").splitlines():
        if "Duration:" not in line:
            continue
        try:
            hhmmss = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hhmmss.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def mux(run_dir: Path) -> Path:
    ffmpeg = find_ffmpeg()
    video = run_dir / "screen.webm"
    audio = run_dir / "voiceover.mp3"
    if not video.is_file():
        raise SystemExit(f"нет {video}")
    if not audio.is_file():
        raise SystemExit(f"нет {audio} — сначала make_voiceover.py")
    out = run_dir / "final.mp4"
    extra = _duration(ffmpeg, audio) - _duration(ffmpeg, video)
    cmd = [ffmpeg, "-y", "-i", str(video), "-i", str(audio)]
    # Озвучка финала длиннее снятого кадра: держим последний кадр, пока речь не кончится.
    if extra > 0.3:
        cmd.extend(["-vf", f"tpad=stop_mode=clone:stop_duration={extra + 0.4:.3f}"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]
    )
    subprocess.run(cmd, check=True)
    print(f"final: {out}")
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    mux(args.run.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
