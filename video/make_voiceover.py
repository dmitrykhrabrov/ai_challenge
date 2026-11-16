"""Собрать дорожку озвучки по timeline.json + script.

TTS:
  1) edge-tts (если сеть/прокси позволяют)
  2) fallback: Windows SAPI (System.Speech) → wav → mp3

  .\\.venv\\Scripts\\python.exe -u video\\make_voiceover.py --run data\\video\\YYYYMMDD_HHMMSS
  .\\.venv\\Scripts\\python.exe -u video\\make_voiceover.py --run ... --engine windows

Голоса edge-tts (приятные RU):
  ru-RU-SvetlanaNeural  — женский, по умолчанию
  ru-RU-DariyaNeural    — женский, чуть мягче
  ru-RU-DmitryNeural    — мужской
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ffmpeg_util import find_ffmpeg

DEFAULT_VOICE = "ru-RU-SvetlanaNeural"


def _probe_duration(ffmpeg: str, path: Path) -> float:
    probe = subprocess.run(
        [ffmpeg, "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    for line in (probe.stderr or "").splitlines():
        if "Duration:" in line:
            try:
                hhmmss = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = hhmmss.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:  # noqa: BLE001
                return 2.0
    return 2.0


def _silence_mp3(path: Path, seconds: float, ffmpeg: str) -> None:
    seconds = max(0.05, float(seconds))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            f"{seconds:.3f}",
            "-q:a",
            "9",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _synth_windows(text: str, out_mp3: Path, ffmpeg: str) -> None:
    """Офлайн TTS через Windows SAPI."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = -1
$s.Volume = 100
try {{
  $ru = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'ru*' }} | Select-Object -First 1
  if ($ru) {{ $s.SelectVoice($ru.VoiceInfo.Name) }}
}} catch {{}}
$s.SetOutputToWaveFile('{str(wav).replace("'", "''")}')
$s.Speak(@'
{text.replace("'", "''")}
'@)
$s.Dispose()
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            check=True,
            capture_output=True,
        )
        if not wav.is_file() or wav.stat().st_size < 100:
            raise RuntimeError("Windows SAPI не создал wav")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(wav), "-q:a", "4", str(out_mp3)],
            check=True,
            capture_output=True,
        )


async def _synth_edge(text: str, voice: str, rate: str, out_mp3: Path) -> None:
    import edge_tts

    # MITM-прокси часто ломает websocket edge-tts
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    os.environ.setdefault("NO_PROXY", "*")
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_mp3))


def synth(
    text: str,
    *,
    voice: str,
    out_mp3: Path,
    ffmpeg: str,
    engine: str = "auto",
    rate: str = "-5%",
) -> None:
    text = " ".join(text.split())
    if not text:
        raise ValueError("пустой narration")

    errors: list[str] = []
    engines = []
    if engine == "auto":
        engines = ["edge", "windows"]
    elif engine == "edge":
        engines = ["edge"]
    elif engine == "windows":
        engines = ["windows"]
    else:
        raise SystemExit(f"unknown engine: {engine}")

    for eng in engines:
        try:
            if eng == "edge":
                asyncio.run(_synth_edge(text, voice, rate, out_mp3))
            else:
                _synth_windows(text, out_mp3, ffmpeg)
            if out_mp3.is_file() and out_mp3.stat().st_size > 200:
                print(f"  tts[{eng}]: {out_mp3.name}", flush=True)
                return
            errors.append(f"{eng}: empty file")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{eng}: {exc}")
            out_mp3.unlink(missing_ok=True)

    raise SystemExit("TTS failed:\n  - " + "\n  - ".join(errors))


def _clip_events(events: list[dict], sid: str) -> list[dict]:
    """События шага с непустой narration, по времени."""
    clips = [
        e
        for e in events
        if e.get("id") == sid and (e.get("narration") or "").strip()
    ]
    clips.sort(key=lambda e: float(e.get("t_s") or 0))
    return clips


def sync_narrations_from_script(events: list[dict], script: dict) -> list[dict]:
    """Подставить narration из script_copy (после ручного разбора теста).

    before/start ← narration_before | narration
    after ← narration_after
    """
    steps = {s.get("id"): s for s in (script.get("steps") or []) if s.get("id")}
    out: list[dict] = []
    for ev in events:
        e = dict(ev)
        step = steps.get(e.get("id") or "") or {}
        phase = e.get("phase") or ""
        if phase in {"before", "start", "narrate"}:
            text = (step.get("narration_before") or step.get("narration") or "").strip()
            if text:
                e["narration"] = text
        elif phase == "after":
            text = (step.get("narration_after") or "").strip()
            if text:
                e["narration"] = text
        out.append(e)
    return out


def build(run_dir: Path, *, engine: str = "auto") -> Path:
    ffmpeg = find_ffmpeg()
    timeline_path = run_dir / "timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    script_path = run_dir / "script_copy.json"
    script = json.loads(script_path.read_text(encoding="utf-8")) if script_path.is_file() else {}
    voice = script.get("voice") or DEFAULT_VOICE
    rate = str(script.get("voice_rate") or "-5%")
    events = sync_narrations_from_script(timeline.get("events") or [], script)
    # сохраняем актуальные тексты озвучки (после правок script_copy)
    timeline["events"] = events
    timeline_path.write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_id: dict[str, list[dict]] = {}
    for ev in events:
        sid = ev.get("id") or ""
        by_id.setdefault(sid, []).append(ev)

    audio_dir = run_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    parts: list[Path] = []
    cursor = 0.0
    idx = 0

    step_order = [s.get("id") for s in (script.get("steps") or [])]
    if not step_order:
        step_order = list(dict.fromkeys(e.get("id") for e in events if e.get("id")))

    def append_silence(seconds: float, tag: str) -> None:
        nonlocal idx, cursor
        if seconds <= 0.05:
            return
        sil = audio_dir / f"{idx:03d}_{tag}.mp3"
        _silence_mp3(sil, seconds, ffmpeg)
        parts.append(sil)
        idx += 1
        cursor += seconds

    for sid in step_order:
        step_events = by_id.get(sid) or []
        if not step_events:
            continue
        t_end = max(float(e.get("t_s") or 0) for e in step_events)
        clips = _clip_events(events, sid)

        # legacy: только start/end с одной narration на start
        if not clips:
            pair = {e.get("phase"): e for e in step_events}
            start = pair.get("start") or pair.get("before") or {}
            narration = (start.get("narration") or "").strip()
            if narration:
                clips = [
                    {
                        "id": sid,
                        "phase": "before",
                        "t_s": float(start.get("t_s") or cursor),
                        "narration": narration,
                    }
                ]

        for clip in clips:
            t_at = float(clip.get("t_s") or cursor)
            narration = (clip.get("narration") or "").strip()
            if not narration:
                continue

            gap = t_at - cursor
            if gap > 0.05:
                append_silence(gap, "silence")
            elif gap < -0.05:
                # речь предыдущего клипа залезла — чуть подожмём старт
                t_at = cursor

            mp3 = audio_dir / f"{idx:03d}_{sid}_{clip.get('phase') or 'n'}.mp3"
            synth(
                narration,
                voice=voice,
                out_mp3=mp3,
                ffmpeg=ffmpeg,
                engine=engine,
                rate=rate,
            )
            parts.append(mp3)
            idx += 1
            narr_dur = _probe_duration(ffmpeg, mp3)
            cursor = t_at + narr_dur

        pad = t_end - cursor
        if pad > 0.05:
            append_silence(pad, "pad")
        else:
            cursor = max(cursor, t_end)

    list_file = audio_dir / "concat.txt"
    lines = []
    for p in parts:
        ap = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{ap}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_mp3 = run_dir / "voiceover.mp3"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(out_mp3),
        ],
        check=True,
        capture_output=True,
    )
    print(f"voiceover: {out_mp3} (voice={voice}, rate={rate})")
    return out_mp3


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="data/video/<run_dir>")
    parser.add_argument(
        "--engine",
        choices=["auto", "edge", "windows"],
        default="auto",
        help="auto: edge-tts → Windows SAPI",
    )
    args = parser.parse_args()
    build(args.run.resolve(), engine=args.engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
