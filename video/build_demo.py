"""Пайплайн демо-видео: record → (разбор агентом) → voiceover → mux.

Типовой день 9:
  1) запись экрана (--record-only) → авторазбор в UI-агенте
  2) правки narration в script_copy.json (по вердикту / своим ответам в чате)
  3) озвучка + склейка (--skip-record)

Перед записью:
  .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_voiceover import build as build_voice  # noqa: E402
from mux import mux  # noqa: E402
from post_record_analyze import analyze_run  # noqa: E402
from record_ui import record  # noqa: E402

DEFAULT_SCRIPT = HERE / "script.json"


def _script_wants_analyze(script_path: Path) -> bool:
    try:
        data = json.loads(script_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("analyze_prompt"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument(
        "--skip-record",
        type=Path,
        default=None,
        help="Пропустить запись: существующий data/video/<run>",
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Только Playwright-запись (+ авторазбор, если есть analyze_prompt)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="После записи запустить разбор агентом (даже без analyze_prompt в скрипте)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Не запускать авторазбор после записи",
    )
    parser.add_argument(
        "--analyze-only",
        type=Path,
        default=None,
        help="Только разбор уже записанного run_dir (без record/TTS)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "edge", "windows"],
        default="auto",
        help="TTS: auto|edge|windows",
    )
    args = parser.parse_args()

    if args.analyze_only:
        analyze_run(args.analyze_only.resolve())
        return 0

    if args.skip_record and args.record_only:
        raise SystemExit("Нельзя одновременно --skip-record и --record-only")

    script_path = args.script.resolve()

    if args.skip_record:
        run_dir = args.skip_record.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"нет run_dir: {run_dir}")
    else:
        run_dir = record(script_path)
        print(f"\nRECORD_ONLY run_dir: {run_dir}", flush=True)

        do_analyze = False
        if args.no_analyze:
            do_analyze = False
        elif args.analyze or args.record_only:
            do_analyze = args.analyze or _script_wants_analyze(script_path)

        if do_analyze:
            print("\n--- post-record analyze ---", flush=True)
            try:
                analyze_run(run_dir)
            except SystemExit as exc:
                print(f"ANALYZE skipped/failed: {exc}", flush=True)
            print(
                "Дальше: правки озвучки в "
                f"{run_dir / 'script_copy.json'} (narration*) → "
                f"build_demo.py --skip-record {run_dir} --engine edge",
                flush=True,
            )
        else:
            print(
                "Дальше: разбор сессий в UI-агенте → правки "
                f"{run_dir / 'script_copy.json'} (narration*) → "
                f"build_demo.py --skip-record {run_dir} --engine edge",
                flush=True,
            )

        if args.record_only:
            return 0

    build_voice(run_dir, engine=args.engine)
    out = mux(run_dir)
    print(f"\nDONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
