"""Запись UI агента по video/script.json (Playwright + video).

Требования:
  1) uvicorn уже запущен: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
  2) pip install -r video/requirements.txt && playwright install chromium

Запуск:
  .\\.venv\\Scripts\\python.exe -u video\\record_ui.py
  .\\.venv\\Scripts\\python.exe -u video\\record_ui.py --script video\\script_day8.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from server_util import restart_server, wait_health

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = Path(__file__).resolve().parent / "script.json"
OUT_DIR = ROOT / "data" / "video"

# ~12.5 симв/с для спокойной русской Neural-речи (оценка hold под озвучку)
CHARS_PER_SEC = 12.5


def _ensure_browsers_path() -> None:
    """Cursor sandbox may point PLAYWRIGHT_BROWSERS_PATH elsewhere; prefer user install."""
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    chrome = local / "chromium-1255" / "chrome-win64" / "chrome.exe"
    if chrome.is_file():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(local)


def _browser_channels(script: dict) -> list[str | None]:
    """Каналы Playwright: msedge/chrome стабильнее bundled (Defender часто лочит chrome.dll)."""
    raw = (
        os.environ.get("PLAYWRIGHT_CHANNEL")
        or script.get("browser_channel")
        or "msedge"
    ).strip().lower()
    if raw in {"bundled", "chromium", "default", "-"}:
        order: list[str | None] = [None, "msedge", "chrome"]
    elif raw == "auto":
        order = ["msedge", "chrome", None]
    else:
        order = [raw]
        for extra in ("msedge", "chrome", None):
            if extra not in order:
                order.append(extra)
    return order


def _launch_browser(p, script: dict):
    last_err: Exception | None = None
    for channel in _browser_channels(script):
        kwargs: dict = {
            "headless": False,
            "args": ["--disable-dev-shm-usage"],
        }
        label = channel or "bundled"
        if channel:
            kwargs["channel"] = channel
        try:
            browser = p.chromium.launch(**kwargs)
            print(f"browser: {label}", flush=True)
            return browser
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"browser launch failed ({label}): {exc}", flush=True)
    raise RuntimeError(f"Не удалось запустить браузер: {last_err}")


def _load_script(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _est_speech_s(text: str) -> float:
    t = " ".join((text or "").split())
    if not t:
        return 0.0
    return max(1.2, len(t) / CHARS_PER_SEC)


def _narration_before(step: dict) -> str:
    return (step.get("narration_before") or step.get("narration") or "").strip()


def _narration_after(step: dict) -> str:
    return (step.get("narration_after") or "").strip()


def _wait_ai_done(page, timeout_s: float) -> None:
    """Ждём done/error в status AI-блока или появление Error-блока после него."""
    try:
        page.wait_for_function(
            """() => {
              const err = document.querySelectorAll('.block.err');
              const nodes = [...document.querySelectorAll('.block.ai .status-line')];
              if (!nodes.length) return false;
              const last = nodes[nodes.length - 1];
              const t = (last.textContent || '').trim();
              if (t.startsWith('done') || t.startsWith('error')) return true;
              // error SSE раньше мог не писать status — смотрим Error после этого AI
              const aiBlocks = [...document.querySelectorAll('.block.ai')];
              const ai = aiBlocks[aiBlocks.length - 1];
              if (!ai) return false;
              let n = ai.nextElementSibling;
              while (n) {
                if (n.classList && n.classList.contains('err')) return true;
                if (n.classList && n.classList.contains('ai')) break;
                n = n.nextElementSibling;
              }
              return false;
            }""",
            timeout=int(timeout_s * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: wait AI done timeout ({timeout_s}s): {exc}", flush=True)


def record(script_path: Path) -> Path:
    _ensure_browsers_path()
    from playwright.sync_api import sync_playwright

    script = _load_script(script_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_url = script.get("base_url", "http://127.0.0.1:8000")
    viewport = script.get("viewport") or {"width": 1280, "height": 720}
    steps = script.get("steps") or []

    timeline: list[dict] = []
    t0 = time.perf_counter()
    server_proc = None

    def mark(step: dict, phase: str, narration: str = "", **extra) -> None:
        timeline.append(
            {
                "id": step.get("id"),
                "phase": phase,
                "t_s": round(time.perf_counter() - t0, 3),
                "narration": narration,
                **extra,
            }
        )

    def apply_context_prefs(page, prefs: dict | None) -> None:
        """Выставить context_mode / keep_last / summarize_every / tail_tokens в UI."""
        if not prefs:
            return
        mode = prefs.get("context_mode") or prefs.get("mode")
        if mode:
            page.wait_for_selector("#contextMode", timeout=10_000)
            page.select_option("#contextMode", str(mode))
        if prefs.get("keep_last") is not None:
            page.fill("#keepLast", str(prefs["keep_last"]))
        if prefs.get("summarize_every") is not None:
            page.fill("#summarizeEvery", str(prefs["summarize_every"]))
        if prefs.get("tail_tokens") is not None:
            page.fill("#tailTokens", str(prefs["tail_tokens"]))
        page.evaluate(
            """() => {
              for (const id of ['contextMode', 'keepLast', 'summarizeEvery', 'tailTokens']) {
                const el = document.getElementById(id);
                if (el) el.dispatchEvent(new Event('change', { bubbles: true }));
              }
            }"""
        )

    def open_ui(page) -> None:
        page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector("#prompt", timeout=30_000)
        sid = script.get("session_id") or "video-demo"
        page.evaluate(
            """(sid) => { localStorage.setItem('agent_session_id', sid); }""",
            sid,
        )
        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector("#prompt", timeout=30_000)
        apply_context_prefs(page, script.get("context"))

    recorded_sessions: list[dict] = []
    pending_mode: str | None = None
    active_prefs: dict = dict(script.get("context") or {})
    if active_prefs:
        pending_mode = str(
            active_prefs.get("context_mode") or active_prefs.get("mode") or ""
        ) or None

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.day10 import reset_contracts, reset_room

    reset_room()
    reset_contracts()
    print("day10: лента писем и версии договора очищены", flush=True)

    with sync_playwright() as p:
        # НЕ launch_persistent_context + new_page(): иначе video пишет пустую
        # первую вкладку, а сценарий крутится на второй → белый экран.
        browser = _launch_browser(p, script)
        context = browser.new_context(
            viewport=viewport,
            record_video_dir=str(raw_dir),
            record_video_size=viewport,
            locale="ru-RU",
        )
        page = context.new_page()
        wait_health(base_url, timeout_s=10)
        open_ui(page)
        # контрольный кадр: если белый — проблема не в mux
        page.screenshot(path=str(run_dir / "debug_after_load.png"), full_page=False)

        def _current_sid() -> str:
            try:
                return str(
                    page.evaluate(
                        "() => localStorage.getItem('agent_session_id') || ''"
                    )
                    or ""
                ).strip()
            except Exception:  # noqa: BLE001
                return ""

        for step in steps:
            action = step.get("action")
            before = _narration_before(step)
            after = _narration_after(step)

            if action == "narrate":
                mark(step, "before", before)
                hold = max(float(step.get("hold_s") or 2), _est_speech_s(before) + 0.4)
                time.sleep(hold)
                mark(step, "end")

            elif action == "reset_session":
                mark(step, "before", before)
                page.once("dialog", lambda dialog: dialog.accept())
                page.click("#reset")
                hold = max(float(step.get("hold_s") or 1), _est_speech_s(before) + 0.3)
                time.sleep(hold)
                mark(step, "end")

            elif action == "new_chat":
                mark(step, "before", before)
                role = str(step.get("role") or "")
                sid = page.evaluate(
                    """async (role) => {
                      const r = await fetch('/api/session/new', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(role ? {role} : {}),
                      });
                      const data = await r.json();
                      localStorage.setItem('agent_session_id', data.session_id);
                      return data.session_id;
                    }""",
                    role,
                )
                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#prompt", timeout=15_000)
                apply_context_prefs(page, active_prefs)
                hold = max(float(step.get("hold_s") or 1.5), _est_speech_s(before) + 0.3)
                time.sleep(hold)
                sid = str(sid or _current_sid() or "").strip()
                if sid:
                    recorded_sessions.append(
                        {
                            "step_id": step.get("id"),
                            "session_id": sid,
                            "context_mode": pending_mode,
                            "t": round(time.perf_counter() - t0, 3),
                        }
                    )
                    print(
                        f"recorded session: {sid} mode={pending_mode or '?'}",
                        flush=True,
                    )
                mark(step, "end")

            elif action == "set_context":
                mark(step, "before", before)
                pending_mode = str(
                    step.get("context_mode") or step.get("mode") or ""
                ) or pending_mode
                active_prefs = {
                    "context_mode": step.get("context_mode") or step.get("mode") or pending_mode,
                    "keep_last": step.get("keep_last", active_prefs.get("keep_last")),
                    "summarize_every": step.get(
                        "summarize_every", active_prefs.get("summarize_every")
                    ),
                    "tail_tokens": step.get("tail_tokens", active_prefs.get("tail_tokens")),
                }
                apply_context_prefs(page, active_prefs)
                hold = max(float(step.get("hold_s") or 2), _est_speech_s(before) + 0.4)
                time.sleep(hold)
                page.screenshot(
                    path=str(run_dir / f"debug_{step.get('id')}_context.png"),
                    full_page=False,
                )
                mark(step, "end")
            elif action == "restart_app":
                mark(step, "before", before)
                server_proc = restart_server(
                    base_url,
                    log_path=run_dir / "uvicorn_restart.log",
                )
                hold = max(float(step.get("hold_s") or 1.5), _est_speech_s(before) + 0.3)
                time.sleep(hold)
                open_ui(page)
                page.screenshot(
                    path=str(run_dir / "debug_after_restart.png"),
                    full_page=False,
                )
                mark(step, "end")

            elif action == "prompt":
                # 1) анонс теста → текст в поле → пауза под речь
                mark(step, "before", before)
                text = step.get("text") or ""
                page.fill("#prompt", text)
                hold_before = float(step.get("hold_before_s") or 0)
                if hold_before <= 0:
                    hold_before = _est_speech_s(before) + 0.6
                time.sleep(hold_before)
                page.click("#send")
                mark(step, "sent", "")
                _wait_ai_done(page, float(step.get("timeout_s") or 120))
                # 2) комментарий к ответу / stats
                mark(step, "after", after)
                hold_after = float(step.get("hold_after_s") or 0)
                if hold_after <= 0:
                    hold_after = max(1.5, _est_speech_s(after) + 0.8)
                else:
                    hold_after = max(hold_after, _est_speech_s(after) + 0.5)
                time.sleep(hold_after)
                page.screenshot(
                    path=str(run_dir / f"debug_{step.get('id')}.png"),
                    full_page=False,
                )
                mark(step, "end")

            elif action == "reset_room":
                mark(step, "before", before)
                page.evaluate(
                    """async () => {
                      await fetch('/api/room/reset', { method: 'POST' });
                      if (window.refreshRoom) await window.refreshRoom();
                    }"""
                )
                hold = max(float(step.get("hold_s") or 0.8), _est_speech_s(before) + 0.3)
                time.sleep(hold)
                mark(step, "end")

            elif action == "switch_session":
                mark(step, "before", before)
                ref = str(step.get("session_ref") or "")
                match = next(
                    (item for item in recorded_sessions if item.get("step_id") == ref),
                    None,
                )
                if not match:
                    raise SystemExit(f"switch_session: нет чата для шага {ref}")
                page.evaluate(
                    """(sid) => { localStorage.setItem('agent_session_id', sid); }""",
                    match["session_id"],
                )
                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#prompt", timeout=15_000)
                apply_context_prefs(page, active_prefs)
                hold = max(float(step.get("hold_s") or 1.5), _est_speech_s(before) + 0.3)
                time.sleep(hold)
                mark(step, "end")

            else:
                raise SystemExit(f"Неизвестный action: {action}")

        # важно: закрыть page до context, чтобы webm финализировался с этой страницы
        page.close()
        context.close()
        browser.close()

    # сервер, который подняли на restart, оставляем живым (удобно для ручной проверки)
    _ = server_proc

    videos = sorted(raw_dir.glob("*.webm"))
    if not videos:
        raise SystemExit(f"Нет webm в {raw_dir}")
    # берём самый большой файл (не пустую заглушку)
    video_src = max(videos, key=lambda p: p.stat().st_size)
    video_path = run_dir / "screen.webm"
    if video_path.exists():
        video_path.unlink()
    video_src.replace(video_path)

    size_kb = video_path.stat().st_size / 1024
    if size_kb < 50:
        print(
            f"WARNING: screen.webm слишком маленький ({size_kb:.1f} KiB) — "
            "возможно снова пустой кадр. Смотри debug_*.png в run_dir.",
            flush=True,
        )

    (run_dir / "timeline.json").write_text(
        json.dumps(
            {
                "script": str(script_path),
                "base_url": base_url,
                "session_id": script.get("session_id"),
                "duration_s": round(time.perf_counter() - t0, 3),
                "events": timeline,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "script_copy.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "recorded_sessions.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "script": str(script_path),
                "sessions": recorded_sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"video: {video_path} ({size_kb:.1f} KiB)")
    print(f"timeline: {run_dir / 'timeline.json'}")
    print(f"recorded_sessions: {run_dir / 'recorded_sessions.json'} ({len(recorded_sessions)})")
    print(f"run_dir: {run_dir}")
    return run_dir

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    args = parser.parse_args()
    record(args.script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
