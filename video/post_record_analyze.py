"""После успешной записи: новый чат с DeepSeek-агентом + разбор прогона.

Читает промпт (по умолчанию prompts/day9_analyze_for_voiceover.txt),
подставляет session_id из recorded_sessions.json, дергает /agent,
пишет ответ в run_dir и открывает UI на этой сессии — можно ответить и поправить.

  .\\.venv\\Scripts\\python.exe -u video\\post_record_analyze.py --run data\\video\\YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = ROOT / "prompts" / "day9_analyze_for_voiceover.txt"


def _http_json(
    method: str,
    url: str,
    body: dict | None = None,
    *,
    timeout: float = 300.0,
) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _control_excerpt(session_id: str) -> str:
    path = ROOT / "data" / "sessions" / f"{session_id}.jsonl"
    if not path.is_file():
        return f"{session_id}: файла нет ({path})"
    pending = ""
    found: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("role") == "user":
            pending = event.get("content") or ""
        if (
            event.get("role") == "assistant"
            and (
                pending.startswith("По пакету документов")
                or pending.startswith("По выпущенному договору")
                or pending.startswith("Можно ли отдать коннектор")
                or pending.startswith("Повтори по текущей версии")
                or pending.startswith("Менеджер передал вторую версию")
                or pending.startswith("Открой первую версию")
            )
            and (event.get("content") or "").strip()
        ):
            prompt_tokens = (event.get("usage") or {}).get("prompt_tokens")
            question = " ".join(pending.split())[:180]
            found.append(
                f"Q: {question}\nrequest_tokens={prompt_tokens}\n{(event.get('content') or '').strip()}"
            )
            pending = ""
    if not found:
        return f"{session_id}: ответа на контрольный вопрос нет"
    return f"{session_id}\n" + "\n\n".join(found)


def _write_control_excerpts(run_dir: Path, sessions: list[dict]) -> Path:
    blocks = []
    for item in sessions:
        sid = str(item.get("session_id") or "").strip()
        mode = item.get("context_mode") or "?"
        if not sid:
            continue
        blocks.append(f"## {mode} {sid}\n{_control_excerpt(sid)}")
    path = run_dir / "control_excerpts.txt"
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return path


def _load_prompt(
    path: Path,
    sessions: list[dict],
    run_dir: Path,
    script: dict | None,
) -> str:
    text = path.read_text(encoding="utf-8").strip()
    excerpts = _write_control_excerpts(run_dir, sessions)
    lines = [
        "",
        "---",
        f"run_dir: {run_dir}",
        f"Сценарий: {run_dir / 'script_copy.json'}",
        f"Ответы на контрольный вопрос: {excerpts}",
        "Чеклист:",
    ]
    checklist = list((script or {}).get("checklist") or [])
    if checklist:
        lines.extend(f"- {item}" for item in checklist)
    else:
        lines.append("- (в сценарии нет поля checklist)")
    lines.append("Сессии этого прогона (используй session_usage по каждому id):")
    if sessions:
        for s in sessions:
            sid = s.get("session_id") or "?"
            mode = s.get("context_mode") or "?"
            step = s.get("step_id") or "?"
            lines.append(f"- {sid}  mode={mode}  step={step}")
    else:
        lines.append(
            "- (recorded_sessions пуст — ищи свежие chat-* по превью "
            "«Карточка проекта» / ООО «Север»)"
        )
    lines.append(
        "Ответ дай в чате; я могу уточнить и попросить поправить вердикт для озвучки."
    )
    return text + "\n" + "\n".join(lines) + "\n"


def _resolve_prompt_path(script: dict | None, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    if script:
        raw = script.get("analyze_prompt")
        if raw:
            p = Path(str(raw))
            if not p.is_absolute():
                p = ROOT / p
            return p
    return DEFAULT_PROMPT


def analyze_run(
    run_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    prompt_path: Path | None = None,
    open_browser: bool = True,
) -> dict:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"нет run_dir: {run_dir}")

    screen = run_dir / "screen.webm"
    if not screen.is_file() or screen.stat().st_size < 50 * 1024:
        raise SystemExit(
            f"снимок экрана не готов или слишком маленький: {screen} — "
            "разбор не запускаю"
        )

    script: dict | None = None
    script_copy = run_dir / "script_copy.json"
    if script_copy.is_file():
        script = json.loads(script_copy.read_text(encoding="utf-8"))
        base_url = str(script.get("base_url") or base_url).rstrip("/")

    sessions_path = run_dir / "recorded_sessions.json"
    sessions: list[dict] = []
    if sessions_path.is_file():
        payload = json.loads(sessions_path.read_text(encoding="utf-8"))
        sessions = list(payload.get("sessions") or [])

    prompt_file = _resolve_prompt_path(script, prompt_path)
    if not prompt_file.is_file():
        raise SystemExit(f"нет промпта: {prompt_file}")

    prompt = _load_prompt(prompt_file, sessions, run_dir, script)
    (run_dir / "analyze_prompt.txt").write_text(prompt, encoding="utf-8")

    print(f"analyze: prompt={prompt_file}", flush=True)
    print(f"analyze: sessions={len(sessions)}", flush=True)

    try:
        new = _http_json("POST", f"{base_url}/api/session/new", timeout=30.0)
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"сервер недоступен ({base_url}): {exc}. "
            "Нужен uvicorn app.main:app"
        ) from exc

    analyze_sid = str(new.get("session_id") or "").strip()
    if not analyze_sid:
        raise SystemExit(f"api/session/new без session_id: {new}")

    print(f"analyze: session={analyze_sid}", flush=True)
    print("analyze: вызываю /agent (session_usage + разбор)…", flush=True)

    try:
        result = _http_json(
            "POST",
            f"{base_url}/agent",
            {
                "prompt": prompt,
                "session_id": analyze_sid,
                "context_mode": "full",
                "temperature": 0.2,
            },
            timeout=600.0,
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise SystemExit(f"/agent HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"/agent failed: {exc}") from exc

    answer = str(result.get("answer") or "").strip()
    (run_dir / "analyze_answer.txt").write_text(answer + ("\n" if answer else ""), encoding="utf-8")
    (run_dir / "analyze_meta.json").write_text(
        json.dumps(
            {
                "analyze_session_id": analyze_sid,
                "prompt_path": str(prompt_file),
                "recorded_sessions": sessions,
                "usage": result.get("usage"),
                "cost": result.get("cost"),
                "model": result.get("model"),
                "ui_url": f"{base_url}/?session={analyze_sid}",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== ANALYZE ANSWER =====\n", flush=True)
    print(answer or "(пустой ответ)", flush=True)
    print("\n===== /ANALYZE =====\n", flush=True)
    print(f"analyze_answer: {run_dir / 'analyze_answer.txt'}", flush=True)
    print(f"analyze_session: {analyze_sid}", flush=True)

    ui = f"{base_url}/?session={analyze_sid}"
    print(f"UI (можно ответить и скорректировать): {ui}", flush=True)
    if open_browser:
        try:
            webbrowser.open(ui)
        except Exception as exc:  # noqa: BLE001
            print(f"не открыл браузер: {exc}", flush=True)

    return {
        "run_dir": str(run_dir),
        "analyze_session_id": analyze_sid,
        "ui_url": ui,
        "answer": answer,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="data/video/<run>")
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать UI после ответа",
    )
    args = parser.parse_args()
    analyze_run(
        args.run,
        base_url=args.base_url.rstrip("/"),
        prompt_path=args.prompt,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
