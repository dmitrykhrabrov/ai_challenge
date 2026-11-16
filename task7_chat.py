"""День 7: чат с агентом + JSONL-сессия (контекст между рестартами).

  .\\.venv\\Scripts\\python.exe -u task7_chat.py
  .\\.venv\\Scripts\\python.exe -u task7_chat.py --session demo --prompt "Который час в Москве?"
  .\\.venv\\Scripts\\python.exe -u task7_chat.py --session demo --prompt "Что ты только что ответил?"
  .\\.venv\\Scripts\\python.exe -u task7_chat.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.agent import AgentResult, run_agent
from app.config import AGENT_SESSION_ID
from app import session_store

DAILY_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "day6_daily_summary.txt"


def _load_daily_prompt() -> str:
    if DAILY_PROMPT_PATH.is_file():
        return DAILY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Подведи итоги сегодняшнего дня: время, почта, календарь, задачи Битрикс. "
        "Не выдумывай данные — только из tools."
    )


def _fmt_usage(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "tokens=?"
    p = usage.get("prompt_tokens", "?")
    c = usage.get("completion_tokens", "?")
    t = usage.get("total_tokens", "?")
    return f"tokens in={p} out={c} total={t}"


def _fmt_ms(ms: float | int | None) -> str:
    if ms is None:
        return "?"
    return f"{float(ms) / 1000:.1f}s"


def _progress(kind: str, data: dict[str, Any]) -> None:
    if kind == "llm_start":
        print(f"\n→ LLM #{data.get('iteration')} …", flush=True)
        return
    if kind == "llm_done":
        step = data.get("step_kind")
        label = "FINAL" if step == "final" else "tools"
        cost = data.get("cost") or {}
        rub = cost.get("rub")
        rub_s = f" · {rub:.4f}₽" if isinstance(rub, (int, float)) else ""
        print(
            f"← LLM #{data.get('iteration')} [{label}] "
            f"model={data.get('model')} "
            f"time={_fmt_ms(data.get('duration_ms'))} "
            f"{_fmt_usage(data.get('usage'))}{rub_s}",
            flush=True,
        )
        names = data.get("tool_names") or []
        if names:
            print(f"  tools: {', '.join(names)}", flush=True)
        return
    if kind == "tool_call":
        args = json.dumps(data.get("arguments") or {}, ensure_ascii=False)
        if len(args) > 120:
            args = args[:120] + "…"
        print(f"  tool {data.get('name')}({args})", flush=True)
        return
    if kind == "tool_done":
        ok = data.get("ok")
        flag = "ok" if ok is True else ("fail" if ok is False else "?")
        print(f"    └─ {flag} {_fmt_ms(data.get('duration_ms'))}", flush=True)
        return
    if kind == "done":
        print(
            f"\n=== ИТОГ === stopped={data.get('stopped_reason')} "
            f"iters={data.get('iterations')} tools={data.get('tools')} "
            f"session={data.get('session_id')} model={data.get('model')}",
            flush=True,
        )
        print(
            f"model_time={_fmt_ms(data.get('model_time_ms'))} "
            f"wall={_fmt_ms(data.get('wall_time_ms'))} "
            f"{_fmt_usage(data.get('usage_total'))}",
            flush=True,
        )


def _print_answer(result: AgentResult, *, verbose: bool) -> None:
    print()
    print("=== ОТВЕТ ===")
    print(result.answer)
    path = session_store.session_path(result.session_id or AGENT_SESSION_ID)
    print(f"\n[session={result.session_id} log={path}]")
    if verbose and result.tool_trace:
        print("=== TOOL TRACE ===")
        for i, t in enumerate(result.tool_trace, 1):
            print(f"{i}. {t.name}({json.dumps(t.arguments, ensure_ascii=False)})")
            preview = json.dumps(t.result, ensure_ascii=False)
            if len(preview) > 500:
                preview = preview[:500] + "…"
            print(f"   → {preview}")


async def _one_shot(
    prompt: str,
    *,
    session_id: str,
    verbose: bool,
    reasoning_effort: str | None,
) -> int:
    result = await run_agent(
        prompt,
        session_id=session_id,
        reasoning_effort=reasoning_effort,
        on_progress=_progress,
    )
    _print_answer(result, verbose=verbose)
    return 0 if result.stopped_reason == "final" else 1


async def _repl(
    *,
    session_id: str,
    verbose: bool,
    reasoning_effort: str | None,
) -> int:
    path = session_store.session_path(session_id)
    events = session_store.load_events(session_id)
    print("Агент дня 7 (JSONL-сессия). Команды: /daily, /reset, /quit, /verbose")
    print(f"session_id={session_id}  events={len(events)}  log={path}")
    print()
    show_trace = verbose
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/q", "/quit", "/exit"}:
            break
        if line == "/verbose":
            show_trace = not show_trace
            print(f"tool_trace={'on' if show_trace else 'off'}")
            continue
        if line == "/reset":
            session_store.reset_session(session_id)
            print(f"session {session_id} cleared")
            continue
        if line == "/daily":
            line = _load_daily_prompt()
            print("(loaded day6_daily_summary.txt)")
        result = await run_agent(
            line,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
            on_progress=_progress,
        )
        _print_answer(result, verbose=show_trace)
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="День 7: агент + JSONL session")
    parser.add_argument("--prompt", "-p", help="Один запрос и выход")
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--session",
        default=AGENT_SESSION_ID,
        help=f"ID сессии (default: {AGENT_SESSION_ID})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Очистить JSONL сессии и выйти",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "high", "max"],
    )
    args = parser.parse_args()

    if args.reset:
        session_store.reset_session(args.session)
        print(f"cleared session={args.session} path={session_store.session_path(args.session)}")
        return 0

    if args.daily:
        return asyncio.run(
            _one_shot(
                _load_daily_prompt(),
                session_id=args.session,
                verbose=args.verbose,
                reasoning_effort=args.reasoning_effort,
            )
        )
    if args.prompt:
        return asyncio.run(
            _one_shot(
                args.prompt,
                session_id=args.session,
                verbose=args.verbose,
                reasoning_effort=args.reasoning_effort,
            )
        )
    return asyncio.run(
        _repl(
            session_id=args.session,
            verbose=args.verbose,
            reasoning_effort=args.reasoning_effort,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
