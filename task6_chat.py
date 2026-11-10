"""День 6: чат с агентом (tools) в терминале.

Один запрос:
  .\\.venv\\Scripts\\python.exe task6_chat.py --prompt "..."

Интерактив:
  .\\.venv\\Scripts\\python.exe task6_chat.py

Готовый сценарий «итоги дня»:
  .\\.venv\\Scripts\\python.exe task6_chat.py --daily
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.agent import AgentResult, run_agent

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
    """Живой лог цикла агента (без финального текста ответа)."""
    if kind == "llm_start":
        n = data.get("iteration")
        print(f"\n→ LLM #{n} …", flush=True)
        return
    if kind == "llm_done":
        step = data.get("step_kind")
        label = "FINAL" if step == "final" else "tools"
        print(
            f"← LLM #{data.get('iteration')} [{label}] "
            f"model={data.get('model')} "
            f"time={_fmt_ms(data.get('duration_ms'))} "
            f"{_fmt_usage(data.get('usage'))}",
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
        print(
            f"    └─ {flag} {_fmt_ms(data.get('duration_ms'))}",
            flush=True,
        )
        return
    if kind == "done":
        print(
            f"\n=== ИТОГ === stopped={data.get('stopped_reason')} "
            f"iters={data.get('iterations')} tools={data.get('tools')} "
            f"model={data.get('model')}",
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
    if verbose and result.tool_trace:
        print()
        print("=== TOOL TRACE ===")
        for i, t in enumerate(result.tool_trace, 1):
            print(f"{i}. {t.name}({json.dumps(t.arguments, ensure_ascii=False)})")
            preview = json.dumps(t.result, ensure_ascii=False)
            if len(preview) > 500:
                preview = preview[:500] + "…"
            print(f"   → {preview}")


async def _one_shot(prompt: str, *, verbose: bool, reasoning_effort: str | None) -> int:
    result = await run_agent(
        prompt,
        reasoning_effort=reasoning_effort,
        on_progress=_progress,
    )
    _print_answer(result, verbose=verbose)
    return 0 if result.stopped_reason == "final" else 1


async def _repl(*, verbose: bool, reasoning_effort: str | None) -> int:
    print("Агент дня 6. Команды: /daily, /quit, /verbose")
    print("Tools: time, calculator, mail, calendar, bitrix")
    print("Прогресс LLM/tools печатается всегда; /verbose — полные результаты tools.")
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
        if line == "/daily":
            line = _load_daily_prompt()
            print("(loaded day6_daily_summary.txt)")
        result = await run_agent(
            line,
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
    parser = argparse.ArgumentParser(description="День 6: чат с агентом")
    parser.add_argument("--prompt", "-p", help="Один запрос и выход")
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Запрос «итоги дня» из prompts/day6_daily_summary.txt",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Печатать полные результаты tools в конце",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "low", "high", "max"],
        help="Переопределить DEEPSEEK_REASONING_EFFORT",
    )
    args = parser.parse_args()

    if args.daily:
        prompt = _load_daily_prompt()
        return asyncio.run(
            _one_shot(prompt, verbose=args.verbose, reasoning_effort=args.reasoning_effort)
        )
    if args.prompt:
        return asyncio.run(
            _one_shot(
                args.prompt,
                verbose=args.verbose,
                reasoning_effort=args.reasoning_effort,
            )
        )
    return asyncio.run(_repl(verbose=args.verbose, reasoning_effort=args.reasoning_effort))


if __name__ == "__main__":
    sys.exit(main())
