"""День 6: агент с циклом messages → API → tool_calls → execute → снова."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import (
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_SAME_TOOL_CALLS,
    DEFAULT_REASONING_EFFORT,
)
from app.deepseek_client import chat_completion_messages
from app.tools import TOOL_SCHEMAS, execute_tool

DEFAULT_SYSTEM = (
    "Ты личный ассистент-агент. У тебя есть инструменты: время, калькулятор, "
    "поиск писем Яндекс.Почты, поиск событий Яндекс.Календаря, поиск задач Битрикс24. "
    "Если для ответа нужны свежие данные — вызывай tools. "
    "Можно вызвать несколько tools подряд. "
    "Не выдумывай письма/задачи/события: опирайся на результаты tools. "
    "В финальном ответе кратко структурируй итог для пользователя на русском. "
    "Без эмодзи."
)

# kind: llm_start | llm_done | tool_call | tool_done | done
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float = 0.0


@dataclass
class LlmStep:
    iteration: int
    kind: str  # tools | final
    model: str | None
    duration_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    tool_names: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    messages: list[dict[str, Any]]
    tool_trace: list[ToolTrace] = field(default_factory=list)
    llm_steps: list[LlmStep] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "final"  # final | max_iterations | repeated_tool
    model: str | None = None
    usage_total: dict[str, int] = field(default_factory=dict)
    model_time_ms: float = 0.0
    wall_time_ms: float = 0.0


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
    if not usage:
        return {}
    return {
        key: int(usage.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _merge_usage(acc: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key, val in _usage_dict(usage).items():
        acc[key] = acc.get(key, 0) + val


def _emit(cb: ProgressCallback | None, event: str, **payload: Any) -> None:
    if cb:
        cb(event, payload)


async def run_agent(
    user_text: str,
    *,
    system: str | None = None,
    history: list[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
    max_iterations: int | None = None,
    temperature: float | None = 0.2,
    on_progress: ProgressCallback | None = None,
) -> AgentResult:
    """Один ход пользователя → цикл tool'ов → финальный ответ."""
    effort = (reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    tool_choice = "auto"
    wall0 = time.perf_counter()

    messages: list[dict[str, Any]] = []
    if history:
        messages.extend(history)
    else:
        messages.append({"role": "system", "content": system or DEFAULT_SYSTEM})
    messages.append({"role": "user", "content": user_text})

    limit = max_iterations or AGENT_MAX_ITERATIONS
    trace: list[ToolTrace] = []
    llm_steps: list[LlmStep] = []
    usage_total: dict[str, int] = {}
    same_counts: dict[str, int] = {}
    model_name: str | None = None
    model_time_ms = 0.0
    stopped = "final"

    for iteration in range(1, limit + 1):
        _emit(on_progress, "llm_start", iteration=iteration, phase="request")
        t0 = time.perf_counter()
        raw = await chat_completion_messages(
            messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,
            reasoning_effort=effort,
            temperature=temperature,
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        model_time_ms += duration_ms
        model_name = raw.get("model") or model_name
        step_usage = _usage_dict(raw.get("usage"))
        _merge_usage(usage_total, raw.get("usage"))

        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        tool_names = [
            ((c.get("function") or {}).get("name") or "") for c in tool_calls
        ]

        step_kind = "tools" if tool_calls else "final"
        llm_steps.append(
            LlmStep(
                iteration=iteration,
                kind=step_kind,
                model=model_name,
                duration_ms=duration_ms,
                usage=step_usage,
                tool_names=[n for n in tool_names if n],
            )
        )
        _emit(
            on_progress,
            "llm_done",
            iteration=iteration,
            step_kind=step_kind,
            model=model_name,
            duration_ms=round(duration_ms),
            usage=step_usage,
            tool_names=[n for n in tool_names if n],
        )

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content"),
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            wall_ms = (time.perf_counter() - wall0) * 1000
            result = AgentResult(
                answer=answer or "(пустой ответ модели)",
                messages=messages,
                tool_trace=trace,
                llm_steps=llm_steps,
                iterations=iteration,
                stopped_reason="final",
                model=model_name,
                usage_total=usage_total,
                model_time_ms=model_time_ms,
                wall_time_ms=wall_ms,
            )
            _emit(
                on_progress,
                "done",
                stopped_reason="final",
                iterations=iteration,
                model=model_name,
                usage_total=usage_total,
                model_time_ms=round(model_time_ms),
                wall_time_ms=round(wall_ms),
                tools=len(trace),
            )
            return result

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            fp = _fingerprint(name, args)
            same_counts[fp] = same_counts.get(fp, 0) + 1
            if same_counts[fp] > AGENT_MAX_SAME_TOOL_CALLS:
                stopped = "repeated_tool"
                wall_ms = (time.perf_counter() - wall0) * 1000
                result = AgentResult(
                    answer=(
                        f"Остановлено: tool `{name}` с теми же аргументами "
                        f"повторяется > {AGENT_MAX_SAME_TOOL_CALLS} раз."
                    ),
                    messages=messages,
                    tool_trace=trace,
                    llm_steps=llm_steps,
                    iterations=iteration,
                    stopped_reason=stopped,
                    model=model_name,
                    usage_total=usage_total,
                    model_time_ms=model_time_ms,
                    wall_time_ms=wall_ms,
                )
                _emit(
                    on_progress,
                    "done",
                    stopped_reason=stopped,
                    iterations=iteration,
                    model=model_name,
                    usage_total=usage_total,
                    model_time_ms=round(model_time_ms),
                    wall_time_ms=round(wall_ms),
                    tools=len(trace),
                )
                return result

            _emit(on_progress, "tool_call", name=name, arguments=args)
            t_tool = time.perf_counter()
            result_payload = execute_tool(name, args)
            tool_ms = (time.perf_counter() - t_tool) * 1000
            trace.append(
                ToolTrace(
                    name=name,
                    arguments=args,
                    result=result_payload,
                    duration_ms=tool_ms,
                )
            )
            ok = result_payload.get("ok")
            _emit(
                on_progress,
                "tool_done",
                name=name,
                ok=ok,
                duration_ms=round(tool_ms),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "content": json.dumps(result_payload, ensure_ascii=False),
                }
            )

        if iteration == limit:
            stopped = "max_iterations"

    wall_ms = (time.perf_counter() - wall0) * 1000
    result = AgentResult(
        answer=(
            f"Достигнут лимит итераций ({limit}). "
            "Частичный контекст собран в tool_trace; финальный текст модель не успела отдать."
        ),
        messages=messages,
        tool_trace=trace,
        llm_steps=llm_steps,
        iterations=limit,
        stopped_reason=stopped,
        model=model_name,
        usage_total=usage_total,
        model_time_ms=model_time_ms,
        wall_time_ms=wall_ms,
    )
    _emit(
        on_progress,
        "done",
        stopped_reason=stopped,
        iterations=limit,
        model=model_name,
        usage_total=usage_total,
        model_time_ms=round(model_time_ms),
        wall_time_ms=round(wall_ms),
        tools=len(trace),
    )
    return result
