"""День 6–7: агент с циклом messages → API → tool_calls → execute → снова.

При session_id история пишется/читается из JSONL (app.session_store).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import (
    AGENT_CONTEXT_MODE,
    AGENT_KEEP_LAST,
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_SAME_TOOL_CALLS,
    AGENT_SUMMARIZE_EVERY,
    DEFAULT_REASONING_EFFORT,
    DEEPSEEK_MODEL,
)
from app.context_compress import prepare_context_messages
from app.deepseek_client import chat_completion_messages
from app.pricing import estimate_usage_cost
from app import session_store
from app.tools import TOOL_SCHEMAS, execute_tool

DEFAULT_SYSTEM = (
    "Ты личный ассистент-агент. У тебя есть инструменты: время, калькулятор, "
    "поиск писем Яндекс.Почты, поиск событий Яндекс.Календаря, поиск задач Битрикс24, "
    "извлечение текста из локального EPUB (epub_to_text), "
    "чтение локального .txt (read_local_text), "
    "сводка токенов и стоимости сессии (session_usage), "
    "сообщение другой роли (ask_agent: manager, lawyer, analyst). "
    "Если для ответа нужны свежие данные — вызывай tools. "
    "Можно вызвать несколько tools подряд. "
    "Не выдумывай письма/задачи/события: опирайся на результаты tools. "
    "Про расход токенов/₽ сессии — вызывай session_usage, не угадывай. "
    "epub_to_text / read_local_text по умолчанию — статистика и preview; "
    "include_full_text=true только если пользователь явно просит полный текст в контекст "
    "(для демо переполнения). "
    "В финальном ответе кратко структурируй итог для пользователя на русском. "
    "Без эмодзи."
)

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
    cost: dict[str, Any] = field(default_factory=dict)


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
    cost_total: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


async def _ask_other_agent(
    args: dict[str, Any],
    *,
    sender_role: str,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Вопрос в чат роли. В тот чат уходит только текст, журнал отправителя не копируется."""
    from app.day10 import latest_version, log_room, pin_version, role_session

    role_name = str(args.get("role") or "").strip().lower()
    body = str(args.get("text") or args.get("message") or "").strip()
    target = role_session(role_name)
    if not target:
        return {"ok": False, "error": f"чат роли {role_name} не открыт"}
    if not body:
        return {"ok": False, "error": "пустой текст"}
    version: int | None = None
    if args.get("attach_version"):
        current = latest_version()
        if not current:
            return {"ok": False, "error": "нет выпущенной версии договора"}
        version = int(current["version"])
        pin_version(target, version)
    result = await run_agent(
        body,
        system=(
            "Ты отвечаешь на одно письмо другой роли. Инструментов нет. "
            "Ответь обычным текстом на русском, коротко. "
            "Не пиши вызовы инструментов и разметку."
        ),
        session_id=target,
        context_mode="sticky_facts",
        tail_tokens=1500,
        keep_last=4,
        tools_enabled=False,
        temperature=0.2,
    )
    from app.day10 import plain_reply

    answer = plain_reply(result.answer)
    log_room(
        kind="ask",
        sender=sender_role or "agent",
        to=role_name,
        text=body,
        answer=answer,
        version=version,
    )
    payload = {
        "ok": True,
        "role": role_name,
        "session_id": target,
        "answer": answer,
        "version": version,
    }
    _emit(on_progress, "room", **payload)
    return payload


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


def _assistant_for_history(msg: dict[str, Any]) -> dict[str, Any]:
    """Только поля, которые можно снова отправить в API (без reasoning_content)."""
    out: dict[str, Any] = {
        "role": "assistant",
        "content": msg.get("content"),
    }
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _persist(
    session_id: str | None,
    message: dict[str, Any],
    *,
    model: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    if not session_id:
        return
    session_store.append_message_event(
        session_id,
        message,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
    )


async def run_agent(
    user_text: str,
    *,
    system: str | None = None,
    history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    reasoning_effort: str | None = None,
    max_iterations: int | None = None,
    temperature: float | None = 0.2,
    context_mode: str | None = None,
    keep_last: int | None = None,
    summarize_every: int | None = None,
    tail_tokens: int | None = None,
    tools_enabled: bool = True,
    on_progress: ProgressCallback | None = None,
) -> AgentResult:
    """Один ход пользователя → цикл tool'ов → финальный ответ.

    session_id: загрузить JSONL-историю и дописывать каждое message-событие.
    history: явный список messages (если задан вместе с session_id — игнорируется в пользу файла).
    context_mode: full | summary_* (день 9) | sliding_window | sticky_facts (день 10).
    """
    effort = (reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    tool_choice = "auto"
    wall0 = time.perf_counter()
    system_text = system or DEFAULT_SYSTEM
    mode = (context_mode if context_mode is not None else AGENT_CONTEXT_MODE) or "full"
    keep = AGENT_KEEP_LAST if keep_last is None else int(keep_last)
    every = AGENT_SUMMARIZE_EVERY if summarize_every is None else int(summarize_every)

    messages: list[dict[str, Any]] = []
    sender_role = ""
    outbound_text = ""
    context_meta: dict[str, Any] = {
        "mode": mode,
        "keep_last": keep,
        "summarize_every": every,
        "compressed": False,
    }

    user_msg = {"role": "user", "content": user_text}

    if session_id:
        session_store.ensure_session_start(
            session_id,
            system=system_text,
            model=DEEPSEEK_MODEL,
        )
        _persist(session_id, user_msg)
        if tools_enabled and mode.strip().lower() == "sticky_facts":
            from app.day10 import apply_user_message

            from app.day10 import outbound_manager_ask, session_role

            side = apply_user_message(session_id, user_text)
            context_meta["facts_count"] = len(side.get("facts") or {})
            context_meta["published_version"] = side.get("published")
            context_meta["handoff"] = side.get("handoff")
            if side.get("published") or side.get("handoff"):
                _emit(on_progress, "room", kind="mail")
            sender_role = session_role(session_store.load_events(session_id))
            ask_text = outbound_manager_ask(user_text)
            outbound_text = ask_text
            if ask_text and sender_role and sender_role != "manager":
                mail = await _ask_other_agent(
                    {"role": "manager", "text": ask_text},
                    sender_role=sender_role,
                    on_progress=on_progress,
                )
                context_meta["mail"] = mail
                if mail.get("ok") and mail.get("answer"):
                    _persist(
                        session_id,
                        {
                            "role": "user",
                            "content": "Ответ менеджера:\n" + str(mail["answer"]),
                        },
                    )
        prepared = await prepare_context_messages(
            session_id,
            system_text=system_text,
            mode=mode,
            keep_last=keep,
            summarize_every=every,
            tail_tokens=tail_tokens,
            on_progress=on_progress,
        )
        messages = list(prepared.messages)
        context_meta.update(
            {
                "mode": prepared.mode,
                "compressed": prepared.compressed,
                "covers_until_seq": prepared.covers_until_seq,
                "raw_tail_count": prepared.raw_tail_count,
                "uncovered_before": prepared.uncovered_before,
                "summary_chars": len(prepared.summary_text or ""),
                "facts_preview": (
                    (
                        (f"версия {context_meta.get('published_version')}\n" if context_meta.get("published_version") else "")
                        + (prepared.summary_text or "")
                    ).strip()
                    if mode.strip().lower() == "sticky_facts"
                    else None
                ),
                "compress_duration_ms": round(prepared.compress_duration_ms),
                "compress_chunk_msgs": prepared.compress_chunk_msgs,
                "note": prepared.note,
            }
        )
        # SSE context_compress_start / context_compress уже из prepare_context_messages
        compress_usage_seed = dict(prepared.compress_usage or {})
    elif history:
        messages.extend(history)
        messages.append(user_msg)
        compress_usage_seed = {}
    else:
        messages.append({"role": "system", "content": system_text})
        messages.append(user_msg)
        compress_usage_seed = {}

    limit = max_iterations or AGENT_MAX_ITERATIONS
    trace: list[ToolTrace] = []
    llm_steps: list[LlmStep] = []
    usage_total: dict[str, int] = {}
    same_counts: dict[str, int] = {}
    model_name: str | None = None
    model_time_ms = 0.0
    stopped = "final"
    _merge_usage(usage_total, compress_usage_seed)

    def _finish(
        *,
        answer: str,
        iterations: int,
        stopped_reason: str,
    ) -> AgentResult:
        wall_ms = (time.perf_counter() - wall0) * 1000
        cost_total = estimate_usage_cost(usage_total, model=model_name)
        result = AgentResult(
            answer=answer,
            messages=messages,
            tool_trace=trace,
            llm_steps=llm_steps,
            iterations=iterations,
            stopped_reason=stopped_reason,
            model=model_name,
            usage_total=usage_total,
            model_time_ms=model_time_ms,
            wall_time_ms=wall_ms,
            cost_total=cost_total,
            session_id=session_id,
            context=context_meta,
        )
        _emit(
            on_progress,
            "done",
            stopped_reason=stopped_reason,
            iterations=iterations,
            model=model_name,
            usage_total=usage_total,
            model_time_ms=round(model_time_ms),
            wall_time_ms=round(wall_ms),
            tools=len(trace),
            cost_total=cost_total,
            session_id=session_id,
            context=context_meta,
        )
        return result

    for iteration in range(1, limit + 1):
        _emit(on_progress, "llm_start", iteration=iteration, phase="request")
        # отдать SSE до долгого HTTP, иначе UI зависает на «сжатие готово»
        await asyncio.sleep(0)
        t0 = time.perf_counter()
        raw = await chat_completion_messages(
            messages,
            tools=TOOL_SCHEMAS if tools_enabled else None,
            tool_choice=tool_choice if tools_enabled else None,
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
        finish_reason = choice.get("finish_reason")
        tool_calls = msg.get("tool_calls") or []
        tool_names = [
            ((c.get("function") or {}).get("name") or "") for c in tool_calls
        ]

        step_kind = "tools" if tool_calls else "final"
        step_cost = estimate_usage_cost(step_usage, model=model_name)
        llm_steps.append(
            LlmStep(
                iteration=iteration,
                kind=step_kind,
                model=model_name,
                duration_ms=duration_ms,
                usage=step_usage,
                tool_names=[n for n in tool_names if n],
                cost=step_cost,
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
            cost=step_cost,
        )

        assistant_msg = _assistant_for_history(msg)
        messages.append(assistant_msg)
        _persist(
            session_id,
            assistant_msg,
            model=model_name,
            finish_reason=finish_reason,
            usage=step_usage,
        )

        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            return _finish(
                answer=answer or "(пустой ответ модели)",
                iterations=iteration,
                stopped_reason="final",
            )

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
                return _finish(
                    answer=(
                        f"Остановлено: tool `{name}` с теми же аргументами "
                        f"повторяется > {AGENT_MAX_SAME_TOOL_CALLS} раз."
                    ),
                    iterations=iteration,
                    stopped_reason="repeated_tool",
                )

            if name == "session_usage" and session_id and not (
                isinstance(args.get("session_id"), str) and args["session_id"].strip()
            ):
                args = {**args, "session_id": session_id}
            _emit(on_progress, "tool_call", name=name, arguments=args)
            t_tool = time.perf_counter()
            if name == "ask_agent":
                prior = context_meta.get("mail") or {}
                same = (
                    str(args.get("role") or "").strip().lower() == str(prior.get("role") or "")
                    and str(args.get("text") or "").strip() == outbound_text
                )
                if prior.get("ok") and same:
                    result_payload = prior
                else:
                    result_payload = await _ask_other_agent(
                        args,
                        sender_role=sender_role,
                        on_progress=on_progress,
                    )
            else:
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
            tool_msg = {
                "role": "tool",
                "tool_call_id": call.get("id") or name,
                "name": name,
                "content": json.dumps(result_payload, ensure_ascii=False),
            }
            messages.append(tool_msg)
            _persist(session_id, tool_msg)

        if iteration == limit:
            stopped = "max_iterations"

    return _finish(
        answer=(
            f"Достигнут лимит итераций ({limit}). "
            "Частичный контекст собран в tool_trace; финальный текст модель не успела отдать."
        ),
        iterations=limit,
        stopped_reason=stopped,
    )
