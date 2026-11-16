"""День 9: сжатие истории → summary + сырой хвост keep_last.

До порога keep_last + summarize_every сырые сообщения не отбрасываются.
Сжатие — отдельный вызов API (без tools), не на каждый user-ход.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.config import DEEPSEEK_MODEL
from app.deepseek_client import chat_completion_messages, extract_text
from app import session_store

ProgressCallback = Callable[[str, dict[str, Any]], None]

ContextMode = Literal["full", "summary_simple", "summary_facts", "summary_rules"]

SUMMARY_SIMPLE_SYSTEM = (
    "Ты сжимаешь историю диалога ассистента с tools. "
    "Краткое связное резюме на русском, 8–12 предложений. "
    "Если в тексте или в предыдущем summary есть клиент, контакт, срок, сумма, "
    "запрет, выбранный вариант, город или код — назови их явно, теми же словами. "
    "Не заменяй значение оборотами «указан», «зафиксирован», «без изменений» "
    "и не пиши, что его нет, если оно уже встречалось. "
    "Без эмодзи и без markdown-заголовков."
)

SUMMARY_FACTS_SYSTEM = (
    "Ты сжимаешь историю диалога. На выходе два блока.\n"
    "ФАКТЫ — маркированный список. Копируй буквально имена людей и компаний, "
    "даты, суммы, города, запреты («нельзя …»), выбранные варианты, коды. "
    "Факт из начала диалога обязателен, даже если дальше длинное описание "
    "и там он не повторяется. Факт из предыдущего summary тоже обязателен, "
    "пока новый фрагмент его прямо не отменяет. "
    "Не пиши «не указано» и «в тексте нет», если значение есть выше. "
    "Не заменяй значение словами «зафиксирован» или «без изменений».\n"
    "КОНТЕКСТ — 2–4 предложения сюжета, без повтора списка.\n"
    "Не выдумывай. Не дублируй один факт. Язык русский. Без эмодзи."
)

SUMMARY_RULES_SYSTEM = (
    "Ты сжимаешь историю делового диалога. Сначала выпиши дословно, отдельным списком, "
    "всё, что относится к этим классам, если оно встретилось в тексте:\n"
    "- финансовые показатели: суммы, пороги, проценты, штрафы, неустойки, аванс, "
    "лимиты платежей, потолок ответственности;\n"
    "- юридические ограничения: запреты, требуемые согласия, применимое право, "
    "подсудность, сроки претензий и ответственность сторон;\n"
    "- сроки и числовые SLA, если от них зависят деньги или ответственность.\n"
    "Не подменяй число общими словами «предусмотрен штраф» или «есть ограничение». "
    "Не пиши, что показателя нет, если он был в тексте. "
    "Прочий сюжет — 4–6 предложений, без пересказа всего черновика. "
    "Не выдумывай. Язык русский. Без эмодзи."
)


@dataclass
class ContextPrepareResult:
    messages: list[dict[str, Any]]
    mode: str
    compressed: bool = False
    summary_text: str | None = None
    covers_until_seq: int | None = None
    raw_tail_count: int = 0
    uncovered_before: int = 0
    compress_usage: dict[str, int] = field(default_factory=dict)
    compress_duration_ms: float = 0.0
    compress_chunk_msgs: int = 0
    note: str = ""


def _norm_mode(mode: str | None) -> ContextMode:
    m = (mode or "full").strip().lower()
    if m in {"summary_simple", "simple"}:
        return "summary_simple"
    if m in {"summary_facts", "facts"}:
        return "summary_facts"
    if m in {"summary_rules", "rules"}:
        return "summary_rules"
    return "full"


def _system_for_mode(mode: ContextMode) -> str:
    if mode == "summary_facts":
        return SUMMARY_FACTS_SYSTEM
    if mode == "summary_rules":
        return SUMMARY_RULES_SYSTEM
    return SUMMARY_SIMPLE_SYSTEM


def format_messages_for_summary(msgs: list[dict[str, Any]], *, max_chars: int = 280_000) -> str:
    lines: list[str] = []
    for m in msgs:
        role = m.get("role")
        if role == "user":
            text = str(m.get("content") or "")
            if len(text) > 36_000:
                text = text[:36_000] + "…"
            lines.append(f"USER: {text}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = []
                for tc in m["tool_calls"]:
                    fn = (tc or {}).get("function") or {}
                    names.append(str(fn.get("name") or "?"))
                lines.append("ASSISTANT: [tool_calls: " + ", ".join(names) + "]")
                content = m.get("content")
                if content:
                    c = str(content)
                    if len(c) > 2_000:
                        c = c[:2_000] + "…"
                    lines.append(f"ASSISTANT_TEXT: {c}")
            else:
                c = str(m.get("content") or "")
                if len(c) > 2_000:
                    c = c[:2_000] + "…"
                lines.append(f"ASSISTANT: {c}")
        elif role == "tool":
            raw = m.get("content") or ""
            if len(raw) > 400:
                raw = raw[:400] + "…"
            lines.append(f"TOOL({m.get('name') or m.get('tool_call_id') or '?'}): {raw}")
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n…[truncated for compress]"
    return body


def raw_tail_start(msgs: list[dict[str, Any]], keep_last: int) -> int:
    """Индекс начала сырого хвоста; не режет tool_calls / tool mid-chain."""
    n = len(msgs)
    if keep_last <= 0 or n <= keep_last:
        return 0
    start = n - keep_last
    while start > 0 and msgs[start].get("role") == "tool":
        start -= 1
    while start > 0:
        prev = msgs[start - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            ids = {
                str((tc or {}).get("id") or "")
                for tc in prev["tool_calls"]
                if (tc or {}).get("id")
            }
            later = {
                str(m.get("tool_call_id") or "")
                for m in msgs[start:]
                if m.get("role") == "tool"
            }
            if ids & later:
                start -= 1
                continue
        break
    return start


def _dialog_api_messages(
    events: list[dict[str, Any]],
    *,
    after_seq: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """API-messages и их seq для message-событий с seq > after_seq."""
    msgs: list[dict[str, Any]] = []
    seqs: list[int] = []
    for e in events:
        if e.get("type") != "message":
            continue
        role = e.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        seq = int(e.get("seq") or 0)
        if seq <= after_seq:
            continue
        m: dict[str, Any] = {"role": role}
        if "content" in e:
            m["content"] = e.get("content")
        if role == "assistant" and e.get("tool_calls"):
            m["tool_calls"] = e["tool_calls"]
            m.setdefault("content", None)
        if role == "tool":
            m["tool_call_id"] = e.get("tool_call_id")
            m["content"] = e.get("content") or ""
            if e.get("name"):
                m["name"] = e.get("name")
        msgs.append(m)
        seqs.append(seq)
    # drop incomplete tool tail (align seqs)
    trimmed = session_store.drop_incomplete_tail(msgs)
    if len(trimmed) < len(msgs):
        seqs = seqs[: len(trimmed)]
        msgs = trimmed
    return msgs, seqs


async def summarize_chunk(
    *,
    mode: ContextMode,
    previous_summary: str | None,
    chunk: list[dict[str, Any]],
) -> tuple[str, dict[str, int], str | None]:
    """Вернуть (summary_text, usage, model)."""
    body = format_messages_for_summary(chunk)
    user_parts = []
    prev = (previous_summary or "").strip()
    if len(prev) > 1200:
        prev = prev[:1200] + "…"
    if prev:
        user_parts.append("ПРЕДЫДУЩЕЕ SUMMARY:\n" + prev)
    user_parts.append("НОВЫЙ ФРАГМЕНТ ДИАЛОГА:\n" + body)
    messages = [
        {"role": "system", "content": _system_for_mode(mode)},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
    # Короткий max_tokens — быстрее и дешевле; лимиты длины уже в system.
    max_out = 1400 if mode == "summary_rules" else 700 if mode == "summary_facts" else 512
    raw = await chat_completion_messages(
        messages,
        tools=None,
        reasoning_effort="none",
        temperature=0.2,
        max_tokens=max_out,
    )
    text = (extract_text(raw) or "").strip()
    usage = raw.get("usage") or {}
    usage_i = {
        k: int(usage.get(k) or 0)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    model = raw.get("model") or DEEPSEEK_MODEL
    if not text:
        text = previous_summary or "(пустое summary)"
    return text, usage_i, model


def _summary_message(text: str, mode: str) -> dict[str, Any]:
    return {
        "role": "system",
        "content": (
            f"[CONTEXT SUMMARY | mode={mode}]\n"
            f"{text.strip()}\n"
            "[END SUMMARY — ниже только недавние сообщения диалога]"
        ),
    }


def _emit(cb: ProgressCallback | None, event: str, **payload: Any) -> None:
    if cb:
        cb(event, payload)


async def prepare_context_messages(
    session_id: str,
    *,
    system_text: str,
    mode: str | None,
    keep_last: int,
    summarize_every: int,
    on_progress: ProgressCallback | None = None,
    tail_tokens: int | None = None,
) -> ContextPrepareResult:
    raw_mode = (mode or "full").strip().lower()
    if raw_mode in {"sliding_window", "sticky_facts", "window"}:
        from app.day10 import prepare_window_messages

        return await prepare_window_messages(
            session_id,
            system_text=system_text,
            mode="sticky_facts" if raw_mode == "sticky_facts" else "sliding_window",
            keep_last=keep_last,
            tail_tokens=tail_tokens,
        )
    mode_n = _norm_mode(mode)
    keep_last = max(1, int(keep_last))
    summarize_every = max(1, int(summarize_every))
    threshold = keep_last + summarize_every

    events = session_store.load_events(session_id)
    head = next((e for e in events if e.get("type") == "session_start"), None)
    system = (head or {}).get("system") or system_text

    if mode_n == "full":
        msgs = session_store.to_messages(events, system_fallback=system_text)
        return ContextPrepareResult(
            messages=msgs,
            mode=mode_n,
            note="full history",
            raw_tail_count=max(0, len(msgs) - (1 if system else 0)),
        )

    latest = session_store.latest_context_summary(events)
    covered = int((latest or {}).get("covers_until_seq") or -1)
    prev_text = ((latest or {}).get("content") or "").strip() or None

    uncovered, seqs = _dialog_api_messages(events, after_seq=covered)
    uncovered_n = len(uncovered)

    compressed = False
    compress_usage: dict[str, int] = {}
    compress_duration_ms = 0.0
    compress_chunk_msgs = 0
    summary_text = prev_text
    covers_until = covered if covered >= 0 else None

    if uncovered_n >= threshold:
        start_tail = raw_tail_start(uncovered, keep_last)
        chunk = uncovered[:start_tail]
        chunk_seqs = seqs[:start_tail]
        if chunk:
            compress_chunk_msgs = len(chunk)
            _emit(
                on_progress,
                "context_compress_start",
                mode=mode_n,
                uncovered_before=uncovered_n,
                chunk_msgs=compress_chunk_msgs,
                threshold=threshold,
                keep_last=keep_last,
            )
            t0 = time.perf_counter()
            summary_text, compress_usage, model = await summarize_chunk(
                mode=mode_n,
                previous_summary=prev_text,
                chunk=chunk,
            )
            compress_duration_ms = (time.perf_counter() - t0) * 1000
            covers_until = int(chunk_seqs[-1])
            session_store.append_context_summary(
                session_id,
                content=summary_text,
                mode=mode_n,
                covers_until_seq=covers_until,
                message_count=len(chunk),
                model=model,
                usage=compress_usage,
                duration_ms=compress_duration_ms,
            )
            compressed = True
            uncovered = uncovered[start_tail:]
            seqs = seqs[start_tail:]
            _emit(
                on_progress,
                "context_compress",
                mode=mode_n,
                covers_until_seq=covers_until,
                raw_tail_count=len(uncovered),
                uncovered_before=uncovered_n,
                usage=compress_usage,
                summary_chars=len(summary_text or ""),
                duration_ms=round(compress_duration_ms),
                chunk_msgs=compress_chunk_msgs,
                phase="done",
                next="llm",
            )

    # Сборка API view
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    if summary_text:
        out.append(_summary_message(summary_text, mode_n))
    # Пока summary нет — весь uncovered (полная история).
    # Если summary есть — весь uncovered (он уже «хвост+накопленное после summary»),
    # без повторного отрезания: отрезание только в момент compress.
    out.extend(uncovered)

    note = (
        f"mode={mode_n}; uncovered={uncovered_n}; "
        f"threshold={threshold}; compressed={compressed}; "
        f"covers_until={covers_until}; compress_ms={round(compress_duration_ms)}"
    )
    return ContextPrepareResult(
        messages=out,
        mode=mode_n,
        compressed=compressed,
        summary_text=summary_text,
        covers_until_seq=covers_until,
        raw_tail_count=len(uncovered),
        uncovered_before=uncovered_n,
        compress_usage=compress_usage,
        compress_duration_ms=compress_duration_ms,
        compress_chunk_msgs=compress_chunk_msgs,
        note=note,
    )
