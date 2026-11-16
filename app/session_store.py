"""День 7: JSONL-лог сессии → messages[] для Chat Completions.

Диск (JSONL) ↔ list[event] ↔ messages (только API-поля).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import AGENT_SESSION_DIR, DEEPSEEK_MODEL
from app.pricing import estimate_usage_cost


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _safe_session_id(session_id: str) -> str:
    sid = (session_id or "default").strip() or "default"
    sid = re.sub(r"[^a-zA-Z0-9._-]+", "_", sid)[:80]
    return sid or "default"


def session_path(session_id: str, *, base_dir: Path | None = None) -> Path:
    root = base_dir or Path(AGENT_SESSION_DIR)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_safe_session_id(session_id)}.jsonl"


def load_events(session_id: str, *, base_dir: Path | None = None) -> list[dict[str, Any]]:
    path = session_path(session_id, base_dir=base_dir)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # обрезанная последняя строка после краша — пропускаем
                continue
    events.sort(key=lambda e: int(e.get("seq") or 0))
    return events


def next_seq(events: list[dict[str, Any]]) -> int:
    if not events:
        return 0
    return max(int(e.get("seq") or 0) for e in events) + 1


def append_event(
    session_id: str,
    event: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Append one JSONL line. seq/id/ts/v проставляются, если нет."""
    path = session_path(session_id, base_dir=base_dir)
    events = load_events(session_id, base_dir=base_dir)
    payload = dict(event)
    payload.setdefault("v", 1)
    payload.setdefault("seq", next_seq(events))
    payload.setdefault("id", f"ev-{uuid.uuid4().hex[:12]}")
    payload.setdefault("ts", _utc_now())
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
    return payload


def ensure_session_start(
    session_id: str,
    *,
    system: str,
    model: str | None = None,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    events = load_events(session_id, base_dir=base_dir)
    if any(e.get("type") == "session_start" for e in events):
        return events
    append_event(
        session_id,
        {
            "type": "session_start",
            "session_id": _safe_session_id(session_id),
            "system": system,
            "model": model or DEEPSEEK_MODEL,
        },
        base_dir=base_dir,
    )
    return load_events(session_id, base_dir=base_dir)


def drop_incomplete_tail(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Убрать висячие tool_calls без role=tool (краш посреди хода)."""
    pending: dict[str, int] = {}
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tc_id = (tc or {}).get("id")
                if tc_id:
                    pending[str(tc_id)] = i
        elif m.get("role") == "tool":
            pending.pop(str(m.get("tool_call_id") or ""), None)
    if not pending:
        return msgs
    cut = min(pending.values())
    return msgs[:cut]


def to_messages(
    events: list[dict[str, Any]],
    *,
    system_fallback: str | None = None,
) -> list[dict[str, Any]]:
    head = next((e for e in events if e.get("type") == "session_start"), None)
    system = (head or {}).get("system") or system_fallback
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})

    for e in events:
        if e.get("type") != "message":
            continue
        role = e.get("role")
        if role not in {"user", "assistant", "tool"}:
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
        msgs.append(m)

    return drop_incomplete_tail(msgs)


def append_message_event(
    session_id: str,
    message: dict[str, Any],
    *,
    model: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Нормализовать chat-message → event type=message и append."""
    role = message.get("role")
    event: dict[str, Any] = {
        "type": "message",
        "role": role,
        "content": message.get("content"),
    }
    if role == "assistant" and message.get("tool_calls"):
        event["tool_calls"] = message["tool_calls"]
    if role == "tool":
        event["tool_call_id"] = message.get("tool_call_id")
        if message.get("name"):
            event["name"] = message.get("name")
    if model:
        event["model"] = model
    if finish_reason:
        event["finish_reason"] = finish_reason
    if usage:
        event["usage"] = {
            k: usage.get(k)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if usage.get(k) is not None
        }
    return append_event(session_id, event, base_dir=base_dir)


def latest_context_summary(
    events: list[dict[str, Any]] | None = None,
    *,
    session_id: str | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Последнее событие type=context_summary."""
    if events is None:
        if not session_id:
            return None
        events = load_events(session_id, base_dir=base_dir)
    summaries = [e for e in events if e.get("type") == "context_summary"]
    return summaries[-1] if summaries else None


def append_context_summary(
    session_id: str,
    *,
    content: str,
    mode: str,
    covers_until_seq: int,
    message_count: int | None = None,
    model: str | None = None,
    usage: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Записать summary отдельно от сырых messages."""
    event: dict[str, Any] = {
        "type": "context_summary",
        "mode": mode,
        "content": content,
        "covers_until_seq": int(covers_until_seq),
    }
    if message_count is not None:
        event["message_count"] = int(message_count)
    if model:
        event["model"] = model
    if duration_ms is not None:
        event["duration_ms"] = round(float(duration_ms), 1)
    if usage:
        event["usage"] = {
            k: usage.get(k)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if usage.get(k) is not None
        }
    return append_event(session_id, event, base_dir=base_dir)


def reset_session(session_id: str, *, base_dir: Path | None = None) -> None:
    path = session_path(session_id, base_dir=base_dir)
    if path.is_file():
        path.unlink()


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"chat-{stamp}-{uuid.uuid4().hex[:6]}"


def list_sessions(*, base_dir: Path | None = None) -> list[dict[str, Any]]:
    root = session_path("default", base_dir=base_dir).parent
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        sid = path.stem
        events = load_events(sid, base_dir=base_dir)
        items.append(
            {
                "session_id": sid,
                "event_count": len(events),
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "preview": session_preview(events),
            }
        )
    return items


def _content_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def session_preview(events: list[dict[str, Any]], *, max_len: int = 72) -> str:
    """Первое user-сообщение — короткая метка для списка/поиска."""
    for e in events:
        if e.get("type") == "message" and e.get("role") == "user":
            text = " ".join(_content_str(e.get("content")).split())
            if not text:
                continue
            if len(text) > max_len:
                return text[: max_len - 1] + "…"
            return text
    return ""


def _snippet(text: str, needle: str, *, radius: int = 56) -> str:
    low = text.lower()
    idx = low.find(needle.lower())
    if idx < 0:
        flat = " ".join(text.split())
        return flat[: radius * 2] + ("…" if len(flat) > radius * 2 else "")
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    chunk = text[start:end]
    chunk = " ".join(chunk.split())
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{chunk}{suffix}"


def _event_search_text(event: dict[str, Any]) -> str | None:
    if event.get("type") != "message":
        return None
    parts: list[str] = []
    content = _content_str(event.get("content")).strip()
    if content:
        parts.append(content)
    if event.get("role") == "assistant" and event.get("tool_calls"):
        for tc in event["tool_calls"] or []:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or ""
            if name or args:
                parts.append(f"{name} {args}".strip())
    if event.get("role") == "tool" and event.get("name"):
        parts.append(str(event.get("name")))
    text = "\n".join(parts).strip()
    return text or None


def search_sessions(
    query: str,
    *,
    limit: int = 40,
    per_session: int = 5,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Поиск подстроки по user/assistant/tool-сообщениям во всех JSONL-сессиях."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    q_low = q.lower()
    limit = max(1, min(int(limit), 100))
    per_session = max(1, min(int(per_session), 20))

    root = session_path("default", base_dir=base_dir).parent
    if not root.is_dir():
        return []

    hits: list[dict[str, Any]] = []
    paths = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        if len(hits) >= limit:
            break
        sid = path.stem
        events = load_events(sid, base_dir=base_dir)
        preview = session_preview(events)
        found_in_session = 0
        # id сессии тоже индексируем
        if q_low in sid.lower() and found_in_session < per_session:
            hits.append(
                {
                    "session_id": sid,
                    "preview": preview,
                    "role": "meta",
                    "seq": None,
                    "snippet": sid,
                    "match": "session_id",
                }
            )
            found_in_session += 1
        for e in events:
            if len(hits) >= limit or found_in_session >= per_session:
                break
            text = _event_search_text(e)
            if not text or q_low not in text.lower():
                continue
            hits.append(
                {
                    "session_id": sid,
                    "preview": preview,
                    "role": e.get("role"),
                    "seq": e.get("seq"),
                    "snippet": _snippet(text, q),
                    "match": "message",
                }
            )
            found_in_session += 1
    return hits


def _parse_event_ts(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def aggregate_session_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Сводка usage/₽/счётчиков по JSONL — для UI Session stats и tool session_usage."""
    prompt = completion = total = 0
    llm_calls = 0
    compress_calls = 0
    usd = rub = 0.0
    models: dict[str, int] = {}
    counts = {"user": 0, "assistant": 0, "tool": 0}
    last_model = DEEPSEEK_MODEL
    first_ts: str | None = None
    last_ts: str | None = None

    for e in events:
        ts = e.get("ts")
        if isinstance(ts, str):
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        et = e.get("type")
        if et == "message":
            role = e.get("role")
            if role in counts:
                counts[role] += 1
            if role != "assistant":
                continue
            usage = e.get("usage")
            if not isinstance(usage, dict):
                continue
            llm_calls += 1
        elif et == "context_summary":
            usage = e.get("usage")
            if not isinstance(usage, dict):
                continue
            compress_calls += 1
            llm_calls += 1
        else:
            continue
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
        prompt += p
        completion += c
        total += t
        model = (e.get("model") or DEEPSEEK_MODEL) or DEEPSEEK_MODEL
        last_model = model
        models[model] = models.get(model, 0) + 1
        step_cost = estimate_usage_cost(
            usage,
            model=model,
            when=_parse_event_ts(e.get("ts")),
        )
        usd += float(step_cost.get("usd") or 0)
        rub += float(step_cost.get("rub") or 0)

    usage_total = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    rolled = estimate_usage_cost(usage_total, model=last_model)
    cost_total = {
        **rolled,
        "usd": round(usd, 8),
        "rub": round(rub, 4),
        "summed_per_llm_call": True,
        "models": models,
    }
    return {
        "turns": counts["user"],
        "tool_calls": counts["tool"],
        "llm_calls": llm_calls,
        "compress_calls": compress_calls,
        "message_counts": counts,
        "usage_total": usage_total,
        "cost_total": cost_total,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def session_public_view(session_id: str, *, base_dir: Path | None = None) -> dict[str, Any]:
    events = load_events(session_id, base_dir=base_dir)
    summary = latest_context_summary(events)
    return {
        "session_id": _safe_session_id(session_id),
        "path": str(session_path(session_id, base_dir=base_dir)),
        "event_count": len(events),
        "preview": session_preview(events),
        "stats": aggregate_session_stats(events),
        "context_summary": (
            {
                "mode": summary.get("mode"),
                "covers_until_seq": summary.get("covers_until_seq"),
                "content": summary.get("content"),
                "chars": len(summary.get("content") or ""),
            }
            if summary
            else None
        ),
        "events": events,
        "messages": to_messages(events),
    }
