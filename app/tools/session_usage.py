"""Tool: сводка токенов и ₽ по JSONL-сессии (без раздувания каждого хода)."""

from __future__ import annotations

from typing import Any

from app import session_store

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "session_usage",
        "description": (
            "Сводка токенов и стоимости (USD/RUB) по сессии чата из JSONL. "
            "Вызывать, когда пользователь спрашивает сколько токенов/денег сожгли, "
            "расход сессии, статистику usage. "
            "Без session_id — текущая сессия (подставляет runtime)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Id сессии (имя JSONL без .jsonl). По умолчанию — текущая.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _clip(text: Any, limit: int = 240) -> str:
    raw = str(text or "").replace("\n", " ")
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "…"


def _tool_names(event: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for call in event.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        names.append(str(fn.get("name") or "?"))
    return names


def _llm_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Шаги из JSONL: токены вызова модели, имя tool и короткий результат."""
    rows: list[dict[str, Any]] = []
    pending_user = ""
    for event in events:
        if event.get("role") == "user":
            pending_user = event.get("content") or ""
        if event.get("type") == "context_summary":
            usage = event.get("usage") or {}
            rows.append(
                {
                    "seq": event.get("seq"),
                    "kind": "context_summary",
                    "mode": event.get("mode"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )
            continue
        if event.get("role") == "tool":
            rows.append(
                {
                    "seq": event.get("seq"),
                    "kind": "tool",
                    "name": event.get("name") or event.get("tool_call_id") or "?",
                    "preview": _clip(event.get("content")),
                }
            )
            continue
        if event.get("role") != "assistant" or not event.get("usage"):
            continue
        usage = event.get("usage") or {}
        row: dict[str, Any] = {
            "seq": event.get("seq"),
            "kind": "assistant",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "user_preview": pending_user.replace("\n", " ")[:120],
        }
        names = _tool_names(event)
        if names:
            row["tools"] = names
        if pending_user.startswith(
            (
                "По пакету документов",
                "По выпущенному договору",
                "Можно ли отдать коннектор",
                "Повтори по текущей версии",
                "Менеджер передал вторую версию",
                "Открой первую версию",
            )
        ) and (event.get("content") or "").strip():
            row["answer"] = event.get("content") or ""
        rows.append(row)
    return rows


def session_usage(session_id: str | None = None) -> dict[str, Any]:
    sid = (session_id or "").strip()
    if not sid:
        return {
            "ok": False,
            "error": "session_id не задан",
            "hint": "Укажи session_id или вызови из агента с активной сессией",
        }

    path = session_store.session_path(sid)
    if not path.is_file():
        return {
            "ok": False,
            "error": f"Файл не найден: {path}",
            "session_id": sid,
            "path": str(path),
        }
    events = session_store.load_events(sid)
    stats = session_store.aggregate_session_stats(events)
    return {
        "ok": True,
        "session_id": sid,
        "path": str(path),
        "event_count": len(events),
        "preview": session_store.session_preview(events),
        "llm_calls": stats["llm_calls"],
        "compress_calls": stats["compress_calls"],
        "message_counts": stats["message_counts"],
        "usage_total": stats["usage_total"],
        "cost_total": stats["cost_total"],
        "llm_steps": _llm_steps(events),
        "first_ts": stats.get("first_ts"),
        "last_ts": stats.get("last_ts"),
        "note": (
            "Сумма usage с assistant + context_summary в JSONL "
            "(каждый шаг tool-loop и сжатие отдельно). "
            "llm_steps включает имя tool и короткий preview результата. "
            "₽/USD — сумма оценок pricing.py по шагам."
        ),
    }
