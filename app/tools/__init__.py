"""Реестр tools дня 6: схемы для LLM + execute по имени.

Лимиты цикла агента (max_iterations, anti-loop) живут в коде Agent/config,
не в system prompt и не в md — модель их не соблюдает сама.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from app.tools import (
    bitrix_tasks,
    calculator,
    epub_text,
    local_text,
    agent_handoff,
    session_usage,
    time_tool,
    yandex_calendar,
    yandex_mail,
)

# Имя функции → callable
HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_current_time": time_tool.get_current_time,
    "calculator": calculator.calculator,
    "yandex_mail_search": yandex_mail.yandex_mail_search,
    "yandex_calendar_search": yandex_calendar.yandex_calendar_search,
    "bitrix_tasks_search": bitrix_tasks.bitrix_tasks_search,
    "epub_to_text": epub_text.epub_to_text,
    "read_local_text": local_text.read_local_text,
    "session_usage": session_usage.session_usage,
    "ask_agent": agent_handoff.ask_agent,
}

# Схемы в формате DeepSeek/OpenAI tools[]
TOOL_SCHEMAS: list[dict[str, Any]] = [
    time_tool.SCHEMA,
    calculator.SCHEMA,
    yandex_mail.SCHEMA,
    yandex_calendar.SCHEMA,
    bitrix_tasks.SCHEMA,
    epub_text.SCHEMA,
    local_text.SCHEMA,
    session_usage.SCHEMA,
    agent_handoff.SCHEMA,
]


def list_tool_names() -> list[str]:
    return list(HANDLERS.keys())


def execute_tool(name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Выполнить tool по имени. arguments — dict или JSON-строка от модели."""
    if name not in HANDLERS:
        return {"ok": False, "error": f"Неизвестный tool: {name}"}
    args: dict[str, Any]
    if arguments is None:
        args = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Некорректный JSON arguments: {exc}"}
    elif isinstance(arguments, dict):
        args = arguments
    else:
        return {"ok": False, "error": f"Неподдерживаемый тип arguments: {type(arguments)}"}

    try:
        return HANDLERS[name](**args)
    except TypeError as exc:
        return {"ok": False, "error": f"Аргументы не подошли к {name}: {exc}", "arguments": args}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "tool": name}
