"""Сообщение другой роли: в её чат уходит текст, не чужой журнал."""

from __future__ import annotations

from typing import Any

from app.day10 import deliver_to_analyst, latest_version, pin_version, role_session

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_agent",
        "description": (
            "Передать сообщение другой роли (manager, lawyer, analyst). "
            "В её чат попадает только этот текст. Если attach_version=true, "
            "ей закрепляется последняя выпущенная версия договора. "
            "Чужой журнал не прикладывается. В ответ вернётся реплика той роли."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["manager", "lawyer", "analyst"],
                    "description": "Кому отправить",
                },
                "text": {
                    "type": "string",
                    "description": "Текст письма",
                },
                "message": {
                    "type": "string",
                    "description": "То же письмо, если удобнее поле message",
                },
                "attach_version": {
                    "type": "boolean",
                    "description": "Закрепить за адресатом последнюю версию договора",
                },
            },
            "required": ["role", "message"],
            "additionalProperties": False,
        },
    },
}


def ask_agent(role: str, text: str, attach_version: bool = False) -> dict[str, Any]:
    """Запасной синхронный вход. Ответ модели собирает цикл агента."""
    role_name = (role or "").strip().lower()
    target = role_session(role_name)
    if not target:
        return {"ok": False, "error": f"чат роли {role_name} не открыт"}
    if attach_version:
        current = latest_version()
        if not current:
            return {"ok": False, "error": "нет выпущенной версии договора"}
        pin_version(target, int(current["version"]))
    if role_name == "analyst" and attach_version:
        return deliver_to_analyst(text)
    return {
        "ok": True,
        "role": role_name,
        "session_id": target,
        "note": "вызов принят циклом агента",
    }
