"""Tool: текущее время."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": (
            "Вернуть текущие дату и время. "
            "Использовать, когда пользователь спрашивает который час / какая дата."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone, например Europe/Moscow. По умолчанию Europe/Moscow.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def get_current_time(timezone: str = "Europe/Moscow") -> dict[str, Any]:
    tz_name = (timezone or "Europe/Moscow").strip() or "Europe/Moscow"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Неизвестная timezone: {tz_name}",
            "hint": "Пример: Europe/Moscow, UTC",
        }
    now = datetime.now(tz)
    return {
        "ok": True,
        "timezone": tz_name,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }
