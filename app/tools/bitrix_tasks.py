"""Tool: поиск задач Битрикс24 (tasks.task.list)."""

from __future__ import annotations

import os
from typing import Any

import httpx


def _httpx_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": 60.0}
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
    if proxy:
        kwargs["proxy"] = proxy
    ca = os.getenv("SSL_CA_BUNDLE", "").strip()
    verify_raw = os.getenv("SSL_VERIFY", "true").strip().lower()
    if ca:
        kwargs["verify"] = ca
    elif verify_raw in {"0", "false", "no", "off"}:
        kwargs["verify"] = False
    return kwargs

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bitrix_tasks_search",
        "description": (
            "Найти задачи в Битрикс24. Фильтры необязательны: можно передать только limit "
            "(список задач по дедлайну). Чтение через REST tasks.task.list. "
            "STATUS: 2=ждёт, 3=в работе, 4=ждёт контроля, 5=завершена, 6=отложена."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Подстрока названия (TITLE, LIKE %title%). Необязательно.",
                },
                "responsible_id": {
                    "type": "integer",
                    "description": "ID исполнителя (RESPONSIBLE_ID)",
                },
                "created_by": {
                    "type": "integer",
                    "description": "ID постановщика (CREATED_BY)",
                },
                "deadline_from": {
                    "type": "string",
                    "description": "Крайний срок от YYYY-MM-DD (>=DEADLINE)",
                },
                "deadline_to": {
                    "type": "string",
                    "description": "Крайний срок до YYYY-MM-DD (<=DEADLINE)",
                },
                "status": {
                    "type": "integer",
                    "description": (
                        "Статус задачи Bitrix (числовой STATUS). "
                        "5 = завершена. Не передавать — без фильтра по статусу."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Сколько задач вернуть (1..50), по умолчанию 20",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def bitrix_tasks_search(
    title: str | None = None,
    responsible_id: int | None = None,
    created_by: int | None = None,
    deadline_from: str | None = None,
    deadline_to: str | None = None,
    status: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    webhook = os.getenv("BITRIX24_WEBHOOK_URL", "").strip().rstrip("/")
    if not webhook:
        return {
            "ok": False,
            "error": "Не задан BITRIX24_WEBHOOK_URL в .env",
            "hint": (
                "Входящий вебхук портала, например "
                "https://example.bitrix24.ru/rest/1/xxxxx"
            ),
        }

    filt: dict[str, Any] = {}
    if title:
        filt["%TITLE"] = title
    if responsible_id is not None:
        filt["RESPONSIBLE_ID"] = int(responsible_id)
    if created_by is not None:
        filt["CREATED_BY"] = int(created_by)
    if deadline_from:
        filt[">=DEADLINE"] = deadline_from
    if deadline_to:
        filt["<=DEADLINE"] = deadline_to
    if status is not None:
        filt["STATUS"] = int(status)

    lim = max(1, min(int(limit or 20), 50))
    # Без фильтра по дедлайну — свежие по ID; иначе — ближайший дедлайн.
    order = {"DEADLINE": "asc"} if (">=DEADLINE" in filt or "<=DEADLINE" in filt) else {"ID": "desc"}
    payload = {
        "order": order,
        "filter": filt,
        "select": [
            "ID",
            "TITLE",
            "DESCRIPTION",
            "STATUS",
            "DEADLINE",
            "CREATED_DATE",
            "RESPONSIBLE_ID",
            "CREATED_BY",
            "PRIORITY",
        ],
        "start": 0,
    }
    url = f"{webhook}/tasks.task.list"
    try:
        with httpx.Client(**_httpx_kwargs()) as client:
            r = client.post(url, json=payload)
        if r.status_code >= 400:
            return {
                "ok": False,
                "error": f"Bitrix HTTP {r.status_code}",
                "body": r.text[:400],
            }
        data = r.json()
        if data.get("error"):
            return {
                "ok": False,
                "error": data.get("error_description") or data.get("error"),
            }
        tasks_map = (data.get("result") or {}).get("tasks") or {}
        # result.tasks — dict id -> task
        items = list(tasks_map.values()) if isinstance(tasks_map, dict) else list(tasks_map)
        items = items[:lim]
        slim = [
            {
                "id": t.get("id") or t.get("ID"),
                "title": t.get("title") or t.get("TITLE"),
                "status": t.get("status") or t.get("STATUS"),
                "deadline": t.get("deadline") or t.get("DEADLINE"),
                "responsible_id": t.get("responsibleId") or t.get("RESPONSIBLE_ID"),
                "created_by": t.get("createdBy") or t.get("CREATED_BY"),
            }
            for t in items
        ]
        return {
            "ok": True,
            "count": len(slim),
            "tasks": slim,
            "filter": filt,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
