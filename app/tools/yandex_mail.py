"""Tool: поиск писем Яндекс.Почта по дате и отправителю (IMAP)."""

from __future__ import annotations

import email
import imaplib
import os
from datetime import datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "yandex_mail_search",
        "description": (
            "Найти письма в Яндекс.Почте по отправителю и/или диапазону дат. "
            "Возвращает краткий список (тема, from, дата), без полного тела."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sender": {
                    "type": "string",
                    "description": "Фрагмент адреса/имени отправителя (FROM)",
                },
                "date_from": {
                    "type": "string",
                    "description": "Начало периода YYYY-MM-DD (включительно)",
                },
                "date_to": {
                    "type": "string",
                    "description": "Конец периода YYYY-MM-DD (включительно)",
                },
                "folder": {
                    "type": "string",
                    "description": "Папка IMAP, по умолчанию INBOX",
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимум писем в ответе (1..20), по умолчанию 10",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _decode_mime(value: str | None) -> str:
    if not value:
        return ""
    parts: list[str] = []
    for chunk, enc in decode_header(value):
        if isinstance(chunk, bytes):
            parts.append(chunk.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(chunk)
    return "".join(parts)


def _imap_date(day: str, *, end: bool = False) -> str:
    """IMAP SEARCH SINCE/BEFORE ждёт дату вида 10-Nov-2026."""
    dt = datetime.strptime(day, "%Y-%m-%d")
    if end:
        # BEFORE — строго меньше дня, поэтому +1 день
        from datetime import timedelta

        dt = dt + timedelta(days=1)
    return dt.strftime("%d-%b-%Y")


def yandex_mail_search(
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    folder: str = "INBOX",
    limit: int = 10,
) -> dict[str, Any]:
    user = os.getenv("YANDEX_MAIL_USER", "").strip()
    password = os.getenv("YANDEX_MAIL_APP_PASSWORD", "").strip()
    host = os.getenv("YANDEX_IMAP_HOST", "imap.yandex.com").strip()
    if not user or not password:
        return {
            "ok": False,
            "error": "Не заданы YANDEX_MAIL_USER / YANDEX_MAIL_APP_PASSWORD в .env",
            "hint": "Пароль приложения: https://id.yandex.ru/security/app-passwords",
        }
    if not sender and not date_from and not date_to:
        return {
            "ok": False,
            "error": "Укажите хотя бы sender или date_from/date_to",
        }

    lim = max(1, min(int(limit or 10), 20))
    criteria: list[str] = []
    try:
        if date_from:
            criteria.extend(["SINCE", _imap_date(date_from)])
        if date_to:
            criteria.extend(["BEFORE", _imap_date(date_to, end=True)])
        if sender:
            # IMAP: FROM "addr"
            criteria.extend(["FROM", f'"{sender}"'])
    except ValueError as exc:
        return {"ok": False, "error": f"Некорректная дата: {exc}"}

    try:
        mail = imaplib.IMAP4_SSL(host, 993)
        mail.login(user, password)
        typ, _ = mail.select(folder or "INBOX", readonly=True)
        if typ != "OK":
            mail.logout()
            return {"ok": False, "error": f"Не удалось открыть папку {folder}"}

        typ, data = mail.search(None, *criteria)
        if typ != "OK":
            mail.logout()
            return {"ok": False, "error": "IMAP SEARCH не удался", "criteria": criteria}

        ids = (data[0] or b"").split()
        ids = ids[-lim:]  # последние lim
        messages: list[dict[str, Any]] = []
        for mid in reversed(ids):
            typ, msg_data = mail.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            date_raw = msg.get("Date")
            try:
                date_iso = parsedate_to_datetime(date_raw).isoformat() if date_raw else None
            except Exception:  # noqa: BLE001
                date_iso = date_raw
            messages.append(
                {
                    "uid": mid.decode(errors="replace"),
                    "from": _decode_mime(msg.get("From")),
                    "subject": _decode_mime(msg.get("Subject")),
                    "date": date_iso,
                }
            )
        mail.logout()
        return {
            "ok": True,
            "folder": folder or "INBOX",
            "count": len(messages),
            "messages": messages,
            "criteria": {
                "sender": sender,
                "date_from": date_from,
                "date_to": date_to,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
