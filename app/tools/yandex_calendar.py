"""Tool: поиск событий Яндекс.Календарь (CalDAV) по диапазону дат и организатору."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse, unquote
from xml.etree import ElementTree as ET

import httpx


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _httpx_kwargs() -> dict[str, Any]:
    """CalDAV часто ломается на локальном MITM-прокси (REPORT/EOF).

    По умолчанию идём напрямую (trust_env=False).
    Прокси только если YANDEX_CALDAV_USE_PROXY=true.
    """
    kwargs: dict[str, Any] = {"timeout": 60.0, "trust_env": False}
    use_proxy = _as_bool(os.getenv("YANDEX_CALDAV_USE_PROXY"), default=False)
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
    if use_proxy and proxy:
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
        "name": "yandex_calendar_search",
        "description": (
            "Найти события в Яндекс.Календаре по диапазону дат и/или организатору. "
            "Использует CalDAV (caldav.yandex.ru)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {
                    "type": "string",
                    "description": "Начало диапазона YYYY-MM-DD",
                },
                "date_to": {
                    "type": "string",
                    "description": "Конец диапазона YYYY-MM-DD",
                },
                "organizer": {
                    "type": "string",
                    "description": "Фрагмент email/имени организатора (фильтр на клиенте)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимум событий (1..30), по умолчанию 15",
                },
            },
            "required": ["date_from", "date_to"],
            "additionalProperties": False,
        },
    },
}

_NS = {
    "d": "DAV:",
    "c": "urn:ietf:params:xml:ns:caldav",
    "cs": "http://calendarserver.org/ns/",
}


def _parse_day(day: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _resolve_caldav_target(
    base: str,
    calendar_path: str,
) -> tuple[str, str | None]:
    """Принять path ИЛИ полный CalDAV URL из настроек Яндекса.

    Возвращает (final_url, warning|None).
    """
    raw = (calendar_path or "").strip()
    base = (base or "https://caldav.yandex.ru").rstrip("/")
    if not raw:
        return base, "пустой YANDEX_CALENDAR_PATH"
    if raw.startswith("http://") or raw.startswith("https://"):
        # Пользователь вставил всю ссылку из вкладки «Экспорт»
        return raw.rstrip("/") + "/", (
            "YANDEX_CALENDAR_PATH содержал полный URL — использован как есть"
        )
    if not raw.startswith("/"):
        raw = "/" + raw
    return base + raw, None


def _ical_prop(block: str, name: str) -> str | None:
    for line in block.splitlines():
        if line.upper().startswith(name.upper() + ":") or line.upper().startswith(
            name.upper() + ";"
        ):
            return line.split(":", 1)[-1].strip()
    return None


def yandex_calendar_search(
    date_from: str,
    date_to: str,
    organizer: str | None = None,
    limit: int = 15,
) -> dict[str, Any]:
    user = os.getenv("YANDEX_CALDAV_USER", "").strip() or os.getenv(
        "YANDEX_MAIL_USER", ""
    ).strip()
    password = os.getenv("YANDEX_CALDAV_APP_PASSWORD", "").strip() or os.getenv(
        "YANDEX_MAIL_APP_PASSWORD", ""
    ).strip()
    base = os.getenv("YANDEX_CALDAV_URL", "https://caldav.yandex.ru").rstrip("/")
    calendar_path = os.getenv("YANDEX_CALENDAR_PATH", "").strip()

    if not user or not password:
        return {
            "ok": False,
            "error": (
                "Не заданы YANDEX_CALDAV_USER/YANDEX_CALDAV_APP_PASSWORD "
                "(или YANDEX_MAIL_* ) в .env"
            ),
        }
    if not calendar_path:
        return {
            "ok": False,
            "error": "Не задан YANDEX_CALENDAR_PATH в .env",
            "hint": (
                "Пример: /calendars/<login>@yandex.ru/events-XXXX/ "
                "(смотрите в настройках календаря / CalDAV)"
            ),
        }

    try:
        start = _parse_day(date_from)
        end = _parse_day(date_to) + timedelta(days=1)
    except ValueError as exc:
        return {"ok": False, "error": f"Некорректная дата: {exc}"}

    lim = max(1, min(int(limit or 15), 30))
    url, url_note = _resolve_caldav_target(base, calendar_path)
    body = f"""<?xml version="1.0" encoding="utf-8" ?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop>
    <d:getetag/>
    <c:calendar-data/>
  </d:prop>
  <c:filter>
    <c:comp-filter name="VCALENDAR">
      <c:comp-filter name="VEVENT">
        <c:time-range start="{start.strftime('%Y%m%dT000000Z')}"
                      end="{end.strftime('%Y%m%dT000000Z')}"/>
      </c:comp-filter>
    </c:comp-filter>
  </c:filter>
</c:calendar-query>
"""
    try:
        # CalDAV REPORT через локальный HTTP-прокси (Clash) часто рвёт TLS.
        # Идём напрямую; при DNS/сети можно включить YANDEX_CALDAV_USE_PROXY=true.
        with httpx.Client(**_httpx_kwargs()) as client:
            r = client.request(
                "REPORT",
                url,
                content=body.encode("utf-8"),
                headers={
                    "Content-Type": "application/xml; charset=utf-8",
                    "Depth": "1",
                },
                auth=(user, password),
            )
        if r.status_code == 401:
            return {
                "ok": False,
                "error": "CalDAV 401: неверный логин/пароль приложения",
                "hint": (
                    "Создайте пароль приложения типа «Календарь» на "
                    "https://id.yandex.ru/security/app-passwords "
                    "и пропишите в YANDEX_CALDAV_APP_PASSWORD "
                    "(пароль для Почты может не подойти)."
                ),
                "url": url,
                "note": url_note,
            }
        if r.status_code >= 400:
            return {
                "ok": False,
                "error": f"CalDAV HTTP {r.status_code}",
                "body": r.text[:400],
                "url": url,
                "note": url_note,
            }
        root = ET.fromstring(r.text)
        events: list[dict[str, Any]] = []
        for resp in root.findall("d:response", _NS):
            href = resp.findtext("d:href", default="", namespaces=_NS)
            data_el = resp.find(".//c:calendar-data", _NS)
            ical = (data_el.text or "") if data_el is not None else ""
            if "BEGIN:VEVENT" not in ical:
                continue
            # берём первый VEVENT в ответе
            chunk = ical.split("BEGIN:VEVENT", 1)[-1].split("END:VEVENT", 1)[0]
            org = _ical_prop(chunk, "ORGANIZER") or ""
            if organizer and organizer.lower() not in org.lower():
                continue
            events.append(
                {
                    "href": href,
                    "summary": _ical_prop(chunk, "SUMMARY"),
                    "dtstart": _ical_prop(chunk, "DTSTART"),
                    "dtend": _ical_prop(chunk, "DTEND"),
                    "organizer": org,
                    "uid": _ical_prop(chunk, "UID"),
                }
            )
            if len(events) >= lim:
                break
        return {
            "ok": True,
            "count": len(events),
            "events": events,
            "url": url,
            "note": url_note,
            "criteria": {
                "date_from": date_from,
                "date_to": date_to,
                "organizer": organizer,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "url": url,
            "note": url_note,
            "hint": (
                "Если SSL EOF — Clash-прокси ломает CalDAV REPORT; "
                "оставьте YANDEX_CALDAV_USE_PROXY не true. "
                "Если 401 — пароль приложения «Календарь»."
            ),
        }
