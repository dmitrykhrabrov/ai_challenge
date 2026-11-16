"""День 10: окно в токенах, факты из размеченного текста, версии договора.

Сводка дня 9 здесь не вызывается. Факты пишет код по строкам «Ключ: значение»
внутри блоков «Карточка» и «Договор». Версия договора — снимок фактов, не копия чата.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from app.config import AGENT_SESSION_DIR
from app import session_store

_LOCK = threading.Lock()
_FACT_LINE = re.compile(r"^([^:\n]{2,80}):\s*(\S.*?)\s*$")
_HEADER = re.compile(r"^(Карточка|Договор)\b", re.IGNORECASE)
_PUBLISH = re.compile(r"выпусти.{0,40}(договор|верси)", re.IGNORECASE)
_AMEND = re.compile(r"Текст правки:\s*(.+)", re.IGNORECASE | re.DOTALL)
_HANDOFF = re.compile(r"передай аналитику", re.IGNORECASE)
_ASK_MANAGER = re.compile(
    r"прошу менеджера|спроси менеджера|уточни у менеджера",
    re.IGNORECASE,
)

ROLE_PROMPTS = {
    "manager": (
        "Роль: менеджер поставщика. Менять договор можешь только ты. "
        "Юрист и аналитик видят выпущенную версию, не твою переписку. "
        "Чужой журнал целиком не пересказывай."
    ),
    "lawyer": (
        "Роль: юрист. Договор сам не меняешь, просишь менеджера согласовать формулировку. "
        "Если в сообщении пользователя уже есть просьба менеджеру, письмо уже доставлено: "
        "ответь по короткому ответу менеджера и закреплённой версии. Тот же текст в ask_agent не повторяй."
    ),
    "analyst": (
        "Роль: аналитик интеграции. Отвечай по закреплённой версии договора. "
        "Новую версию применяй только после сообщения менеджера о передаче."
    ),
}


def _room_path() -> Path:
    return _store_path().parent / "room.jsonl"


def reset_contracts() -> None:
    """Новый дубль ролика начинает версии с 1, без прошлых закреплений."""
    with _LOCK:
        _save_store({"versions": [], "roles": {}})


def reset_room() -> None:
    path = _room_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def log_room(
    *,
    kind: str,
    sender: str,
    to: str = "",
    text: str = "",
    answer: str = "",
    version: int | None = None,
) -> dict[str, Any]:
    """Лента писем. В запрос модели не подставляется."""
    from datetime import datetime, timezone

    item = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "kind": kind,
        "from": sender,
        "to": to,
        "text": (text or "")[:1500],
        "answer": (answer or "")[:1500],
        "version": version,
    }
    line = json.dumps(item, ensure_ascii=False) + "\n"
    with _LOCK:
        path = _room_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return item


def load_room(limit: int = 40) -> list[dict[str, Any]]:
    path = _room_path()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-max(1, int(limit)) :]


def plain_reply(text: str) -> str:
    """Ответ письма без разметки вызова инструмента."""
    raw = (text or "").strip()
    if "tool_call" not in raw and "DSML" not in raw and "ask_agent" not in raw:
        return raw
    match = re.search(r'name="message"[^>]*>(.*?)</', raw, re.DOTALL)
    if not match:
        match = re.search(r'"message"\s*:\s*"(.*?)"\s*}', raw, re.DOTALL)
    if match:
        extracted = match.group(1).replace("\\n", "\n").strip()
        extracted = re.sub(r"<[^>]+>", "", extracted).strip()
        if extracted:
            return extracted
    return "Ответ не получился текстом: в письме снова был вызов инструмента."


def outbound_manager_ask(text: str) -> str:
    if _ASK_MANAGER.search(text or ""):
        return (text or "").strip()
    return ""


def _store_path() -> Path:
    root = Path(AGENT_SESSION_DIR)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    root.mkdir(parents=True, exist_ok=True)
    return root / "contracts.json"


def _load_store() -> dict[str, Any]:
    path = _store_path()
    if not path.is_file():
        return {"versions": [], "roles": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"versions": [], "roles": {}}
    data.setdefault("versions", [])
    data.setdefault("roles", {})
    return data


def _save_store(data: dict[str, Any]) -> None:
    _store_path().write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 2)


def parse_structured_facts(text: str) -> dict[str, str]:
    """Строки «Ключ: значение» только внутри блоков Карточка / Договор."""
    facts: dict[str, str] = {}
    in_block = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _HEADER.match(line):
            in_block = True
            continue
        if not in_block:
            continue
        match = _FACT_LINE.match(line)
        if not match:
            in_block = False
            continue
        facts[match.group(1).strip()] = match.group(2).strip()
    amend = _AMEND.search(text or "")
    if amend:
        facts["Субподряд"] = " ".join(amend.group(1).split())
    return facts


def latest_facts(events: list[dict[str, Any]]) -> dict[str, str]:
    found: dict[str, str] = {}
    for event in events:
        if event.get("type") == "facts":
            items = event.get("items") or {}
            if isinstance(items, dict):
                found = {str(k): str(v) for k, v in items.items()}
    return found


def session_role(events: list[dict[str, Any]]) -> str:
    role = ""
    for event in events:
        if event.get("type") == "agent_role" and event.get("role"):
            role = str(event["role"])
    return role


def pinned_version(events: list[dict[str, Any]]) -> int | None:
    version: int | None = None
    for event in events:
        if event.get("type") == "contract_pin":
            raw = event.get("version")
            version = int(raw) if raw else None
    return version


def merge_facts(session_id: str, updates: dict[str, str]) -> dict[str, str]:
    if not updates:
        events = session_store.load_events(session_id)
        return latest_facts(events)
    events = session_store.load_events(session_id)
    merged = latest_facts(events)
    merged.update(updates)
    session_store.append_event(
        session_id,
        {"type": "facts", "items": merged},
    )
    return merged


def register_role(session_id: str, role: str) -> None:
    role = (role or "").strip().lower()
    if role not in ROLE_PROMPTS:
        return
    session_store.append_event(
        session_id,
        {"type": "agent_role", "role": role},
    )
    with _LOCK:
        data = _load_store()
        data["roles"][role] = session_id
        _save_store(data)
    if role in {"lawyer", "analyst"}:
        latest = latest_version()
        if latest:
            number = int(latest["version"])
            pin_version(session_id, number)
            log_room(
                kind="pin",
                sender="manager",
                to=role,
                text=f"Закреплена версия {number}. Журнал менеджера не приложен.",
                version=number,
            )


def latest_version() -> dict[str, Any] | None:
    with _LOCK:
        versions = _load_store().get("versions") or []
    if not versions:
        return None
    return versions[-1]


def version_by_number(number: int) -> dict[str, Any] | None:
    with _LOCK:
        versions = list(_load_store().get("versions") or [])
    for item in versions:
        if int(item.get("version") or 0) == int(number):
            return item
    return None


def publish_version(session_id: str, facts: dict[str, str]) -> dict[str, Any]:
    with _LOCK:
        data = _load_store()
        versions = data.setdefault("versions", [])
        number = len(versions) + 1
        item = {
            "version": number,
            "facts": dict(facts),
            "source_session": session_id,
        }
        versions.append(item)
        data["roles"]["manager"] = session_id
        _save_store(data)
    session_store.append_event(
        session_id,
        {"type": "contract_published", "version": number},
    )
    log_room(
        kind="published",
        sender="manager",
        text="Выпущена версия договора. Закрепления других ролей не сдвинуты.",
        version=number,
    )
    return item


def pin_version(session_id: str, version: int) -> None:
    session_store.append_event(
        session_id,
        {"type": "contract_pin", "version": int(version)},
    )


def role_session(role: str) -> str:
    with _LOCK:
        return str((_load_store().get("roles") or {}).get(role) or "")


def facts_for_request(session_id: str) -> tuple[dict[str, str], int | None]:
    events = session_store.load_events(session_id)
    version = pinned_version(events)
    if version:
        item = version_by_number(version)
        if item:
            return dict(item.get("facts") or {}), version
    return latest_facts(events), None


def deliver_to_analyst(text: str, version: int | None = None) -> dict[str, Any]:
    target = role_session("analyst")
    if not target:
        return {"ok": False, "error": "чат аналитика ещё не открыт"}
    item = version_by_number(version) if version else latest_version()
    if not item:
        return {"ok": False, "error": "нет выпущенной версии договора"}
    number = int(item["version"])
    pin_version(target, number)
    notice = (
        "Сообщение менеджера. Передана версия договора "
        f"{number}.\n{text.strip()}"
    )
    session_store.append_message_event(
        target,
        {"role": "user", "content": notice},
    )
    log_room(
        kind="handoff",
        sender="manager",
        to="analyst",
        text=notice,
        version=number,
    )
    return {"ok": True, "session_id": target, "version": number}


def apply_user_message(session_id: str, text: str) -> dict[str, Any]:
    """Разбор размеченного текста и публикация версии. Без вызова модели."""
    updates = parse_structured_facts(text)
    facts = merge_facts(session_id, updates)
    published = None
    role = session_role(session_store.load_events(session_id))
    if role == "manager" and _PUBLISH.search(text or ""):
        published = publish_version(session_id, facts)
    handed = None
    if _HANDOFF.search(text or ""):
        version = int(published["version"]) if published else None
        handed = deliver_to_analyst(text, version)
    return {
        "facts": facts,
        "published": published["version"] if published else None,
        "handoff": handed,
    }


def message_tokens(message: dict[str, Any]) -> int:
    parts = [str(message.get("content") or "")]
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        parts.append(str(fn.get("name") or ""))
        parts.append(str(fn.get("arguments") or ""))
    return approx_tokens("\n".join(parts)) + 4


def tail_by_tokens(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Хвост с конца, пока влезает в budget. Последнее сообщение берём даже если оно длиннее."""
    if not messages:
        return []
    budget = max(1, int(budget))
    chosen: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        cost = message_tokens(message)
        if chosen and used + cost > budget:
            break
        chosen.append(message)
        used += cost
    chosen.reverse()
    while len(chosen) > 1 and chosen[0].get("role") == "tool":
        chosen = chosen[1:]
    return chosen


def tail_by_messages(messages: list[dict[str, Any]], keep_last: int) -> list[dict[str, Any]]:
    from app.context_compress import raw_tail_start

    if not messages:
        return []
    start = raw_tail_start(messages, max(1, int(keep_last)))
    return messages[start:]


def format_facts_message(facts: dict[str, str], version: int | None) -> dict[str, str] | None:
    if not facts and not version:
        return None
    lines = [f"{key}: {value}" for key, value in facts.items()]
    head = f"[FACTS version={version}]" if version else "[FACTS]"
    return {
        "role": "system",
        "content": head + "\n" + "\n".join(lines) + "\n[END FACTS]",
    }


def role_suffix(events: list[dict[str, Any]]) -> str:
    role = session_role(events)
    return ROLE_PROMPTS.get(role, "")


async def prepare_window_messages(
    session_id: str,
    *,
    system_text: str,
    mode: str,
    keep_last: int,
    tail_tokens: int | None,
):
    from app.context_compress import ContextPrepareResult, _dialog_api_messages

    events = session_store.load_events(session_id)
    suffix = role_suffix(events)
    system = system_text if not suffix else f"{system_text}\n\n{suffix}"
    dialog, _seqs = _dialog_api_messages(events, after_seq=-1)
    if tail_tokens and int(tail_tokens) > 0:
        tail = tail_by_tokens(dialog, int(tail_tokens))
        budget_note = f"tail_tokens={int(tail_tokens)}"
    else:
        tail = tail_by_messages(dialog, keep_last)
        budget_note = f"keep_last={keep_last}"
    facts: dict[str, str] = {}
    version: int | None = None
    if mode == "sticky_facts":
        facts, version = facts_for_request(session_id)
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    fact_msg = format_facts_message(facts, version) if mode == "sticky_facts" else None
    if fact_msg:
        out.append(fact_msg)
    out.extend(tail)
    preview = "\n".join(f"{key}: {value}" for key, value in facts.items())
    if version:
        preview = f"версия {version}\n{preview}".strip()
    return ContextPrepareResult(
        messages=out,
        mode=mode,
        compressed=False,
        raw_tail_count=len(tail),
        uncovered_before=len(dialog),
        note=(
            f"mode={mode}; {budget_note}; tail={len(tail)}/{len(dialog)}; "
            f"facts={len(facts)}; version={version}"
        ),
        summary_text=preview or None,
    )
