"""День 2: один сценарий — разные уровни контроля ответа.

Идея stop vs max_tokens:
  Просим нумерованный список 1..10.
  - max_tokens режет «вслепую» по бюджету токенов (часто mid-word / mid-item).
  - stop=["\\n5."] обрывает, когда модель собралась писать пункт 5 —
    в ответе остаются пункты 1..4 целиком. Stop работает только если
    в формате ответа есть предсказуемый маркер.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx

URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 120.0

# Базовый сценарий: нумерованный список — удобно для max_tokens / stop.
LIST_PROMPT = """Перечисли ровно 10 крупных городов России.
Формат СТРОГО такой (без вступления и заключения):
1. Город
2. Город
...
10. Город"""

# Логика для сравнения reasoning.
# Классическая «ловушка»: без рассуждения часто отвечают 0.10, верно — 0.05.
LOGIC_PROMPT = """Бита и мяч вместе стоят 1.10 доллара.
Бита стоит на 1 доллар дороже мяча.
Сколько стоит мяч в долларах?

Сначала в уме проверь уравнение, не угадывай «круглое» число.
Ответь одной строкой строго в формате: ОТВЕТ=<число с точкой>"""

BOOK = {
    "title": "Мастер и Маргарита",
    "year": 1967,
    "tags": ["роман", "сатира"],
}

XML_PROMPT = f"""Опиши книгу данными ниже ТОЛЬКО в XML, без markdown и пояснений.
Данные: title={BOOK['title']!r}, year={BOOK['year']}, tags={BOOK['tags']}.
Корень: <book>, внутри <title>, <year>, <tags><tag>...</tag></tags>."""

JSON_FREE_PROMPT = f"""Опиши книгу данными ниже ТОЛЬКО валидным JSON-объектом, без markdown.
Данные: title={BOOK['title']!r}, year={BOOK['year']}, tags={BOOK['tags']}.
Поля на твой вкус, но смысл сохрани."""

JSON_SCHEMA = {
    "type": "object",
    "required": ["title", "year", "tags"],
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "additionalProperties": False,
}

JSON_STRICT_PROMPT = f"""Верни ТОЛЬКО валидный JSON-объект по схеме (без markdown):
{json.dumps(JSON_SCHEMA, ensure_ascii=False)}
Данные книги: title={BOOK['title']!r}, year={BOOK['year']}, tags={BOOK['tags']}.
Не добавляй поля вне схемы."""


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def call_chat(**payload: Any) -> dict[str, Any]:
    response = httpx.post(URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show(label: str, payload: dict[str, Any], data: dict[str, Any]) -> None:
    print(f"\n--- {label} ---")
    print("request:", json.dumps(payload, ensure_ascii=False, indent=2))
    print("finish_reason:", data.get("finish_reason"))
    print("usage:", json.dumps(data.get("usage"), ensure_ascii=False))
    print("answer:")
    print(data.get("answer", ""))


def try_parse_json(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        return False, str(exc)


def validate_schema(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["корень должен быть object"]
    for key in ("title", "year", "tags"):
        if key not in obj:
            errors.append(f"нет поля {key}")
    if "title" in obj and not isinstance(obj["title"], str):
        errors.append("title должен быть string")
    if "year" in obj and not isinstance(obj["year"], int):
        errors.append("year должен быть integer")
    if "tags" in obj:
        if not isinstance(obj["tags"], list) or not obj["tags"]:
            errors.append("tags должен быть непустым array")
        elif any(not isinstance(x, str) for x in obj["tags"]):
            errors.append("элементы tags должны быть string")
    extra = set(obj) - {"title", "year", "tags"}
    if extra:
        errors.append(f"лишние поля: {sorted(extra)}")
    return errors


def run_list_controls() -> None:
    section("A. Список городов: baseline / max_tokens / stop")
    # temperature=0: убираем случайность, чтобы сравнивать именно max_tokens/stop/формат.
    # Разные temperature — тема дня 4.

    baseline = {
        "prompt": LIST_PROMPT,
        "reasoning_effort": "none",
        "temperature": 0,
    }
    show("A1 baseline (без лимитов)", baseline, call_chat(**baseline))

    capped = {
        **baseline,
        "max_tokens": 40,
    }
    show(
        "A2 max_tokens=40 (слепой обрез; смотри finish_reason=length)",
        capped,
        call_chat(**capped),
    )

    # Маркер следующего пункта — обрыв перед 5-м элементом.
    stopped = {
        **baseline,
        "stop": ["\n5.", "5."],
    }
    show(
        "A3 stop=['\\n5.','5.'] → ожидаем пункты 1..4 целиком",
        stopped,
        call_chat(**stopped),
    )


def run_reasoning() -> None:
    section("B. Ловушка бита+мяч × reasoning_effort (верно ОТВЕТ=0.05)")
    print(
        "Ожидание: none чаще даёт 0.10 (быстрый ответ-ошибка);\n"
        "low/high чаще доходят до 0.05, но тратят reasoning_tokens.\n"
        "temperature=0 — чтобы сравнивать effort, а не рандом."
    )
    for effort in ("none", "low", "high"):
        payload = {
            "prompt": LOGIC_PROMPT,
            "reasoning_effort": effort,
            "temperature": 0,
            # high может жечь много reasoning_tokens — запас побольше
            "max_tokens": 1024,
        }
        data = call_chat(**payload)
        show(f"B reasoning_effort={effort}", payload, data)
        answer = (data.get("answer") or "").strip()
        ok = "0.05" in answer.replace(",", ".")
        print("проверка (есть 0.05):", "верно" if ok else "скорее ошибка/другой формат")


def run_formats() -> None:
    section("C. Один объект книги: XML / JSON free / JSON+схема в промпте")

    xml_payload = {
        "prompt": XML_PROMPT,
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": 300,
    }
    xml_data = call_chat(**xml_payload)
    show("C1 XML (только промпт, без response_format)", xml_payload, xml_data)
    print("parse hint: это текст; API XML не валидирует.")

    json_free = {
        "prompt": JSON_FREE_PROMPT,
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": 300,
        "response_format": "json_object",
    }
    free_data = call_chat(**json_free)
    show("C2 json_object без схемы в промпте", json_free, free_data)
    ok, parsed = try_parse_json(free_data.get("answer", ""))
    print("json.loads:", "OK" if ok else f"FAIL ({parsed})")
    if ok:
        print("keys:", list(parsed) if isinstance(parsed, dict) else type(parsed))

    json_strict = {
        "prompt": JSON_STRICT_PROMPT,
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": 300,
        "response_format": "json_object",
    }
    strict_data = call_chat(**json_strict)
    show("C3 json_object + схема в промпте + валидация у нас", json_strict, strict_data)
    ok, parsed = try_parse_json(strict_data.get("answer", ""))
    print("json.loads:", "OK" if ok else f"FAIL ({parsed})")
    if ok:
        errs = validate_schema(parsed)
        print("schema:", "OK" if not errs else f"FAIL {errs}")


def main() -> int:
    _utf8_stdout()
    print("AI Challenge — день 2")
    print("Эндпоинт:", URL)
    print(
        "\nКак читать stop:\n"
        "  Вы заранее договариваетесь о формате ответа (1.\\n2.\\n3....).\n"
        "  stop — это НЕ «оставь N элементов магически», а «остановись, когда\n"
        "  встретится эта подстрока». Чтобы оставить 4 пункта из 10, ставим\n"
        "  stop на маркер пятого пункта. max_tokens так не умеет — он режет\n"
        "  по счётчику токенов, часто посередине строки."
    )
    try:
        run_list_controls()
        run_reasoning()
        run_formats()
    except httpx.ConnectError:
        print(f"\nНе удалось подключиться к {URL}. Запущен ли uvicorn?", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"\nHTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1

    section("Готово")
    print("Сравните finish_reason, целостность списка, reasoning_tokens и parse/schema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
