"""Ручная проверка tools без агента/LLM.

  python -u task6_tools_smoke.py
  python -u task6_tools_smoke.py --tool calculator --arg expression=15*7

PowerShell (JSON неудобен — лучше --arg):
  .\\.venv\\Scripts\\python.exe -u task6_tools_smoke.py --tool yandex_mail_search --arg sender=yandex.ru --arg limit=3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from app.tools import TOOL_SCHEMAS, execute_tool, list_tool_names  # noqa: E402


def _parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Ожидалось key=value, получено: {item}")
        key, raw = item.split("=", 1)
        raw = raw.strip()
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            out[key] = int(raw)
        elif raw.lower() in {"true", "false"}:
            out[key] = raw.lower() == "true"
        else:
            out[key] = raw
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default="", help="Имя tool; пусто — список + демо time/calc")
    parser.add_argument(
        "--args",
        default="",
        help='JSON arguments (в PowerShell лучше не использовать; см. --arg)',
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        dest="arg_kv",
        help="Аргумент key=value (можно несколько раз). Удобно в PowerShell.",
    )
    ns = parser.parse_args()

    print("tools:", list_tool_names())
    print("schemas:", len(TOOL_SCHEMAS))
    if not ns.tool:
        for name, payload in (
            ("get_current_time", {}),
            ("calculator", {"expression": "(15*7)+3"}),
        ):
            print(f"\n=== {name} ===")
            print(json.dumps(execute_tool(name, payload), ensure_ascii=False, indent=2))
        print(
            "\nИнтеграции (нужен .env): "
            "yandex_mail_search / yandex_calendar_search / bitrix_tasks_search"
        )
        print(
            "Пример PowerShell:\n"
            "  .\\.venv\\Scripts\\python.exe -u task6_tools_smoke.py "
            "--tool yandex_mail_search --arg sender=yandex.ru --arg limit=3"
        )
        return 0

    payload: dict[str, Any] | str
    if ns.arg_kv:
        payload = _parse_kv(ns.arg_kv)
    elif ns.args:
        payload = ns.args
    else:
        payload = {}

    result = execute_tool(ns.tool, payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

