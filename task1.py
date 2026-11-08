"""День 1: вызвать локальный /chat и вывести ответ модели."""

from __future__ import annotations

import argparse
import json
import sys

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/chat"
DEFAULT_PROMPT = "Скажи одним словом: готов"


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Challenge — день 1")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Текст промпта")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL локального endpoint")
    args = parser.parse_args()

    # Чтобы кириллица в консоли Windows не превращалась в Ð¾Ñ...
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        response = httpx.post(
            args.url,
            json={"prompt": args.prompt},
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print(f"Не удалось подключиться к {args.url}. Запущен ли uvicorn?", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1

    data = response.json()
    answer = data.get("answer", "")
    print(answer)
    print("---")
    print(f"model: {data.get('model')}")
    print(f"usage: {json.dumps(data.get('usage'), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
