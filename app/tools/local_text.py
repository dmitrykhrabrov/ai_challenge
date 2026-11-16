"""Tool: прочитать локальный .txt (для дня 8 — можно вернуть полный текст в контекст)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.epub_text import approx_tokens

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_local_text",
        "description": (
            "Прочитать локальный текстовый файл (.txt, .json, .jsonl, .md). "
            "По умолчанию — статистика и preview с начала. "
            "tail_chars — кусок с конца файла (ответ в длинном JSONL). "
            "include_full_text=true — весь файл в tool result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к .txt, .json, .jsonl или .md",
                },
                "tail_chars": {
                    "type": "integer",
                    "description": "Вернуть последние N символов (0 = не резать с конца)",
                },
                "preview_chars": {
                    "type": "integer",
                    "description": "Длина preview (по умолчанию 1500)",
                },
                "include_full_text": {
                    "type": "boolean",
                    "description": "Вернуть полный text в ответе tool",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Прочитать не больше N символов с начала (0 = весь файл)",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


def read_local_text(
    path: str,
    preview_chars: int = 1500,
    include_full_text: bool = False,
    max_chars: int = 0,
    tail_chars: int = 0,
) -> dict[str, Any]:
    src = Path(path).expanduser()
    if not src.is_file():
        return {"ok": False, "error": f"Файл не найден: {src}"}
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = src.read_text(encoding="utf-8", errors="replace")
    tailed = bool(tail_chars and int(tail_chars) > 0)
    if tailed:
        text = text[-int(tail_chars) :]
    elif max_chars and max_chars > 0:
        text = text[: int(max_chars)]
    chars = len(text)
    preview_n = max(0, int(preview_chars or 0))
    out: dict[str, Any] = {
        "ok": True,
        "path": str(src.resolve()),
        "chars": chars,
        "approx_tokens": approx_tokens(chars),
        "preview": text[:preview_n],
        "preview_chars": min(preview_n, chars),
        "include_full_text": bool(include_full_text),
    }
    if include_full_text or tailed:
        out["text"] = text
        if include_full_text:
            out["warning"] = "Полный текст в tool result раздувает контекст / может превысить лимит."
    return out
