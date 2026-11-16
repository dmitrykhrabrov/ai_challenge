"""Tool + библиотека: EPUB → plain text (stdlib, без DRM)."""

from __future__ import annotations

import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "epub_to_text",
        "description": (
            "Извлечь текст из локального EPUB-файла в .txt. "
            "По умолчанию в ответ модели отдаёт только статистику и превью "
            "(не весь текст — он огромный). Полный текст пишется на диск. "
            "Для дня 8 / переполнения контекста: include_full_text=true "
            "(осторожно: может превысить лимит модели)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "epub_path": {
                    "type": "string",
                    "description": "Абсолютный или относительный путь к .epub",
                },
                "out_path": {
                    "type": "string",
                    "description": "Куда сохранить .txt (опционально)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Ограничить извлечение первыми N символами (0 = без лимита)",
                },
                "preview_chars": {
                    "type": "integer",
                    "description": "Сколько символов вернуть в preview (по умолчанию 1500)",
                },
                "include_full_text": {
                    "type": "boolean",
                    "description": "Вернуть весь текст в ответе tool (опасно для контекста)",
                },
            },
            "required": ["epub_path"],
            "additionalProperties": False,
        },
    },
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "head"}:
            self._skip += 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "head"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _is_html_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".xhtml", ".html", ".htm", ".xml")) and "meta-inf" not in lower


def _sort_key(name: str) -> tuple[int, str]:
    # page_123.html → числовой порядок, иначе имя
    m = re.search(r"(\d+)", Path(name).stem)
    if m:
        return (0, f"{int(m.group(1)):08d}")
    return (1, name.lower())


def extract_epub_text(epub_path: Path, *, max_chars: int = 0) -> str:
    if not epub_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {epub_path}")
    parts: list[str] = []
    total = 0
    with zipfile.ZipFile(epub_path, "r") as zf:
        names = sorted(
            (n for n in zf.namelist() if _is_html_name(n) and not n.endswith("/")),
            key=_sort_key,
        )
        if not names:
            raise ValueError("В EPUB нет HTML/XHTML страниц")
        for name in names:
            raw = zf.read(name)
            try:
                html = raw.decode("utf-8")
            except UnicodeDecodeError:
                html = raw.decode("latin-1", errors="replace")
            parser = _HTMLTextExtractor()
            try:
                parser.feed(html)
                parser.close()
            except Exception:  # noqa: BLE001
                continue
            chunk = parser.text()
            if not chunk:
                continue
            parts.append(chunk)
            total += len(chunk)
            if max_chars > 0 and total >= max_chars:
                break
    text = "\n\n".join(parts)
    if max_chars > 0:
        text = text[:max_chars]
    return text


def approx_tokens(chars: int) -> int:
    # грубо для смешанного FR/EN/RU: ~3–4 символа на токен; берём /3.5
    return max(0, int(chars / 3.5))


def epub_to_text(
    epub_path: str,
    out_path: str | None = None,
    max_chars: int = 0,
    preview_chars: int = 1500,
    include_full_text: bool = False,
) -> dict[str, Any]:
    src = Path(epub_path).expanduser()
    try:
        text = extract_epub_text(src, max_chars=int(max_chars or 0))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "epub_path": str(src)}

    if out_path:
        dest = Path(out_path).expanduser()
    else:
        dest = src.with_suffix(".txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")

    chars = len(text)
    preview_n = max(0, int(preview_chars or 0))
    result: dict[str, Any] = {
        "ok": True,
        "epub_path": str(src.resolve()),
        "out_path": str(dest.resolve()),
        "chars": chars,
        "approx_tokens": approx_tokens(chars),
        "approx_tokens_note": "оценка chars/3.5, не usage API",
        "preview": text[:preview_n],
        "preview_chars": min(preview_n, chars),
        "truncated": bool(max_chars and chars >= int(max_chars)),
        "include_full_text": bool(include_full_text),
    }
    if include_full_text:
        result["text"] = text
        result["warning"] = (
            "Полный текст в tool result раздует контекст и может превысить лимит модели."
        )
    return result
