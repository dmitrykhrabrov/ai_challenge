"""CLI: EPUB → UTF-8 .txt (+ оценка токенов).

  .\\.venv\\Scripts\\python.exe -u epub_to_txt.py "C:\\path\\book.epub"
  .\\.venv\\Scripts\\python.exe -u epub_to_txt.py book.epub --out data\\artamene.txt
  .\\.venv\\Scripts\\python.exe -u epub_to_txt.py book.epub --max-chars 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.tools.epub_text import approx_tokens, extract_epub_text


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="EPUB → text")
    parser.add_argument("epub", type=Path, help="Путь к .epub")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Выходной .txt")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Обрезать после N символов (0 = весь текст)",
    )
    args = parser.parse_args()

    src = args.epub.expanduser()
    if not src.is_file():
        print(f"Файл не найден: {src}", file=sys.stderr)
        return 1

    print(f"reading: {src}", flush=True)
    text = extract_epub_text(src, max_chars=args.max_chars)
    dest = args.out.expanduser() if args.out else src.with_suffix(".txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")

    chars = len(text)
    print(f"out: {dest}")
    print(f"chars: {chars}")
    print(f"approx_tokens: {approx_tokens(chars)} (chars/3.5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
