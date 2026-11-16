import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key or key.startswith("sk-your-key"):
        raise RuntimeError(
            "Задайте DEEPSEEK_API_KEY в файле .env "
            "(ключ создаётся на https://platform.deepseek.com → API Keys)."
        )
    return key


DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
# Дефолт API у DeepSeek ≈ high; для челленджа берём low и даём переопределять в запросе.
DEFAULT_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "low").strip().lower()
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# День 6 — предохранители цикла агента (в КОДЕ, не в system prompt)
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
AGENT_MAX_SAME_TOOL_CALLS = int(os.getenv("AGENT_MAX_SAME_TOOL_CALLS", "2"))

# День 7 — JSONL-сессия (история messages между рестартами)
AGENT_SESSION_DIR = os.getenv("AGENT_SESSION_DIR", "data/sessions").strip() or "data/sessions"
AGENT_SESSION_ID = os.getenv("AGENT_SESSION_ID", "default").strip() or "default"

# День 9 — управление контекстом (сжатие истории)
# full | summary_simple | summary_facts | summary_rules
AGENT_CONTEXT_MODE = os.getenv("AGENT_CONTEXT_MODE", "full").strip().lower() or "full"
AGENT_KEEP_LAST = int(os.getenv("AGENT_KEEP_LAST", "6"))
AGENT_SUMMARIZE_EVERY = int(os.getenv("AGENT_SUMMARIZE_EVERY", "10"))

# Оценка стоимости (off-peak USD / 1M tokens). Peak в коде ×2.
# Источник: https://api-docs.deepseek.com/quick_start/pricing/
DEEPSEEK_PRICE_FLASH_INPUT_HIT_1M = float(
    os.getenv("DEEPSEEK_PRICE_FLASH_INPUT_HIT_1M", "0.003")
)
DEEPSEEK_PRICE_FLASH_INPUT_MISS_1M = float(
    os.getenv("DEEPSEEK_PRICE_FLASH_INPUT_MISS_1M", "0.15")
)
DEEPSEEK_PRICE_FLASH_OUTPUT_1M = float(
    os.getenv("DEEPSEEK_PRICE_FLASH_OUTPUT_1M", "0.60")
)
DEEPSEEK_PRICE_PRO_INPUT_HIT_1M = float(
    os.getenv("DEEPSEEK_PRICE_PRO_INPUT_HIT_1M", "0.022")
)
DEEPSEEK_PRICE_PRO_INPUT_MISS_1M = float(
    os.getenv("DEEPSEEK_PRICE_PRO_INPUT_MISS_1M", "0.66")
)
DEEPSEEK_PRICE_PRO_OUTPUT_1M = float(
    os.getenv("DEEPSEEK_PRICE_PRO_OUTPUT_1M", "1.98")
)
# Курс для UI (не биржевой фид) — поправьте под себя
USD_RUB_RATE = float(os.getenv("USD_RUB_RATE", "95"))


# День 5 — OpenRouter (сравнение моделей; ключ в .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "ai-challenge-day5").strip()


def require_openrouter_api_key() -> str:
    key = OPENROUTER_API_KEY
    if not key or key.startswith("sk-or-v1-your-key"):
        raise RuntimeError(
            "Задайте OPENROUTER_API_KEY в файле .env "
            "(ключ: https://openrouter.ai/keys)."
        )
    return key


# Локальный MITM-прокси (Clash/V2Ray и т.п.) подменяет TLS-сертификат.
# VS Code это обходит через http.proxyStrictSSL=false; Python — нет.
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
SSL_VERIFY = _as_bool(os.getenv("SSL_VERIFY"), default=True)
# Путь к CA прокси, если хотите verify=true, но через MITM.
SSL_CA_BUNDLE = os.getenv("SSL_CA_BUNDLE", "").strip()
