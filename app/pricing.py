"""Оценка стоимости DeepSeek → USD / RUB.

Цены: https://api-docs.deepseek.com/quick_start/pricing/
Peak (UTC, Пн–Пт, без китайских праздников — праздники не моделируем):
  01:00–04:00 и 06:00–10:00 UTC.
Без cache hit/miss в usage считаем весь input как cache miss (верхняя оценка).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import (
    DEEPSEEK_MODEL,
    DEEPSEEK_PRICE_FLASH_INPUT_HIT_1M,
    DEEPSEEK_PRICE_FLASH_INPUT_MISS_1M,
    DEEPSEEK_PRICE_FLASH_OUTPUT_1M,
    DEEPSEEK_PRICE_PRO_INPUT_HIT_1M,
    DEEPSEEK_PRICE_PRO_INPUT_MISS_1M,
    DEEPSEEK_PRICE_PRO_OUTPUT_1M,
    USD_RUB_RATE,
)


@dataclass(frozen=True)
class ModelRates:
    input_hit_per_1m: float
    input_miss_per_1m: float
    output_per_1m: float


def is_peak_utc(when: datetime | None = None) -> bool:
    dt = when or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    if dt.weekday() >= 5:  # Sat/Sun
        return False
    h = dt.hour
    return (1 <= h < 4) or (6 <= h < 10)


def rates_for_model(model: str | None, *, peak: bool | None = None) -> ModelRates:
    name = (model or DEEPSEEK_MODEL or "deepseek-flash").strip().lower()
    peak = is_peak_utc() if peak is None else peak
    # В .env храним off-peak; peak = ×2 (как у DeepSeek).
    mult = 2.0 if peak else 1.0
    if "pro" in name:
        return ModelRates(
            input_hit_per_1m=DEEPSEEK_PRICE_PRO_INPUT_HIT_1M * mult,
            input_miss_per_1m=DEEPSEEK_PRICE_PRO_INPUT_MISS_1M * mult,
            output_per_1m=DEEPSEEK_PRICE_PRO_OUTPUT_1M * mult,
        )
    return ModelRates(
        input_hit_per_1m=DEEPSEEK_PRICE_FLASH_INPUT_HIT_1M * mult,
        input_miss_per_1m=DEEPSEEK_PRICE_FLASH_INPUT_MISS_1M * mult,
        output_per_1m=DEEPSEEK_PRICE_FLASH_OUTPUT_1M * mult,
    )


def _split_input_tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    """(hit, miss). Если cache-полей нет — все prompt_tokens = miss."""
    if not usage:
        return 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is not None or miss is not None:
        hit_i = int(hit or 0)
        miss_i = int(miss or 0)
        if hit_i + miss_i == 0 and prompt:
            return 0, prompt
        return hit_i, miss_i
    return 0, prompt


def estimate_usage_cost(
    usage: dict[str, Any] | None,
    *,
    model: str | None = None,
    when: datetime | None = None,
    usd_rub: float | None = None,
) -> dict[str, Any]:
    peak = is_peak_utc(when)
    rates = rates_for_model(model, peak=peak)
    hit, miss = _split_input_tokens(usage)
    out = int((usage or {}).get("completion_tokens") or 0)
    usd = (
        hit * rates.input_hit_per_1m / 1_000_000
        + miss * rates.input_miss_per_1m / 1_000_000
        + out * rates.output_per_1m / 1_000_000
    )
    rate = float(usd_rub if usd_rub is not None else USD_RUB_RATE)
    rub = usd * rate
    return {
        "usd": round(usd, 8),
        "rub": round(rub, 4),
        "usd_rub_rate": rate,
        "peak": peak,
        "model": model or DEEPSEEK_MODEL,
        "input_hit_tokens": hit,
        "input_miss_tokens": miss,
        "output_tokens": out,
        "rates_per_1m": {
            "input_hit": rates.input_hit_per_1m,
            "input_miss": rates.input_miss_per_1m,
            "output": rates.output_per_1m,
        },
        "note": (
            "Оценка по прайсу DeepSeek; без cache-полей в usage input = miss. "
            "RUB = USD × USD_RUB_RATE из .env."
        ),
    }
