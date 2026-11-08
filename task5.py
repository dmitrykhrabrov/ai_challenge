"""День 5. Версии моделей — small / mid / flagship.

Сетка:
  • 5 самых сложных олимпиадных задач из дня 3
  • пробы A/B/C из фазы 2 дня 3
      A) роль «только перебор» vs subsets_40
      B) JSON Schema в промпте vs response_format=xml
      C) противоречие в условии (3 бага) без подсказки ПРОТИВОРЕЧИЕ

По умолчанию:
  small/mid — OpenRouter free
  flagship — deepseek-v4-pro (DEEPSEEK_API_KEY)

Запуск:
  python -u task5.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from task3 import (
    BUG_TASKS,
    GRADE3_BRUTE_ONLY,
    HARD_TASK,
    JSON_SCHEMA_FOR_B,
    TASKS,
    Task,
    answers_match,
    final_tail,
    normalize_answer,
    normalize_hard,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "ai-challenge-day5")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
TIMEOUT = 180.0
DEFAULT_CONCURRENCY = 3
MAX_TOKENS = 2000
MAX_TOKENS_A = 8000  # как в task3 COMMON_CHAT — иначе A сразу length
TEMPERATURE = 0.0

# 5 самых сложных по дню 5 (токены/смысл): без meet/paper/queue
OLYMP_HARD5_IDS = ("candies", "milestones", "masses", "bikes", "balls")

DEEPSEEK_V4_PRO_PRICE_PROMPT_PER_1M = 0.66
DEEPSEEK_V4_PRO_PRICE_COMPLETION_PER_1M = 1.98

DEFAULT_MODELS: list[dict[str, str]] = [
    {
        "tier": "small",
        "label": "маленькая",
        "provider": "openrouter",
        "model_id": "liquid/lfm-2.5-2.6b:free",
    },
    {
        "tier": "mid",
        "label": "средняя",
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3.5-lightning:free",
    },
    {
        "tier": "flagship",
        "label": "флагман",
        "provider": "deepseek",
        "model_id": "deepseek-v4-pro",
    },
]


@dataclass
class ModelSpec:
    tier: str
    label: str
    model_id: str
    provider: str = "openrouter"
    price_prompt_per_token: float = 0.0
    price_completion_per_token: float = 0.0
    price_note: str = ""


@dataclass
class CallJob:
    suite: str  # olympiad | A | B | C
    task_id: str
    prompt: str
    expected: str
    max_tokens: int = MAX_TOKENS
    response_format: str | None = None
    score_mode: str = "olympiad"  # olympiad | role_a | format_b | logic_c
    trap: str | None = None


@dataclass
class TaskRun:
    suite: str
    task_id: str
    model_id: str
    tier: str
    ok: bool
    expected: str
    got: str
    elapsed_sec: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None
    answer_preview: str = ""
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAgg:
    tier: str
    label: str
    model_id: str
    suite: str
    correct: int = 0
    total: int = 0
    elapsed_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    runs: list[TaskRun] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.total, 3) if self.total else 0.0


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def require_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("sk-or-v1-your-key"):
        raise RuntimeError(
            "Задайте OPENROUTER_API_KEY в .env (https://openrouter.ai/keys)."
        )
    return key


def require_deepseek_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key or key.startswith("sk-your-key"):
        raise RuntimeError(
            "Задайте DEEPSEEK_API_KEY в .env (https://platform.deepseek.com → API Keys)."
        )
    return key


def _client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": TIMEOUT}
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
    if proxy:
        kwargs["proxy"] = proxy
    ca = os.getenv("SSL_CA_BUNDLE", "").strip()
    verify_raw = os.getenv("SSL_VERIFY", "true").strip().lower()
    if ca:
        kwargs["verify"] = ca
    elif verify_raw in {"0", "false", "no", "off"}:
        kwargs["verify"] = False
    return kwargs


def or_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }


def ds_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def olympiad_match(expected: str, got: str) -> bool:
    """Как answers_match, но пустой got всегда fail."""
    if not (got or "").strip():
        return False
    return answers_match(expected, got)


def build_jobs(olymp: list[Task]) -> list[CallJob]:
    jobs: list[CallJob] = []
    for t in olymp:
        jobs.append(
            CallJob(
                suite="olympiad",
                task_id=t.tid,
                prompt=final_tail(t.text),
                expected=t.expected,
                max_tokens=MAX_TOKENS,
                score_mode="olympiad",
            )
        )

    # A — роль vs невозможный перебор
    jobs.append(
        CallJob(
            suite="A",
            task_id=HARD_TASK["tid"],
            prompt=(
                f"{GRADE3_BRUTE_ONLY}\n\n"
                f"Задача:\n{HARD_TASK['text']}\n\n"
                "В конце одной строкой: ОТВЕТ=<значение>"
            ),
            expected=HARD_TASK["expected"],
            max_tokens=MAX_TOKENS_A,
            score_mode="role_a",
        )
    )

    # B — JSON Schema в тексте vs response_format=xml
    schema_txt = json.dumps(JSON_SCHEMA_FOR_B, ensure_ascii=False, indent=2)
    jobs.append(
        CallJob(
            suite="B",
            task_id="format_schema_vs_xml",
            prompt=(
                "Верни ответ СТРОГО как JSON-объект, валидный относительно этой JSON Schema:\n"
                f"{schema_txt}\n\n"
                "Задача: у Маши 2 яблока, ей дали ещё 2. Сколько яблок стало?\n"
                'Поле value — число яблок, unit всегда "apples".\n'
                "Никакого XML, никакого markdown — только JSON по схеме."
            ),
            expected="value=4",
            max_tokens=MAX_TOKENS,
            response_format="xml",
            score_mode="format_b",
        )
    )

    # C — противоречия без подсказки
    for bug in BUG_TASKS:
        jobs.append(
            CallJob(
                suite="C",
                task_id=bug.tid,
                prompt=(
                    f"Задача:\n{bug.text}\n\n"
                    "В конце одной строкой: ОТВЕТ=<число>"
                ),
                expected="ПРОТИВОРЕЧИЕ",
                max_tokens=MAX_TOKENS,
                score_mode="logic_c",
                trap=bug.trap,
            )
        )
    return jobs


async def fetch_pricing(
    client: httpx.AsyncClient,
    api_key: str,
    model_ids: list[str],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {mid: (0.0, 0.0) for mid in model_ids}
    try:
        r = await client.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=or_headers(api_key),
        )
        r.raise_for_status()
        data = r.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        print(f"pricing_warn: каталог моделей недоступен ({exc}); cost=0")
        return out
    by_id = {m.get("id"): m for m in data if m.get("id")}
    for mid in model_ids:
        m = by_id.get(mid)
        if not m:
            continue
        pricing = m.get("pricing") or {}
        try:
            p = float(pricing.get("prompt") or 0)
            c = float(pricing.get("completion") or 0)
        except (TypeError, ValueError):
            p, c = 0.0, 0.0
        out[mid] = (p, c)
    return out


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_prompt: float,
    price_completion: float,
) -> float:
    return round(
        prompt_tokens * price_prompt + completion_tokens * price_completion,
        8,
    )


def usage_from_or(data: dict[str, Any]) -> tuple[int, int, int, int]:
    usage = data.get("usage") or {}
    p = int(usage.get("prompt_tokens") or 0)
    c = int(usage.get("completion_tokens") or 0)
    tot = int(usage.get("total_tokens") or (p + c))
    details = usage.get("completion_tokens_details") or {}
    r = int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    return p, c, r, tot


def extract_content(data: dict[str, Any]) -> tuple[str, str | None]:
    try:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        finish = choice.get("finish_reason")
        return content, finish
    except (KeyError, IndexError, TypeError):
        return "", None


def build_http(
    model: ModelSpec,
    job: CallJob,
    *,
    deepseek_effort: str,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    messages = [{"role": "user", "content": job.prompt}]
    if model.provider == "deepseek":
        effort = deepseek_effort.strip().lower()
        body: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": job.max_tokens,
            "reasoning_effort": effort,
            "thinking": {"type": "disabled" if effort == "none" else "enabled"},
        }
        if job.response_format is not None:
            body["response_format"] = {"type": job.response_format}
        return (
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            ds_headers(require_deepseek_key()),
            body,
        )
    body = {
        "model": model.model_id,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": job.max_tokens,
    }
    if job.response_format is not None:
        body["response_format"] = {"type": job.response_format}
    return (
        f"{OPENROUTER_BASE_URL}/chat/completions",
        or_headers(require_openrouter_key()),
        body,
    )


def score_run(
    job: CallJob,
    content: str,
    *,
    http_status: int,
    finish: str | None,
    completion_tokens: int,
) -> tuple[bool, str, dict[str, Any]]:
    if job.score_mode == "olympiad":
        got = normalize_answer(content)
        ok = olympiad_match(job.expected, got)
        return ok, got or content[:120], {"mode": "olympiad"}

    if job.score_mode == "role_a":
        got = normalize_hard(content)
        empty = not (content or "").strip()
        correct = (not empty) and got == normalize_hard(job.expected)
        hit_cap = finish == "length" and completion_tokens >= job.max_tokens
        labels = {
            "mode": "role_a",
            "broke_role_but_correct": correct,
            "hit_our_max_tokens": hit_cap,
            "why_hard": HARD_TASK["why_hard"],
            "how_to_read": (
                "ok/correct=True — верное число, роль «только перебор» нарушена"
            ),
        }
        return correct, got or content[:120], labels

    if job.score_mode == "format_b":
        looks_xml = "<" in content and ">" in content
        looks_json = False
        parsed = None
        try:
            parsed = json.loads(content)
            looks_json = True
        except json.JSONDecodeError:
            looks_json = False
        if http_status >= 400:
            winner = "api_rejected_response_format_xml"
        elif looks_json:
            winner = "prompt_json_schema_won"
        elif looks_xml:
            winner = "response_format_xml_won"
        elif not (content or "").strip():
            winner = "empty_or_broken"
        else:
            winner = "neither_or_hybrid"
        schema_ok = False
        if isinstance(parsed, dict):
            schema_ok = parsed.get("value") == 4 and parsed.get("unit") == "apples"
        # «Справилась» = промпт победил: валидный JSON по схеме (не XML и не 400)
        ok = winner == "prompt_json_schema_won" and schema_ok
        labels = {
            "mode": "format_b",
            "winner_guess": winner,
            "looks_json": looks_json,
            "looks_xml": looks_xml,
            "json_parsed": parsed,
            "schema_ok": schema_ok,
            "how_to_read": (
                "ok=True — API принял запрос и модель вернула JSON по схеме "
                "(промпт победил response_format=xml)"
            ),
        }
        return ok, content[:160], labels

    if job.score_mode == "logic_c":
        got = normalize_hard(content)
        detected = got == "ПРОТИВОРЕЧИЕ"
        fell = got == normalize_hard(job.trap or "") or got == (job.trap or "")
        naive = bool(re.fullmatch(r"\d+", got)) and not detected
        labels = {
            "mode": "logic_c",
            "detected_contradiction": detected,
            "fell_for_trap": fell,
            "gave_some_number": naive,
            "how_to_read": (
                "ok=True — сама нашла ПРОТИВОРЕЧИЕ без подсказки в промпте"
            ),
        }
        return detected, got or content[:120], labels

    return False, content[:120], {"mode": "unknown"}


async def one_job(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    *,
    model: ModelSpec,
    job: CallJob,
    deepseek_effort: str,
) -> TaskRun:
    url, headers, body = build_http(model, job, deepseek_effort=deepseek_effort)
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(url, headers=headers, json=body)
            elapsed = round(time.perf_counter() - t0, 2)
            status = r.status_code
            if status >= 400:
                # B: 400 по xml — ожидаемый исход, не «транспортная» авария
                if job.score_mode == "format_b":
                    ok, got, labels = score_run(
                        job,
                        "",
                        http_status=status,
                        finish=None,
                        completion_tokens=0,
                    )
                    return TaskRun(
                        suite=job.suite,
                        task_id=job.task_id,
                        model_id=model.model_id,
                        tier=model.tier,
                        ok=ok,
                        expected=job.expected,
                        got=got,
                        elapsed_sec=elapsed,
                        http_status=status,
                        error=f"HTTP {status}: {r.text[:300]}",
                        labels=labels,
                    )
                return TaskRun(
                    suite=job.suite,
                    task_id=job.task_id,
                    model_id=model.model_id,
                    tier=model.tier,
                    ok=False,
                    expected=job.expected,
                    got="",
                    elapsed_sec=elapsed,
                    http_status=status,
                    error=f"HTTP {status}: {r.text[:300]}",
                )
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            return TaskRun(
                suite=job.suite,
                task_id=job.task_id,
                model_id=model.model_id,
                tier=model.tier,
                ok=False,
                expected=job.expected,
                got="",
                elapsed_sec=round(time.perf_counter() - t0, 2),
                error=str(exc),
            )

    content, finish = extract_content(data)
    p, c, reason, tot = usage_from_or(data)
    ok, got, labels = score_run(
        job,
        content,
        http_status=200,
        finish=finish,
        completion_tokens=c,
    )
    cost = estimate_cost(
        p, c, model.price_prompt_per_token, model.price_completion_per_token
    )
    return TaskRun(
        suite=job.suite,
        task_id=job.task_id,
        model_id=model.model_id,
        tier=model.tier,
        ok=ok,
        expected=job.expected,
        got=got,
        elapsed_sec=elapsed,
        prompt_tokens=p,
        completion_tokens=c,
        reasoning_tokens=reason,
        total_tokens=tot,
        cost_usd=cost,
        finish_reason=finish,
        http_status=200,
        answer_preview=(content or "")[:240],
        labels=labels,
    )


def aggregate_suite(
    models: list[ModelSpec],
    runs: list[TaskRun],
    suite: str,
) -> list[ModelAgg]:
    by_tier = {
        m.tier: ModelAgg(
            tier=m.tier, label=m.label, model_id=m.model_id, suite=suite
        )
        for m in models
    }
    for run in runs:
        if run.suite != suite:
            continue
        agg = by_tier[run.tier]
        agg.total += 1
        crash = bool(run.error) and run.suite != "B"
        agg.correct += int(run.ok and not crash)
        agg.elapsed_sec += run.elapsed_sec
        agg.prompt_tokens += run.prompt_tokens
        agg.completion_tokens += run.completion_tokens
        agg.reasoning_tokens += run.reasoning_tokens
        agg.total_tokens += run.total_tokens
        agg.cost_usd += run.cost_usd
        if crash:
            agg.errors += 1
        agg.runs.append(run)
    for agg in by_tier.values():
        agg.elapsed_sec = round(agg.elapsed_sec, 2)
        agg.cost_usd = round(agg.cost_usd, 8)
    return [by_tier[m.tier] for m in models]


def build_conclusion(
    olymp_aggs: list[ModelAgg],
    runs: list[TaskRun],
) -> dict[str, Any]:
    if not olymp_aggs:
        return {}
    small = next((a for a in olymp_aggs if a.tier == "small"), olymp_aggs[0])
    flag = next((a for a in olymp_aggs if a.tier == "flagship"), olymp_aggs[-1])
    small_ok = {r.task_id for r in small.runs if r.ok}
    enough = sorted(
        {
            r.task_id
            for r in flag.runs
            if r.ok and r.task_id in small_ok
        }
    )

    cost_ratio: Any = None
    if small.cost_usd > 0:
        cost_ratio = round(flag.cost_usd / small.cost_usd, 2)
    elif flag.cost_usd == 0 and small.cost_usd == 0:
        cost_ratio = "оба $0 (free)"
    elif small.cost_usd == 0 and flag.cost_usd > 0:
        cost_ratio = f"small=$0; flagship≈${flag.cost_usd:.6f}"

    probes: dict[str, Any] = {}
    for suite in ("A", "B", "C"):
        probes[suite] = {}
        for tier in ("small", "mid", "flagship"):
            rs = [r for r in runs if r.suite == suite and r.tier == tier]
            if not rs:
                continue
            probes[suite][tier] = [
                {
                    "task_id": r.task_id,
                    "ok": r.ok,
                    "got": r.got,
                    "labels": r.labels,
                    "http_status": r.http_status,
                    "error": r.error,
                    "elapsed_sec": r.elapsed_sec,
                    "total_tokens": r.total_tokens,
                }
                for r in rs
            ]

    return {
        "olympiad_small_enough_tasks": enough,
        "olympiad_flagship_vs_small_accuracy_delta": round(
            flag.accuracy - small.accuracy, 3
        ),
        "olympiad_flagship_vs_small_cost_ratio": cost_ratio,
        "probes": probes,
        "note": (
            "Олимпиада: 5 сложных задач. A — верный ответ = слом роли перебора. "
            "B — ok если JSON по схеме победил xml. "
            "C — ok если сама написала ПРОТИВОРЕЧИЕ. "
            "Стоимость Pro — оценка off-peak cache-miss."
        ),
    }


def print_table(title: str, aggs: list[ModelAgg]) -> None:
    print(f"\n{title}")
    print(
        f"{'tier':<10} {'model':<48} {'ok':>6} {'время_с':>8} "
        f"{'tok':>8} {'$':>10} {'err':>4}"
    )
    for a in aggs:
        print(
            f"{a.tier:<10} {a.model_id:<48} "
            f"{a.correct}/{a.total:<3} {a.elapsed_sec:>8.1f} "
            f"{a.total_tokens:>8} {a.cost_usd:>10.6f} {a.errors:>4}"
        )


def print_probe_summary(runs: list[TaskRun]) -> None:
    print("\nПробы A/B/C (по моделям)")
    for suite in ("A", "B", "C"):
        print(f"  [{suite}]")
        for tier in ("small", "mid", "flagship"):
            rs = [r for r in runs if r.suite == suite and r.tier == tier]
            for r in rs:
                mark = "OK" if r.ok else ("ERR" if (r.error and suite != "B") else "NO")
                lab = r.labels
                hint = ""
                if suite == "A":
                    hint = f" broke_role={lab.get('broke_role_but_correct')} hit_cap={lab.get('hit_our_max_tokens')}"
                elif suite == "B":
                    hint = f" winner={lab.get('winner_guess')} http={r.http_status}"
                elif suite == "C":
                    hint = (
                        f" contradiction={lab.get('detected_contradiction')} "
                        f"number={lab.get('gave_some_number')}"
                    )
                print(
                    f"    [{mark}] {tier:<9} {r.task_id:<22} "
                    f"{r.elapsed_sec:>5.1f}s tok={r.total_tokens} got={r.got!r}{hint}"
                )


async def amain() -> int:
    _utf8()
    parser = argparse.ArgumentParser(description="AI Challenge — день 5")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Параллельных запросов (free: лучше 2–3)",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Ограничить олимпиадные tid (пробы A/B/C всё равно идут)",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Только олимпиада, без A/B/C",
    )
    parser.add_argument(
        "--models-json",
        default="",
        help='JSON-список [{tier,label,provider,model_id}, ...]',
    )
    parser.add_argument(
        "--deepseek-effort",
        default="none",
        choices=("none", "low", "high", "max"),
        help="reasoning_effort для DeepSeek-флагмана (default none)",
    )
    args = parser.parse_args()

    raw_models = DEFAULT_MODELS
    if args.models_json:
        raw_models = json.loads(args.models_json)
    for m in raw_models:
        m.setdefault("provider", "openrouter")

    providers = {m["provider"] for m in raw_models}
    or_key = require_openrouter_key() if "openrouter" in providers else ""
    if "deepseek" in providers:
        require_deepseek_key()

    by_id = {t.tid: t for t in TASKS}
    olymp_ids = list(OLYMP_HARD5_IDS)
    if args.tasks:
        olymp_ids = list(args.tasks)
    missing = [tid for tid in olymp_ids if tid not in by_id]
    if missing:
        print(f"unknown tasks: {missing}")
        return 2
    olymp = [by_id[tid] for tid in olymp_ids]

    jobs = build_jobs(olymp)
    if args.skip_probes:
        jobs = [j for j in jobs if j.suite == "olympiad"]

    print("settings (день 5 — версии моделей)")
    print(f"  openrouter={OPENROUTER_BASE_URL}")
    print(f"  deepseek={DEEPSEEK_BASE_URL}")
    print(f"  concurrency={args.concurrency}")
    print(f"  temperature={TEMPERATURE} max_tokens={MAX_TOKENS} max_tokens_A={MAX_TOKENS_A}")
    print(f"  deepseek_effort={args.deepseek_effort}")
    print(f"  olympiad={olymp_ids} (n={len(olymp)})")
    print(
        f"  probes={'off' if args.skip_probes else 'A=subsets_40; B=schema_vs_xml; C=' + ','.join(b.tid for b in BUG_TASKS)}"
    )
    print(f"  jobs_per_model={len(jobs)} total={len(jobs) * len(raw_models)}")
    print("  models:")
    for m in raw_models:
        print(f"    {m['tier']}: {m['provider']}/{m['model_id']}")
    print()

    async with httpx.AsyncClient(**_client_kwargs()) as client:
        or_ids = [m["model_id"] for m in raw_models if m["provider"] == "openrouter"]
        prices = await fetch_pricing(client, or_key, or_ids) if or_ids else {}
        models: list[ModelSpec] = []
        for m in raw_models:
            if m["provider"] == "deepseek":
                models.append(
                    ModelSpec(
                        tier=m["tier"],
                        label=m["label"],
                        model_id=m["model_id"],
                        provider="deepseek",
                        price_prompt_per_token=DEEPSEEK_V4_PRO_PRICE_PROMPT_PER_1M
                        / 1_000_000,
                        price_completion_per_token=DEEPSEEK_V4_PRO_PRICE_COMPLETION_PER_1M
                        / 1_000_000,
                        price_note="deepseek-v4-pro off-peak cache-miss (docs)",
                    )
                )
            else:
                p, c = prices.get(m["model_id"], (0.0, 0.0))
                models.append(
                    ModelSpec(
                        tier=m["tier"],
                        label=m["label"],
                        model_id=m["model_id"],
                        provider="openrouter",
                        price_prompt_per_token=p,
                        price_completion_per_token=c,
                        price_note="openrouter catalog",
                    )
                )
        for m in models:
            print(
                f"  price {m.provider}/{m.model_id}: "
                f"prompt={m.price_prompt_per_token} "
                f"completion={m.price_completion_per_token} $/tok ({m.price_note})"
            )
        print()

        sem = asyncio.Semaphore(max(1, args.concurrency))
        coros = [
            one_job(
                client,
                sem,
                model=model,
                job=job,
                deepseek_effort=args.deepseek_effort,
            )
            for model in models
            for job in jobs
        ]
        print(f"api: {len(coros)} запросов...")
        t_wall0 = time.perf_counter()
        runs = list(await asyncio.gather(*coros))
        wall = round(time.perf_counter() - t_wall0, 2)
        print(f"api: готово, время_стены_сек={wall}")

    for run in runs:
        if run.suite != "olympiad":
            continue
        mark = "OK" if run.ok and not run.error else ("ERR" if run.error else "NO")
        extra = f" err={run.error[:80]}" if run.error else f" got={run.got}"
        print(
            f"  [{mark}] {run.tier:<9} {run.task_id:<12} "
            f"{run.elapsed_sec:>5.1f}s tok={run.total_tokens}{extra}"
        )

    olymp_aggs = aggregate_suite(models, runs, "olympiad")
    print_table("Олимпиада (5 сложных): модель → верных / время / токены / $", olymp_aggs)
    print_probe_summary(runs)

    conclusion = build_conclusion(olymp_aggs, runs)
    print("\nВывод (черновик)")
    print(
        f"- olympiad small достаточно: "
        f"{conclusion.get('olympiad_small_enough_tasks')}"
    )
    print(
        f"- olympiad flagship−small accuracy: "
        f"{conclusion.get('olympiad_flagship_vs_small_accuracy_delta')}"
    )
    print(
        f"- olympiad cost ratio: "
        f"{conclusion.get('olympiad_flagship_vs_small_cost_ratio')}"
    )
    print(f"- {conclusion.get('note')}")

    out = {
        "settings": {
            "openrouter_base_url": OPENROUTER_BASE_URL,
            "deepseek_base_url": DEEPSEEK_BASE_URL,
            "concurrency": args.concurrency,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "max_tokens_A": MAX_TOKENS_A,
            "deepseek_effort": args.deepseek_effort,
            "wall_sec": wall,
            "olympiad_tasks": olymp_ids,
            "olympiad_why": (
                "5 самых сложных из дня 3 по токенам/смыслу прошлого прогона; "
                "убраны meet, paper, queue"
            ),
            "probes": None
            if args.skip_probes
            else {
                "A": HARD_TASK["tid"],
                "B": "format_schema_vs_xml",
                "C": [b.tid for b in BUG_TASKS],
            },
            "models": [asdict(m) for m in models],
        },
        "table_olympiad": [
            {
                "tier": a.tier,
                "label": a.label,
                "model_id": a.model_id,
                "correct": a.correct,
                "total": a.total,
                "accuracy": a.accuracy,
                "elapsed_sec": a.elapsed_sec,
                "total_tokens": a.total_tokens,
                "prompt_tokens": a.prompt_tokens,
                "completion_tokens": a.completion_tokens,
                "reasoning_tokens": a.reasoning_tokens,
                "cost_usd": a.cost_usd,
                "errors": a.errors,
            }
            for a in olymp_aggs
        ],
        "conclusion": conclusion,
        "runs": [asdict(r) for r in runs],
    }
    path = Path("task5_results.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{path}")

    try:
        from task5_report import build_docx

        docx_path = Path("task5_result.docx")
        build_docx(out, docx_path)
        print(docx_path)
    except Exception as exc:  # noqa: BLE001
        print(f"docx_error: {exc}")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
