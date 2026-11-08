"""День 4. Temperature / top_p на deepseek-flash.

Основной прогон: reasoning=none × разные temperature и top_p.
Крайности: reasoning=high — по докам temperature игнорируется;
            top_p на flash в thinking узкий (0.95–1.0).

Запуск (сервер /chat уже поднят):
  python task4.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 180.0
MAX_TOKENS = 8000
REPEATS_ACCURACY = 5
REPEATS_GEN = 5
DEFAULT_CONCURRENCY = 8

# Локальный скорер PPL на CPU (~125M, русский). Не сервер — import в процессе.
PPL_MODEL_ID = "ai-forever/rugpt3small_based_on_gpt2"
PPL_MAX_LENGTH = 256
PPL_BATCH_SIZE = 1  # pad-batch на ruGPT искажает PPL; один load, тексты подряд

# --- Почему такие параметры (DeepSeek API, deepseek-flash) ---
# Источник: https://api-docs.deepseek.com/guides/thinking_mode.md
#           https://api-docs.deepseek.com/api/create-chat-completion.md
DOCS_WHY = {
    "model": "deepseek-flash",
    "source": [
        "https://api-docs.deepseek.com/guides/thinking_mode.md",
        "https://api-docs.deepseek.com/api/create-chat-completion.md",
    ],
    "why_chosen": (
        "На flash temperature работает только при reasoning=none; при thinking "
        "temperature принимается, но не влияет. top_p наоборот: при thinking "
        "действует в 0.95–1.0 (<0.95 → 0.95); при none зафиксирован в 1.0 и игнорируется. "
        "Финальная сетка (без лишних запросов): none×temperature — основной эффект; "
        "high×top_p на story_s — единственный sampling-рычаг в thinking. "
        "Проверки «none×top_p игнорируется» и «high×temperature игнорируется» "
        "уже подтверждены прошлым прогоном и из сетки убраны."
    ),
    "matrix": {
        "temperature": {
            "non_thinking": "действует, диапазон 0–2",
            "thinking": "игнорируется (ошибки нет)",
        },
        "top_p": {
            "non_thinking": "игнорируется, фикс 1.0",
            "thinking": "действует 0.95–1.0; ниже 0.95 → как 0.95",
        },
        "top_k": "нет / игнор",
        "presence_frequency_penalty": "нет эффекта",
    },
}

# Результаты уже выполненных проверок, убранных из финальной сетки (экономия запросов).
PRIOR_CONFIRMED_CHECKS = {
    "source": "предыдущий полный прогон task4_results.json + доки DeepSeek flash",
    "items": [
        {
            "check": "none × top_p ∈ {0.5, 0.95, 1.0} при temperature=0",
            "result": (
                "perplexity/letter_ratio почти не менялись (story_s ppl ~69–72). "
                "Подтверждает док: в non-thinking top_p на flash зафиксирован в 1.0."
            ),
            "in_final_run": False,
        },
        {
            "check": "high × temperature ∈ {0, 2} на story_s",
            "result": (
                "стоимость почти одинаковая (~70 с; reasoning ~8459 vs ~9099), "
                "letter_ratio=1.0, ppl ~60–75. Temperature в thinking не рулит; "
                "дороговизна — сам reasoning под constraint «слова на С»."
            ),
            "in_final_run": False,
        },
        {
            "check": "high × temperature на bat_ball/sheep",
            "result": (
                "accuracy=1.0 при t=0 и t=2; reasoning умеренный (сотни токенов на 5 вызовов). "
                "Ещё одно подтверждение игнора temperature при high."
            ),
            "in_final_run": False,
        },
    ],
}

ANSWER_RE = re.compile(r"ОТВЕТ\s*=\s*(.+?)(?:\s*$|\n)", re.IGNORECASE | re.DOTALL)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)

_active_url = URL


@dataclass
class AccTask:
    tid: str
    prompt: str
    expected: str


@dataclass
class GenTask:
    tid: str
    prompt: str
    letter: str  # ожидаемая первая буква слов (нижний регистр)


ACC_TASKS: list[AccTask] = [
    AccTask(
        tid="bat_ball",
        prompt=(
            "Бита и мяч вместе стоят 1.10 доллара. Бита стоит на 1 доллар дороже мяча. "
            "Сколько стоит мяч?\n"
            "В конце одной строкой: ОТВЕТ=<число>"
        ),
        expected="0.05",
    ),
    AccTask(
        tid="sheep",
        prompt=(
            "У фермера было 17 овец. Все кроме 9 умерли. Сколько овец осталось?\n"
            "В конце одной строкой: ОТВЕТ=<число>"
        ),
        expected="9",
    ),
]

GEN_TASKS: list[GenTask] = [
    GenTask(
        tid="story_s",
        prompt=(
            "Напиши короткий рассказ (4–6 предложений) про северного странника. "
            "Жёсткое правило: КАЖДОЕ слово в тексте должно начинаться с буквы «С» "
            "(регистр не важен; союзы/предлоги тоже на «С»: с, со, среди…). "
            "Цифры и знаки препинания можно. Без заголовка и пояснений — только рассказ."
        ),
        letter="с",
    ),
    GenTask(
        tid="list_s",
        prompt=(
            "Перечисли ровно 8 существительных в именительном падеже, "
            "каждое начинается с буквы «С». Формат: одна строка, через запятую, "
            "без нумерации и пояснений."
        ),
        letter="с",
    ),
]


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_docs_why() -> None:
    print("docs_why (deepseek-flash)")
    print(f"  source={DOCS_WHY['source']}")
    print(f"  {DOCS_WHY['why_chosen']}")
    print("  temperature: non-thinking=да; thinking=игнор")
    print("  top_p: non-thinking=игнор(фикс 1.0); thinking=0.95..1.0")
    print("  top_k / penalties: нет эффекта")
    print()


def normalize_acc(raw: str) -> str:
    text = (raw or "").strip()
    m = ANSWER_RE.search(text)
    if m:
        text = m.group(1).strip()
    text = text.strip().strip("`\"'").replace(" ", "").replace(",", ".")
    # оставить число вида 0.05
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[0] if nums else text.lower()


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def letter_compliance(text: str, letter: str) -> dict[str, Any]:
    ws = words_of(text)
    if not ws:
        return {"words": 0, "ok": 0, "ratio": 0.0, "bad": []}
    letter = letter.lower().replace("ё", "е")
    ok = 0
    bad: list[str] = []
    for w in ws:
        first = w[0].lower().replace("ё", "е")
        if first == letter:
            ok += 1
        else:
            bad.append(w)
    return {
        "words": len(ws),
        "ok": ok,
        "ratio": round(ok / len(ws), 3),
        "bad": bad[:12],
    }


@dataclass
class CallJob:
    """Один HTTP-вызов; потом склеиваем в строки отчёта."""

    phase: str
    kind: str  # accuracy | generation
    task_id: str
    prompt: str
    effort: str
    temperature: float | None
    top_p: float | None
    repeat_idx: int
    expected: str | None = None
    letter: str | None = None


def api_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Стоимость/время только API DeepSeek (без локального PPL)."""
    return {
        # сумма длительностей запросов (при async > wall-clock)
        "api_elapsed_sec": round(sum(float(r.get("elapsed_sec") or 0) for r in runs), 2),
        "reasoning_tokens": int(sum(int(r.get("reasoning_tokens") or 0) for r in runs)),
        "total_tokens": int(sum(int(r.get("total_tokens") or 0) for r in runs)),
    }


async def one_call_async(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    job: CallJob,
) -> tuple[CallJob, dict[str, Any]]:
    body: dict[str, Any] = {
        "prompt": job.prompt,
        "reasoning_effort": job.effort,
        "max_tokens": MAX_TOKENS,
    }
    if job.temperature is not None:
        body["temperature"] = job.temperature
    if job.top_p is not None:
        body["top_p"] = job.top_p
    async with sem:
        t0 = time.perf_counter()
        response = await client.post(_active_url, json=body)
        response.raise_for_status()
        data = response.json()
        elapsed = round(time.perf_counter() - t0, 2)
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    raw = {
        "sent": body,
        "elapsed_sec": elapsed,
        "answer": data.get("answer") or "",
        "finish_reason": data.get("finish_reason"),
        "model": data.get("model"),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "repeat_idx": job.repeat_idx,
    }
    return job, raw


def build_jobs(
    *,
    repeats_acc: int,
    repeats_gen: int,
    skip_high: bool,
) -> list[CallJob]:
    jobs: list[CallJob] = []
    for temp in (0.0, 1.0, 2.0):
        for task in ACC_TASKS:
            for i in range(repeats_acc):
                jobs.append(
                    CallJob(
                        phase="none_temperature",
                        kind="accuracy",
                        task_id=task.tid,
                        prompt=task.prompt,
                        effort="none",
                        temperature=temp,
                        top_p=None,
                        repeat_idx=i,
                        expected=task.expected,
                    )
                )
        for task in GEN_TASKS:
            for i in range(repeats_gen):
                jobs.append(
                    CallJob(
                        phase="none_temperature",
                        kind="generation",
                        task_id=task.tid,
                        prompt=task.prompt,
                        effort="none",
                        temperature=temp,
                        top_p=None,
                        repeat_idx=i,
                        letter=task.letter,
                    )
                )
    if not skip_high:
        story = next(t for t in GEN_TASKS if t.tid == "story_s")
        for top_p in (0.95, 1.0):
            for i in range(repeats_gen):
                jobs.append(
                    CallJob(
                        phase="high_top_p",
                        kind="generation",
                        task_id=story.tid,
                        prompt=story.prompt,
                        effort="high",
                        temperature=None,
                        top_p=top_p,
                        repeat_idx=i,
                        letter=story.letter,
                    )
                )
    return jobs


def aggregate_rows(
    jobs_and_raw: list[tuple[CallJob, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Склеить параллельные ответы в строки отчёта (порядок repeat_idx)."""
    buckets: dict[tuple[Any, ...], list[tuple[CallJob, dict[str, Any]]]] = defaultdict(list)
    for job, raw in jobs_and_raw:
        key = (
            job.phase,
            job.kind,
            job.task_id,
            job.effort,
            job.temperature,
            job.top_p,
        )
        buckets[key].append((job, raw))

    rows: list[dict[str, Any]] = []
    # стабильный порядок: как в build_jobs
    seen: list[tuple[Any, ...]] = []
    for job, _ in jobs_and_raw:
        key = (
            job.phase,
            job.kind,
            job.task_id,
            job.effort,
            job.temperature,
            job.top_p,
        )
        if key not in seen:
            seen.append(key)

    for key in seen:
        items = sorted(buckets[key], key=lambda jr: jr[0].repeat_idx)
        phase, kind, task_id, effort, temperature, top_p = key
        if kind == "accuracy":
            runs: list[dict[str, Any]] = []
            correct = 0
            gots: list[str] = []
            expected = items[0][0].expected or ""
            for job, raw in items:
                got = normalize_acc(raw["answer"])
                ok = got == expected
                correct += int(ok)
                gots.append(got)
                runs.append({**raw, "parsed_got": got, "ok": ok})
            unique = sorted(set(gots))
            stats = api_stats(runs)
            rows.append(
                {
                    "kind": "accuracy",
                    "phase": phase,
                    "task": task_id,
                    "expected": expected,
                    "effort": effort,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeats": len(runs),
                    "correct": correct,
                    "accuracy": round(correct / len(runs), 3) if runs else 0.0,
                    "unique_answers": unique,
                    "n_unique": len(unique),
                    **stats,
                    "runs": runs,
                }
            )
        else:
            runs = []
            texts: list[str] = []
            ratios: list[float] = []
            letter = items[0][0].letter or "с"
            for job, raw in items:
                ans = (raw["answer"] or "").strip()
                comp = letter_compliance(ans, letter)
                texts.append(ans)
                ratios.append(float(comp["ratio"]))
                runs.append({**raw, "letter_stats": comp})
            unique_n = len(set(texts))
            stats = api_stats(runs)
            rows.append(
                {
                    "kind": "generation",
                    "phase": phase,
                    "task": task_id,
                    "letter": letter,
                    "effort": effort,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeats": len(runs),
                    "n_unique_texts": unique_n,
                    "diversity": round(unique_n / len(runs), 3) if runs else 0.0,
                    "mean_letter_ratio": round(sum(ratios) / len(ratios), 3) if ratios else 0.0,
                    "mean_perplexity": None,
                    **stats,
                    "runs": runs,
                }
            )
    return rows


async def run_all_api(
    jobs: list[CallJob],
    *,
    concurrency: int,
) -> tuple[list[dict[str, Any]], float]:
    """Все /chat параллельно (semaphore). Возвращает rows и wall-clock сек."""
    sem = asyncio.Semaphore(max(1, concurrency))
    timeout = httpx.Timeout(TIMEOUT)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(
            *[one_call_async(client, sem, job) for job in jobs],
            return_exceptions=False,
        )
    wall = round(time.perf_counter() - t0, 2)
    rows = aggregate_rows(list(results))
    return rows, wall


class CpuPerplexityScorer:
    """Causal LM на CPU: batch forward → perplexity по каждому тексту."""

    def __init__(self, model_id: str = PPL_MODEL_ID) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def batch_perplexity(self, texts: list[str]) -> list[float]:
        """Все тексты в одной фазе (один load модели).

        Считаем по одному: у GPT-2/ruGPT pad-batch с разными длинами
        искажает PPL (проверено solo vs batch). На CPU для коротких
        ответов это приемлемо; модель в память грузится один раз.
        """
        torch = self.torch
        if not texts:
            return []
        out: list[float] = []
        for raw in texts:
            t = (raw or "").strip() or "."
            enc = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=PPL_MAX_LENGTH,
            )
            input_ids = enc["input_ids"].to(self.device)
            with torch.no_grad():
                loss = self.model(input_ids=input_ids, labels=input_ids).loss
            out.append(float(torch.exp(loss)))
        return out


def attach_perplexity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Оцениваем ВСЕ generation-ответы одним загрузом модели, пачками на CPU."""
    gen_rows = [r for r in rows if r.get("kind") == "generation"]
    flat_texts: list[str] = []
    index: list[tuple[int, int]] = []  # (row_idx in rows, run_idx)
    for ri, row in enumerate(rows):
        if row.get("kind") != "generation":
            continue
        for rj, run in enumerate(row.get("runs") or []):
            flat_texts.append(run.get("answer") or "")
            index.append((ri, rj))

    meta: dict[str, Any] = {
        "model": PPL_MODEL_ID,
        "device": "cpu",
        "n_texts": len(flat_texts),
        "batch_size": PPL_BATCH_SIZE,
        "max_length": PPL_MAX_LENGTH,
        "load_and_score_sec": None,
        "note": "Время PPL не входит во время_api_сек / токены_reasoning строк прогона",
    }
    if not flat_texts:
        return meta

    t0 = time.perf_counter()
    scorer = CpuPerplexityScorer(PPL_MODEL_ID)
    scores = scorer.batch_perplexity(flat_texts)
    meta["load_and_score_sec"] = round(time.perf_counter() - t0, 2)

    for (ri, rj), ppl in zip(index, scores, strict=True):
        rows[ri]["runs"][rj]["perplexity"] = round(ppl, 3)

    for row in gen_rows:
        ppls = [float(r["perplexity"]) for r in row["runs"] if r.get("perplexity") is not None]
        row["mean_perplexity"] = round(sum(ppls) / len(ppls), 3) if ppls else None

    return meta


def print_block_summary(row: dict[str, Any]) -> None:
    api = (
        f"время_api_сек={row.get('api_elapsed_sec')} "
        f"токены_reasoning={row.get('reasoning_tokens')} "
        f"токены_всего={row.get('total_tokens')}"
    )
    if row["kind"] == "accuracy":
        print(
            f"{row['task']:<10} effort={row['effort']:<5} "
            f"t={row['temperature']!s:<4} p={row['top_p']!s:<5} "
            f"acc={row['correct']}/{row['repeats']} "
            f"unique={row['n_unique']} {row['unique_answers']} | {api}"
        )
    else:
        ppl = row.get("mean_perplexity")
        ppl_s = f"perplexity={ppl}" if ppl is not None else "perplexity=pending"
        print(
            f"{row['task']:<10} effort={row['effort']:<5} "
            f"t={row['temperature']!s:<4} p={row['top_p']!s:<5} "
            f"div={row['diversity']} "
            f"letter_ratio={row['mean_letter_ratio']} {ppl_s} | {api}"
        )


def _first_answer(row: dict[str, Any], *, max_len: int = 400) -> str:
    for run in row.get("runs") or []:
        ans = (run.get("answer") or "").strip()
        if ans:
            return ans if len(ans) <= max_len else ans[: max_len - 1] + "…"
    return ""


def build_result_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Результат по условию дня 4: примеры при разной temperature + выводы по использованию."""
    examples: list[dict[str, Any]] = []
    for temp in (0.0, 1.0, 2.0):
        block: dict[str, Any] = {"temperature": temp, "effort": "none", "samples": []}
        for row in rows:
            if row.get("phase") != "none_temperature" or row.get("temperature") != temp:
                continue
            sample: dict[str, Any] = {
                "task": row["task"],
                "kind": row["kind"],
                "answer_example": _first_answer(row),
                "время_api_сек": row.get("api_elapsed_sec"),
                "токены_reasoning": row.get("reasoning_tokens"),
                "токены_всего": row.get("total_tokens"),
            }
            if row["kind"] == "accuracy":
                sample["accuracy"] = f"{row['correct']}/{row['repeats']}"
                sample["unique_answers"] = row.get("unique_answers")
            else:
                sample["diversity"] = row.get("diversity")
                sample["letter_ratio"] = row.get("mean_letter_ratio")
                sample["perplexity"] = row.get("mean_perplexity")
            block["samples"].append(sample)
        examples.append(block)

    # краткие цифры accuracy @ none × temperature
    acc_by_temp: dict[str, str] = {}
    for temp in (0.0, 1.0, 2.0):
        bits = []
        for row in rows:
            if (
                row.get("phase") == "none_temperature"
                and row.get("kind") == "accuracy"
                and row.get("temperature") == temp
            ):
                bits.append(f"{row['task']} {row['correct']}/{row['repeats']}")
        acc_by_temp[str(temp)] = ", ".join(bits) if bits else "—"

    usage = [
        "temperature=0 (reasoning=none): ответы почти детерминированные — для фактов, "
        "коротких точных ответов, когда важна стабильность (у нас выше доля верных на ловушках).",
        "temperature≈1: баланс разнообразия и связности — черновики, списки, мягкий креатив; "
        "смотреть letter_ratio и perplexity на генерации.",
        "temperature=2: максимум случайности — идеи/варианты, но чаще срыв правила «слова на С», "
        "рост perplexity и падение accuracy на задачах с одним верным ответом.",
        "На deepseek-flash при reasoning=high temperature не крутим (игнор по докам); "
        "для sampling в thinking смотреть top_p в узком окне 0.95–1.0 (фаза high_top_p).",
        "top_p при reasoning=none на flash игнорируется — в финальном прогоне не гоняли.",
    ]
    return {
        "title": "Результат: примеры ответов с разной температурой и выводы по их использованию",
        "examples_by_temperature": examples,
        "accuracy_by_temperature": acc_by_temp,
        "usage_conclusions": usage,
        "answers_saved": (
            "Полные ответы API лежат в task4_results.json → rows[].runs[].answer "
            "(все повторы, не только пример в Результате)."
        ),
    }


def print_result(result: dict[str, Any]) -> None:
    print("\nРезультат:")
    print(result["title"])
    print()
    for block in result.get("examples_by_temperature") or []:
        print(f"--- temperature={block['temperature']} (effort=none) ---")
        for s in block.get("samples") or []:
            print(
                f"  [{s['task']}] "
                f"время_api_сек={s.get('время_api_сек')} "
                f"токены_reasoning={s.get('токены_reasoning')} "
                f"токены_всего={s.get('токены_всего')}"
            )
            if s["kind"] == "accuracy":
                print(f"    accuracy={s.get('accuracy')} unique={s.get('unique_answers')}")
            else:
                print(
                    f"    diversity={s.get('diversity')} "
                    f"letter_ratio={s.get('letter_ratio')} "
                    f"perplexity={s.get('perplexity')}"
                )
            ans = s.get("answer_example") or ""
            preview = ans.replace("\n", " / ")
            print(f"    пример: {preview}")
        print()
    print("Выводы по использованию temperature:")
    for line in result.get("usage_conclusions") or []:
        print(f"- {line}")
    print()
    print(result.get("answers_saved") or "")


def build_conclusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    none_temp = [r for r in rows if r.get("phase") == "none_temperature"]
    high_top = [r for r in rows if r.get("phase") == "high_top_p"]

    def acc_line(rs: list[dict[str, Any]]) -> str:
        bits = [
            f"{r['task']}@t={r['temperature']}/p={r['top_p']}→{r['accuracy']}"
            for r in rs
            if r["kind"] == "accuracy"
        ]
        return "; ".join(bits) if bits else "—"

    def gen_line(rs: list[dict[str, Any]]) -> str:
        bits = []
        for r in rs:
            if r["kind"] != "generation":
                continue
            bits.append(
                f"{r['task']}@t={r['temperature']}/p={r['top_p']}"
                f"→div={r['diversity']},S={r['mean_letter_ratio']},ppl={r.get('mean_perplexity')}"
            )
        return "; ".join(bits) if bits else "—"

    def cost_line(rs: list[dict[str, Any]]) -> str:
        if not rs:
            return "—"
        api_s = sum(float(r.get("api_elapsed_sec") or 0) for r in rs)
        reason = sum(int(r.get("reasoning_tokens") or 0) for r in rs)
        tok = sum(int(r.get("total_tokens") or 0) for r in rs)
        return (
            f"время_api_сек={api_s:.1f} "
            f"токены_reasoning={reason} "
            f"токены_всего={tok}"
        )

    return {
        "docs_reminder": DOCS_WHY["why_chosen"],
        "none_temperature": {
            "accuracy": acc_line(none_temp),
            "generation": gen_line(none_temp),
            "api_cost": cost_line(none_temp),
            "expect": "с ростом temperature: accuracy↓ / diversity↑ / perplexity↑",
        },
        "high_top_p": {
            "accuracy": acc_line(high_top),
            "generation": gen_line(high_top),
            "api_cost": cost_line(high_top),
            "expect": (
                "при high на flash смотрим top_p 0.95 vs 1.0 (узкий диапазон); "
                "temperature в thinking не варьируем — по докам игнор"
            ),
        },
        "skipped_as_confirmed": (
            "Убраны none×top_p и high×temperature — уже подтверждены прошлым прогоном/доками."
        ),
    }


async def amain() -> int:
    _utf8()
    global _active_url
    parser = argparse.ArgumentParser(description="AI Challenge — день 4")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--repeats-acc", type=int, default=REPEATS_ACCURACY)
    parser.add_argument("--repeats-gen", type=int, default=REPEATS_GEN)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Параллельных HTTP-запросов к /chat (default 8)",
    )
    parser.add_argument(
        "--skip-high",
        action="store_true",
        help="Не гонять крайности с reasoning=high",
    )
    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Не считать локальный perplexity (CPU LM)",
    )
    args = parser.parse_args()
    _active_url = args.url

    print_docs_why()
    print("settings (финальный прогон для видео)")
    print(f"  url={_active_url}")
    print(f"  max_tokens={MAX_TOKENS}")
    print(f"  repeats_acc={args.repeats_acc} repeats_gen={args.repeats_gen}")
    print(f"  concurrency={args.concurrency} (async httpx)")
    print(f"  acc_tasks={[t.tid for t in ACC_TASKS]}")
    print(f"  gen_tasks={[t.tid for t in GEN_TASKS]}")
    print("  сетка: none×temperature{0,1,2}; high×top_p{0.95,1.0} только story_s")
    print("  убрано: none×top_p, high×temperature (уже подтверждено)")
    print(f"  ppl_model={PPL_MODEL_ID} device=cpu")
    print()

    jobs = build_jobs(
        repeats_acc=args.repeats_acc,
        repeats_gen=args.repeats_gen,
        skip_high=args.skip_high,
    )
    print(f"api: {len(jobs)} запросов параллельно (semaphore={args.concurrency})...")
    rows, api_wall_s = await run_all_api(jobs, concurrency=args.concurrency)
    print(f"api: готово, время_стены_сек={api_wall_s}")
    for row in rows:
        print_block_summary(row)

    # После всех API: один load модели + batch PPL по всем generation-ответам
    ppl_meta: dict[str, Any] | None = None
    if not args.skip_ppl:
        print("\nperplexity (локально CPU; не входит во время_api_сек)")
        ppl_meta = attach_perplexity(rows)
        print(
            f"  model={ppl_meta['model']} n={ppl_meta['n_texts']} "
            f"время_расчёта_ppl_сек={ppl_meta['load_and_score_sec']}"
        )
        for row in rows:
            if row.get("kind") == "generation":
                print_block_summary(row)

    api_sum_s = sum(float(r.get("api_elapsed_sec") or 0) for r in rows)
    api_reason = sum(int(r.get("reasoning_tokens") or 0) for r in rows)
    api_tok = sum(int(r.get("total_tokens") or 0) for r in rows)

    conclusion = build_conclusion(rows)
    result = build_result_payload(rows)
    print_result(result)

    print("\nСводка метрик (дополнительно)")
    print(f"- none×temperature accuracy: {conclusion['none_temperature']['accuracy']}")
    print(f"- none×temperature generation: {conclusion['none_temperature']['generation']}")
    print(f"- none×temperature DeepSeek: {conclusion['none_temperature']['api_cost']}")
    print(f"- high×top_p: {conclusion['high_top_p']['expect']}")
    print(f"  {conclusion['high_top_p']['generation']}")
    print(f"  DeepSeek: {conclusion['high_top_p']['api_cost']}")
    print(f"- {conclusion['skipped_as_confirmed']}")
    print(
        f"- всего DeepSeek: время_стены_сек={api_wall_s:.1f} "
        f"время_api_сумма_сек={api_sum_s:.1f} "
        f"токены_reasoning={api_reason} токены_всего={api_tok}"
    )
    if ppl_meta and ppl_meta.get("load_and_score_sec") is not None:
        print(
            f"- локальный perplexity отдельно: "
            f"время_расчёта_ppl_сек={ppl_meta['load_and_score_sec']} "
            f"(model={ppl_meta['model']}, cpu)"
        )

    out = {
        "docs_why": DOCS_WHY,
        "settings": {
            "url": _active_url,
            "max_tokens": MAX_TOKENS,
            "repeats_acc": args.repeats_acc,
            "repeats_gen": args.repeats_gen,
            "concurrency": args.concurrency,
            "main_effort": "none",
            "temperatures_none": [0.0, 1.0, 2.0],
            "high_top_p": [0.95, 1.0],
            "high_top_p_task": "story_s",
            "skipped": ["none_top_p", "high_temperature"],
            "ppl_model": PPL_MODEL_ID,
            "ppl_device": "cpu",
        },
        "api_totals": {
            "время_стены_сек": api_wall_s,
            "время_api_сумма_сек": round(api_sum_s, 2),
            "время_api_сек": round(api_sum_s, 2),  # совместимость со старым ключом
            "токены_reasoning": api_reason,
            "токены_всего": api_tok,
            "note": (
                "время_стены_сек — wall-clock async-пакета; "
                "время_api_сумма_сек — сумма длительностей запросов; "
                "без локального perplexity"
            ),
        },
        "prior_confirmed_checks": PRIOR_CONFIRMED_CHECKS,
        "ppl_scoring": ppl_meta,
        "result": result,
        "conclusion": conclusion,
        "rows": rows,
    }
    path = "task4_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n{path}")

    try:
        from task4_report import build_docx

        docx_path = Path("task4_result.docx")
        build_docx(out, docx_path)
        print(docx_path)
    except Exception as exc:  # noqa: BLE001
        print(f"docx_error: {exc}")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
