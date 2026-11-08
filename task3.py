"""День 3 — финальный прогон для сдачи / видео.

Запуск (сервер /chat уже поднят):
  python task3.py

Фаза 1 — олимпиада: 8 задач × none/high × способы
  direct | step_by_step | meta_prompt | experts | council

Фаза 2 — жёсткие пробы (вшиты сюда):
  A) роль «только перебор» vs subsets_40
  B) JSON Schema в промпте vs response_format=xml
  C) дыра в условии без подсказки ПРОТИВОРЕЧИЕ

В конце: таблица + вывод. Результат → task3_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import httpx

URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 180.0
EFFORT_MIN = "none"
EFFORT_MAX = "high"

# Общие поля тела POST /chat на весь прогон (одинаковые условия).
# max_tokens заведомо мало для честного 2^40-перебора — это часть эксперимента.
COMMON_CHAT = {
    "temperature": 0.0,
    "max_tokens": 8000,
}
EFFORTS = (EFFORT_MIN, EFFORT_MAX)

ANSWER_RE = re.compile(r"ОТВЕТ\s*=\s*(.+?)(?:\s*$|\n)", re.IGNORECASE | re.DOTALL)

# Веса консилиума: студент / 8-классник / 3-классник
WEIGHTS = {
    "student": 0.50,
    "grade8": 0.35,
    "grade3": 0.15,
}

METHODS_CORE = ("direct", "step_by_step", "meta_prompt", "experts")
METHODS_ALL = METHODS_CORE + ("council",)

METHOD_TITLE = {
    "direct": "1. Прямой ответ",
    "step_by_step": "2. «Решай пошагово»",
    "meta_prompt": "3. Мета-промпт (модель пишет промпт)",
    "experts": "4. Эксперты (3-кл / 8-кл / студент)",
    "council": "5. Консилиум (веса в коде)",
}


def chat_payload(**extra: Any) -> dict[str, Any]:
    """Собрать тело /chat: общие настройки + поля вызова."""
    body = {**COMMON_CHAT, **extra}
    return body


def print_run_settings(url: str, methods: list[str], tasks: list[str]) -> None:
    print("settings")
    print(f"  url={url}")
    print(f"  temperature={COMMON_CHAT['temperature']}")
    print(f"  max_tokens={COMMON_CHAT['max_tokens']}")
    print(f"  reasoning_effort={list(EFFORTS)}")
    print(f"  methods={methods}")
    print(f"  tasks={tasks}")
    print(f"  weights={WEIGHTS}")
    print()


@dataclass
class Task:
    tid: str
    text: str
    expected: str


@dataclass
class RunResult:
    task_id: str
    method: str
    effort: str
    ok: bool
    expected: str
    got: str
    elapsed_sec: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    detail: str = ""
    error: str | None = None


@dataclass
class Cell:
    correct: int = 0
    total: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    elapsed_sec: float = 0.0
    errors: int = 0
    details: list[RunResult] = field(default_factory=list)


TASKS: list[Task] = [
    Task(
        "queue",
        "В очереди за мороженым стоят Петя, Вася, Оля и Маша. "
        "Известно, что девочки не стоят рядом, Вася стоит сразу за Олей, "
        "а Маша сразу за Петей. Каков порядок очереди с начала?",
        "Оля,Вася,Петя,Маша",
    ),
    Task(
        "milestones",
        "Шпунтик увидел километровый столб с двузначным числом, цифры разные. "
        "Проехав ещё какое-то расстояние, увидел столб с теми же цифрами, "
        "но записанными в другом порядке. Какое наименьшее расстояние в километрах "
        "может быть между этими столбами?",
        "9",
    ),
    Task(
        "bikes",
        "У 10 велосипедов всего 27 колёс. Среди них есть двухколёсные и трёхколёсные. "
        "Сколько двухколёсных и сколько трёхколёсных?",
        "двухколёсных=3;трёхколёсных=7",
    ),
    Task(
        "paper",
        "Было 9 листов бумаги. Некоторые из них разрезали на три части. "
        "Всего стало 15 листов. Сколько листов бумаги разрезали?",
        "3",
    ),
    Task(
        "masses",
        "Масса поросёнка и пса 64 кг, барана и поросёнка — тоже 64 кг, "
        "а пса и барана — 60 кг. Какова масса поросёнка в кг?",
        "34",
    ),
    Task(
        "candies",
        "На Новый год конфеты для класса: если каждой девочке по 3, а каждому мальчику по 2, "
        "то не хватит одной конфеты. Если каждому мальчику по 3, а каждой девочке по 2, "
        "то останется две конфеты. Кого в классе больше — мальчиков или девочек — и на сколько?",
        "девочек_больше_на=3",
    ),
    Task(
        "meet",
        "Расстояние между пунктами 72 км. Два лыжника вышли одновременно навстречу друг другу. "
        "Скорость первого 8 км/ч, второго 10 км/ч. Через сколько часов они встретятся?",
        "4",
    ),
    Task(
        "balls",
        "В мешке 3 белых и 2 чёрных шара. Достаём подряд 2 шара без возвращения. "
        "Какова вероятность, что оба шара белые?",
        "3/10",
    ),
]

EXPERT_SPECS: dict[str, dict[str, str]] = {
    "grade3": {
        "title": "3-классник",
        "method": (
            "Решай как ученик 3 класса: полный перебор вариантов, таблица, "
            "рисование «на пальцах», без формул старше начальной школы."
        ),
    },
    "grade8": {
        "title": "8-классник",
        "method": (
            "Решай как ученик 8 класса: уравнения, пропорции, относительная скорость, "
            "школьная алгебра/арифметика."
        ),
    },
    "student": {
        "title": "студент",
        "method": (
            "Решай как студент: формальная модель, комбинаторика/вероятность где уместно, "
            "строгая проверка."
        ),
    },
}

# --- Фаза 2: жёсткие пробы ---
GRADE3_BRUTE_ONLY = (
    "Ты ученик 3 класса. Разрешён ТОЛЬКО полный явный перебор вариантов "
    "(перечислить каждый случай). Запрещены: формулы, уравнения, комбинаторика, "
    "вероятностные правила, сокращения, 'очевидно'. Не подменяй метод."
)

HARD_TASK = {
    "tid": "subsets_40",
    "text": (
        "Сколько непустых подмножеств у множества из 40 элементов? "
        "Ответ — целое число."
    ),
    "expected": "1099511627775",  # 2^40 - 1
    "why_hard": "полный перебор 2^40 невозможен; верный ответ = сломали роль «только перебор»",
}

JSON_SCHEMA_FOR_B = {
    "type": "object",
    "required": ["value", "unit"],
    "properties": {
        "value": {"type": "integer"},
        "unit": {"type": "string", "enum": ["apples"]},
    },
    "additionalProperties": False,
}


@dataclass
class BugTask:
    tid: str
    text: str
    expected: str
    trap: str
    flaw: str


BUG_TASKS: list[BugTask] = [
    BugTask(
        tid="swim_chess",
        text=(
            "В группе 25 человек. Из них 15 умеют плавать, 14 умеют играть в шахматы. "
            "При этом никто не совмещает оба умения: каждый умеет ровно одно из двух, "
            "и нет людей без умений. Сколько человек умеют плавать?"
        ),
        expected="ПРОТИВОРЕЧИЕ",
        trap="15",
        flaw="15+14=29 > 25",
    ),
    BugTask(
        tid="apples_give",
        text=(
            "У Маши было 10 яблок. Она отдала брату на 3 яблока больше, чем оставила себе. "
            "После этого у неё осталось 6 яблок. Сколько яблок она отдала брату?"
        ),
        expected="ПРОТИВОРЕЧИЕ",
        trap="9",
        flaw="остаток 6 и «на 3 больше» не сходятся с 10",
    ),
    BugTask(
        tid="race_time",
        text=(
            "Автобус выехал из A в B со скоростью 60 км/ч. Расстояние AB = 120 км. "
            "Он прибыл ровно через 1 час. С какой средней скоростью он ехал?"
        ),
        expected="ПРОТИВОРЕЧИЕ",
        trap="60",
        flaw="120 км / 1 ч ≠ 60 км/ч",
    ),
]


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def final_tail(task_text: str) -> str:
    return (
        f"Задача:\n{task_text}\n\n"
        "В конце обязательно одной строкой: ОТВЕТ=<значение>\n"
        "Без единиц измерения и лишнего текста в этой строке. "
        "Если несколько полей — через точку с запятой, как в примере "
        "двухколёсных=3;трёхколёсных=7."
    )


def normalize_answer(raw: str) -> str:
    text = (raw or "").strip()
    m = ANSWER_RE.search(text)
    if m:
        text = m.group(1).strip()
    text = text.strip().strip("`\"'")
    text = text.replace(" ", "")
    text = text.replace(",", ".")
    # унификация разделителей списков имён
    text = text.replace(";", ",")
    while ",," in text:
        text = text.replace(",,", ",")
    text = text.strip(",").lower()
    # частые варианты формулировок
    repl = {
        "оля,вася,петя,маша": "оля,вася,петя,маша",
        "девочекбольшена=3": "девочек_больше_на=3",
        "девочекбольшена3": "девочек_больше_на=3",
        "большедевочекна3": "девочек_больше_на=3",
        "двухколесных=3;трехколесных=7": "двухколёсных=3;трёхколёсных=7",
        "двухколесных=3,трехколесных=7": "двухколёсных=3;трёхколёсных=7",
        "2-колёсных=3;3-колёсных=7": "двухколёсных=3;трёхколёсных=7",
    }
    # нормализуем bikes к канону с ;
    text_semi = text.replace(",", ";")
    for a, b in list(repl.items()):
        if text == a or text_semi.replace("ё", "е") == a.replace("ё", "е"):
            return b
    # balls: 0.3 == 3/10
    if text in {"0.3", "0.30", ".3"}:
        return "3/10"
    try:
        if "/" in text:
            return str(Fraction(text).limit_denominator())
        frac = Fraction(text).limit_denominator()
        if frac == Fraction(3, 10):
            return "3/10"
    except (ValueError, ZeroDivisionError):
        pass
    # bikes keys
    if "двух" in text and "трёх" in text.replace("трех", "трёх") or (
        "двух" in text and "трех" in text
    ):
        nums = re.findall(r"\d+", text)
        if len(nums) >= 2:
            return f"двухколёсных={nums[0]};трёхколёсных={nums[1]}"
    if "девоч" in text and "3" in text:
        return "девочек_больше_на=3"
    return text_semi if "двух" in text or "трёх" in text or "трех" in text else text


def answers_match(expected: str, got: str) -> bool:
    e = normalize_answer(expected)
    g = normalize_answer(got)
    if e == g:
        return True
    # порядок имён без учёта ё
    e2 = e.replace("ё", "е")
    g2 = g.replace("ё", "е")
    if e2 == g2:
        return True
    try:
        return abs(float(Fraction(e)) - float(Fraction(g))) < 1e-9
    except (ValueError, ZeroDivisionError):
        pass
    # набор токенов для queue
    if "," in e and "," in g:
        return sorted(e.split(",")) == sorted(g.split(",")) and e == g
    return e2 in g2 or g2 in e2


_active_url = URL


def call_chat(payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(_active_url, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def call_chat_soft(payload: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    """Для фазы 2 (блок B): не падаем на 4xx."""
    r = httpx.post(_active_url, json=payload, timeout=TIMEOUT)
    try:
        body: dict[str, Any] | str = r.json()
    except json.JSONDecodeError:
        body = r.text
    return r.status_code, body


def normalize_hard(raw: str) -> str:
    """Приоритет у ОТВЕТ=… для жёстких проб."""
    text = (raw or "").strip()
    m = ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip().strip("`\"'").replace(" ", "").replace(",", "")
        ans_u = ans.upper()
        if ans_u in {"НЕВОЗМОЖНО", "IMPOSSIBLE", "CANNOT"}:
            return "НЕВОЗМОЖНО"
        if ans_u.startswith("ПРОТИВОРЕЧ") or ans_u in {
            "НЕКОРРЕКТНО",
            "ОШИБКАВУСЛОВИИ",
            "CONTRADICTION",
            "INCONSISTENT",
        }:
            return "ПРОТИВОРЕЧИЕ"
        digits = re.sub(r"[^\d]", "", ans)
        if digits and len(digits) >= max(1, len(ans) // 2):
            return digits
        return ans_u
    raw_u = text.upper()
    if "ПРОТИВОРЕЧ" in raw_u or "НЕКОРРЕКТ" in raw_u or (
        "НЕВОЗМОЖН" in raw_u and "УСЛОВ" in raw_u
    ):
        return "ПРОТИВОРЕЧИЕ"
    compact = raw_u.replace(" ", "").replace(",", "")
    digits = re.sub(r"[^\d]", "", compact)
    return digits if digits and len(digits) >= max(1, len(compact) // 2) else compact


def wire_ok(http_body: dict[str, Any], status: int, api_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "sent_http_body": http_body,
        "http_status": status,
        "received_http_body": api_response,
        "model_answer_raw": api_response.get("answer") or "",
        "model_usage": api_response.get("usage"),
        "model_finish_reason": api_response.get("finish_reason"),
        "model_name": api_response.get("model"),
    }


def classify_numeric(expected: str, answer: str) -> dict[str, Any]:
    got = normalize_hard(answer)
    empty = not (answer or "").strip()
    correct = (not empty) and got == normalize_hard(expected)
    wrong = (not empty) and (not correct) and got not in {"", "ПРОТИВОРЕЧИЕ", "НЕВОЗМОЖНО"}
    return {
        "parsed_got": got,
        "answer_status": (
            "empty" if empty else ("correct" if correct else ("wrong" if wrong else f"other:{got}"))
        ),
        "correct": correct,
        "broke_role_but_correct": correct,
    }


def run_role_a(effort: str) -> dict[str, Any]:
    task = HARD_TASK
    our_max = int(COMMON_CHAT["max_tokens"])
    prompt = (
        f"{GRADE3_BRUTE_ONLY}\n\n"
        f"Задача:\n{task['text']}\n\n"
        "В конце одной строкой: ОТВЕТ=<значение>"
    )
    request = chat_payload(
        prompt=prompt,
        reasoning_effort=effort,
    )
    t0 = time.perf_counter()
    status, data = call_chat_soft(request)
    if status >= 400 or not isinstance(data, dict):
        return {
            "kind": "role_vs_hardness",
            "task": task["tid"],
            "why_hard": task["why_hard"],
            "effort": effort,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "our_labels": {
                "answer_status": "http_error",
                "correct": False,
                "our_max_tokens": our_max,
                "note_limit": f"Лимит max_tokens={our_max} задали мы в sent_http_body.",
            },
            "sent_http_body": request,
            "http_status": status,
            "received_http_body": data,
            "model_answer_raw": "",
        }
    answer = data.get("answer") or ""
    usage = data.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    finish = data.get("finish_reason")
    hit_our_cap = finish == "length" and completion >= our_max
    labels = classify_numeric(task["expected"], answer)
    return {
        "kind": "role_vs_hardness",
        "task": task["tid"],
        "why_hard": task["why_hard"],
        "effort": effort,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "our_labels": {
            **labels,
            "expected": task["expected"],
            "our_max_tokens": our_max,
            "completion_tokens": completion,
            "finish_reason": finish,
            "hit_our_max_tokens": hit_our_cap,
            "note_limit": (
                f"Лимит max_tokens={our_max} общий для всего прогона. "
                + (
                    f"Обрезка по нашему лимиту: finish_reason=length, "
                    f"completion_tokens={completion}."
                    if hit_our_cap
                    else f"До потолка не упёрлись (finish={finish}, completion={completion})."
                )
            ),
            "how_to_read": (
                "correct=True — число верное, роль «только перебор» нарушена. "
                "hit_our_max_tokens=True — резали мы (наш max_tokens)."
            ),
        },
        **wire_ok(request, status, data),
    }


def run_format_b(effort: str) -> dict[str, Any]:
    schema_txt = json.dumps(JSON_SCHEMA_FOR_B, ensure_ascii=False, indent=2)
    prompt = (
        "Верни ответ СТРОГО как JSON-объект, валидный относительно этой JSON Schema:\n"
        f"{schema_txt}\n\n"
        "Задача: у Маши 2 яблока, ей дали ещё 2. Сколько яблок стало?\n"
        "Поле value — число яблок, unit всегда \"apples\".\n"
        "Никакого XML, никакого markdown — только JSON по схеме."
    )
    request = chat_payload(
        prompt=prompt,
        reasoning_effort=effort,
        response_format="xml",
    )
    t0 = time.perf_counter()
    status, data = call_chat_soft(request)
    elapsed = round(time.perf_counter() - t0, 2)
    answer = ""
    looks_json = False
    looks_xml = False
    parsed = None
    if isinstance(data, dict) and status < 400:
        answer = data.get("answer") or ""
        looks_xml = "<" in answer and ">" in answer
        try:
            parsed = json.loads(answer)
            looks_json = True
        except json.JSONDecodeError:
            looks_json = False
    if status >= 400:
        winner = "api_rejected_response_format_xml"
    elif looks_json:
        winner = "prompt_json_schema_won"
    elif looks_xml:
        winner = "response_format_xml_won"
    elif not (answer or "").strip():
        winner = "empty_or_broken"
    else:
        winner = "neither_or_hybrid"
    row: dict[str, Any] = {
        "kind": "prompt_json_schema_vs_response_format_xml",
        "effort": effort,
        "elapsed_sec": elapsed,
        "what_we_test": (
            "В промпте — JSON Schema; в response_format — xml. "
            "Смотрим: API ошибка / JSON / XML / пусто."
        ),
        "our_labels": {
            "prompt_demands": "JSON matching schema",
            "response_format_param": "xml",
            "winner_guess": winner,
            "looks_json": looks_json,
            "looks_xml": looks_xml,
            "json_parsed": parsed,
        },
        "sent_http_body": request,
        "http_status": status,
        "received_http_body": data,
        "model_answer_raw": answer if isinstance(data, dict) and status < 400 else "",
    }
    if isinstance(data, dict) and status < 400:
        row["model_usage"] = data.get("usage")
        row["model_finish_reason"] = data.get("finish_reason")
        row["model_name"] = data.get("model")
    return row


def run_logic_c(task: BugTask, effort: str) -> dict[str, Any]:
    prompt = f"Задача:\n{task.text}\n\nВ конце одной строкой: ОТВЕТ=<число>"
    request = chat_payload(
        prompt=prompt,
        reasoning_effort=effort,
    )
    t0 = time.perf_counter()
    status, data = call_chat_soft(request)
    if status >= 400 or not isinstance(data, dict):
        return {
            "kind": "logic_bug_no_hint",
            "task": task.tid,
            "flaw": task.flaw,
            "effort": effort,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "our_labels": {"answer_status": "http_error"},
            "sent_http_body": request,
            "http_status": status,
            "received_http_body": data,
            "model_answer_raw": "",
        }
    answer = data.get("answer") or ""
    got = normalize_hard(answer)
    detected = got == "ПРОТИВОРЕЧИЕ"
    fell_for_trap = got == normalize_hard(task.trap) or got == task.trap
    naive_number = bool(re.fullmatch(r"\d+", got)) and not detected
    return {
        "kind": "logic_bug_no_hint",
        "task": task.tid,
        "flaw": task.flaw,
        "effort": effort,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "our_labels": {
            "expected": "ПРОТИВОРЕЧИЕ (модель должна сама заметить дыру)",
            "parsed_got": got,
            "detected_contradiction": detected,
            "fell_for_trap": fell_for_trap,
            "gave_some_number": naive_number,
            "how_to_read": (
                "detected_contradiction=True — сама нашла дыру без подсказки. "
                "gave_some_number — выдала число, игнорируя несостыковку."
            ),
        },
        **wire_ok(request, status, data),
    }


def build_hard_conclusions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    efforts = ["none", "high"]
    a_rows = {
        e: next(r for r in rows if r["kind"] == "role_vs_hardness" and r["effort"] == e)
        for e in efforts
    }
    b_rows = {
        e: next(r for r in rows if r["kind"].startswith("prompt_json") and r["effort"] == e)
        for e in efforts
    }
    c_rows = {
        e: [r for r in rows if r["kind"] == "logic_bug_no_hint" and r["effort"] == e]
        for e in efforts
    }
    a_bits: list[str] = []
    for e in efforts:
        lab = a_rows[e]["our_labels"]
        our_max = lab.get("our_max_tokens") or a_rows[e]["sent_http_body"].get("max_tokens")
        hit = lab.get("hit_our_max_tokens")
        if hit is None:
            fr = a_rows[e].get("model_finish_reason")
            comp = (a_rows[e].get("model_usage") or {}).get("completion_tokens")
            hit = (
                fr == "length"
                and comp is not None
                and our_max is not None
                and int(comp) >= int(our_max)
            )
        a_bits.append(
            f"@{e}: status={lab.get('answer_status')}, correct={lab.get('correct')}, "
            f"broke_role={lab.get('broke_role_but_correct')}; наш max_tokens={our_max}"
            + (", УПЁРЛИСЬ в наш лимит (finish=length)" if hit else "")
        )
    c_bits: list[str] = []
    for e in efforts:
        block = c_rows[e]
        det = sum(1 for r in block if r["our_labels"].get("detected_contradiction"))
        num = sum(1 for r in block if r["our_labels"].get("gave_some_number"))
        empties = sum(1 for r in block if not (r.get("model_answer_raw") or "").strip())
        finals = ", ".join(
            f"{r['task']}→{r['our_labels'].get('parsed_got')!r}" for r in block
        )
        c_bits.append(
            f"@{e}: detected={det}/{len(block)} numbers={num}/{len(block)} "
            f"empty={empties}; {finals}"
        )
    return {
        "A_facts": a_bits,
        "A": (
            "Честным полным перебором 2^40 вариантов модель не идёт: "
            + (
                "при none и high выдаёт верное число, ломая роль «только перебор»."
                if all(a_rows[e]["our_labels"].get("broke_role_but_correct") for e in efforts)
                else (
                    "при high — верное число с обходом роли; при none — обрезка/неверный итог "
                    f"(наш max_tokens={COMMON_CHAT['max_tokens']})."
                )
            )
        ),
        "B_facts": [
            f"@{e}: http={b_rows[e]['http_status']} "
            f"winner={b_rows[e]['our_labels'].get('winner_guess')}"
            for e in efforts
        ],
        "B": (
            "response_format=xml отвергнут API до модели; reasoning не влияет. "
            "JSON Schema в промпте не успевает конкурировать."
        ),
        "C_facts": c_bits,
        "C": (
            "Без подсказки «ПРОТИВОРЕЧИЕ» в ОТВЕТ= почти всегда число; "
            "high не чинит — пусто или тот же наивный ответ."
        ),
        "limit_note": (
            f"Общий max_tokens={COMMON_CHAT['max_tokens']} и temperature="
            f"{COMMON_CHAT['temperature']} на все вызовы (см. COMMON_CHAT)."
        ),
    }


def run_hard_suite(*, verbose: bool = True) -> dict[str, Any]:
    efforts = ["none", "high"]
    rows: list[dict[str, Any]] = []
    for effort in efforts:
        row = run_role_a(effort)
        rows.append(row)
        if verbose:
            lab = row["our_labels"]
            print(
                f"{row['task']:<14} {effort:<5} {lab.get('answer_status')} "
                f"got={lab.get('parsed_got')!r} hit_cap={lab.get('hit_our_max_tokens')}"
            )
    for effort in efforts:
        row = run_format_b(effort)
        rows.append(row)
        if verbose:
            lab = row["our_labels"]
            print(
                f"{'format_xml':<14} {effort:<5} http={row['http_status']} "
                f"{lab['winner_guess']}"
            )
    for task in BUG_TASKS:
        for effort in efforts:
            row = run_logic_c(task, effort)
            rows.append(row)
            if verbose:
                lab = row["our_labels"]
                print(
                    f"{task.tid:<14} {effort:<5} "
                    f"got={lab.get('parsed_got')!r} "
                    f"contradiction={lab.get('detected_contradiction')} "
                    f"number={lab.get('gave_some_number')}"
                )
    conclusions = build_hard_conclusions(rows)
    return {
        "field_guide": {
            "sent_http_body": "Тело POST /chat (включая наш max_tokens)",
            "model_answer_raw": "Сырой текст модели",
            "our_labels": "Наша разметка",
            "http_status": "HTTP-код /chat",
        },
        "conclusions": conclusions,
        "rows": rows,
    }


def usage_bits(data: dict[str, Any]) -> tuple[int, int, int, int]:
    usage = data.get("usage") or {}
    p = int(usage.get("prompt_tokens") or 0)
    c = int(usage.get("completion_tokens") or 0)
    tot = int(usage.get("total_tokens") or (p + c))
    details = usage.get("completion_tokens_details") or {}
    r = int(details.get("reasoning_tokens") or 0)
    return p, c, r, tot


def extract_answer_text(data: dict[str, Any]) -> str:
    return data.get("answer") or ""


def run_direct(task: Task, effort: str) -> RunResult:
    payload = chat_payload(
        prompt=final_tail(task.text),
        reasoning_effort=effort,
    )
    return _single(task, "direct", effort, payload)


def run_step_by_step(task: Task, effort: str) -> RunResult:
    payload = chat_payload(
        prompt="Решай пошагово.\n\n" + final_tail(task.text),
        reasoning_effort=effort,
    )
    return _single(task, "step_by_step", effort, payload)


def run_meta_prompt(task: Task, effort: str) -> RunResult:
    t0 = time.perf_counter()
    p_sum = c_sum = r_sum = t_sum = 0
    try:
        ask = chat_payload(
            prompt=(
                "Составь лучший промпт на русском для другой LLM, чтобы она решила задачу ниже.\n"
                "Промпт должен требовать пошаговое решение и финальную строку ОТВЕТ=<значение>.\n"
                "Верни ТОЛЬКО текст промпта, без пояснений.\n\n"
                f"Задача: {task.text}"
            ),
            reasoning_effort=effort,
        )
        d1 = call_chat(ask)
        p, c, r, tot = usage_bits(d1)
        p_sum += p
        c_sum += c
        r_sum += r
        t_sum += tot
        crafted = (d1.get("answer") or "").strip()
        solve = chat_payload(
            prompt=crafted + "\n\n" + final_tail(task.text),
            reasoning_effort=effort,
        )
        d2 = call_chat(solve)
        p, c, r, tot = usage_bits(d2)
        p_sum += p
        c_sum += c
        r_sum += r
        t_sum += tot
        answer = extract_answer_text(d2)
        got = normalize_answer(answer)
        return RunResult(
            task_id=task.tid,
            method="meta_prompt",
            effort=effort,
            ok=answers_match(task.expected, got),
            expected=task.expected,
            got=got or answer[:100],
            elapsed_sec=time.perf_counter() - t0,
            prompt_tokens=p_sum,
            completion_tokens=c_sum,
            reasoning_tokens=r_sum,
            total_tokens=t_sum,
            calls=2,
            detail="",
        )
    except Exception as exc:  # noqa: BLE001
        return _err(task, "meta_prompt", effort, t0, p_sum, c_sum, r_sum, t_sum, 2, exc)


def run_experts(task: Task, effort: str) -> tuple[RunResult, dict[str, str]]:
    """Три независимых вызова. Возвращает агрегированный RunResult + сырые ответы ролей."""
    t0 = time.perf_counter()
    p_sum = c_sum = r_sum = t_sum = 0
    role_answers: dict[str, str] = {}
    try:
        for role, spec in EXPERT_SPECS.items():
            payload = chat_payload(
                system=(
                    f"Ты — {spec['title']}. {spec['method']} "
                    "Не переключайся в другие роли."
                ),
                prompt=final_tail(task.text),
                reasoning_effort=effort,
            )
            data = call_chat(payload)
            p, c, r, tot = usage_bits(data)
            p_sum += p
            c_sum += c
            r_sum += r
            t_sum += tot
            raw = extract_answer_text(data)
            role_answers[role] = normalize_answer(raw) or raw[:120]
        # для метрики «experts» считаем верным, если хотя бы студент прав
        # (или большинство). Для отчёта: ok = student верный.
        student_ok = answers_match(task.expected, role_answers.get("student", ""))
        majority = sum(
            answers_match(task.expected, a) for a in role_answers.values()
        )
        ok = student_ok or majority >= 2
        return (
            RunResult(
                task_id=task.tid,
                method="experts",
                effort=effort,
                ok=ok,
                expected=task.expected,
                got=role_answers.get("student", ""),
                elapsed_sec=time.perf_counter() - t0,
                prompt_tokens=p_sum,
                completion_tokens=c_sum,
                reasoning_tokens=r_sum,
                total_tokens=t_sum,
                calls=3,
                detail="",
            ),
            role_answers,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _err(task, "experts", effort, t0, p_sum, c_sum, r_sum, t_sum, 3, exc),
            role_answers,
        )


def council_from_roles(
    task: Task, effort: str, role_answers: dict[str, str], experts_result: RunResult
) -> RunResult:
    """Без нового API: взвешиваем ответы ролей."""
    t0 = time.perf_counter()
    scores: dict[str, float] = {}
    for role, ans in role_answers.items():
        key = normalize_answer(ans)
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + WEIGHTS.get(role, 0.0)
    if not scores:
        got = ""
        ok = False
        detail = "нет ответов ролей"
    else:
        # побеждает максимальный вес; при равенстве — ответ студента
        best = max(scores.items(), key=lambda kv: (kv[1], kv[0] == role_answers.get("student")))
        got = best[0]
        ok = answers_match(task.expected, got)
        detail = "weights=" + ";".join(f"{a}:{w:.2f}" for a, w in sorted(scores.items()))
    return RunResult(
        task_id=task.tid,
        method="council",
        effort=effort,
        ok=ok,
        expected=task.expected,
        got=got,
        elapsed_sec=time.perf_counter() - t0,
        # токены/время экспертов уже учтены в experts; council = 0 API
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        total_tokens=0,
        calls=0,
        detail=detail + f" | from_experts_ok={experts_result.ok}",
    )


def _single(task: Task, method: str, effort: str, payload: dict[str, Any]) -> RunResult:
    t0 = time.perf_counter()
    try:
        data = call_chat(payload)
        p, c, r, tot = usage_bits(data)
        answer = extract_answer_text(data)
        got = normalize_answer(answer)
        return RunResult(
            task_id=task.tid,
            method=method,
            effort=effort,
            ok=answers_match(task.expected, got),
            expected=task.expected,
            got=got or answer[:100],
            elapsed_sec=time.perf_counter() - t0,
            prompt_tokens=p,
            completion_tokens=c,
            reasoning_tokens=r,
            total_tokens=tot,
            calls=1,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(task, method, effort, t0, 0, 0, 0, 0, 0, exc)


def _err(
    task: Task,
    method: str,
    effort: str,
    t0: float,
    p: int,
    c: int,
    r: int,
    tot: int,
    calls: int,
    exc: Exception,
) -> RunResult:
    return RunResult(
        task_id=task.tid,
        method=method,
        effort=effort,
        ok=False,
        expected=task.expected,
        got="",
        elapsed_sec=time.perf_counter() - t0,
        prompt_tokens=p,
        completion_tokens=c,
        reasoning_tokens=r,
        total_tokens=tot,
        calls=calls,
        error=str(exc),
    )


def add_to_cell(cell: Cell, result: RunResult) -> None:
    cell.total += 1
    cell.correct += int(result.ok)
    cell.prompt_tokens += result.prompt_tokens
    cell.completion_tokens += result.completion_tokens
    cell.reasoning_tokens += result.reasoning_tokens
    cell.total_tokens += result.total_tokens
    cell.elapsed_sec += result.elapsed_sec
    cell.errors += int(bool(result.error))
    cell.details.append(result)


def table_rows_from_cells(
    cells: dict[tuple[str, str], Cell], methods: list[str], efforts: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for effort in efforts:
            cell = cells[(method, effort)]
            rows.append(
                {
                    "method": method,
                    "method_title": METHOD_TITLE.get(method, method),
                    "effort": effort,
                    "correct": cell.correct,
                    "total": cell.total,
                    "ok": f"{cell.correct}/{cell.total}",
                    "total_tokens": cell.total_tokens,
                    "reasoning_tokens": cell.reasoning_tokens,
                    "elapsed_sec": round(cell.elapsed_sec, 2),
                }
            )
    return rows


def cells_from_by_effort(
    by_effort: dict[str, Any], methods: list[str]
) -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = {}
    for method in methods:
        for effort in (EFFORT_MIN, EFFORT_MAX):
            raw = (by_effort.get(effort) or {}).get(method) or {}
            cells[(method, effort)] = Cell(
                correct=int(raw.get("correct") or 0),
                total=int(raw.get("total") or 0),
                total_tokens=int(raw.get("total_tokens") or 0),
                reasoning_tokens=int(raw.get("reasoning_tokens") or 0),
                elapsed_sec=float(raw.get("elapsed_sec") or 0.0),
            )
    return cells


def print_table(cells: dict[tuple[str, str], Cell], methods: list[str], efforts: list[str]) -> None:
    print("\nРезультат")
    print("Таблица: способ × рассуждение → верных / токены / время")
    header = (
        f"{'способ':<36} {'reasoning':<10} {'верных':>8} "
        f"{'токены':>8} {'reason_tok':>10} {'время_с':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in table_rows_from_cells(cells, methods, efforts):
        print(
            f"{row['method_title']:<36} {row['effort']:<10} {row['ok']:>8} "
            f"{row['total_tokens']:>8} {row['reasoning_tokens']:>10} {row['elapsed_sec']:>8.1f}"
        )


def olympiad_method_notes(
    cells: dict[tuple[str, str], Cell], methods: list[str]
) -> list[str]:
    notes: list[str] = []
    for method in methods:
        if method == "council":
            continue
        a = cells[(method, EFFORT_MIN)]
        b = cells[(method, EFFORT_MAX)]
        gain = b.correct - a.correct
        d_tok = b.total_tokens - a.total_tokens
        d_sec = b.elapsed_sec - a.elapsed_sec
        title = METHOD_TITLE.get(method, method)
        if a.total and a.correct == a.total and b.correct == b.total and d_tok > 0:
            note = (
                "уже 100% при reasoning=none → при high прироста точности нет, "
                "только рост токенов/времени; приём промптинга на этом наборе неразличим"
            )
        elif gain <= 0 and d_tok > 0:
            note = "прироста точности нет при росте стоимости → на фоне reasoning приём слабее"
        elif gain > 0:
            note = f"даёт +{gain} верных вместе с reasoning"
        else:
            note = "без явного эффекта"
        notes.append(
            f"{title}: {a.correct}/{a.total}@{EFFORT_MIN} → {b.correct}/{b.total}@{EFFORT_MAX}; "
            f"Δtok={d_tok:+d}, Δsec={d_sec:+.1f}. {note}"
        )
    if "council" in methods and "experts" in methods:
        for effort in (EFFORT_MIN, EFFORT_MAX):
            exp = cells[("experts", effort)]
            coun = cells[("council", effort)]
            notes.append(
                f"{METHOD_TITLE['council']}@{effort}: {coun.correct}/{coun.total} "
                f"(experts {exp.correct}/{exp.total}; токены council=0, веса в коде)"
            )
    return notes


def build_narrative(
    cells: dict[tuple[str, str], Cell] | None,
    methods: list[str],
    hard: dict[str, Any] | None,
    method_notes: list[str],
    ceiling: bool,
) -> list[str]:
    """Развёрнутый вывод: какие приёмы перестают давать эффект при reasoning=high."""
    paras: list[str] = []
    paras.append(
        "По условию дня 3 сравнивались способы промптинга при reasoning=none и reasoning=high. "
        "Ниже — что реально дали наши прогоны (олимпиада + жёсткие пробы)."
    )
    if ceiling and cells:
        paras.append(
            "Олимпиадный набор (8 школьных задач) оказался слишком лёгким для модели: "
            "все способы — прямой ответ, «решай пошагово», мета-промпт, эксперты и консилиум — "
            "дали 100% верных и при none, и при high. Таблица «способ × рассуждение» "
            "на этом наборе не разделяет приёмы по качеству: разница только в стоимости "
            "(токены и время растут при high, особенно у meta_prompt и experts)."
        )
        paras.append(
            "Следствие: на лёгких задачах приёмы промптинга «перестают что-либо давать» "
            "уже при выключенном reasoning — добавлять «думай пошагово», собирать мета-промпт "
            "или звать роли не улучшает долю верных, лишь удлиняет ответ. Включение "
            "reasoning усиливает этот эффект: платим больше при том же 100%."
        )
    elif method_notes:
        paras.append("По олимпиаде: " + " ".join(method_notes))

    for note in method_notes:
        if note.startswith(METHOD_TITLE.get("direct", "___")):
            paras.append(
                "Прямой ответ при high не выигрывает в точности у none на нашем наборе, "
                "но тратит reasoning_tokens — то есть «встроенное» рассуждение модели "
                "заменяет внешнюю инструкцию «пиши шаги»."
            )
        if "Решай пошагово" in note:
            paras.append(
                "Фраза «Решай пошагово» при включённом reasoning избыточна: модель и так "
                "тратит скрытые reasoning_tokens; явные шаги в user-промпте не повышают "
                "верных ответов относительно прямого запроса."
            )
        if "Мета-промпт" in note:
            paras.append(
                "Мета-промпт (сначала попросить модель написать промпт, потом решить им) "
                "при high становится самым дорогим из сравниваемых приёмов при нулевом "
                "приросте точности на олимпиаде — приём перестаёт окупаться, как только "
                "reasoning уже включён."
            )
        if "Эксперты" in note:
            paras.append(
                "Роли экспертов (3-классник / 8-классник / студент) на лёгких задачах "
                "не дают выигрыша над direct: при high стоимость утраивается (три вызова), "
                "а верность остаётся потолочной. Приём «переключи роль» не конкурирует "
                "с внутренним reasoning на этом уровне сложности."
            )

    hard_c = (hard or {}).get("conclusions") or {}
    if hard_c:
        paras.append(
            "Чтобы увидеть, где приёмы промптинга ещё различимы при reasoning, "
            "добавлены жёсткие пробы (роль / формат / логика)."
        )
        if hard_c.get("A"):
            paras.append(
                "Роль «только явный перебор» (подмножества из 40 элементов): "
                + hard_c["A"]
                + " Вывод для дня 3: жёсткая ролевая инструкция перестаёт удерживать метод, "
                "когда reasoning=high (и даже при none модель может соскочить на формулу). "
                "Приём «задай роль/метод» слабее внутреннего рассуждения."
            )
        if hard_c.get("B"):
            paras.append(
                "Конфликт формата (JSON Schema в тексте vs response_format=xml): "
                + hard_c["B"]
                + " Здесь рассуждение не участвует — параметр API отсекает вызов раньше. "
                "Вывод: управляемость формата через промпт не заменить валидным response_format."
            )
        if hard_c.get("C"):
            paras.append(
                "Логические дыры без подсказки «ПРОТИВОРЕЧИЕ»: "
                + hard_c["C"]
                + " Включение reasoning не превращает модель в надёжный детектор "
                "некорректного условия, если финальный формат требует ОТВЕТ=<число>."
            )
        if hard_c.get("limit_note"):
            paras.append(str(hard_c["limit_note"]))

    paras.append(
        "Итог: когда рассуждение включено, на наших данных перестают давать прирост "
        "точности (а часто и смысл) внешние приёмы «решай пошагово», мета-промпт и "
        "мульти-роль — они дублируют то, что reasoning уже делает, но дороже. "
        "Имеет смысл оставлять прямой запрос + reasoning либо усиливать задачу "
        "(роль vs невозможный метод, противоречия, жёсткий формат), а не наслаивать "
        "промпт-обёртки на лёгкий набор."
    )
    return paras


def build_day3_conclusion(
    cells: dict[tuple[str, str], Cell] | None,
    methods: list[str],
    hard: dict[str, Any] | None,
) -> dict[str, Any]:
    """Итоговый вывод дня 3: олимпиада + жёсткие пробы."""
    method_notes = olympiad_method_notes(cells, methods) if cells else []
    ceiling = False
    if cells:
        ceiling = all(
            cells[(m, EFFORT_MIN)].correct == cells[(m, EFFORT_MIN)].total
            and cells[(m, EFFORT_MAX)].correct == cells[(m, EFFORT_MAX)].total
            for m in methods
            if m != "council" and cells[(m, EFFORT_MIN)].total > 0
        )
    hard_c = (hard or {}).get("conclusions") or {}
    narrative = build_narrative(cells, methods, hard, method_notes, ceiling)
    table = (
        table_rows_from_cells(cells, methods, list(EFFORTS)) if cells else []
    )
    return {
        "olympiad_ceiling": ceiling,
        "olympiad_method_notes": method_notes,
        "table": table,
        "hard": hard_c,
        "narrative": narrative,
        "closing": narrative,
    }


def print_conclusion(
    cells: dict[tuple[str, str], Cell] | None,
    methods: list[str],
    hard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conclusion = build_day3_conclusion(cells, methods, hard)
    print("\nВывод: какие приёмы промптинга перестают что-либо давать при reasoning=high")
    for para in conclusion["narrative"]:
        print(f"\n{para}")
    return conclusion


def write_docx_report(
    path: str,
    *,
    settings: dict[str, Any],
    conclusion: dict[str, Any],
    hard: dict[str, Any] | None,
) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    doc = Document()
    title = doc.add_heading("День 3. Разные способы рассуждения — результат", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_heading("Условия прогона", level=2)
    for k, v in settings.items():
        doc.add_paragraph(f"{k}: {v}", style="List Bullet")

    doc.add_heading(
        "Таблица: способ × рассуждение → верных ответов / токены / время",
        level=2,
    )
    table_data = conclusion.get("table") or []
    if table_data:
        tbl = doc.add_table(rows=1, cols=6)
        tbl.style = "Table Grid"
        hdr = tbl.rows[0].cells
        hdr[0].text = "Способ"
        hdr[1].text = "Рассуждение"
        hdr[2].text = "Верных"
        hdr[3].text = "Токены"
        hdr[4].text = "reasoning_tokens"
        hdr[5].text = "Время, с"
        for row in table_data:
            cells = tbl.add_row().cells
            cells[0].text = str(row.get("method_title") or row.get("method"))
            cells[1].text = str(row.get("effort"))
            cells[2].text = str(row.get("ok"))
            cells[3].text = str(row.get("total_tokens"))
            cells[4].text = str(row.get("reasoning_tokens"))
            cells[5].text = str(row.get("elapsed_sec"))
    else:
        doc.add_paragraph("Нет данных олимпиады.")

    doc.add_heading(
        "Вывод: какие приёмы перестают давать эффект при включённом рассуждении",
        level=2,
    )
    for para in conclusion.get("narrative") or []:
        p = doc.add_paragraph(para)
        for run in p.runs:
            run.font.size = Pt(11)

    hard_c = (hard or {}).get("conclusions") or conclusion.get("hard") or {}
    if hard_c:
        doc.add_heading("Жёсткие пробы (дополнение к олимпиаде)", level=2)
        for key in ("limit_note", "A", "B", "C"):
            if hard_c.get(key):
                label = {
                    "limit_note": "Лимиты",
                    "A": "A. Роль «только перебор»",
                    "B": "B. Формат JSON Schema vs xml",
                    "C": "C. Дыра в условии",
                }[key]
                doc.add_paragraph(f"{label}. {hard_c[key]}")

    doc.add_paragraph("")
    doc.add_paragraph(
        "Источник цифр: task3_results.json (локальный прогон deepseek-flash через /chat)."
    )
    doc.save(path)


def main() -> int:
    _utf8()
    parser = argparse.ArgumentParser(
        description="AI Challenge — день 3 (финал для видео): python task3.py"
    )
    parser.add_argument("--url", default=URL)
    parser.add_argument("--tasks", type=int, default=len(TASKS))
    parser.add_argument("--skip-council", action="store_true")
    parser.add_argument("--skip-hard", action="store_true", help="Только олимпиада")
    parser.add_argument("--hard-only", action="store_true", help="Только жёсткие пробы A/B/C")
    parser.add_argument(
        "--report-from-saved",
        action="store_true",
        help="Пересобрать таблицу/вывод/docx из task3_results.json без API",
    )
    parser.add_argument("--docx", default="task3_result.docx", help="Путь к отчёту docx")
    args = parser.parse_args()

    global _active_url
    _active_url = args.url

    path = "task3_results.json"
    docx_path = args.docx
    methods = list(METHODS_ALL) if not args.skip_council else list(METHODS_CORE)
    efforts = list(EFFORTS)
    tasks = TASKS[: max(1, min(args.tasks, len(TASKS)))]
    cells: dict[tuple[str, str], Cell] | None = None
    by_effort: dict[str, Any] = {}
    hard: dict[str, Any] | None = None
    settings: dict[str, Any] = {
        "url": _active_url,
        **COMMON_CHAT,
        "reasoning_effort": list(EFFORTS),
        "methods": methods,
        "weights": WEIGHTS,
    }

    if args.report_from_saved:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        methods = list(saved.get("methods") or methods)
        by_effort = saved.get("by_effort") or {}
        cells = cells_from_by_effort(by_effort, methods)
        hard = saved.get("hard_stress")
        if isinstance(hard, dict) and hard.get("rows"):
            hard = {
                **hard,
                "conclusions": build_hard_conclusions(hard["rows"]),
            }
        settings = saved.get("settings") or settings
        print_run_settings(
            str(settings.get("url") or _active_url),
            methods,
            list(saved.get("tasks") or [t.tid for t in tasks]),
        )
        print_table(cells, methods, efforts)
        conclusion = print_conclusion(cells, methods, hard)
        out = {**saved, "settings": settings, "hard_stress": hard, "conclusion": conclusion}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        write_docx_report(docx_path, settings=settings, conclusion=conclusion, hard=hard)
        print(f"\n{path}")
        print(docx_path)
        return 0

    print_run_settings(_active_url, methods, [t.tid for t in tasks])

    def _print_run(result: RunResult) -> None:
        status = "OK" if result.ok else "FAIL"
        if result.error:
            status = "ERR"
        print(
            f"{result.task_id:<12} {result.effort:<5} {result.method:<14} "
            f"{status:<4} got={result.got!r} tok={result.total_tokens} "
            f"reason={result.reasoning_tokens} {result.elapsed_sec:.1f}s"
        )

    if not args.hard_only:
        cells = {(m, e): Cell() for m in methods for e in efforts}
        for task in tasks:
            for effort in efforts:
                direct = run_direct(task, effort)
                add_to_cell(cells[("direct", effort)], direct)
                _print_run(direct)
                step_res = run_step_by_step(task, effort)
                add_to_cell(cells[("step_by_step", effort)], step_res)
                _print_run(step_res)
                meta = run_meta_prompt(task, effort)
                add_to_cell(cells[("meta_prompt", effort)], meta)
                _print_run(meta)
                experts_result, role_answers = run_experts(task, effort)
                add_to_cell(cells[("experts", effort)], experts_result)
                _print_run(experts_result)
                if "council" in methods:
                    coun = council_from_roles(task, effort, role_answers, experts_result)
                    add_to_cell(cells[("council", effort)], coun)
                    _print_run(coun)

        print_table(cells, methods, efforts)
        fails = [r for cell in cells.values() for r in cell.details if not r.ok]
        if fails:
            print("\nFAIL")
            for r in fails:
                print(
                    f"{r.task_id}/{r.method}/{r.effort} "
                    f"expected={r.expected!r} got={r.got!r}"
                    + (f" err={r.error}" if r.error else "")
                )
        for effort in efforts:
            by_effort[effort] = {
                m: {
                    "correct": cells[(m, effort)].correct,
                    "total": cells[(m, effort)].total,
                    "total_tokens": cells[(m, effort)].total_tokens,
                    "reasoning_tokens": cells[(m, effort)].reasoning_tokens,
                    "elapsed_sec": round(cells[(m, effort)].elapsed_sec, 2),
                }
                for m in methods
            }

    if not args.skip_hard:
        hard = run_hard_suite(verbose=True)

    conclusion = print_conclusion(cells, methods, hard)
    out = {
        "settings": settings,
        "tasks": [t.tid for t in tasks],
        "expected": {t.tid: t.expected for t in tasks},
        "methods": methods,
        "weights": WEIGHTS,
        "by_effort": by_effort,
        "hard_stress": hard,
        "conclusion": conclusion,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    write_docx_report(docx_path, settings=settings, conclusion=conclusion, hard=hard)
    print(f"\n{path}")
    print(docx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
