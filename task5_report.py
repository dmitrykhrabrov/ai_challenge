"""Собрать task5_result.docx по task5_results.json (без нового API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def _utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def add_para(doc: Document, text: str, *, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.bold = bold


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        for run in p.runs:
            run.font.size = Pt(11)


def table_from_dicts(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    for i, h in enumerate(headers):
        tbl.rows[0].cells[i].text = h
    for row in rows:
        cells = tbl.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val


def olympiad_matrix(runs: list[dict[str, Any]], task_ids: list[str]) -> list[list[str]]:
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        if r.get("suite") != "olympiad":
            continue
        by.setdefault(r["task_id"], {})[r["tier"]] = r
    rows: list[list[str]] = []
    for tid in task_ids:
        cells = [tid]
        for tier in ("small", "mid", "flagship"):
            r = (by.get(tid) or {}).get(tier)
            if not r:
                cells.append("—")
                continue
            mark = "верно" if r.get("ok") else "нет"
            got = str(r.get("got") or "")[:40]
            cells.append(f"{mark}; got={got}; {r.get('elapsed_sec')}с; {r.get('total_tokens')} tok")
        rows.append(cells)
    return rows


def probe_rows(probes: dict[str, Any], suite: str) -> list[list[str]]:
    block = probes.get(suite) or {}
    out: list[list[str]] = []
    for tier in ("small", "mid", "flagship"):
        for item in block.get(tier) or []:
            lab = item.get("labels") or {}
            if suite == "A":
                detail = (
                    f"broke_role={lab.get('broke_role_but_correct')}; "
                    f"hit_cap={lab.get('hit_our_max_tokens')}"
                )
            elif suite == "B":
                detail = (
                    f"winner={lab.get('winner_guess')}; "
                    f"http={item.get('http_status')}"
                )
            else:
                detail = (
                    f"contradiction={lab.get('detected_contradiction')}; "
                    f"trap={lab.get('fell_for_trap')}; "
                    f"number={lab.get('gave_some_number')}"
                )
            out.append(
                [
                    tier,
                    str(item.get("task_id")),
                    "да" if item.get("ok") else "нет",
                    str(item.get("got") or "")[:80],
                    detail,
                    f"{item.get('elapsed_sec')}с / {item.get('total_tokens')} tok",
                ]
            )
    return out


def build_docx(data: dict[str, Any], out_path: Path) -> None:
    settings = data.get("settings") or {}
    table = data.get("table_olympiad") or []
    conclusion = data.get("conclusion") or {}
    runs = data.get("runs") or []
    models = settings.get("models") or []
    probes_meta = settings.get("probes") or {}
    probes = conclusion.get("probes") or {}

    doc = Document()
    title = doc.add_heading("День 5. Версии моделей — отчёт по прогону", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_heading("1. Цель и постановка", level=2)
    add_para(
        doc,
        "Сравнить маленькую, среднюю и флагманскую модели на одном наборе задач: "
        "сколько верных ответов, время, токены и стоимость. Источник задания — "
        "пост челленджа «День 5. Версии моделей» (таблица модель → метрики + вывод, "
        "на каких задачах маленькой модели достаточно).",
    )
    add_para(doc, "Что гоняли в этом прогоне:", bold=True)
    add_bullets(
        doc,
        [
            f"Олимпиада (5 сложных из дня 3): {', '.join(settings.get('olympiad_tasks') or [])}",
            f"Почему эти 5: {settings.get('olympiad_why') or '—'}",
            f"Проба A: роль «только перебор» vs {probes_meta.get('A')}",
            f"Проба B: JSON Schema в промпте vs response_format=xml ({probes_meta.get('B')})",
            f"Проба C: противоречие без подсказки — {', '.join(probes_meta.get('C') or [])}",
        ],
    )

    doc.add_heading("2. Модели и настройки", level=2)
    add_bullets(
        doc,
        [
            f"wall-clock прогона: {settings.get('wall_sec')} с (async, concurrency={settings.get('concurrency')})",
            f"temperature={settings.get('temperature')}, max_tokens={settings.get('max_tokens')}, "
            f"max_tokens_A={settings.get('max_tokens_A')}",
            f"deepseek_effort={settings.get('deepseek_effort')} (у флагмана thinking выключен при none)",
            f"OpenRouter: {settings.get('openrouter_base_url')}",
            f"DeepSeek: {settings.get('deepseek_base_url')}",
        ],
    )
    table_from_dicts(
        doc,
        ["tier", "label", "provider", "model_id", "цена (заметка)"],
        [
            [
                str(m.get("tier")),
                str(m.get("label")),
                str(m.get("provider")),
                str(m.get("model_id")),
                str(m.get("price_note") or ""),
            ]
            for m in models
        ],
    )
    add_para(
        doc,
        "Стоимость deepseek-v4-pro — оценка off-peak / cache-miss по официальному прайсу "
        "($0.66 / $1.98 за 1M prompt/completion). Free-модели OpenRouter — $0.",
    )

    doc.add_heading("3. Олимпиада: сводная таблица", level=2)
    table_from_dicts(
        doc,
        [
            "tier",
            "model",
            "верных",
            "accuracy",
            "время_с",
            "токены",
            "reasoning",
            "стоимость_$",
        ],
        [
            [
                str(r.get("tier")),
                str(r.get("model_id")),
                f"{r.get('correct')}/{r.get('total')}",
                str(r.get("accuracy")),
                str(r.get("elapsed_sec")),
                str(r.get("total_tokens")),
                str(r.get("reasoning_tokens")),
                f"{float(r.get('cost_usd') or 0):.6f}",
            ]
            for r in table
        ],
    )

    doc.add_heading("3.1. Матрица задач × модели", level=3)
    table_from_dicts(
        doc,
        ["задача", "small", "mid", "flagship"],
        olympiad_matrix(runs, list(settings.get("olympiad_tasks") or [])),
    )

    doc.add_heading("4. Пробы A / B / C", level=2)

    doc.add_heading("4.1. A — роль «только перебор» vs subsets_40", level=3)
    add_para(
        doc,
        "Промпт запрещает формулы и требует полный перебор; задача — число непустых "
        "подмножеств множества из 40 элементов (2^40−1). Верный ответ означает, "
        "что модель обошла роль и воспользовалась формулой/знанием.",
    )
    table_from_dicts(
        doc,
        ["tier", "task", "ok", "got", "детали", "время/токены"],
        probe_rows(probes, "A"),
    )

    doc.add_heading("4.2. B — JSON Schema в тексте vs response_format=xml", level=3)
    add_para(
        doc,
        "В промпте требуется JSON по схеме; в параметре API передан response_format=xml. "
        "Смотрим, дойдёт ли запрос до модели и победит ли промпт.",
    )
    table_from_dicts(
        doc,
        ["tier", "task", "ok", "got", "детали", "время/токены"],
        probe_rows(probes, "B"),
    )

    doc.add_heading("4.3. C — противоречие в условии без подсказки", level=3)
    add_para(
        doc,
        "В конце требуется ОТВЕТ=<число>, слова «ПРОТИВОРЕЧИЕ» в инструкции нет. "
        "Успех — модель сама пишет ПРОТИВОРЕЧИЕ (или эквивалент), а не наивное число.",
    )
    table_from_dicts(
        doc,
        ["tier", "task", "ok", "got", "детали", "время/токены"],
        probe_rows(probes, "C"),
    )

    doc.add_heading("5. Выводы", level=2)

    small = next((r for r in table if r.get("tier") == "small"), {})
    mid = next((r for r in table if r.get("tier") == "mid"), {})
    flag = next((r for r in table if r.get("tier") == "flagship"), {})

    add_para(doc, "Олимпиада (5 сложных задач):", bold=True)
    add_bullets(
        doc,
        [
            f"small ({small.get('model_id')}): {small.get('correct')}/{small.get('total')} "
            f"(accuracy={small.get('accuracy')}), Σ время {small.get('elapsed_sec')} с, "
            f"{small.get('total_tokens')} tok, ${float(small.get('cost_usd') or 0):.6f}",
            f"mid ({mid.get('model_id')}): {mid.get('correct')}/{mid.get('total')} "
            f"(accuracy={mid.get('accuracy')}), Σ время {mid.get('elapsed_sec')} с, "
            f"{mid.get('total_tokens')} tok, ${float(mid.get('cost_usd') or 0):.6f}",
            f"flagship ({flag.get('model_id')}): {flag.get('correct')}/{flag.get('total')} "
            f"(accuracy={flag.get('accuracy')}), Σ время {flag.get('elapsed_sec')} с, "
            f"{flag.get('total_tokens')} tok, ≈${float(flag.get('cost_usd') or 0):.6f}",
            f"delta accuracy flagship−small: "
            f"{conclusion.get('olympiad_flagship_vs_small_accuracy_delta')}",
            f"отношение стоимости: {conclusion.get('olympiad_flagship_vs_small_cost_ratio')}",
            f"где small тоже справился (и flagship тоже): "
            f"{', '.join(conclusion.get('olympiad_small_enough_tasks') or []) or '—'}",
        ],
    )

    add_para(doc, "Интерпретация олимпиады:", bold=True)
    add_bullets(
        doc,
        [
            "На урезанном «сложном» наборе флагман заметно точнее: 5/5 против 2/5 у small "
            "и 3/5 у mid (accuracy delta +0.6 к small).",
            "Флагман при этом быстрее и экономнее по токенам (reasoning_tokens=0 при "
            "effort=none); free mid/small тратят тысячи токенов на внутреннее reasoning.",
            "Маленькой модели «достаточно» только на части задач (в прогоне: balls, bikes). "
            "На candies / milestones / masses small уже не тянет — нужен более сильный класс.",
            "Стоимость Pro на 5 задач ≈ $0.002 при free baseline: по деньгам флагман дешёвый, "
            "выигрыш — в качестве и latency, не в «экономии долларов на free».",
        ],
    )

    add_para(doc, "Пробы A/B/C:", bold=True)
    add_bullets(
        doc,
        [
            "A: все три модели дали верное 2^40−1 и тем самым сломали роль «только перебор». "
            "Разница — в цене/скорости (flagship ~11 с / 962 tok vs mid ~91 с / 7.5k tok). "
            "Вывод: запрет метода в system/prompt не удерживает даже маленькую модель на "
            "комбинаторно невозможном переборе.",
            "B: у всех провайдеров запрос с response_format=xml отвергнут API до модели "
            "(400/422; xml не в списке допустимых type). Промпт со схемой не конкурирует — "
            "конфликт решается на валидации API, не на уровне модели. Это повторяет вывод дня 3.",
            "C: ни одна модель не написала ПРОТИВОРЕЧИЕ без подсказки (0/3 у всех). "
            "Small/mid часто пустой или мусорный ответ / упор в max_tokens; flagship стабильно "
            "даёт наивное число (ловушка или «исправленная» арифметика: 15, 4, 120). "
            "Флагман здесь не «умнее в смысле детекции дыры» — он увереннее ошибается.",
        ],
    )

    add_para(doc, "Итог для сдачи дня 5:", bold=True)
    add_bullets(
        doc,
        [
            "Для сложных школьных задач с известным ответом флагман (deepseek-v4-pro) "
            "окупается точностью и скоростью; маленькой free-модели достаточно лишь на "
            "части набора.",
            "Средняя free (Nemotron Lightning) занимает промежуток по accuracy, но хуже "
            "флагмана по времени и не закрывает противоречия/форматный конфликт.",
            "Жёсткие пробы показывают границы: роль не держит (A), невалидный "
            "response_format ломает вызов до LLM (B), скрытое противоречие без явной "
            "инструкции не ловится даже флагманом при effort=none (C).",
        ],
    )

    add_para(
        doc,
        f"Источник цифр: task5_results.json (wall={settings.get('wall_sec')} с). "
        f"{conclusion.get('note') or ''}",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)


def main() -> int:
    _utf8()
    src = Path("task5_results.json")
    if not src.exists():
        print(f"нет файла {src}")
        return 1
    data = json.loads(src.read_text(encoding="utf-8"))
    out = Path("task5_result.docx")
    build_docx(data, out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
