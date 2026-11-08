"""Собрать task4_result.docx по task4_results.json (без нового API)."""

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


def first_answer(row: dict[str, Any], max_len: int = 350) -> str:
    for run in row.get("runs") or []:
        ans = (run.get("answer") or "").strip()
        if ans:
            return ans if len(ans) <= max_len else ans[: max_len - 1] + "…"
    return "—"


def gen_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in (data.get("rows") or []) if r.get("kind") == "generation"]


def acc_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in (data.get("rows") or []) if r.get("kind") == "accuracy"]


def find(
    rows: list[dict[str, Any]],
    *,
    task: str | None = None,
    phase: str | None = None,
    effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any] | None:
    for r in rows:
        if task is not None and r.get("task") != task:
            continue
        if phase is not None and r.get("phase") != phase:
            continue
        if effort is not None and r.get("effort") != effort:
            continue
        if temperature is not None and r.get("temperature") != temperature:
            continue
        if top_p is not None and r.get("top_p") != top_p:
            continue
        return r
    return None


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


def build_docx(data: dict[str, Any], out_path: Path) -> None:
    gens = gen_rows(data)
    accs = acc_rows(data)
    settings = data.get("settings") or {}
    docs_why = data.get("docs_why") or {}
    api_totals = data.get("api_totals") or {}
    ppl_meta = data.get("ppl_scoring") or {}

    story_n0 = find(gens, task="story_s", phase="none_temperature", temperature=0.0)
    story_n1 = find(gens, task="story_s", phase="none_temperature", temperature=1.0)
    story_n2 = find(gens, task="story_s", phase="none_temperature", temperature=2.0)
    # финальная сетка: high×top_p; старые прогоны могли иметь high_extreme×temperature
    story_hp95 = find(gens, task="story_s", phase="high_top_p", top_p=0.95)
    story_hp10 = find(gens, task="story_s", phase="high_top_p", top_p=1.0)
    story_h0 = find(gens, task="story_s", phase="high_extreme", temperature=0.0)
    story_h2 = find(gens, task="story_s", phase="high_extreme", temperature=2.0)
    list_n0 = find(gens, task="list_s", phase="none_temperature", temperature=0.0)
    list_n1 = find(gens, task="list_s", phase="none_temperature", temperature=1.0)
    list_n2 = find(gens, task="list_s", phase="none_temperature", temperature=2.0)
    prior = data.get("prior_confirmed_checks") or {}

    doc = Document()
    title = doc.add_heading("День 4. Temperature / top_p — отчёт по прогону", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_heading("1. Цель и постановка", level=2)
    add_para(
        doc,
        "Проверить влияние temperature при reasoning=none и top_p при reasoning=high "
        "на deepseek-flash; оценить генерацию с ограничением «все слова на С» "
        "(diversity, letter_ratio, perplexity) и задачи с одним верным ответом.",
    )
    add_para(doc, "Почему выбраны параметры (по документации API):", bold=True)
    add_para(doc, str(docs_why.get("why_chosen") or "—"))
    add_bullets(
        doc,
        [
            f"Модель в прогоне: {docs_why.get('model') or settings.get('model') or 'deepseek-flash'}",
            f"Источники: {', '.join(docs_why.get('source') or [])}",
            f"max_tokens={settings.get('max_tokens')}, repeats_acc={settings.get('repeats_acc')}, "
            f"repeats_gen={settings.get('repeats_gen')}",
            f"Локальный скорер perplexity: {ppl_meta.get('model')} на CPU; "
            f"время_расчёта_ppl_сек={ppl_meta.get('load_and_score_sec')} "
            "(не входит во время DeepSeek).",
        ],
    )

    doc.add_heading("2. Сетка этого прогона", level=2)
    skipped = settings.get("skipped") or []
    add_bullets(
        doc,
        [
            "Фаза none×temperature: reasoning=none, temperature ∈ {0, 1, 2}",
            "Фаза high×top_p: reasoning=high, top_p ∈ {0.95, 1.0}, задача story_s "
            "(единственный sampling-рычаг в thinking по докам flash)",
            "Accuracy: bat_ball (0.05), sheep (9) — в фазе none×temperature",
            "Generation: story_s, list_s — в none×temperature; story_s ещё в high×top_p",
            f"Убрано из сетки (экономия): {skipped or ['none_top_p', 'high_temperature']}",
        ],
    )

    doc.add_heading("2.1. Уже выполненные проверки (не повторяли в финальном прогоне)", level=2)
    add_para(
        doc,
        f"Источник: {prior.get('source') or 'предыдущий полный прогон + документация'}.",
    )
    for item in prior.get("items") or []:
        add_para(doc, str(item.get("check") or ""), bold=True)
        add_para(doc, str(item.get("result") or "—"))
    if not prior.get("items"):
        add_para(
            doc,
            "none×top_p: на прошлом прогоне ppl/letter_ratio почти не менялись — top_p @none на flash игнор. "
            "high×temperature 0 vs 2: стоимость и ppl почти одинаковы — temperature @high игнор; "
            "дороговизна story_s — reasoning под constraint.",
        )

    doc.add_heading("3. Сводка стоимости DeepSeek", level=2)
    add_para(
        doc,
        f"Итого по API (без локального PPL): время_api_сек={api_totals.get('время_api_сек') or api_totals.get('api_elapsed_sec')}, "
        f"токены_reasoning={api_totals.get('токены_reasoning') or api_totals.get('reasoning_tokens')}, "
        f"токены_всего={api_totals.get('токены_всего') or api_totals.get('total_tokens')}.",
    )

    doc.add_heading("4. Таблица generation (story_s / list_s)", level=2)
    g_headers = [
        "task",
        "effort",
        "t",
        "top_p",
        "diversity",
        "letter_ratio",
        "perplexity",
        "время_api_сек",
        "токены_reasoning",
        "токены_всего",
    ]
    g_table: list[list[str]] = []
    for r in gens:
        g_table.append(
            [
                str(r.get("task")),
                str(r.get("effort")),
                str(r.get("temperature")),
                str(r.get("top_p")),
                str(r.get("diversity")),
                str(r.get("mean_letter_ratio")),
                str(r.get("mean_perplexity")),
                str(r.get("api_elapsed_sec")),
                str(r.get("reasoning_tokens")),
                str(r.get("total_tokens")),
            ]
        )
    table_from_dicts(doc, g_headers, g_table)

    doc.add_heading("5. Таблица accuracy", level=2)
    a_headers = [
        "task",
        "effort",
        "t",
        "top_p",
        "accuracy",
        "n_unique",
        "время_api_сек",
        "токены_reasoning",
        "токены_всего",
    ]
    a_table: list[list[str]] = []
    for r in accs:
        a_table.append(
            [
                str(r.get("task")),
                str(r.get("effort")),
                str(r.get("temperature")),
                str(r.get("top_p")),
                str(r.get("accuracy")),
                str(r.get("n_unique")),
                str(r.get("api_elapsed_sec")),
                str(r.get("reasoning_tokens")),
                str(r.get("total_tokens")),
            ]
        )
    table_from_dicts(doc, a_headers, a_table)

    doc.add_heading("6. Примеры ответов с разной temperature (effort=none)", level=2)
    for temp, row in ((0.0, story_n0), (1.0, story_n1), (2.0, story_n2)):
        doc.add_heading(f"temperature = {temp} — story_s", level=3)
        if not row:
            add_para(doc, "Нет данных.")
            continue
        add_bullets(
            doc,
            [
                f"letter_ratio={row.get('mean_letter_ratio')}, diversity={row.get('diversity')}, "
                f"perplexity={row.get('mean_perplexity')}",
                f"время_api_сек={row.get('api_elapsed_sec')}, токены_reasoning={row.get('reasoning_tokens')}, "
                f"токены_всего={row.get('total_tokens')}",
            ],
        )
        add_para(doc, "Пример ответа:", bold=True)
        add_para(doc, first_answer(row))

    doc.add_heading("7. Reasoning=high: стоимость story_s и роль top_p", level=2)
    high_ref = story_hp95 or story_hp10 or story_h0
    if story_n0 and high_ref:
        add_para(
            doc,
            f"Сравнение story_s: none×t=0 → время_api_сек={story_n0.get('api_elapsed_sec')}, "
            f"токены_reasoning={story_n0.get('reasoning_tokens')}, "
            f"токены_всего={story_n0.get('total_tokens')}; "
            f"high (пример top_p={high_ref.get('top_p')}, t={high_ref.get('temperature')}) → "
            f"время_api_сек={high_ref.get('api_elapsed_sec')}, "
            f"токены_reasoning={high_ref.get('reasoning_tokens')}, "
            f"токены_всего={high_ref.get('total_tokens')}.",
        )
    add_para(doc, "Почему story_s дорогой при high:", bold=True)
    add_bullets(
        doc,
        [
            "Constraint «каждое слово на С» на рассказе заставляет thinking долго планировать "
            "и проверять слова — токены уходят в reasoning_tokens.",
            "У коротких accuracy-задач reasoning умеренный: один числовой ответ.",
            "В финальном прогоне при high варьируем top_p (0.95 vs 1.0), не temperature — "
            "по докам flash именно top_p действует в thinking.",
        ],
    )
    if story_hp95 or story_hp10:
        doc.add_heading("7.1. high × top_p (этот прогон)", level=3)
        for r in (story_hp95, story_hp10):
            if not r:
                continue
            add_bullets(
                doc,
                [
                    f"top_p={r.get('top_p')}: ppl={r.get('mean_perplexity')}, "
                    f"letter_ratio={r.get('mean_letter_ratio')}, diversity={r.get('diversity')}, "
                    f"время_api_сек={r.get('api_elapsed_sec')}, "
                    f"токены_reasoning={r.get('reasoning_tokens')}",
                ],
            )
        add_para(doc, "Пример ответа high×top_p=0.95:" if story_hp95 else "Пример:", bold=True)
        add_para(doc, first_answer(story_hp95 or story_hp10 or {}))
    elif story_h0 and story_h2:
        add_para(
            doc,
            "В этом JSON ещё старая фаза high×temperature (из прошлого прогона): "
            f"t=0 reason={story_h0.get('reasoning_tokens')}, "
            f"t=2 reason={story_h2.get('reasoning_tokens')} — почти без разницы.",
        )

    doc.add_heading("8. Perplexity: влияние temperature, top_p и reasoning", level=2)
    add_para(doc, "Как читать perplexity:", bold=True)
    add_para(
        doc,
        "Считал локальный ruGPT3-small на CPU по готовым ответам. Ниже значение → текст "
        "«естественнее» для языковой модели. Метрика не требует эталона и не путает разные "
        "валидные стили на одну тему так, как cosine эмбеддингов.",
    )

    doc.add_heading("8.1. Temperature (reasoning=none)", level=3)
    if story_n0 and story_n1 and story_n2:
        add_bullets(
            doc,
            [
                f"story_s: t=0 → ppl={story_n0.get('mean_perplexity')}, "
                f"letter_ratio={story_n0.get('mean_letter_ratio')}",
                f"story_s: t=1 → ppl={story_n1.get('mean_perplexity')}, "
                f"letter_ratio={story_n1.get('mean_letter_ratio')}",
                f"story_s: t=2 → ppl={story_n2.get('mean_perplexity')}, "
                f"letter_ratio={story_n2.get('mean_letter_ratio')} "
                "(текст ломается: смесь языков, мусорные токены — perplexity взлетает).",
            ],
        )
    if list_n0 and list_n1 and list_n2:
        add_bullets(
            doc,
            [
                f"list_s: t=0 → ppl={list_n0.get('mean_perplexity')}, S={list_n0.get('mean_letter_ratio')}",
                f"list_s: t=1 → ppl={list_n1.get('mean_perplexity')}, S={list_n1.get('mean_letter_ratio')}",
                f"list_s: t=2 → ppl={list_n2.get('mean_perplexity')}, S={list_n2.get('mean_letter_ratio')}",
            ],
        )
    add_para(
        doc,
        "Вывод: на flash при none temperature реально живой. Переход 0→1 даёт умеренный сдвиг "
        "perplexity; t=2 резко ухудшает адекватность (ppl на порядки выше) и соблюдение правила буквы С. "
        "Для креатива с контролем качества разумнее зона около 0–1, не 2.",
    )

    doc.add_heading("8.2. top_p", level=3)
    top_p_bits: list[str] = []
    for p in (0.5, 0.95, 1.0):
        r = find(gens, task="story_s", phase="none_top_p", top_p=p)
        if r:
            top_p_bits.append(
                f"none×top_p={p}: ppl={r.get('mean_perplexity')}, "
                f"letter_ratio={r.get('mean_letter_ratio')}, "
                f"diversity={r.get('diversity')}"
            )
    if top_p_bits:
        add_bullets(doc, top_p_bits)
        add_para(
            doc,
            "В этом JSON ещё есть none×top_p: сдвига почти нет → top_p @none на flash игнор.",
        )
    else:
        add_para(
            doc,
            "Фазу none×top_p в финальном прогоне не гоняли (уже подтверждена ранее, см. §2.1). "
            "При none крутим temperature; top_p имеет смысл смотреть при high (см. §7–8.3).",
        )

    doc.add_heading("8.3. Reasoning=high и sampling", level=3)
    if story_hp95 and story_hp10 and story_n0:
        add_bullets(
            doc,
            [
                f"high × top_p=0.95: ppl={story_hp95.get('mean_perplexity')}, "
                f"S={story_hp95.get('mean_letter_ratio')}, reason={story_hp95.get('reasoning_tokens')}",
                f"high × top_p=1.0: ppl={story_hp10.get('mean_perplexity')}, "
                f"S={story_hp10.get('mean_letter_ratio')}, reason={story_hp10.get('reasoning_tokens')}",
                f"none × t=0: ppl={story_n0.get('mean_perplexity')}, "
                f"S={story_n0.get('mean_letter_ratio')}",
            ],
        )
        add_para(
            doc,
            "Смотрим, сдвигает ли узкое окно top_p perplexity/diversity при thinking. "
            "Temperature при high в финальном прогоне не варьировали (игнор по докам; "
            "подтверждено прошлым прогоном — см. §2.1).",
        )
    elif story_h0 and story_h2 and story_n0:
        add_bullets(
            doc,
            [
                f"high × t=0: ppl={story_h0.get('mean_perplexity')}, "
                f"reason={story_h0.get('reasoning_tokens')}",
                f"high × t=2: ppl={story_h2.get('mean_perplexity')}, "
                f"reason={story_h2.get('reasoning_tokens')}",
                f"none × t=0: ppl={story_n0.get('mean_perplexity')}",
            ],
        )
        add_para(
            doc,
            "Старый прогон high×temperature: ppl в норме, t=0≈t=2 → temperature не рулит; "
            "платим reasoning. Для разнообразия дешевле none + умеренная temperature.",
        )
    else:
        add_para(doc, "Нет данных high для сравнения perplexity.")

    doc.add_heading("9. Accuracy и temperature", level=2)
    for temp in (0.0, 1.0, 2.0):
        bits = []
        for task in ("bat_ball", "sheep"):
            r = find(accs, task=task, phase="none_temperature", temperature=temp)
            if r:
                bits.append(f"{task}: accuracy={r.get('accuracy')}, unique={r.get('n_unique')}")
        add_para(doc, f"temperature={temp}: " + ("; ".join(bits) if bits else "—"))
    add_para(
        doc,
        "На ловушках t=0 и t=1 дают стабильные верные ответы; t=2 заметно роняет accuracy "
        "(особенно bat_ball) и раздувает число уникальных неверных вариантов — ещё один аргумент "
        "не использовать максимальную temperature для задач с одним правильным ответом.",
    )

    doc.add_heading("10. Выводы по использованию параметров", level=2)
    add_bullets(
        doc,
        [
            "temperature=0 + reasoning=none — точные/стабильные ответы, предсказуемая генерация под правило.",
            "temperature≈1 + none — умеренное разнообразие при ещё приемлемом perplexity и letter_ratio.",
            "temperature=2 + none — максимум шума: ломается и смысл (ppl↑), и формальное правило (letter_ratio↓), "
            "accuracy на ловушках падает; для продакшена обычно непригодно.",
            "top_p при none на flash не инструмент управления — смотреть temperature.",
            "reasoning=high — дорого на constrained story (тысячи reasoning_tokens), но держит правило и "
            "адекватность; temperature при high не использовать как рычаг разнообразия.",
            "Полные ответы API сохранены в task4_results.json → rows[].runs[].answer.",
        ],
    )

    doc.add_paragraph("")
    add_para(
        doc,
        "Источник: task4_results.json (локальный прогон deepseek-flash через /chat + ruGPT3-small PPL).",
    )
    doc.save(out_path)


def main() -> int:
    _utf8()
    src = Path("task4_results.json")
    out = Path("task4_result.docx")
    data = json.loads(src.read_text(encoding="utf-8"))
    build_docx(data, out)
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
