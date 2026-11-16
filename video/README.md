# Демо-видео агента (Playwright + TTS)

Минимальный пайплайн как у коллег: сценарий → запись UI → озвучка → mp4.

## Один раз

```powershell
.\.venv\Scripts\python.exe -m pip install -r video\requirements.txt
# важно: браузеры в %LOCALAPPDATA%\ms-playwright (не в sandbox Cursor)
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\ms-playwright"
.\.venv\Scripts\python.exe -m playwright install chromium
winget install ffmpeg
```

## Запуск

Терминал 1 — API + UI (нужен до первого шага; сценарий сам **перезапустит** uvicorn на шаге `restart_app`):

```powershell
# только из venv (системный python часто без httpx)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Шаг `restart_app` сам поднимает сервер через `.venv\Scripts\python.exe`.

Терминал 2 — сборка:

```powershell
.\.venv\Scripts\python.exe -u video\build_demo.py --engine windows
```

Сценарий дня 7 (`script.json`): tool-вопрос → `restart_app` → follow-up из JSONL.

День 8 — токены (`script_day8.json`): короткий → жирный полгода → переполнение книгами:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\ms-playwright"
.\.venv\Scripts\python.exe -u video\build_demo.py --script video\script_day8.json --engine edge
```

День 9 — сжатие контекста (`script_day9.json`). Рекомендуемый порядок:

```powershell
# 1) запись экрана → авторазбор DeepSeek-агентом (промпт из analyze_prompt /
#    prompts/day9_analyze_for_voiceover.txt). Откроется UI ?session=… — можно ответить и поправить.
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\ms-playwright"
.\.venv\Scripts\python.exe -u video\build_demo.py --script video\script_day9.json --record-only

# только разбор уже записанного run (если автозапуск пропустили / упал):
# .\.venv\Scripts\python.exe -u video\build_demo.py --analyze-only data\video\<run>

# 2) правки озвучки: data\video\<run>\script_copy.json → narration / narration_before / narration_after
#    (ориентир: analyze_answer.txt + ваш диалог в UI)

# 3) озвучка + mp4 из уже записанного экрана
.\.venv\Scripts\python.exe -u video\build_demo.py --skip-record data\video\<run> --engine edge
```

После записи в `run_dir`: `recorded_sessions.json`, `analyze_answer.txt`, `analyze_meta.json`.
Флаги: `--no-analyze` (пропуск), `--analyze` (форс), `--analyze-only <run>`.

`make_voiceover` подхватывает тексты из `script_copy.json` (не только из timeline на момент съёмки).

По умолчанию запись идёт через **системный Edge** (`browser_channel: msedge`) — bundled Chromium на Windows часто падает с `chrome.dll … занят другим процессом` (Defender). Fallback: Chrome → bundled. Переопределение: `$env:PLAYWRIGHT_CHANNEL='chrome'`.

Или по шагам:

```powershell
.\.venv\Scripts\python.exe -u video\record_ui.py
# запомни run_dir из вывода, например data\video\20261114_143000
.\.venv\Scripts\python.exe -u video\make_voiceover.py --run data\video\20261114_143000
.\.venv\Scripts\python.exe -u video\mux.py --run data\video\20261114_143000
```

Результат: `data/video/<run>/final.mp4` (+ `screen.webm`, `voiceover.mp3`, `timeline.json`).

## Сценарий и озвучка

Правь скрипт: шаги `narrate` / `new_chat` / `set_context` / `reset_session` / `restart_app` / `prompt`.

`set_context`: `context_mode` (`full`|`summary_simple`|`summary_facts`), `keep_last`, `summarize_every`. Можно задать блок `context` на весь скрипт.

Поля TTS:
- `voice` — edge-tts: по умолчанию `ru-RU-SvetlanaNeural` (приятный женский). Альтернативы: `ru-RU-DariyaNeural`, `ru-RU-DmitryNeural`.
- `voice_rate` — например `"-5%"` (чуть медленнее, разборчивее).
- `narration` — одна реплика (intro / new_chat / restart).
- у `prompt`: `narration_before` (анонс теста, пока текст в поле) и `narration_after` (комментарий к ответу и stats).

Запись сама держит паузы под длину речи, чтобы озвучка совпадала с экраном.

Для Neural-голоса лучше edge (нужна сеть без ломающего MITM-прокси):

```powershell
.\.venv\Scripts\python.exe -u video\build_demo.py --script video\script_day8.json --engine edge
```

`--engine windows` — офлайн SAPI (голос системы, не Svetlana).
