import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from httpx import ConnectError, HTTPStatusError, ProxyError, TimeoutException
from pydantic import BaseModel, Field

from app.agent import AgentResult, run_agent
from app.config import (
    AGENT_CONTEXT_MODE,
    AGENT_KEEP_LAST,
    AGENT_SESSION_ID,
    AGENT_SUMMARIZE_EVERY,
    DEFAULT_REASONING_EFFORT,
    DEEPSEEK_MODEL,
    USD_RUB_RATE,
)
from app.deepseek_client import chat_completion, extract_finish_reason, extract_text
from app.pricing import estimate_usage_cost
from app import session_store

app = FastAPI(title="AI Challenge", version="0.5.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ReasoningEffort = Literal["none", "low", "high", "max"]


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Промпт для модели")
    system: str | None = Field(default=None, description="System-сообщение")
    reasoning_effort: ReasoningEffort = Field(
        default_factory=lambda: DEFAULT_REASONING_EFFORT,  # type: ignore[arg-type]
        description="none|low|high|max",
    )
    max_tokens: int | None = Field(default=None, ge=1, le=393216)
    stop: str | list[str] | None = None
    response_format: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)


class ChatResponse(BaseModel):
    answer: str
    model: str | None = None
    usage: dict | None = None
    finish_reason: str | None = None
    reasoning_effort: str | None = None
    request: dict | None = None
    cost: dict[str, Any] | None = None


ContextMode = Literal[
    "full",
    "summary_simple",
    "summary_facts",
    "summary_rules",
    "sliding_window",
    "sticky_facts",
]


class AgentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Запрос пользователя агенту")
    system: str | None = Field(default=None, description="Переопределить system prompt")
    session_id: str | None = Field(
        default=None,
        description="ID JSONL-сессии (день 7). По умолчанию AGENT_SESSION_ID.",
    )
    reasoning_effort: ReasoningEffort = Field(
        default_factory=lambda: DEFAULT_REASONING_EFFORT,  # type: ignore[arg-type]
    )
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_iterations: int | None = Field(default=None, ge=1, le=20)
    context_mode: ContextMode | None = Field(
        default=None,
        description="full | summary_* | sliding_window | sticky_facts",
    )
    keep_last: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Сырой хвост после сжатия",
    )
    summarize_every: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Сжимать когда uncovered >= keep_last + summarize_every",
    )
    tail_tokens: int | None = Field(
        default=None,
        ge=200,
        le=20000,
        description="Бюджет хвоста в токенах для sliding_window и sticky_facts",
    )


def _session_id(body: AgentRequest) -> str:
    return (body.session_id or AGENT_SESSION_ID).strip() or AGENT_SESSION_ID


class AgentToolTraceItem(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float | None = None


class AgentLlmStep(BaseModel):
    iteration: int
    kind: str  # tools | final
    model: str | None = None
    duration_ms: float
    usage: dict[str, int] = Field(default_factory=dict)
    tool_names: list[str] = Field(default_factory=list)
    cost: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    answer: str
    model: str | None = None
    iterations: int
    stopped_reason: str
    session_id: str | None = None
    usage_total: dict[str, int] | None = None
    model_time_ms: float | None = None
    wall_time_ms: float | None = None
    cost_total: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    llm_steps: list[AgentLlmStep] = Field(default_factory=list)
    tool_trace: list[AgentToolTraceItem] = Field(default_factory=list)


def _to_agent_response(result: AgentResult) -> AgentResponse:
    return AgentResponse(
        answer=result.answer,
        model=result.model,
        iterations=result.iterations,
        stopped_reason=result.stopped_reason,
        session_id=result.session_id,
        usage_total=result.usage_total or None,
        model_time_ms=round(result.model_time_ms),
        wall_time_ms=round(result.wall_time_ms),
        cost_total=result.cost_total or None,
        context=result.context or None,
        llm_steps=[
            AgentLlmStep(
                iteration=s.iteration,
                kind=s.kind,
                model=s.model,
                duration_ms=round(s.duration_ms),
                usage=s.usage,
                tool_names=s.tool_names,
                cost=s.cost,
            )
            for s in result.llm_steps
        ],
        tool_trace=[
            AgentToolTraceItem(
                name=t.name,
                arguments=t.arguments,
                result=t.result,
                duration_ms=round(t.duration_ms),
            )
            for t in result.tool_trace
        ],
    )


def _agent_http_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    if isinstance(exc, HTTPStatusError):
        return HTTPException(
            status_code=exc.response.status_code,
            detail=f"DeepSeek API error: {exc.response.text}",
        )
    if isinstance(exc, (ConnectError, ProxyError, TimeoutException)):
        return HTTPException(status_code=502, detail=f"Сеть/TLS до DeepSeek: {exc}")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def ui_index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Нет app/static/index.html")
    return FileResponse(index)


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    return {
        "model": DEEPSEEK_MODEL,
        "usd_rub_rate": USD_RUB_RATE,
        "tools_ring": 5,
        "default_session_id": AGENT_SESSION_ID,
        "context_mode": AGENT_CONTEXT_MODE,
        "keep_last": AGENT_KEEP_LAST,
        "summarize_every": AGENT_SUMMARIZE_EVERY,
    }


@app.get("/api/sessions")
def api_sessions_list() -> dict[str, Any]:
    return {"sessions": session_store.list_sessions()}


@app.get("/api/sessions/search")
def api_sessions_search(
    q: str = "",
    limit: int = 40,
    per_session: int = 5,
) -> dict[str, Any]:
    """Поиск по текстам сообщений во всех чатах (JSONL)."""
    query = (q or "").strip()
    hits = session_store.search_sessions(
        query,
        limit=limit,
        per_session=per_session,
    )
    return {"q": query, "count": len(hits), "hits": hits}


@app.post("/api/session/new")
async def api_session_new(request: Request) -> dict[str, str]:
    sid = session_store.new_session_id()
    role = ""
    if request.headers.get("content-type", "").startswith("application/json"):
        raw = await request.body()
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            role = str((payload or {}).get("role") or "").strip().lower()
    if role:
        from app.day10 import register_role

        register_role(sid, role)
    return {"status": "ok", "session_id": sid, "role": role}


@app.get("/api/room")
def api_room(limit: int = 40) -> dict[str, Any]:
    from app.day10 import load_room

    messages = load_room(limit)
    return {"count": len(messages), "messages": messages}


@app.post("/api/room/reset")
def api_room_reset() -> dict[str, str]:
    from app.day10 import reset_room

    reset_room()
    return {"status": "ok"}


@app.get("/api/session/{session_id}")
def api_session_get(session_id: str) -> dict[str, Any]:
    return session_store.session_public_view(session_id)


@app.post("/api/session/{session_id}/reset")
def api_session_reset(session_id: str) -> dict[str, str]:
    session_store.reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """Принять промпт и параметры → вызвать DeepSeek → вернуть ответ."""
    request_echo = body.model_dump(exclude_none=True)
    try:
        raw = await chat_completion(
            body.prompt,
            system=body.system,
            reasoning_effort=body.reasoning_effort,
            max_tokens=body.max_tokens,
            stop=body.stop,
            response_format=body.response_format,
            temperature=body.temperature,
            top_p=body.top_p,
        )
        answer = extract_text(raw)
        finish_reason = extract_finish_reason(raw)
    except Exception as exc:  # noqa: BLE001
        raise _agent_http_errors(exc) from exc

    usage = raw.get("usage")
    return ChatResponse(
        answer=answer,
        model=raw.get("model"),
        usage=usage,
        finish_reason=finish_reason,
        reasoning_effort=body.reasoning_effort,
        request=request_echo,
        cost=estimate_usage_cost(usage, model=raw.get("model")),
    )


@app.post("/chat/text", response_class=PlainTextResponse)
async def chat_text(body: ChatRequest) -> str:
    result = await chat(body)
    return result.answer


@app.post("/agent", response_model=AgentResponse)
async def agent(body: AgentRequest) -> AgentResponse:
    """Агент: цикл LLM ↔ tools + JSONL-сессия (день 7)."""
    try:
        result = await run_agent(
            body.prompt,
            system=body.system,
            session_id=_session_id(body),
            reasoning_effort=body.reasoning_effort,
            temperature=body.temperature,
            max_iterations=body.max_iterations,
            context_mode=body.context_mode,
            keep_last=body.keep_last,
            summarize_every=body.summarize_every,
            tail_tokens=body.tail_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        raise _agent_http_errors(exc) from exc
    return _to_agent_response(result)


@app.post("/agent/text", response_class=PlainTextResponse)
async def agent_text(body: AgentRequest) -> str:
    result = await agent(body)
    return result.answer


@app.post("/agent/stream")
async def agent_stream(body: AgentRequest) -> StreamingResponse:
    """SSE: live progress (llm/tools) + финальный result."""

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def on_progress(kind: str, data: dict[str, Any]) -> None:
        queue.put_nowait({"event": kind, "data": data})

    async def runner() -> None:
        try:
            result = await run_agent(
                body.prompt,
                system=body.system,
                session_id=_session_id(body),
                reasoning_effort=body.reasoning_effort,
                temperature=body.temperature,
                max_iterations=body.max_iterations,
                context_mode=body.context_mode,
                keep_last=body.keep_last,
                summarize_every=body.summarize_every,
                tail_tokens=body.tail_tokens,
                on_progress=on_progress,
            )
            await queue.put(
                {
                    "event": "result",
                    "data": _to_agent_response(result).model_dump(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            http_exc = _agent_http_errors(exc)
            await queue.put(
                {
                    "event": "error",
                    "data": {"status": http_exc.status_code, "detail": http_exc.detail},
                }
            )
        finally:
            await queue.put(None)

    async def event_gen():
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                payload = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {item['event']}\ndata: {payload}\n\n"
        finally:
            await task

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
