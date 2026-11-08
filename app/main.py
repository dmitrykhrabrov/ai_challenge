from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from httpx import ConnectError, HTTPStatusError, ProxyError, TimeoutException
from pydantic import BaseModel, Field

from app.config import DEFAULT_REASONING_EFFORT
from app.deepseek_client import chat_completion, extract_finish_reason, extract_text

app = FastAPI(title="AI Challenge", version="0.2.0")

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
    # DeepSeek официально: text | json_object. Прочие (напр. xml) — пробрасываем, API может ответить 400.
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPStatusError as exc:
        detail = exc.response.text
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"DeepSeek API error: {detail}",
        ) from exc
    except (ConnectError, ProxyError, TimeoutException) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Сеть/TLS до DeepSeek: {exc}. "
                "Если стоит локальный прокси (127.0.0.1:10809), "
                "в .env задайте HTTPS_PROXY и SSL_VERIFY=false "
                "или SSL_CA_BUNDLE=путь_к_CA_прокси."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        model=raw.get("model"),
        usage=raw.get("usage"),
        finish_reason=finish_reason,
        reasoning_effort=body.reasoning_effort,
        request=request_echo,
    )


@app.post("/chat/text", response_class=PlainTextResponse)
async def chat_text(body: ChatRequest) -> str:
    result = await chat(body)
    return result.answer
