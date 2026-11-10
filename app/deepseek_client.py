from typing import Any

import httpx

from app.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEFAULT_REASONING_EFFORT,
    HTTPS_PROXY,
    SSL_CA_BUNDLE,
    SSL_VERIFY,
    require_api_key,
)


def _client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": 90.0}

    if HTTPS_PROXY:
        kwargs["proxy"] = HTTPS_PROXY

    if SSL_CA_BUNDLE:
        kwargs["verify"] = SSL_CA_BUNDLE
    else:
        kwargs["verify"] = SSL_VERIFY

    return kwargs


async def chat_completion_messages(
    messages: list[dict[str, Any]],
    *,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    stop: str | list[str] | None = None,
    response_format: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Низкоуровневый chat/completions: готовый список messages (+ tools)."""
    api_key = require_api_key()
    effort = (reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()

    payload: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": False,
        "reasoning_effort": effort,
    }
    if effort == "none":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["thinking"] = {"type": "enabled"}

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop is not None:
        payload["stop"] = stop
    if response_format is not None:
        payload["response_format"] = {"type": response_format}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if tools is not None:
        payload["tools"] = tools
        # auto — безопасный дефолт; required в thinking у DeepSeek часто 400
        payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"

    async with httpx.AsyncClient(**_client_kwargs()) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


async def chat_completion(
    prompt: str,
    *,
    system: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    stop: str | list[str] | None = None,
    response_format: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    """Вызов DeepSeek Chat Completions с опциональными параметрами дня 2+."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return await chat_completion_messages(
        messages,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        stop=stop,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
    )


def extract_text(completion: dict[str, Any]) -> str:
    try:
        return completion["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Неожиданный ответ DeepSeek: {completion}") from exc


def extract_finish_reason(completion: dict[str, Any]) -> str | None:
    try:
        return completion["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None
