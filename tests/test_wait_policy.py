"""Both explicit sign-in modes share a cancellable healthy-generation wait."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_openai_chatgpt import provider as module


def configured(mode, timeout=None):
    p = module.ChatGPTProvider(
        {"auth_mode": mode, "default_model": "fake", "timeout": timeout}, None, None
    )
    p._ensure_valid_tokens = AsyncMock()
    p._build_headers = dict
    return p


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("timeout", [None, 45])
async def test_actual_transport_timeout_extensions(monkeypatch, mode, timeout):
    sent = []

    async def receive(request):
        sent.append(request.extensions["timeout"])
        event = {
            "type": "response.completed",
            "response": {
                "id": "fake",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "OK"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
        return httpx.Response(
            200,
            text="data: "
            + json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": event["response"]["output"][0],
                }
            )
            + "\n\ndata: "
            + json.dumps(event)
            + "\n\n",
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real_client(**kw, transport=httpx.MockTransport(receive)),
    )
    result = await configured(mode, timeout).complete(
        ChatRequest(messages=[Message(role="user", content="Hello")])
    )
    assert result.content[0].text == "OK"
    assert sent == [{"connect": 5.0, "pool": 5.0, "read": timeout, "write": timeout}]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
async def test_user_cancellation_does_not_retry(monkeypatch, mode):
    entered = asyncio.Event()
    calls = []

    async def receive(request):
        calls.append(request)
        entered.set()
        await asyncio.Event().wait()

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real_client(**kw, transport=httpx.MockTransport(receive)),
    )
    task = asyncio.create_task(
        configured(mode).complete(
            ChatRequest(messages=[Message(role="user", content="Hello")])
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1
