"""No network/LLM calls: exercise the actual HTTPX timeout extension and stream lifecycle."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider
import amplifier_module_provider_openai_chatgpt.provider as provider_module

MODEL = "synthetic-wait-model"


def make_provider(**config):
    return ChatGPTProvider(
        {
            "default_model": MODEL,
            "experimental_compaction": True,
            "token_file_path": "/nonexistent/synthetic-wait-tokens",
            **config,
        },
        coordinator=MagicMock(hooks=MagicMock(emit=AsyncMock())),
        tokens={
            "access_token": "synthetic",
            "account_id": "synthetic-account",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    )


def request(**kwargs):
    return ChatRequest(
        model=MODEL,
        messages=[Message(role="user", content="Synthetic delay")],
        **kwargs,
    )


class ControlledStream(httpx.AsyncByteStream):
    def __init__(self, *, native, failure=None):
        self.native = native
        self.failure = failure
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.read_past_terminal = False

    @staticmethod
    def encode(event):
        return ("data: " + json.dumps(event) + "\n\n").encode()

    async def __aiter__(self):
        yield self.encode(
            {
                "type": "response.created",
                "response": {"id": "synthetic-response", "model": MODEL},
            }
        )
        if not self.native:
            yield self.encode(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message"},
                }
            )
            yield self.encode(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "Waiting",
                }
            )
        self.waiting.set()
        await self.release.wait()
        if self.failure:
            raise self.failure
        item = (
            {"type": "compaction", "encrypted_content": "synthetic-opaque"}
            if self.native
            else {
                "type": "message",
                "content": [{"type": "output_text", "text": "Completed after delay"}],
            }
        )
        yield self.encode(
            {"type": "response.output_item.done", "output_index": 0, "item": item}
        )
        yield self.encode(
            {
                "type": "response.completed",
                "response": {
                    "id": "synthetic-response",
                    "model": MODEL,
                    "status": "completed",
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            }
        )
        # A live server can keep the socket open after terminal completion.
        # Reading again is itself the regression, without adding a test sleep.
        self.read_past_terminal = True
        raise AssertionError("Provider waited past completed model work")

    async def aclose(self):
        self.closed = True


def instrument_httpx(monkeypatch, stream):
    actual_client = httpx.AsyncClient
    observed = []

    def handle(req):
        observed.append(req.extensions["timeout"].copy())
        return httpx.Response(200, stream=stream)

    def client(**kwargs):
        return actual_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(provider_module.httpx, "AsyncClient", client)

    def no_disk(*args, **kwargs):
        raise AssertionError("Unit fixture must not read live auth")

    monkeypatch.setattr(provider_module, "load_tokens", no_disk)
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_healthy_delayed_model_has_no_hidden_deadline_and_stops_at_terminal(
    monkeypatch, native, streaming
):
    p = make_provider(use_streaming=streaming)
    stream = ControlledStream(native=native)
    observed = instrument_httpx(monkeypatch, stream)
    clock = [0.0]
    # Advance only provider bookkeeping time, without changing asyncio's clock.
    monkeypatch.setattr(
        provider_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    task = asyncio.create_task(
        p.compact_checkpoint(request()) if native else p.complete(request())
    )
    await stream.waiting.wait()
    clock[0] = 24 * 60 * 60.0
    await asyncio.sleep(0)
    assert not task.done()
    assert observed == [{"connect": 10.0, "read": None, "write": None, "pool": 10.0}]
    stream.release.set()
    result = await task
    assert result.usage.input_tokens == 12
    assert stream.closed and not stream.read_past_terminal
    responses = [
        c.args[1]
        for c in p._coordinator.hooks.emit.call_args_list
        if c.args[0] == "llm:response"
    ]
    assert len(responses) == 1 and responses[0]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_explicit_cancellation_closes_pending_stream_without_retry(
    monkeypatch, native
):
    p = make_provider()
    stream = ControlledStream(native=native)
    observed = instrument_httpx(monkeypatch, stream)
    task = asyncio.create_task(
        p.compact_checkpoint(request()) if native else p.complete(request())
    )
    await stream.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed and len(observed) == 1
    responses = [
        c.args[1]
        for c in p._coordinator.hooks.emit.call_args_list
        if c.args[0] == "llm:response"
    ]
    assert len(responses) == 1 and responses[0]["status"] == "cancelled"
    aborts = [
        c
        for c in p._coordinator.hooks.emit.call_args_list
        if c.args[0] == "llm:stream_aborted"
    ]
    assert len(aborts) == (0 if native else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "failure,expected",
    [
        (
            httpx.RemoteProtocolError("synthetic disconnect"),
            llm_errors.ProviderUnavailableError,
        ),
        (httpx.ReadTimeout("explicit configured timeout"), llm_errors.LLMTimeoutError),
    ],
)
async def test_connection_failure_and_explicit_timeout_still_end_model_work(
    monkeypatch, native, failure, expected
):
    p = make_provider(timeout=23.0)
    stream = ControlledStream(native=native, failure=failure)
    observed = instrument_httpx(monkeypatch, stream)
    task = asyncio.create_task(
        p.compact_checkpoint(request()) if native else p.complete(request())
    )
    await stream.waiting.wait()
    stream.release.set()
    with pytest.raises(expected):
        await task
    assert stream.closed and len(observed) == 1
    assert observed[0] == {"connect": 23.0, "read": 23.0, "write": 23.0, "pool": 23.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,requested,expected",
    [(31.0, 7.0, 7.0), (31.0, None, None), (31.0, 0.0, 0.0)],
)
@pytest.mark.parametrize("native", [False, True])
async def test_explicit_request_limit_or_none_overrides_provider_default(
    monkeypatch, configured, requested, expected, native
):
    p = make_provider(timeout=configured)
    stream = ControlledStream(native=native)
    stream.release.set()
    observed = instrument_httpx(monkeypatch, stream)
    req = request(timeout=requested)
    await (p.compact_checkpoint(req) if native else p.complete(req))
    assert (
        observed[0]["read"] is None
        if expected is None
        else observed[0]["read"] == expected
    )
    assert p.timeout == configured


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1])
def test_invalid_explicit_request_timeout_rejected(value):
    with pytest.raises(llm_errors.InvalidRequestError):
        make_provider()._http_timeout(request(timeout=value))
