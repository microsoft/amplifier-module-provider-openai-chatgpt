"""Image wire and private-admission tests through actual HTTPX serialization."""

import base64
import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolCall,
)
from PIL import Image

from amplifier_module_provider_openai_chatgpt import _media
from amplifier_module_provider_openai_chatgpt import provider as module


def encoded_image(image_format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (19, 87, 163)).save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode()


PNG = encoded_image()


def success():
    events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "OK"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "fixture",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ]
    return httpx.Response(
        200,
        headers={"x-request-id": "req-fixture"},
        text="".join("data: " + json.dumps(e) + "\n\n" for e in events),
    )


class Capability:
    version = 1

    def __init__(self, request):
        self.request, self.calls, self.valid = request, [], True

    def validate(self, request, *, purpose, request_options):
        assert request is self.request
        self.calls.append((purpose, copy.deepcopy(request_options)))
        if not self.valid:
            raise ValueError("revoked")
        return {"valid": True}


def request(role="user", image_format="PNG"):
    parts = [
        TextBlock(text="before"),
        ImageBlock(
            source={
                "type": "base64",
                "media_type": "image/png" if image_format == "PNG" else "image/jpeg",
                "data": encoded_image(image_format),
            }
        ),
        TextBlock(text="after"),
    ]
    if role == "tool":
        return ChatRequest(
            messages=[
                Message(role="user", content="read"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call1", name="read_file", arguments={})],
                ),
                Message(role="tool", tool_call_id="call1", content=parts),
            ]
        )
    return ChatRequest(messages=[Message(role=role, content=parts)])


def harness(
    monkeypatch, req, *, model="gpt-6.1-sol", mode="chatgpt_codex", receive=None
):
    cap = Capability(req)
    events, calls = [], []

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))

    p = module.ChatGPTProvider(
        {
            "default_model": model,
            "auth_mode": mode,
            "raw": True,
            "min_retry_delay": 0,
            "max_retry_delay": 0,
        },
        SimpleNamespace(
            get_capability=lambda key: cap, hooks=SimpleNamespace(emit=emit)
        ),
    )
    p._ensure_valid_tokens = AsyncMock()
    p._build_headers = lambda: {"authorization": "Bearer offline"}
    real = httpx.AsyncClient

    async def handle(r):
        calls.append(r)
        return await receive(r, cap) if receive else success()

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real(**kw, transport=httpx.MockTransport(handle)),
    )
    return p, cap, calls, events


@pytest.mark.asyncio
@pytest.mark.parametrize("model", sorted(_media.VISION_MODELS))
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize("display_stream", [True, False])
async def test_each_exact_model_real_http_wire(
    monkeypatch, model, role, image_format, display_stream
):
    req = request(role, image_format)
    req.metadata = {"stream": display_stream}
    encoded = encoded_image(image_format)
    before = req.model_dump_json()
    p, cap, calls, events = harness(monkeypatch, req, model=model)
    await p.complete(req)
    assert len(calls) == 1
    body = json.loads(calls[0].content)
    assert body["model"] == model
    item = body["input"][-1]
    if role == "tool":
        assert item["type"] == "function_call_output" and item["call_id"] == "call1"
        parts = item["output"]
    else:
        parts = item["content"]
    assert [x["type"] for x in parts] == ["input_text", "input_image", "input_text"]
    assert base64.b64decode(
        parts[1]["image_url"].split(",", 1)[1], validate=True
    ) == base64.b64decode(encoded)
    assert req.model_dump_json() == before
    assert cap.calls and all(x == "transport" for x, _ in cap.calls)
    assert encoded not in json.dumps(events)
    assert "not retained" in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,model", [("chatgpt_plan", "gpt-6.1-sol"), ("chatgpt_codex", "unknown")]
)
@pytest.mark.parametrize("role", ["user", "tool"])
async def test_unqualified_original_position_not_stringified_pixels(
    monkeypatch, mode, model, role
):
    req = request(role)
    p, _, calls, events = harness(monkeypatch, req, model=model, mode=mode)
    assert p.get_media_request_support(model=model)["positions"] == []
    await p.complete(req)
    raw = calls[0].content.decode()
    assert PNG not in raw and "Image unavailable" in raw
    body = json.loads(raw)
    if role == "tool":
        assert body["input"][-1]["call_id"] == "call1"
    assert PNG not in json.dumps(events)


@pytest.mark.asyncio
async def test_retry_revalidates_after_known_rejection(monkeypatch):
    req = request()

    async def receive(r, cap):
        cap.valid = False
        return httpx.Response(503, json={"error": {"message": "known rejection"}})

    p, cap, calls, _ = harness(monkeypatch, req, receive=receive)
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(req)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "connection"])
async def test_unknown_transport_outcome_never_replays(monkeypatch, kind):
    req = request()

    async def receive(r, cap):
        raise (
            httpx.ReadTimeout(PNG)
            if kind == "timeout"
            else httpx.RemoteProtocolError(PNG)
        )

    p, _, calls, events = harness(monkeypatch, req, receive=receive)
    with pytest.raises(llm_errors.LLMError) as caught:
        await p.complete(req)
    assert not caught.value.retryable and len(calls) == 1
    assert PNG not in json.dumps(events)


@pytest.mark.asyncio
async def test_request_observer_mutation_refuses_before_http(monkeypatch):
    req = request()
    p, _, calls, _ = harness(monkeypatch, req)

    async def mutate(name, data):
        if name == "llm:request":
            p.extra_request_params["input"] = "changed"

    p._coordinator.hooks.emit = mutate
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(req)
    assert calls == []


def test_diagnostic_copy_known_wire_only():
    source = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c",
                "output": [
                    {"type": "input_image", "image_url": "data:image/png;base64," + PNG}
                ],
            }
        ],
        "metadata": {"opaque": "keep"},
    }
    old = copy.deepcopy(source)
    safe = _media.diagnostic_copy(source)
    assert PNG not in json.dumps(safe) and source == old
    assert safe["metadata"] == {"opaque": "keep"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_terminal_observer_never_invites_outer_replay(monkeypatch, status):
    req = request()

    async def receive(r, cap):
        if status == 503:
            return httpx.Response(503, json={"error": {"message": "known rejection"}})
        return success()

    p, _, calls, _ = harness(monkeypatch, req, receive=receive)
    p._retry_config = module.RetryConfig(max_retries=0)

    async def observe(name, data):
        if name == "llm:response":
            raise llm_errors.ProviderUnavailableError(
                PNG, provider=p.name, retryable=True
            )

    p._coordinator.hooks.emit = observe
    with pytest.raises(llm_errors.LLMError) as caught:
        await p.complete(req)
    assert not caught.value.retryable and PNG not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_model_option_mismatch_refuses_before_http(monkeypatch):
    req = request()
    req.model = "gpt-6.1-sol"
    p, _, calls, _ = harness(monkeypatch, req)
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(req, model="gpt-6-luna")
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("display_stream", [True, False])
@pytest.mark.parametrize(
    "error_kind", ["error", "response.failed", "response.incomplete"]
)
async def test_first_sse_then_echoed_pixels_error_private_at_hook_entry(
    monkeypatch, display_stream, error_kind
):
    req = request()
    req.metadata = {"stream": display_stream}
    canonical = req.model_dump_json()
    url = "data:image/png;base64," + PNG

    async def receive(r, cap):
        error = {"message": "echo " + url, "code": "server_error"}
        final = (
            {"type": error_kind, "error": error}
            if error_kind == "error"
            else {
                "type": error_kind,
                "response": {"error": error},
            }
        )
        wire = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message"},
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "first text",
            },
            final,
        ]
        return httpx.Response(
            200, text="".join("data: " + json.dumps(e) + "\n\n" for e in wire)
        )

    p, _, calls, events = harness(monkeypatch, req, receive=receive)
    with pytest.raises(llm_errors.LLMError) as caught:
        await p.complete(req)
    assert not caught.value.retryable and len(calls) == 1
    assert req.model_dump_json() == canonical
    assert PNG not in json.dumps(events) and url not in json.dumps(events)
    if display_stream:
        assert any(name == "llm:stream_aborted" for name, _ in events)


@pytest.mark.asyncio
async def test_account_change_during_401_refresh_blocks_second_send(monkeypatch):
    req = request()

    async def reject(r, cap):
        return httpx.Response(401, json={"error": {"message": "expired"}})

    p, _, calls, _ = harness(monkeypatch, req, receive=reject)
    p._tokens = {
        "account_id": "selected",
        "access_token": "offline",
        "subject": "selected",
    }
    ensures = []

    async def ensure():
        ensures.append(True)
        if len(ensures) == 2:
            p._tokens = {**p._tokens, "account_id": "different", "subject": "different"}

    p._ensure_valid_tokens = ensure
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(req)
    assert len(calls) == 1 and len(ensures) == 2


@pytest.mark.asyncio
async def test_nonzero_backoff_mutation_blocks_second_send(monkeypatch):
    import asyncio

    req = request()

    async def reject(r, cap):
        return httpx.Response(503, json={"error": {"message": "known rejection"}})

    p, _, calls, _ = harness(monkeypatch, req, receive=reject)
    p._retry_config = module.RetryConfig(
        max_retries=1, min_delay=0.05, max_delay=0.05, jitter=False
    )

    async def mutate():
        while not calls:
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        req.messages[0].content[0].text = "changed during wait"

    mutation = asyncio.create_task(mutate())
    try:
        with pytest.raises(llm_errors.InvalidRequestError):
            await p.complete(req)
        assert len(calls) == 1
    finally:
        await mutation


@pytest.mark.asyncio
async def test_initial_auth_cannot_change_already_selected_account(monkeypatch):
    req = request()
    p, _, calls, _ = harness(monkeypatch, req)
    p._tokens = {
        "account_id": "selected",
        "access_token": "offline",
        "subject": "selected",
    }

    async def ensure():
        p._tokens = {**p._tokens, "account_id": "different", "subject": "different"}

    p._ensure_valid_tokens = ensure
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(req)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("during_401", [False, True])
async def test_same_account_token_rotation_requires_fresh_media_candidate(
    monkeypatch, during_401
):
    req = request()

    async def reject(r, cap):
        return httpx.Response(401, json={"error": {"message": "expired"}})

    p, _, calls, _ = harness(monkeypatch, req, receive=reject)
    p._tokens = {
        "account_id": "same",
        "subject": "same",
        "access_token": "old",
        "id_token": "old-id",
    }
    p._tokens_injected = True
    ensures = []

    async def ensure():
        ensures.append(True)
        if len(ensures) == (2 if during_401 else 1):
            p._tokens = {**p._tokens, "access_token": "new", "id_token": "new-id"}

    p._ensure_valid_tokens = ensure
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(req)
    assert len(calls) == (1 if during_401 else 0)
    assert p._tokens["account_id"] == p._tokens["subject"] == "same"
