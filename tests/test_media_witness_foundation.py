"""Optional actual Foundation private-witness pairing, entirely offline.

The caller supplies an explicit installed/pinned Foundation test graph. An
absent optional package is a skip, never installed-graph or DTU evidence.
"""
# ruff: noqa: E402 -- optional test graph gate must precede its imports

import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock
from amplifier_core.utils.retry import RetryConfig
from PIL import Image

media = pytest.importorskip("amplifier_foundation.session.media")
from amplifier_foundation.session.media_provider import MediaPreparedProvider

from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("boundary", ["auth", "401", "backoff", "valid-retry"])
async def test_actual_private_witness_at_each_physical_send(
    tmp_path, monkeypatch, mode, role, boundary,
):
    image = io.BytesIO()
    Image.new("RGB", (3, 2), "blue").save(image, format="PNG")
    pixels = image.getvalue()
    encoded = base64.b64encode(pixels).decode()
    store = media.SessionMediaStore(
        tmp_path / "media", "witness-fixture",
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    cap = media.SessionMediaCapability(store, enabled=True)
    origin = {"kind": "tool", "session_id": "witness-fixture", "id": "actual"}
    ref = (await cap.ingest(
        pixels, source_format="image-bytes", origin=origin, policy=None,
    ))["value"]
    await cap.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    parts = [TextBlock(text="existing result"), TextBlock(text=f"[Session image {ref['sha256']}]")]
    row = Message(
        role=role, content=parts,
        tool_call_id="actual" if role == "tool" else None,
        metadata={"experimental.media.v1": {"refs": [ref]}},
    )
    messages = ([Message(role="user", content="read"), Message(
        role="assistant", content=[ToolCallBlock(id="actual", name="read_file", input={})],
    ), row] if role == "tool" else [row])
    request = ChatRequest(messages=messages, model="gpt-6.1-sol")
    before = request.model_dump()
    calls, validations = [], []
    real_validate = cap.validate

    def validate(private, *, purpose, request_options=None):
        result = real_validate(private, purpose=purpose, request_options=request_options)
        if purpose == "provider-send":
            validations.append((id(private), request_options.copy()))
        return result

    cap.validate = validate
    p = ChatGPTProvider(
        {"auth_mode": mode, "default_model": "gpt-6.1-sol"},
        SimpleNamespace(get_capability=lambda _: cap),
        tokens={"auth_mode": mode, "access_token": "offline", "account_id": "fixture"},
    )
    p._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0, jitter=False)

    async def ensure():
        if boundary == "auth" or boundary == "401" and calls:
            store.epoch += 1

    async def emit(name, _data):
        if boundary == "backoff" and name == "provider:retry":
            store.epoch += 1

    p._ensure_valid_tokens = AsyncMock(side_effect=ensure)
    p._coordinator.hooks = SimpleNamespace(emit=emit)
    real_client = httpx.AsyncClient

    def receive(wire):
        calls.append((wire.content, len(validations)))
        body = json.loads(wire.content)
        assert validations and validations[-1][1] == {"model": "gpt-6.1-sol", "fixture": "exact"}
        output = body["input"][-1]
        if role == "tool":
            assert output["type"] == "function_call_output" and output["call_id"] == "actual"
        if mode == "chatgpt_codex":
            assert encoded in wire.content.decode()
        else:
            assert encoded not in wire.content.decode()
            assert "Image unavailable" in wire.content.decode()
        if len(calls) == 1:
            return httpx.Response(
                401 if boundary == "401" else 503,
                json={"error": {"code": "server_error", "message": "first outcome"}},
            )
        return httpx.Response(200, text='data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n')

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(
        **kw, transport=httpx.MockTransport(receive),
    ))
    wrapped = MediaPreparedProvider(
        p, cap, provider_identity={"provider": p.name, "account": "fixture", "generation": 1},
        model="gpt-6.1-sol", execution_scope={},
    )
    try:
        if boundary == "valid-retry":
            await wrapped.complete(request, request_options={"model": "gpt-6.1-sol", "fixture": "exact"})
            assert len(calls) == 2 and calls[0][0] == calls[1][0]
            assert calls[1][1] > calls[0][1]
        else:
            expected = (llm_errors.AuthenticationError
                        if mode == "chatgpt_plan" and boundary == "401"
                        else llm_errors.InvalidRequestError)
            with pytest.raises(expected):
                await wrapped.complete(request, request_options={"model": "gpt-6.1-sol", "fixture": "exact"})
            assert len(calls) == (0 if boundary == "auth" else 1)
        assert request.model_dump() == before
    finally:
        wrapped.close()
        store.close()