"""Optional actual Foundation private-witness pairing, entirely offline.

The caller supplies an explicit installed/pinned Foundation test graph. An
absent optional package is a skip, never installed-graph or DTU evidence.
"""
# ruff: noqa: E402 -- optional test graph gate must precede its imports

import base64
import asyncio
import copy
import io
import json
import time
from datetime import datetime, timedelta, timezone
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
        assert encoded in wire.content.decode()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "path_style"), [
    ("chatgpt_codex", "absolute"), ("chatgpt_codex", "relative"),
    ("chatgpt_plan", "absolute"), ("chatgpt_plan", "relative"), ("chatgpt_plan", "tilde"),
])
@pytest.mark.parametrize("boundary", [
    "same-selected", "initial-account-change", "initial-host-selection-change",
    "401-account-change", "backoff-account-change",
])
async def test_cold_file_auth_and_same_session_resume_preserve_selected_account(
    tmp_path, monkeypatch, mode, path_style, boundary,
):
    """Actual unprimed provider auth reads synthetic file; no login or auth HTTP."""
    from amplifier_module_provider_openai_chatgpt import plan_auth

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    auth_path = tmp_path / "synthetic-auth.json"
    configured_path = (
        "~/synthetic-auth.json" if path_style == "tilde"
        else "synthetic-auth.json" if path_style == "relative" else str(auth_path)
    )
    selected = {
        "auth_mode": mode, "access_token": "offline-synthetic-only",
        **(
            {"subject": "selected-subject", "client_id": "selected-client",
             "scopes": [plan_auth.PLAN_SCOPE], "expires_at": time.time() + 3600}
            if mode == "chatgpt_plan" else
            {"account_id": "selected-account",
             "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
        ),
    }
    auth_path.write_text(json.dumps(selected))
    account_key = "subject" if mode == "chatgpt_plan" else "account_id"
    image = io.BytesIO()
    Image.new("RGB", (3, 2), "blue").save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode()
    media_path = tmp_path / "session-media"
    store = media.SessionMediaStore(
        media_path, "cold-fixture", checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    cap = media.SessionMediaCapability(store, enabled=True)
    origin = {"kind": "tool", "session_id": "cold-fixture", "id": "call-cold"}
    ref = (await cap.ingest(
        image.getvalue(), source_format="image-bytes", origin=origin, policy=None,
    ))["value"]
    await cap.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    request = ChatRequest(model="gpt-6.1-sol", messages=[
        Message(role="user", content="read"),
        Message(role="assistant", content=[ToolCallBlock(id="call-cold", name="read_file", input={})]),
        Message(role="tool", tool_call_id="call-cold", content=[TextBlock(text="result")],
                metadata={"experimental.media.v1": {"refs": [ref]}}),
    ])
    original = request.model_dump()
    real_client = httpx.AsyncClient
    # Success runs twice: both initial and reopened same-session graphs are cold.
    for generation in range(2 if boundary == "same-selected" else 1):
        if generation:
            store = media.SessionMediaStore(
                media_path, "cold-fixture", checkpoint_verifier=lambda receipt: receipt["refs"],
            )
            store.acknowledge_checkpoint({"refs": [ref]})
            cap = media.SessionMediaCapability(store, enabled=True)
        identity = {"provider": "openai-chatgpt", "account": selected[account_key],
                    "generation": generation}
        p = ChatGPTProvider(
            {"auth_mode": mode, "default_model": "gpt-6.1-sol",
             "token_file_path": configured_path},
            SimpleNamespace(get_capability=lambda _: cap),
        )
        assert p._tokens is None  # No public plaintext warm-up before projection.
        p._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0, jitter=False)
        sent = []
        auth_reads = []
        real_ensure = p._ensure_valid_tokens

        async def ensure():
            await asyncio.sleep(0)
            if boundary == "initial-account-change":
                auth_path.write_text(json.dumps({**selected, account_key: "different-account"}))
            await real_ensure()  # Actual provider + mode-specific file authentication.
            auth_reads.append(True)
            if boundary == "initial-host-selection-change":
                identity["account"] = "different-host-selection"
            if boundary == "401-account-change" and sent:
                p._tokens = {**p._tokens, account_key: "different-account"}

        async def emit(name, _data):
            if boundary == "backoff-account-change" and name == "provider:retry":
                p._tokens = {**p._tokens, account_key: "different-account"}

        p._ensure_valid_tokens = ensure
        p._coordinator.hooks = SimpleNamespace(emit=emit)

        def receive(wire):
            sent.append(wire.content)
            assert wire.headers["authorization"] == "Bearer offline-synthetic-only"
            if mode == "chatgpt_codex":
                assert wire.headers["chatgpt-account-id"] == "selected-account"
            assert encoded in wire.content.decode()
            if boundary in {"401-account-change", "backoff-account-change"}:
                return httpx.Response(
                    401 if boundary == "401-account-change" else 503,
                    json={"error": {"code": "server_error", "message": "first outcome"}},
                )
            return httpx.Response(200, text='data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n')

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(
            **kw, transport=httpx.MockTransport(receive),
        ))
        wrapped = MediaPreparedProvider(
            p, cap, provider_identity=lambda: copy.deepcopy(identity),
            model="gpt-6.1-sol", execution_scope={},
        )
        try:
            if boundary == "same-selected":
                await wrapped.complete(request, request_options={"model": "gpt-6.1-sol"})
                assert len(sent) == 1 and p._tokens[account_key] == selected[account_key]
            else:
                expected = (llm_errors.AuthenticationError
                            if mode == "chatgpt_plan" and boundary == "401-account-change"
                            else llm_errors.InvalidRequestError)
                with pytest.raises(expected):
                    await wrapped.complete(request, request_options={"model": "gpt-6.1-sol"})
                assert len(sent) == (1 if boundary.startswith(("401", "backoff")) else 0)
            assert auth_reads and request.model_dump() == original
        finally:
            wrapped.close()
            store.close()