"""Optional installed Foundation public-wrapper pairing, no inference."""
# ruff: noqa: E402 -- optional dependency gate must precede its imports

import base64
import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock
from PIL import Image

media = pytest.importorskip("amplifier_foundation.session.media")
from amplifier_foundation.session.media_provider import (
    MediaPreparedProvider,
)

from amplifier_module_provider_openai_chatgpt.provider import (
    ChatGPTProvider,
)
from tests.test_media import success


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize("revoke", [False, True])
async def test_actual_store_wrapper_to_http(
    tmp_path, monkeypatch, mode, role, image_format, revoke
):
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (17, 89, 167)).save(output, format=image_format)
    pixels = output.getvalue()
    encoded = base64.b64encode(pixels).decode()
    store = media.SessionMediaStore(
        tmp_path / "media",
        "fixture",
        policy=media.MediaPolicy(100000, 10000, 8),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    cap = media.SessionMediaCapability(store, enabled=True)
    origin = {"kind": "tool", "session_id": "fixture", "id": "call1"}
    ref = (
        await cap.ingest(
            pixels, source_format="image-bytes", origin=origin, policy=None
        )
    )["value"]
    await cap.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    parts = [
        TextBlock(text="before"),
        TextBlock(text=f"[Session image {ref['sha256']}]"),
        TextBlock(text="after"),
    ]
    metadata = {"experimental.media.v1": {"refs": [ref]}, "opaque": {"keep": True}}
    messages = (
        [Message(role="user", content=parts, metadata=metadata)]
        if role == "user"
        else [
            Message(role="user", content="read"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="call1", name="read_file", input={})],
            ),
            Message(
                role="tool", tool_call_id="call1", content=parts, metadata=metadata
            ),
        ]
    )
    req = ChatRequest(messages=messages)
    before = req.model_dump_json()
    events, sent = [], []

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))

    coordinator = SimpleNamespace(
        get_capability=lambda _: cap, hooks=SimpleNamespace(emit=emit)
    )
    p = ChatGPTProvider(
        {
            "auth_mode": mode,
            "default_model": "gpt-6.1-sol",
            "raw": True,
            "max_retries": 0,
        },
        coordinator,
    )
    p._ensure_valid_tokens = AsyncMock()
    p._build_headers = lambda: {"authorization": "Bearer offline"}
    real = httpx.AsyncClient

    def receive(r):
        sent.append(json.loads(r.content))
        return success()

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(**kw, transport=httpx.MockTransport(receive)),
    )
    wrapped = MediaPreparedProvider(
        p,
        cap,
        provider_identity={
            "provider": "openai-chatgpt",
            "account": "offline",
            "generation": 1,
        },
        model="gpt-6.1-sol",
        execution_scope={},
    )
    try:
        if revoke:
            prepared = await cap.prepare_request(
                req,
                provider_identity={
                    "provider": "openai-chatgpt",
                    "account": "offline",
                    "media_support": p.get_media_request_support(model="gpt-6.1-sol"),
                },
                model="gpt-6.1-sol",
                request_options={},
                purpose="transport",
                execution_scope={},
            )
            store.epoch += 1
            with pytest.raises(InvalidRequestError):
                await p.complete(prepared.request)
            prepared.release()
            assert sent == []
        else:
            await wrapped.complete(req)
            assert len(sent) == 1
            raw = json.dumps(sent[0])
            if mode == "chatgpt_codex":
                assert encoded in raw
                item = sent[0]["input"][-1]
                actual = item["content"] if role == "user" else item["output"]
                assert [x["type"] for x in actual] == [
                    "input_text",
                    "input_image",
                    "input_text",
                ]
                assert (
                    base64.b64decode(
                        actual[1]["image_url"].split(",", 1)[1], validate=True
                    )
                    == pixels
                )
            else:
                assert encoded not in raw and "Image unavailable" in raw
        assert req.model_dump_json() == before and encoded not in before
        assert encoded not in json.dumps(events)
    finally:
        wrapped.close()
        store.close()
        await p.close()
