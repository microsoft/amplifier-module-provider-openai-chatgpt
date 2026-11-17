"""Offline native-image contracts, not authenticated service acceptance."""

import base64
import copy
import io
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import (
    ChatRequest, ImageBlock, Message, TextBlock, ToolCall, ToolCallBlock, ToolResultBlock,
)
from PIL import Image

from amplifier_module_provider_openai_chatgpt import _images
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider


@pytest.fixture(params=["chatgpt_codex", "chatgpt_plan"])
def provider(request):
    p = ChatGPTProvider(
        {"auth_mode": request.param, "default_model": "exact-model", "max_retries": 0},
        tokens={"auth_mode": request.param, "access_token": "offline-token", "account_id": "offline-account"},
    )
    p._ensure_valid_tokens = AsyncMock()
    return p


def source(fmt="PNG"):
    out = io.BytesIO()
    Image.new("RGB", (3, 2), "blue").save(out, format=fmt)
    return {
        "type": "base64", "media_type": {"PNG": "image/png", "JPEG": "image/jpeg"}[fmt],
        "data": base64.b64encode(out.getvalue()).decode(),
    }


def request_for(src, role="user"):
    return ChatRequest(messages=[Message(role=role, content=[ImageBlock(source=src)])])


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_order_history_lossless_and_nonmutating(provider, fmt):
    src = source(fmt)
    request = ChatRequest(messages=[
        Message(role="system", content="instructions"),
        Message(role="user", content=[
            TextBlock(text="before"), ImageBlock(source=src), TextBlock(text="after"),
            ImageBlock(source=src),
        ]),
        Message(role="assistant", content="saw it"),
        Message(role="user", content="follow-up"),
    ])
    original = copy.deepcopy(request.model_dump())
    payload = provider._build_payload(request)
    assert payload["instructions"] == "instructions"
    content = payload["input"][0]["content"]
    assert [c["type"] for c in content] == ["input_text", "input_image", "input_text", "input_image"]
    assert content[0]["text"] == "before" and content[2]["text"] == "after"
    assert content[1] == {
        "type": "input_image",
        "image_url": f"data:{src['media_type']};base64,{src['data']}",
    }
    assert payload["input"][-1]["content"][0]["text"] == "follow-up"
    assert request.model_dump() == original
    assert payload["stream"] is True and payload["store"] is False


@pytest.mark.parametrize("src", [
    {"type": "url", "url": "https://private.invalid/SECRET"},
    {"type": "file", "path": "/private/SECRET"},
    {"type": "base64", "media_type": "image/gif", "data": "SECRET"},
    {"type": "base64", "media_type": "image/webp", "data": "SECRET"},
    {"type": "base64", "media_type": "image/png", "data": "SECRET%%%"},
    {"type": "base64", "media_type": "image/png", "data": ""},
    {"type": "base64", "media_type": "image/png", "data": "U0VDUkVU"},
])
def test_unsupported_is_safe_nonretryable_and_no_instance_poison(provider, src):
    request = request_for(src)
    original = request.model_dump()
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        provider._build_payload(request)
    assert error.value.retryable is False
    assert error.value.error_code == "unsupported_image_input"
    assert error.value.error_param == "messages.content.image"
    assert "SECRET" not in str(error.value)
    assert request.model_dump() == original
    assert provider._build_payload(ChatRequest(messages=[Message(role="user", content="next")]))[
        "input"
    ][0]["content"][0]["text"] == "next"
    # Reusing an unrepaired unsupported history must remain explicit, not be pruned.
    request.messages.append(Message(role="user", content="next"))
    with pytest.raises(llm_errors.InvalidRequestError):
        provider._build_payload(request)


@pytest.mark.parametrize("role", ["assistant", "system", "developer", "tool", "function"])
def test_unsupported_image_roles_never_silently_drop(provider, role):
    with pytest.raises(llm_errors.InvalidRequestError):
        provider._build_payload(request_for(source(), role))


def test_mime_mismatch(provider):
    src = source()
    src["media_type"] = "image/jpeg"
    with pytest.raises(llm_errors.InvalidRequestError, match="does not match"):
        provider._build_payload(request_for(src))


@pytest.mark.parametrize("budget", ["PLAN_MAX_PAYLOAD_BYTES", "PLAN_MAX_IMAGES"])
def test_bounded_images_and_original_unchanged(provider, monkeypatch, budget):
    request = request_for(source())
    original = request.model_dump()
    monkeypatch.setattr(_images, budget, 0)
    if provider.auth_mode == "chatgpt_plan":
        with pytest.raises(llm_errors.InvalidRequestError, match="limit"):
            provider._build_payload(request)
    else:
        provider._build_payload(request)  # No fabricated Codex transport ceiling.
    assert request.model_dump() == original


def test_native_tool_output_keeps_call_position_and_order(provider):
    src = source()
    output = [{"type": "text", "text": "tool before"}, {"type": "image", "source": src},
              {"type": "text", "text": "tool after"}]
    request = ChatRequest(messages=[
        Message(role="user", content="look"),
        Message(role="assistant", content=[ToolCallBlock(id="call-image", name="screen", input={})]),
        Message(role="tool", content=[ToolResultBlock(tool_call_id="call-image", output=output)]),
        Message(role="user", content="continue"),
    ])
    original = request.model_dump()
    payload = provider._build_payload(request)
    items = payload["input"]
    assert [i.get("type", i.get("role")) for i in items] == [
        "user", "function_call", "function_call_output", "user",
    ]
    assert items[1]["call_id"] == items[2]["call_id"] == "call-image"
    assert items[2]["output"] == [
        {"type": "input_text", "text": "tool before"},
        {"type": "input_image", "image_url": f"data:image/png;base64,{src['data']}"},
        {"type": "input_text", "text": "tool after"},
    ]
    assert request.model_dump() == original


@pytest.mark.parametrize("output", [
    {"type": "image", "not_content": "ordinary JSON"},
    [{"type": "text", "text": "ordinary JSON"}],
    '{"type":"image","source":{"data":"not-an-image"}}',
])
def test_ordinary_json_tool_outputs_are_not_promoted(provider, output):
    payload = provider._build_payload(ChatRequest(messages=[
        Message(role="tool", content=[ToolResultBlock(tool_call_id="call", output=output)])
    ]))
    assert payload["input"][0]["output"] == (
        output if isinstance(output, str) else json.dumps(output)
    )


def test_image_bearing_tool_lists_reject_ambiguous_entries(provider):
    output = [{"type": "image", "source": source()}, {"arbitrary": "JSON"}]
    with pytest.raises(llm_errors.InvalidRequestError):
        provider._build_payload(ChatRequest(messages=[
            Message(role="tool", content=[ToolResultBlock(tool_call_id="call", output=output)])
        ]))


def install_transport(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(
        transport=httpx.MockTransport(handler), **kw
    ))


def completed(text="done"):
    return "\n".join("data: " + json.dumps(event) + "\n" for event in [
        {"type": "response.output_item.done", "item": {
            "type": "message", "content": [{"type": "output_text", "text": text}],
        }},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 5, "output_tokens": 1,
        }}},
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 400, 401, 500])
async def test_native_wire_and_all_diagnostics_hide_image(provider, monkeypatch, status):
    src = source()
    url = f"data:image/png;base64,{src['data']}"
    events = []
    provider.raw = True
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(side_effect=lambda name, data: events.append((name, data)))
    calls = []

    def handle(request):
        body = json.loads(request.content)
        assert body["input"][0]["content"][0]["image_url"] == url
        calls.append(str(request.url))
        if status == 200:
            return httpx.Response(200, text=completed("echo " + url + " " + src["data"]))
        return httpx.Response(status, json={"error": {
            "message": "echo " + url + " " + src["data"], "code": "invalid_request_error",
        }})

    install_transport(monkeypatch, handle)
    original = request_for(src)
    before = original.model_dump()
    if status == 200:
        result = await provider.complete(original)
        # Actual model output is preserved; diagnostics are a separate copy.
        assert src["data"] in str(result)
    else:
        with pytest.raises(llm_errors.LLMError) as error:
            await provider.complete(original)
        assert src["data"] not in str(error.value)
        assert src["data"] not in str(getattr(error.value, "response_body", ""))
    assert src["data"] not in json.dumps(events)
    assert url not in json.dumps(events)
    expected = ("https://api.openai.com/v1/responses" if provider.auth_mode == "chatgpt_plan"
                else "https://chatgpt.com/backend-api/codex/responses")
    assert calls and set(calls) == {expected}
    assert original.model_dump() == before


@pytest.mark.asyncio
async def test_invalid_image_is_never_sent_or_emitted(provider, monkeypatch):
    def forbidden(request):
        pytest.fail("invalid image reached HTTP")

    install_transport(monkeypatch, forbidden)
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock()
    with pytest.raises(llm_errors.InvalidRequestError):
        await provider.complete(request_for({"type": "base64", "data": "SECRET"}))
    provider._coordinator.hooks.emit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["error", "response.failed", "response.incomplete"])
@pytest.mark.parametrize("display", [True, False])
async def test_sse_errors_scrub_before_hook_and_error(provider, monkeypatch, kind, display):
    src = source()
    events = []
    provider.raw = True
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )
    detail = {"message": "echo " + src["data"], "code": "invalid_request_error"}
    final = ({"type": kind, "error": detail} if kind == "error"
             else {"type": kind, "response": {"error": detail}})
    wire = [
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "partial"},
        final,
    ]
    install_transport(monkeypatch, lambda _: httpx.Response(
        200, text="".join("data: " + json.dumps(e) + "\n\n" for e in wire)
    ))
    request = request_for(src)
    request.metadata = {"stream": display}
    with pytest.raises(llm_errors.LLMError) as error:
        await provider.complete(request)
    assert src["data"] not in json.dumps(events)
    assert src["data"] not in str(error.value)
    assert src["data"] not in str(getattr(error.value, "response_body", ""))


@pytest.mark.asyncio
async def test_escaped_json_error_echo_is_decoded_before_scrubbing(provider, monkeypatch):
    src = source()
    body = json.dumps({"error": {"code": "invalid_request_error", "message": src["data"]}})
    body = body.replace("iVBOR", r"\u0069VBOR")
    events = []
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )
    install_transport(monkeypatch, lambda _: httpx.Response(400, text=body))
    with pytest.raises(llm_errors.LLMError) as error:
        await provider.complete(request_for(src))
    assert src["data"] not in str(error.value)
    assert src["data"] not in str(getattr(error.value, "response_body", ""))
    assert r"\u0069VBOR" not in json.dumps(events)


@pytest.mark.asyncio
async def test_fragmented_stream_echo_cannot_reconstruct_pixels(provider, monkeypatch):
    src = source()
    mid = len(src["data"]) // 2
    wire = [
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": src["data"][:mid]},
        {"type": "response.output_text.delta", "output_index": 0, "delta": src["data"][mid:]},
        {"type": "response.output_item.done", "output_index": 0, "item": {
            "type": "message", "content": [{"type": "output_text", "text": src["data"]}],
        }},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 1}}},
    ]
    events = []
    provider.raw = True
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )
    install_transport(monkeypatch, lambda _: httpx.Response(
        200, text="".join("data: " + json.dumps(e) + "\n\n" for e in wire)
    ))
    result = await provider.complete(request_for(src))
    assert result.content[0].text == src["data"]
    assert "".join(data.get("text", "") for _, data in events) != src["data"]
    assert src["data"][:mid] not in json.dumps(events)
    assert src["data"][mid:] not in json.dumps(events)


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer", "function"])
def test_misplaced_tool_image_carrier_is_not_silently_lost(provider, role):
    request = ChatRequest(messages=[Message(role=role, content=[
        ToolResultBlock(tool_call_id="call", output=[{"type": "image", "source": source()}])
    ])])
    with pytest.raises(llm_errors.InvalidRequestError, match="tool message role"):
        provider._build_payload(request)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("call_shape", ["blocks", "attribute"])
def test_actual_foundation_tool_message_shape(provider, fmt, call_shape):
    src = source(fmt)
    assistant = (
        Message(role="assistant", content=[ToolCallBlock(id="actualid", name="read_file", input={})])
        if call_shape == "blocks" else
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="actualid", name="read_file", arguments={})
        ])
    )
    request = ChatRequest(messages=[
        Message(role="user", content="read"),
        assistant,
        Message(role="tool", name="read_file", tool_call_id="actualid", content=[
            TextBlock(text="existing serialized tool content"), ImageBlock(source=src),
        ], metadata={"experimental.media.v1": {"refs": [{"not": "authority"}]}}),
    ])
    original = request.model_dump()
    payload = provider._build_payload(request)
    assert payload["input"][-1] == {
        "type": "function_call_output", "call_id": "actualid", "output": [
            {"type": "input_text", "text": "existing serialized tool content"},
            {"type": "input_image", "image_url": f"data:{src['media_type']};base64,{src['data']}"},
        ],
    }
    assert payload["input"][-2]["type"] == "function_call"
    assert payload["input"][-2]["call_id"] == "actualid"
    assert request.model_dump() == original


@pytest.mark.parametrize("model", sorted(_images.CODEX_IMAGE_MODELS))
def test_indexed_facts_exact_models_positions_and_nonmutation(provider, model):
    request = ChatRequest(messages=[
        Message(role="system", content="instructions"),
        Message(role="user", content="read"),
        Message(role="assistant", content=[ToolCallBlock(id="actual", name="read", input={})]),
        Message(role="tool", tool_call_id="actual", content="serialized tool text",
                metadata={"experimental.media.v1": {"refs": [{"opaque": "not authority"}]}}),
        Message(role="tool", tool_call_id="actual", content="duplicate consumed call"),
        Message(role="tool", tool_call_id="orphan", content="orphan"),
    ])
    before = request.model_dump()
    options = {"model": model}
    facts = provider.get_media_request_support_for_request(
        request=request, model=model, request_options=options,
    )
    assert facts == {
        "v": 2, "mime_types": ["image/png", "image/jpeg"],
        "positions": [{"message_index": 1, "role": "user"}, {"message_index": 3, "role": "tool"}]
        if provider.auth_mode == "chatgpt_codex" else [],
    }
    assert request.model_dump() == before and options == {"model": model}
    facts["positions"].clear()
    assert provider.get_media_request_support_for_request(
        request=request, model=model, request_options=options,
    )["positions"] == (
        [{"message_index": 1, "role": "user"}, {"message_index": 3, "role": "tool"}]
        if provider.auth_mode == "chatgpt_codex" else []
    )


@pytest.mark.parametrize("model", ["latest", "unknown", "gpt-6-unproven"])
def test_indexed_support_never_guesses_model_modality(provider, model):
    assert provider.get_media_request_support_for_request(
        request=request_for(source()), model=model,
    )["positions"] == []


def test_indexed_support_declines_option_override(provider):
    request = request_for(source())
    assert provider.get_media_request_support_for_request(
        request=request, model="gpt-6.1-sol", request_options={"model": "different"},
    )["positions"] == []
    provider.extra_request_params["input"] = "different"
    assert provider.get_media_request_support_for_request(
        request=request, model="gpt-6.1-sol",
    )["positions"] == []


def test_fragment_filter_bounded_memory_and_ordinary_text():
    secret = "A" * 100_000 + "B"
    redactor = _images.ImageEchoStream((secret,))
    assert redactor.feed("hello world") == "hello world"
    collected = []
    for _ in range(1000):
        collected.append(redactor.feed("A" * 100))
        assert len(redactor.pending) <= 63
    collected.append(redactor.feed("B done", final=True))
    assert "".join(collected) == "[IMAGE REDACTED] done"


@pytest.mark.asyncio
async def test_actual_tool_arguments_remain_exact_while_raw_diagnostics_hide_image(provider, monkeypatch):
    src = source()
    args = {"opaque": src["data"]}
    wire = [
        {"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "call", "name": "keep",
            "arguments": json.dumps(args),
        }},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 5, "output_tokens": 1,
        }}},
    ]
    events = []
    provider.raw = True
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )
    install_transport(monkeypatch, lambda _: httpx.Response(
        200, text="".join("data: " + json.dumps(e) + "\n\n" for e in wire)
    ))
    result = await provider.complete(request_for(src))
    assert result.tool_calls[0].arguments == args
    assert src["data"] not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", [httpx.ReadTimeout, httpx.RemoteProtocolError, ValueError])
async def test_transport_exception_echo_scrubbed(provider, monkeypatch, error_class):
    import traceback

    src = source()
    events = []
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )

    def handle(_):
        raise error_class(src["data"])

    install_transport(monkeypatch, handle)
    with pytest.raises(llm_errors.LLMError) as error:
        await provider.complete(request_for(src))
    assert src["data"] not in str(error.value)
    assert src["data"] not in "".join(traceback.format_exception(error.value))
    assert src["data"] not in json.dumps(events)


def test_plan_size_is_exact_httpx_json_and_counts_nonimage_content(provider, monkeypatch):
    request = request_for(source())
    payload = provider._build_payload(request)
    size = len(httpx.Request("POST", "https://unused.invalid", json=payload).content)
    monkeypatch.setattr(_images, "PLAN_MAX_PAYLOAD_BYTES", size)
    assert provider._build_payload(request) == payload
    monkeypatch.setattr(_images, "PLAN_MAX_PAYLOAD_BYTES", size - 1)
    if provider.auth_mode == "chatgpt_plan":
        with pytest.raises(llm_errors.InvalidRequestError, match="512 MB"):
            provider._build_payload(request)
    else:
        provider._build_payload(request)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_truncated_frames_fail_locally(provider, fmt):
    src = source(fmt)
    src["data"] = base64.b64encode(base64.b64decode(src["data"])[:-2]).decode()
    with pytest.raises(llm_errors.InvalidRequestError, match="truncated"):
        provider._build_payload(request_for(src))


def test_image_input_cannot_be_silently_replaced_by_escape_hatch(provider):
    provider.extra_request_params["input"] = [{"role": "user", "content": "lost"}]
    with pytest.raises(llm_errors.InvalidRequestError, match="replace"):
        provider._build_payload(request_for(source()))


def test_untranslated_image_detail_is_explicit(provider):
    request = request_for(source())
    request.messages[0].content[0].detail = "original"
    with pytest.raises(llm_errors.InvalidRequestError, match="extra fields"):
        provider._build_payload(request)