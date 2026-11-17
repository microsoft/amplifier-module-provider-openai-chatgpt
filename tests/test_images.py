"""Offline native-image contracts, not authenticated service acceptance."""

import base64
import copy
import io
import json
from types import SimpleNamespace
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


@pytest.mark.parametrize("reverse", [True, False])
def test_fragment_filter_shared_image_prefixes(reverse):
    images = []
    for color in ("red", "blue"):
        out = io.BytesIO()
        Image.new("RGB", (100, 100), color).save(out, format="PNG")
        images.append(base64.b64encode(out.getvalue()).decode())
    values = tuple("data:image/png;base64," + item for item in images)
    assert values[0][:64] == values[1][:64]
    for value in reversed(values) if reverse else values:
        redactor = _images.ImageEchoStream(values + tuple(images))
        visible = "".join(redactor.feed(value[i:i+31]) for i in range(0, len(value), 31))
        visible += redactor.feed("", final=True)
        assert visible == "[IMAGE REDACTED]"


@pytest.mark.asyncio
async def test_escaped_canonical_tool_arguments_are_not_diagnostic_pixels(provider, monkeypatch):
    src = source()
    arguments = json.dumps({"image": src["data"]}).replace("iVBOR", r"\u0069VBOR")
    wire = [
        {"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "call", "name": "keep", "arguments": arguments,
        }},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 1, "output_tokens": 1}}},
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
    response = await provider.complete(request_for(src))
    assert response.tool_calls[0].arguments == {"image": src["data"]}
    assert src["data"] not in json.dumps(events)
    assert r"\u0069VBOR" not in json.dumps(events)


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
def test_signature_check_does_not_pretend_to_validate_raster_integrity(provider, fmt):
    src = source(fmt)
    src["data"] = base64.b64encode(base64.b64decode(src["data"])[:-2]).decode()
    # Only the decoder/backend can judge raster completeness. No fabricated
    # terminal framing policy and no repair/re-encoding of the source.
    assert provider._build_payload(request_for(src))["input"][0]["content"][0][
        "image_url"
    ].endswith(src["data"])


def test_image_input_cannot_be_silently_replaced_by_escape_hatch(provider):
    provider.extra_request_params["input"] = [{"role": "user", "content": "lost"}]
    with pytest.raises(llm_errors.InvalidRequestError, match="replace"):
        provider._build_payload(request_for(source()))


def test_untranslated_image_detail_is_explicit(provider):
    request = request_for(source())
    request.messages[0].content[0].detail = "original"
    with pytest.raises(llm_errors.InvalidRequestError, match="extra fields"):
        provider._build_payload(request)


def serialized_call_request(content):
    return ChatRequest(messages=[
        Message.model_validate({
            "role": "assistant", "content": "", "tool_calls": [
                {"id": "cold-call", "name": "read_file", "arguments": {"path": "fixture"}}
            ],
        }),
        Message(role="tool", tool_call_id="cold-call", content=content),
    ])


def test_cold_history_dictionary_tool_calls(provider):
    request = serialized_call_request([TextBlock(text="read"), ImageBlock(source=source())])
    before = request.model_dump()
    facts = provider.get_media_request_support_for_request(request=request, model="gpt-6.1-sol")
    assert facts["positions"] == (
        [{"message_index": 1, "role": "tool"}] if provider.auth_mode == "chatgpt_codex" else []
    )
    payload = provider._build_payload(request)
    assert payload["input"][-2] == {
        "type": "function_call", "call_id": "cold-call", "name": "read_file",
        "arguments": '{"path": "fixture"}',
    }
    assert payload["input"][-1]["call_id"] == "cold-call"
    assert payload["input"][-1]["output"][1]["type"] == "input_image"
    assert request.model_dump() == before


@pytest.mark.parametrize("bad", [
    {"id": "SECRET", "name": "read", "arguments": "SECRET"},
    {"name": "SECRET", "arguments": {}},
    {"id": [], "name": "SECRET", "arguments": {}},
    "SECRET", None,
])
def test_invalid_dictionary_calls_raise_safe_typed_errors(provider, bad):
    import traceback

    request = ChatRequest(messages=[Message.model_validate({
        "role": "assistant", "content": "", "tool_calls": [bad],
    })])
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        provider._build_payload(request)
    assert error.value.retryable is False and "SECRET" not in str(error.value)
    assert "SECRET" not in "".join(traceback.format_exception(error.value))
    if provider.auth_mode == "chatgpt_codex":
        with pytest.raises(llm_errors.InvalidRequestError):
            provider.get_media_request_support_for_request(request=request, model="gpt-6.1-sol")


@pytest.mark.parametrize("first_result", ["string", "blocks", "nested-text", "raw-image", "nested-image"])
@pytest.mark.parametrize("second_result", ["raw", "nested"])
@pytest.mark.parametrize("call_shape", ["dict", "object"])
def test_completed_call_never_qualifies_a_second_image_result(
    provider, first_result, second_result, call_shape,
):
    src = source()
    results = {
        "string": Message(role="tool", tool_call_id="cold-call", content="done"),
        "blocks": Message(role="tool", tool_call_id="cold-call", content=[TextBlock(text="done")]),
        "nested-text": Message(role="tool", content=[
            ToolResultBlock(tool_call_id="cold-call", output="done"),
        ]),
        "raw-image": Message(role="tool", tool_call_id="cold-call", content=[ImageBlock(source=src)]),
        "nested-image": Message(role="tool", content=[ToolResultBlock(
            tool_call_id="cold-call", output=[{"type": "image", "source": src}],
        )]),
    }
    second = (
        Message(role="tool", tool_call_id="cold-call", content=[ImageBlock(source=src)])
        if second_result == "raw" else Message(role="tool", content=[ToolResultBlock(
            tool_call_id="cold-call", output=[{"type": "image", "source": src}],
        )])
    )
    request = serialized_call_request("unused")
    if call_shape == "object":
        request.messages[0].tool_calls = [ToolCall(id="cold-call", name="read_file", arguments={})]
    request.messages[1:] = [results[first_result], second]
    facts = provider.get_media_request_support_for_request(request=request, model="gpt-6.1-sol")
    assert {"message_index": 2, "role": "tool"} not in facts["positions"]
    with pytest.raises(llm_errors.InvalidRequestError, match="outstanding"):
        provider._build_payload(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True, "remote"])
@pytest.mark.parametrize("array_input", [True, False])
async def test_extra_input_images_rejected_before_any_hook_or_http(
    provider, monkeypatch, native, array_input,
):
    src = source()
    image = (
        {"type": "input_image", "image_url": "https://private.invalid/SECRET"}
        if native == "remote" else
        {"type": "input_image", "image_url": f"data:image/png;base64,{src['data']}"}
        if native else {"type": "image", "source": src}
    )
    row = {"role": "user", "content": [image]}
    provider.extra_request_params["input"] = [row] if array_input else row
    provider.raw = True
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock()
    install_transport(monkeypatch, lambda _: pytest.fail("override image reached HTTP"))
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        await provider.complete(ChatRequest(messages=[Message(role="user", content="text only")]))
    assert src["data"] not in str(error.value) and "SECRET" not in str(error.value)
    provider._coordinator.hooks.emit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("delta_type", ["response.output_text.delta", "response.reasoning_text.delta"])
@pytest.mark.parametrize("suffix", ["i", "iVBORw0KGgoAAA"])
async def test_partial_nonimage_suffix_flushed_before_transport_abort(
    provider, monkeypatch, delta_type, suffix,
):
    events = []
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )
    wire = [
        {"type": "response.output_item.added", "output_index": 0, "item": {
            "type": "message" if "output_text" in delta_type else "reasoning",
        }},
        {"type": delta_type, "output_index": 0, "delta": "kiw" + suffix},
    ]

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self):
            for event in wire:
                yield ("data: " + json.dumps(event) + "\n\n").encode()
            raise httpx.ReadError("offline interrupted")

    install_transport(monkeypatch, lambda _: httpx.Response(200, stream=Interrupted()))
    with pytest.raises(llm_errors.LLMError):
        await provider.complete(request_for(source()))
    visible = "".join(data.get("text", "") for name, data in events if name == "llm:stream_block_delta")
    assert visible == "kiw" + suffix
    assert [name for name, _ in events].index("llm:stream_aborted") > max(
        i for i, (name, _) in enumerate(events) if name == "llm:stream_block_delta"
    )


@pytest.mark.parametrize("container", ["function_call_output", "tool_result", "message"])
def test_override_scanner_rejects_nested_known_carriers_but_leaves_opaque_json(provider, container):
    image = {"type": "image", "source": source()}
    item = ({"type": container, "role": "tool", "content": [image]} if container == "message"
            else {"type": container, "call_id": "c", "output": [image]})
    provider.extra_request_params["input"] = [item]
    with pytest.raises(llm_errors.InvalidRequestError, match="typed message"):
        provider._build_payload(ChatRequest(messages=[]))
    opaque = {
        "type": "function_call", "call_id": "c", "name": "opaque",
        "arguments": json.dumps({"type": "image", "source": source()}),
        "metadata": {"type": "image", "source": source()},
    }
    provider.extra_request_params["input"] = [opaque]
    assert provider._build_payload(ChatRequest(messages=[]))["input"] == [opaque]


@pytest.mark.asyncio
async def test_cold_history_dict_calls_reach_actual_native_http_wire(provider, monkeypatch):
    src = source()
    request = serialized_call_request([TextBlock(text="original"), ImageBlock(source=src)])
    before = request.model_dump()
    sent = []

    def handle(wire):
        body = json.loads(wire.content)
        sent.append(body)
        assert body["input"][-1] == {
            "type": "function_call_output", "call_id": "cold-call", "output": [
                {"type": "input_text", "text": "original"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{src['data']}"},
            ],
        }
        return httpx.Response(200, text=completed())

    install_transport(monkeypatch, handle)
    await provider.complete(request)
    assert len(sent) == 1 and request.model_dump() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", ["message", "function_call_output", "computer_call_output"])
@pytest.mark.parametrize("location", ["data", "remote"])
async def test_constructor_tuple_and_computer_override_images_never_escape(
    provider, monkeypatch, carrier, location,
):
    src = source()
    url = (f"data:image/png;base64,{src['data']}" if location == "data"
           else "https://private.invalid/SECRET")
    content = ({"type": "input_image", "image_url": url},)
    row = (
        {"role": "user", "content": content} if carrier == "message" else
        {"type": carrier, "call_id": "c", "output": (
            {"type": "computer_screenshot", "image_url": url}
            if carrier == "computer_call_output" else content
        )}
    )
    p = ChatGPTProvider({
        "auth_mode": provider.auth_mode, "default_model": "offline", "raw": True,
        "extra_request_params": {"input": (row,)},
    })
    p._ensure_valid_tokens = AsyncMock()
    p._coordinator = MagicMock()
    p._coordinator.hooks.emit = AsyncMock()
    install_transport(monkeypatch, lambda _: pytest.fail("override image reached HTTP"))
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        await p.complete(ChatRequest(messages=[Message(role="user", content="ordinary")]))
    assert src["data"] not in str(error.value) and "SECRET" not in str(error.value)
    p._coordinator.hooks.emit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [["first", "second"], ["[Image unavailable: no pixels supplied]"]])
async def test_plain_typed_tool_text_roundtrip_no_media_capability(provider, monkeypatch, texts):
    request = serialized_call_request([TextBlock(text=text) for text in texts])
    original = request.model_dump()
    sent = []

    def handle(wire):
        body = json.loads(wire.content)
        sent.append(body)
        assert body["input"][-1] == {
            "type": "function_call_output", "call_id": "cold-call", "output": "\n".join(texts),
        }
        return httpx.Response(200, text=completed("continued"))

    install_transport(monkeypatch, handle)
    assert (await provider.complete(request)).content[0].text == "continued"
    assert len(sent) == 1 and request.model_dump() == original


@pytest.mark.parametrize("image", [
    {"type": "image", "url": "https://example.invalid/logo", "title": "logo"},
    {"type": "image", "source": "stock photography"},
    {"type": "image", "source": {"type": "catalog-record", "id": "ordinary"}},
])
def test_ordinary_business_image_json_is_not_promoted_or_rejected(provider, image):
    output = [
        image,
        {"type": "web", "title": "ordinary result"},
    ]
    request = ChatRequest(messages=[Message(role="tool", content=[
        ToolResultBlock(tool_call_id="c", output=output),
    ])])
    assert provider._build_payload(request)["input"][-1]["output"] == json.dumps(output)
    # Reusing this history for the following text request must remain possible.
    request.messages.append(Message(role="user", content="continue"))
    assert provider._build_payload(request)["input"][-1]["content"][0]["text"] == "continue"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_valid_image_with_trailer_keeps_exact_original_bytes(provider, fmt):
    src = source(fmt)
    original = base64.b64decode(src["data"]) + b"TRAILER"
    with Image.open(io.BytesIO(original)) as image:
        image.load()  # Fixture really is decoder-accepted.
    src["data"] = base64.b64encode(original).decode()
    actual = provider._build_payload(request_for(src))["input"][0]["content"][0]["image_url"]
    assert base64.b64decode(actual.split(",", 1)[1]) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["error", "response.failed", "response.incomplete"])
async def test_sse_error_never_emits_held_text_after_abort(provider, monkeypatch, kind):
    error = {"message": "offline error", "code": "invalid_request_error"}
    terminal = ({"type": kind, "error": error} if kind == "error" else
                {"type": kind, "response": {"error": error}})
    wire = [
        {"type": "response.output_text.delta", "output_index": 0, "delta": "I rea"},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "d"},
        terminal,
    ]
    events = []
    provider._coordinator = MagicMock()
    provider._coordinator.hooks.emit = AsyncMock(
        side_effect=lambda name, data: events.append((name, data))
    )
    install_transport(monkeypatch, lambda _: httpx.Response(
        200, text="".join("data: " + json.dumps(e) + "\n\n" for e in wire),
    ))
    with pytest.raises(llm_errors.LLMError):
        await provider.complete(request_for(source()))
    names = [name for name, _ in events]
    assert "".join(data["text"] for name, data in events if name == "llm:stream_block_delta") == "I read"
    assert all(name != "llm:stream_block_delta" for name in names[names.index("llm:stream_aborted") + 1:])


class PrivateWitness:
    """Structural unit fixture, NOT installed Foundation composition evidence."""
    version = 1

    def __init__(self, request):
        self.request = request
        self.source = request.model_dump()
        self.valid = True
        self.calls = []

    def validate(self, request, *, purpose, request_options):
        assert request is self.request and purpose == "provider-send"
        self.calls.append(copy.deepcopy(request_options))
        if not self.valid or request.model_dump() != self.source:
            raise ValueError("SECRET private refusal")
        return {"valid": True}


def witness_request(provider):
    request = request_for(source())
    request.model = "gpt-6.1-sol"
    request.messages[0].metadata = {"experimental.media.v1": {"refs": []}}
    witness = PrivateWitness(request)
    provider._coordinator = SimpleNamespace(get_capability=lambda _: witness)
    return request, witness


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["auth", "model", "request-hook"])
async def test_perwire_witness_invalidation_before_first_send(provider, monkeypatch, where):
    request, witness = witness_request(provider)
    if where == "auth":
        provider._ensure_valid_tokens = AsyncMock(side_effect=lambda: setattr(witness, "valid", False))
    elif where == "model":
        async def resolve():
            witness.valid = False
            return "gpt-6.1-sol"
        provider._resolve_default_model = resolve
    else:
        provider._coordinator.hooks = SimpleNamespace(emit=AsyncMock(
            side_effect=lambda *_: setattr(witness, "valid", False),
        ))
    install_transport(monkeypatch, lambda _: pytest.fail("revoked witness reached HTTP"))
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        await provider.complete(request, request_options={"model": request.model, "caller": "unit"})
    assert error.value.retryable is False and "SECRET" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["401", "backoff"])
@pytest.mark.parametrize("change", ["witness", "capability", "options", "request", "account", "config"])
async def test_perwire_invalidation_after_known_rejection_never_sends_twice(
    provider, monkeypatch, retry, change,
):
    from amplifier_core.utils.retry import RetryConfig

    request, witness = witness_request(provider)
    options = {"model": request.model, "caller": "unit"}
    sent = []
    provider._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0, jitter=False)

    def mutate():
        if change == "witness":
            witness.valid = False
        elif change == "capability":
            provider._coordinator.get_capability = lambda _: None
        elif change == "options":
            options["caller"] = "changed"
        elif change == "request":
            request.messages[0].content[0].source["data"] = "changed"
        elif change == "account":
            provider._tokens["account_id"] = "changed"
        else:
            provider.extra_request_params["reasoning"] = {"effort": "changed"}

    if retry == "401":
        async def ensure():
            if sent:
                mutate()
        provider._ensure_valid_tokens = ensure
    else:
        async def emit(name, data):
            if name == "provider:retry":
                mutate()
        provider._coordinator.hooks = SimpleNamespace(emit=emit)

    def handle(wire):
        sent.append(json.loads(wire.content))
        return httpx.Response(int(retry) if retry == "401" else 503,
                              json={"error": {"code": "server_error", "message": "first rejection"}})

    install_transport(monkeypatch, handle)
    error_type = (llm_errors.AuthenticationError
                  if retry == "401" and provider.auth_mode == "chatgpt_plan"
                  else llm_errors.InvalidRequestError)
    with pytest.raises(error_type):
        await provider.complete(request, request_options=options)
    assert len(sent) == 1
    assert sent[0]["input"][0]["content"][0]["type"] == "input_image"


@pytest.mark.asyncio
async def test_same_private_candidate_validates_before_each_successful_retry(provider, monkeypatch):
    from amplifier_core.utils.retry import RetryConfig

    request, witness = witness_request(provider)
    options = {"model": request.model, "caller": "unit"}
    provider._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0, jitter=False)
    bodies, validation_counts = [], []

    def handle(wire):
        bodies.append(wire.content)
        validation_counts.append(len(witness.calls))
        return (httpx.Response(503, json={"error": {"code": "server_error"}})
                if len(bodies) == 1 else httpx.Response(200, text=completed()))

    install_transport(monkeypatch, handle)
    await provider.complete(request, request_options=options)
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert validation_counts[1] > validation_counts[0]
    assert witness.calls and all(call == options for call in witness.calls)


@pytest.mark.asyncio
async def test_refs_without_actual_witness_fail_before_auth_or_wire(provider):
    request = request_for(source())
    request.messages[0].metadata = {"experimental.media.v1": {"refs": []}}
    with pytest.raises(llm_errors.InvalidRequestError):
        await provider.complete(request)
    provider._ensure_valid_tokens.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant_content", ["string", "blocks"])
async def test_real_loop_tool_key_two_http_continuation(
    provider, monkeypatch, assistant_content,
):
    """Exact basic/streaming/events serialized call key, no media capability."""
    sent = []
    arguments = {"path": "fixture"}

    def handle(wire):
        body = json.loads(wire.content)
        sent.append(body)
        if len(sent) == 1:
            events = [
                {"type": "response.output_item.done", "item": {
                    "type": "function_call", "call_id": "loop-call", "name": "read_file",
                    "arguments": json.dumps(arguments),
                }},
                {"type": "response.completed", "response": {
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }},
            ]
            return httpx.Response(200, text="".join(
                "data: " + json.dumps(event) + "\n\n" for event in events
            ))
        calls = [item for item in body["input"] if item.get("type") == "function_call"]
        assert calls == [{
            "type": "function_call", "call_id": "loop-call", "name": "read_file",
            "arguments": json.dumps(arguments),
        }]
        assert body["input"][-1] == {
            "type": "function_call_output", "call_id": "loop-call",
            "output": "existing tool result",
        }
        return httpx.Response(200, text=completed("continued"))

    install_transport(monkeypatch, handle)
    request = ChatRequest(messages=[Message(role="user", content="read fixture")])
    first = await provider.complete(request)
    call = first.tool_calls[0]
    # These are the actual loop dictionary fields, not ToolCall.model_dump().
    row = Message.model_validate({
        "role": "assistant",
        "content": "" if assistant_content == "string" else first.content,
        "tool_calls": [{"id": call.id, "tool": call.name, "arguments": call.arguments}],
    })
    request.messages.extend([
        row,
        Message(role="tool", tool_call_id=call.id, content=[TextBlock(text="existing tool result")]),
    ])
    original = copy.deepcopy(request.model_dump())
    facts = provider.get_media_request_support_for_request(request=request, model="gpt-6.1-sol")
    if provider.auth_mode == "chatgpt_codex":
        assert {"message_index": 2, "role": "tool"} in facts["positions"]
    else:
        assert facts["positions"] == []
    second = await provider.complete(request)
    assert second.content[0].text == "continued"
    assert len(sent) == 2 and request.model_dump() == original
    assert row.tool_calls == [{"id": "loop-call", "tool": "read_file", "arguments": arguments}]


@pytest.mark.parametrize("fields", [
    {"tool": 5}, {"tool": ""}, {"tool": None},
    {"name": "", "tool": "read_file"},
    {"name": None, "tool": "read_file"},
    {"name": 5, "tool": "read_file"},
])
def test_loop_tool_alias_does_not_override_invalid_explicit_name(provider, fields):
    request = ChatRequest(messages=[Message.model_validate({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "loop-call", "arguments": {}, **fields}],
    })])
    before = request.model_dump()
    with pytest.raises(llm_errors.InvalidRequestError) as error:
        provider._build_payload(request)
    assert error.value.error_code == "invalid_tool_call" and error.value.retryable is False
    if provider.auth_mode == "chatgpt_codex":
        with pytest.raises(llm_errors.InvalidRequestError):
            provider.get_media_request_support_for_request(request=request, model="gpt-6.1-sol")
    assert request.model_dump() == before


def test_explicit_name_wins_over_loop_tool_alias_without_extra_equality_conflict(provider):
    request = ChatRequest(messages=[Message.model_validate({
        "role": "assistant",
        "content": [ToolCallBlock(id="c", name="explicit", input={})],
        "tool_calls": [{"id": "c", "name": "explicit", "tool": "ignored", "arguments": {}}],
    })])
    payload = provider._build_payload(request)
    assert [item for item in payload["input"] if item.get("type") == "function_call"] == [{
        "type": "function_call", "call_id": "c", "name": "explicit", "arguments": "{}",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["same", "different", "missing-selected-file", "known-memory-account"])
async def test_cold_media_guard_pins_selected_file_not_unknown_wildcard(
    provider, monkeypatch, tmp_path, change,
):
    from datetime import datetime, timedelta, timezone
    import time
    from amplifier_module_provider_openai_chatgpt import plan_auth

    selected = {
        "auth_mode": provider.auth_mode, "access_token": "offline-cold-unit",
        **(
            {"subject": "selected", "client_id": "selected-client",
             "scopes": [plan_auth.PLAN_SCOPE], "expires_at": time.time() + 3600}
            if provider.auth_mode == "chatgpt_plan" else
            {"account_id": "selected",
             "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
        ),
    }
    account_key = "subject" if provider.auth_mode == "chatgpt_plan" else "account_id"
    auth_path = tmp_path / "synthetic-cold.json"
    if change != "missing-selected-file":
        auth_path.write_text(json.dumps(selected))
    provider._token_file_path = str(auth_path)
    provider._tokens = selected.copy() if change == "known-memory-account" else None
    provider._tokens_injected = False
    request, witness = witness_request(provider)
    sends = []
    real_ensure = ChatGPTProvider._ensure_valid_tokens.__get__(provider)

    async def ensure():
        if change in {"different", "known-memory-account"}:
            changed = {**selected, account_key: "different"}
            # Simulate normal loading/refresh resolving another identity; the
            # guard must reject it, not accept any account after a cold None.
            provider._tokens = changed
        elif change == "missing-selected-file":
            provider._tokens = selected.copy()
        else:
            await real_ensure()

    provider._ensure_valid_tokens = ensure

    def handle(wire):
        sends.append(wire.content)
        return httpx.Response(200, text=completed())

    install_transport(monkeypatch, handle)
    if change == "same":
        await provider.complete(request)
        assert len(sends) == 1 and witness.calls
    else:
        with pytest.raises(llm_errors.InvalidRequestError):
            await provider.complete(request)
        assert sends == []