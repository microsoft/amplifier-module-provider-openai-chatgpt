import base64
import hashlib
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock

from amplifier_module_provider_openai_chatgpt.artwork import CodexArtwork
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider


@pytest.fixture
def artwork(tmp_path):
    provider = ChatGPTProvider(
        {"auth_mode": "chatgpt_codex", "default_model": "gpt-6.2-sol"},
        tokens={
            "account_id": "fixture",
            "access_token": "fixture",
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    )
    return CodexArtwork(provider, tmp_path)


def png(color):
    out = io.BytesIO()
    Image.new("RGB", (512, 256), color).save(out, "PNG")
    return out.getvalue()


def install_stream(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handler), **kw),
    )


def events(*items):
    return httpx.Response(
        200, text="".join("data: " + json.dumps(item) + "\n\n" for item in items)
    )


def generated(data):
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "image_generation_call",
            "result": base64.b64encode(data).decode(),
        },
    }


@pytest.mark.asyncio
async def test_generate_inspect_and_reference_edit(artwork, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    original, revised = png("teal"), png("orange")
    sends = []

    def send(req):
        assert req.url.host == "chatgpt.com"
        body = json.loads(req.content)
        sends.append(body)
        return events(
            generated(original if len(sends) == 1 else revised),
            {"type": "response.completed"},
        )

    install_stream(monkeypatch, send)
    result = await artwork.execute({"prompt": "draw"})
    receipt = result.output
    assert receipt["artifact_id"] == hashlib.sha256(original).hexdigest()
    assert Path(receipt["path"]).read_bytes() == original
    request = ChatRequest(
        messages=[
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="one", name=artwork.name, input={"prompt": "draw"})
                ],
            ),
            Message(
                role="tool", tool_call_id="one", content=result.get_serialized_output()
            ),
        ]
    )
    restored = artwork.restore_recent_preview(request)
    output = artwork.provider._build_payload(restored)["input"][-1]["output"]
    preview = base64.b64decode(output[1]["image_url"].split(",")[1])
    assert Image.open(io.BytesIO(preview)).size == (384, 192)
    assert request.messages[-1].content == result.get_serialized_output()
    if hasattr(result, "content"):
        assert result.content[1].source["data"] == base64.b64encode(preview).decode()
    # CLI canonicalizes object ordering and puts the receipt in a successful Core envelope.
    message = request.messages[-1]
    message.content = json.dumps(
        {"output": receipt, "success": True, "error": None}, sort_keys=True
    )
    assert artwork.restore_recent_preview(request) is not request
    message.content = json.dumps(
        {"output": receipt, "success": False, "error": "denied"}
    )
    assert artwork.restore_recent_preview(request) is request
    message.content = json.dumps({**receipt, "width": 999})
    assert artwork.restore_recent_preview(request) is request
    message.content = result.get_serialized_output()
    assert (
        artwork.artifact_for_preview(receipt["preview_sha256"], {"prompt": "draw"}, 0)
        == receipt["path"]
    )
    assert (
        artwork.artifact_for_preview(receipt["preview_sha256"], {"prompt": "edit"}, 0)
        is None
    )
    assert (
        artwork.artifact_for_preview(
            receipt["preview_sha256"], {"prompt": "draw"}, artwork.generation
        )
        is None
    )
    artwork.provider._tokens["account_id"] = "different"
    assert artwork.restore_recent_preview(request) is request
    artwork.provider._tokens["account_id"] = "fixture"
    result = await artwork.execute(
        {"prompt": "revise", "reference": receipt["artifact_id"]}
    )
    assert Path(result.output["path"]).read_bytes() == revised
    request.messages.extend(
        [
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="two",
                        name=artwork.name,
                        input={"prompt": "revise", "reference": receipt["artifact_id"]},
                    )
                ],
            ),
            Message(
                role="tool", tool_call_id="two", content=result.get_serialized_output()
            ),
        ]
    )
    comparison = artwork.provider._build_payload(
        artwork.restore_recent_preview(request)
    )
    outputs = [
        item["output"]
        for item in comparison["input"]
        if item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 2 and all(
        parts[1]["type"] == "input_image" for parts in outputs
    )
    request.messages.append(Message(role="user", content="A new turn"))
    assert artwork.restore_recent_preview(request) is request
    assert len(sends) == 2
    assert sends[1]["input"][0]["content"][1]["image_url"].endswith(
        base64.b64encode(original).decode()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["response.incomplete", "error", None])
async def test_interrupted_operation_never_replays_or_publishes(
    artwork, monkeypatch, terminal
):
    sends = []

    def send(req):
        sends.append(1)
        return events(
            generated(png("blue")), *([{"type": terminal}] if terminal else [])
        )

    install_stream(monkeypatch, send)
    with pytest.raises(RuntimeError, match="completed image"):
        await artwork.execute({"prompt": "draw"})
    assert sends == [1] and not list(artwork.directory.iterdir())


@pytest.mark.asyncio
async def test_rejects_untrusted_reference_before_submission(artwork):
    with pytest.raises(ValueError, match="ID"):
        await artwork.execute({"prompt": "edit", "reference": "../elsewhere.png"})
    identity = "a" * 64
    (artwork.directory / (identity + ".png")).write_bytes(png("red"))
    with pytest.raises(ValueError, match="receipt"):
        await artwork.execute({"prompt": "edit", "reference": identity})


@pytest.mark.asyncio
async def test_final_response_can_carry_the_image(artwork, monkeypatch):
    data = png("indigo")
    item = generated(data)["item"]
    install_stream(
        monkeypatch,
        lambda req: events(
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [item]},
            }
        ),
    )
    result = await artwork.execute({"prompt": "draw"})
    assert Path(result.output["path"]).read_bytes() == data


@pytest.mark.asyncio
async def test_conflicting_terminal_image_does_not_publish(artwork, monkeypatch):
    first = generated(png("red"))
    second = generated(png("blue"))
    install_stream(
        monkeypatch,
        lambda req: events(
            first,
            {"type": "response.completed", "response": {"output": [second["item"]]}},
        ),
    )
    with pytest.raises(ValueError, match="Inconsistent"):
        await artwork.execute({"prompt": "draw"})
    assert not list(artwork.directory.iterdir())


@pytest.mark.asyncio
async def test_workspace_reference_bytes_and_output_options(artwork, monkeypatch):
    references = [png("orange"), png("green")]
    calls = []

    def send(request):
        calls.append(json.loads(request.content))
        return events(generated(png("white")), {"type": "response.completed"})

    install_stream(monkeypatch, send)
    with pytest.raises(ValueError, match="quality"):
        await artwork.generate("edit", references, quality="unknown")
    receipt, parts = await artwork.generate(
        "edit", references, size="1536x1024", quality="medium", background="transparent"
    )
    (body,) = calls
    assert body["tools"][0] == {
        "type": "image_generation",
        "output_format": "png",
        "size": "1536x1024",
        "quality": "medium",
        "background": "transparent",
    }
    images = body["input"][0]["content"][1:]
    assert [
        base64.b64decode(p["image_url"].split(",")[1]) for p in images
    ] == references
    assert receipt["artifact_id"] == hashlib.sha256(png("white")).hexdigest()
    assert (
        hashlib.sha256(base64.b64decode(parts[1]["source"]["data"])).hexdigest()
        == receipt["preview_sha256"]
    )
