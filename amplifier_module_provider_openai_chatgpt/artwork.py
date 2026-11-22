"""Codex subscription artwork as a Core tool result with a durable original."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import base64
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from pathlib import Path

import httpx
from amplifier_core import ToolResult
from PIL import Image

from .provider import CHATGPT_CODEX_ENDPOINT, CODEX_MODE

MAX_BYTES = 16 * 1024 * 1024


class CodexArtwork:
    name = "codex_artwork"
    description = (
        "Generate an image, or revise a previous artifact by ID. Inspect the returned "
        "visual preview before deciding how to revise it. The receipt locates the original PNG."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Art direction or revision"},
            "reference": {"type": "string", "description": "Prior artwork SHA-256 ID"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def __init__(self, provider, directory):
        self.provider = provider
        self._previews = OrderedDict()
        self._issued = OrderedDict()
        self.generation = 0
        self.directory = Path(directory).expanduser().resolve()

    def artifact_for_preview(self, digest, arguments, after):
        """Resolve a locally issued preview for a host witnessing this invocation."""
        issued = self._issued.get(digest)
        if issued and issued[0] > after and issued[1] == arguments:
            return issued[2]
        return None

    def as_agent_tool(self):
        """Register this instance through the amplifier-agent Python SDK."""
        from amplifier_agent import ImagePart, TextPart, Tool

        async def handle(arguments, context):
            _, (receipt, image) = await self._render(arguments)
            return [
                TextPart(receipt["text"]),
                ImagePart(media_type="image/png", data=image["source"]["data"]),
            ]

        return Tool(
            self.name,
            self.description,
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                **self.input_schema,
            },
            handle,
        )

    def restore_recent_preview(self, request):
        """Legacy loops retain the receipt as text. Never open paths from history."""
        from amplifier_core.message_models import Message

        messages = list(request.messages)
        changed = False
        index = len(messages)
        while index and messages[index - 1].role == "tool":
            index -= 1
        if not index or messages[index - 1].role != "assistant":
            return request
        calls = self.provider._message_tool_calls(messages[index - 1])
        inputs = {call.id: call.arguments for call in calls if call.name == self.name}
        for pos in range(index, len(messages)):
            message = messages[pos]
            if message.tool_call_id not in inputs or not isinstance(
                message.content, str
            ):
                continue
            if len(message.content) > 8192:
                continue
            try:
                receipt = json.loads(message.content)
            except (ValueError, RecursionError):
                continue
            if not isinstance(receipt, dict):
                continue
            if (
                set(receipt) == {"success", "error", "output"}
                and receipt["success"] is True
                and receipt["error"] is None
            ):
                receipt = receipt["output"]
            cached = self._previews.get(json.dumps(receipt, sort_keys=True))
            if (
                cached
                and cached[0] == (self.provider._tokens or {}).get("account_id")
                and cached[2] == inputs[message.tool_call_id]
            ):
                messages[pos] = Message(
                    role="tool", tool_call_id=message.tool_call_id, content=cached[1]
                )
                changed = True
        return request.model_copy(update={"messages": messages}) if changed else request

    async def execute(self, input):
        receipt, parts = await self._render(input)
        rich = {"content": parts} if "content" in ToolResult.model_fields else {}
        return ToolResult(output=receipt, **rich)

    async def _render(self, input):
        if self.provider.auth_mode != CODEX_MODE:
            raise ValueError("Codex subscription sign-in is required for artwork")
        prompt = input.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Supply an artwork prompt")
        parts = [{"type": "input_text", "text": prompt}]
        if "reference" in input:
            reference = input["reference"]
            if not isinstance(reference, str) or not re.fullmatch(
                "[0-9a-f]{64}", reference
            ):
                raise ValueError("Invalid artwork ID")
            with os.fdopen(
                os.open(
                    self.directory / (reference + ".png"),
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                ),
                "rb",
            ) as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Artwork reference must be a regular file")
                data = stream.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != reference:
                raise ValueError("Artwork does not match its receipt")
            with Image.open(io.BytesIO(data), formats=["PNG"]) as original:
                if original.width * original.height > 16_777_216:
                    raise ValueError("Reference image dimensions exceed the limit")
                original.verify()
            parts.append(
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(data).decode(),
                }
            )

        model = await self.provider._resolve_default_model()
        await self.provider._ensure_valid_tokens(fresh_from_file=True)
        account = self.provider._tokens.get("account_id")
        headers = self.provider._build_headers()
        body = {
            "model": model.removesuffix("-fast"),
            "store": False,
            "stream": True,
            "instructions": "Create the requested artwork.",
            "input": [{"role": "user", "content": parts}],
            "tools": [{"type": "image_generation", "output_format": "png"}],
            "tool_choice": "required",
        }
        if model.endswith("-fast"):
            body["service_tier"] = "priority"
        # One submission. A disconnected or incomplete paid operation must never replay.
        async with httpx.AsyncClient(timeout=self.provider.timeout) as client:
            async with client.stream(
                "POST",
                CHATGPT_CODEX_ENDPOINT,
                headers=headers,
                json=body,
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Codex artwork request failed (HTTP {response.status_code})"
                    )
                receipt, content = await self._receive(response.aiter_lines())
                if account:
                    key = json.dumps(receipt, sort_keys=True)
                    self._previews[key] = (account, content, dict(input))
                    if len(self._previews) > 4:
                        self._previews.popitem(last=False)
                self.generation += 1
                self._issued[receipt["preview_sha256"]] = (
                    self.generation,
                    dict(input),
                    receipt["path"],
                )
                if len(self._issued) > 16:
                    self._issued.popitem(last=False)
                return receipt, content

    async def _receive(self, lines):
        encoded = None
        async for line in lines:
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind in {"error", "response.failed", "response.incomplete"}:
                break
            if kind == "response.output_item.done":
                item = event["item"]
                if item.get("type") == "image_generation_call":
                    if encoded is not None:
                        raise ValueError("Expected one artwork result")
                    encoded = item.get("result")
            if kind == "response.completed":
                final = event.get("response", {})
                if final.get("status", "completed") != "completed":
                    break
                images = [
                    item
                    for item in final.get("output", [])
                    if item.get("type") == "image_generation_call"
                ]
                if len(images) > 1:
                    raise ValueError("Expected one artwork result")
                if images and images[0].get("result") is not None:
                    final_image = images[0]["result"]
                    if encoded is not None and encoded != final_image:
                        raise ValueError("Inconsistent final artwork")
                    encoded = final_image
                if not isinstance(encoded, str):
                    break
                if len(encoded) > 4 * ((MAX_BYTES + 2) // 3):
                    raise ValueError("Generated artwork exceeds the size limit")
                return await asyncio.to_thread(
                    self._save, base64.b64decode(encoded, validate=True)
                )
        raise RuntimeError("Codex did not confirm a completed image")

    def _save(self, data):
        if len(data) > MAX_BYTES:
            raise ValueError("Generated artwork exceeds the size limit")
        with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
            width, height = image.size
            if width * height > 16_777_216:
                raise ValueError("Generated image dimensions exceed the limit")
            image.load()
            image.thumbnail((384, 384), Image.Resampling.LANCZOS)
            preview = io.BytesIO()
            image = image.convert("RGBA")
            image.info.clear()
            image.save(preview, "PNG")
            # Leave room for base64 and the receipt in Agent's default 128 KiB result.
            if preview.tell() > 88 * 1024:
                image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                preview = io.BytesIO()
                image.quantize(colors=256).save(preview, "PNG")
        identity = hashlib.sha256(data).hexdigest()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.directory / (identity + ".png")
        temp = tempfile.NamedTemporaryFile(dir=self.directory, delete=False)
        try:
            with temp:
                temp.write(data)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp.name, path)
        finally:
            Path(temp.name).unlink(missing_ok=True)
        receipt = {
            "artifact_id": identity,
            "path": str(path),
            "mime_type": "image/png",
            "width": width,
            "height": height,
            "bytes": len(data),
            "preview_sha256": hashlib.sha256(preview.getvalue()).hexdigest(),
        }
        parts = [
            {"type": "text", "text": json.dumps(receipt)},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(preview.getvalue()).decode(),
                },
            },
        ]
        return receipt, parts
