"""Narrow, lossless native image input; no file/network reads or resizing."""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any

from amplifier_core import llm_errors

# Public Responses image guide, not an independently established Codex ceiling.
PLAN_MAX_PAYLOAD_BYTES = 512_000_000
PLAN_MAX_IMAGES = 1500
_FORMATS = {"image/png", "image/jpeg"}
_DATA_URL = re.compile(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=]+")
# Exact first-party input_modalities facts, not entitlement or plan discovery.
IMAGE_MODEL_EVIDENCE = "https://github.com/openai/codex/blob/65a5410f72dac1f4b40862c34ef0a8b71b12834a/codex-rs/models-manager/models.json"
CODEX_IMAGE_MODELS = frozenset({
    "gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-sol", "gpt-6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
    "gpt-daybreak-blue-latest", "gpt-daybreak-red-latest", "codex-auto-review",
})


def unsupported_image(explanation: str) -> None:
    error = llm_errors.InvalidRequestError(
        "ChatGPT image input: " + explanation,
        provider="openai-chatgpt",
        retryable=False,
    )
    error.error_code = "unsupported_image_input"
    error.error_param = "messages.content.image"
    raise error


def image_part(source: Any, role: str = "user") -> dict[str, str]:
    """Check base64 and format framing, not full raster validity; never decode pixels."""
    if role != "user" and role != "tool":
        unsupported_image("images are supported only in user or tool-result content.")
    if not isinstance(source, dict) or source.get("type") != "base64":
        unsupported_image(
            "use a base64 PNG/JPEG source; URL, file and path sources are not supported. "
            "The provider will not fetch or upload them."
        )
    if set(source) - {"type", "media_type", "data"}:
        unsupported_image("base64 sources accept only type, media_type and data.")
    media_type = source.get("media_type")
    if not isinstance(media_type, str) or media_type not in _FORMATS:
        unsupported_image(
            "only image/png and image/jpeg are supported by this adapter. "
            "Explicitly convert GIF/WebP/other formats before attaching; originals are never changed."
        )
    data = source.get("data")
    if not isinstance(data, str) or not data:
        unsupported_image("base64 data must be a nonempty string.")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        unsupported_image("invalid base64 data; attach an encoded PNG/JPEG image.")
    valid = (
        len(decoded) >= 45
        and decoded.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        and decoded.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
        if media_type == "image/png"
        else len(decoded) >= 4
        and decoded.startswith(b"\xff\xd8\xff")
        and decoded.endswith(b"\xff\xd9")
    )
    if not valid:
        unsupported_image(
            "declared media type does not match PNG/JPEG framing, or image is truncated. "
            "Re-export explicitly; the provider does not decode or repair pixels."
        )
    return {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}


def has_tool_images(output: Any) -> bool:
    return isinstance(output, list) and any(
        isinstance(item, dict) and item.get("type") == "image" for item in output
    )


def tool_output(output: Any) -> Any:
    """Promote only an explicit image-bearing Core content list, never JSON text."""
    if not has_tool_images(output):
        return output if isinstance(output, str) else json.dumps(output)
    parts = []
    for item in output:
        if not isinstance(item, dict):
            unsupported_image(
                "tool image content must be a list of text/image block objects."
            )
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            if set(item) - {"type", "text"}:
                unsupported_image("tool text blocks accept only type and text.")
            parts.append({"type": "input_text", "text": item["text"]})
        elif item.get("type") == "image" and set(item) <= {"type", "source"}:
            parts.append(image_part(item.get("source"), role="tool"))
        else:
            unsupported_image(
                "tool image content supports only explicit text/image blocks."
            )
    return parts


def reject_override_images(items: Any) -> None:
    """Reject image carriers in the raw input escape hatch, not arbitrary JSON.

    Typed request conversion owns validation/position policy. Inspect only
    message content and tool-result output; never parse argument/text strings
    or walk metadata, function arguments, or other application dictionaries.
    """
    for item in items if isinstance(items, (list, tuple)) else [items]:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind is not None and not isinstance(kind, str):
            unsupported_image("extra_request_params.input item types must be strings.")
        if kind in {"image", "input_image", "image_url", "computer_screenshot"}:
            unsupported_image(
                "images in extra_request_params.input are unsupported; use typed message images."
            )
        if kind in {"function_call_output", "tool_result", "computer_call_output"}:
            reject_override_images(item.get("output"))
        elif kind == "message" or (
            isinstance(item.get("role"), str) and item["role"] in {
            "user", "tool", "assistant", "system", "developer", "function",
        }):
            reject_override_images(item.get("content"))


def image_values(value: Any) -> tuple[str, ...]:
    """Collect only native image values for private, invocation-local redaction."""
    found: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "input_image":
                url = obj.get("image_url")
                if isinstance(url, str):
                    found.add(url)
                    if ";base64," in url:
                        found.add(url.split(";base64,", 1)[1])
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(value)
    return tuple(sorted(found, key=len, reverse=True))


def safe_image_text(value: str, known: tuple[str, ...] = ()) -> str:
    for secret in known:
        value = value.replace(secret, "[IMAGE REDACTED]")
    return _DATA_URL.sub("[IMAGE REDACTED]", value)


def diagnostic_copy(value: Any, known: tuple[str, ...] = ()) -> Any:
    if isinstance(value, str):
        return safe_image_text(value, known)
    if isinstance(value, dict):
        if known and value.get("type") == "function_call":
            return {
                key: ("[IMAGE ECHO NOT RETAINED]" if key == "arguments"
                      else diagnostic_copy(child, known))
                for key, child in value.items()
            }
        if known and isinstance(value.get("type"), str) and value["type"].endswith(".delta"):
            # Redundant fragments can reconstruct exact image values across
            # records. Actual model/parser events remain untouched.
            return {key: ("[IMAGE ECHO NOT RETAINED]" if key in {"delta", "arguments"}
                          else diagnostic_copy(child, known))
                    for key, child in value.items()}
        return {key: diagnostic_copy(child, known) for key, child in value.items()}
    if isinstance(value, list):
        return [diagnostic_copy(child, known) for child in value]
    return value


class ImageEchoStream:
    """Bounded lookbehind for known image prefixes, including split echoes.

    A 64-character matching prefix is enough to omit a diagnostic echo. Only
    display copies change; actual parsed model output is untouched.
    """

    def __init__(self, known: tuple[str, ...]):
        self.known = known
        self.pending = ""
        self.active: list[str] = []
        self.offset = 0

    def feed(self, value: str, *, final: bool = False) -> str:
        if not self.known:
            return value
        text = self.pending + value
        self.pending = ""
        result = []
        while text:
            if self.active:
                size = min(len(text), min(len(s) - self.offset for s in self.active))
                matched = [s for s in self.active
                           if text[:size] == s[self.offset:self.offset + size]]
                if matched:
                    text = text[size:]
                    self.offset += size
                    self.active = [s for s in matched if self.offset < len(s)]
                    continue
                # A shared prefix may diverge mid-chunk. Keep every candidate
                # until its first mismatching character, not one arbitrary image.
                common = 0
                for secret in self.active:
                    length = 0
                    for left, right in zip(text[:size], secret[self.offset:self.offset + size]):
                        if left != right:
                            break
                        length += 1
                    common = max(common, length)
                if common:
                    prefix = text[:common]
                    self.active = [s for s in self.active
                                   if s[self.offset:self.offset + common] == prefix]
                    self.offset += common
                    text = text[common:]
                    continue
                # Prefix was an image-like echo but then diverged. Do not
                # retain the matched transport prefix in diagnostics.
                self.active = []
            matches = [(text.find(secret[:64]), secret) for secret in self.known]
            matches = [(index, secret) for index, secret in matches if index >= 0]
            if matches:
                index, secret = min(matches, key=lambda pair: pair[0])
                result.extend([text[:index], "[IMAGE REDACTED]"])
                consumed = min(64, len(secret))
                text = text[index + consumed:]
                self.active = [s for _, s in matches
                               if s[:consumed] == secret[:consumed] and len(s) > consumed]
                self.offset = consumed
                continue
            hold = 0
            if not final:
                for secret in self.known:
                    prefix = secret[:64]
                    for size in range(min(len(text), len(prefix) - 1), hold, -1):
                        if text.endswith(prefix[:size]):
                            hold = size
                            break
            if hold:
                text, self.pending = text[:-hold], text[-hold:]
            result.append(text)
            break
        return "".join(result)


def safe_error_body(body: bytes, known: tuple[str, ...]) -> bytes:
    """Decode JSON escapes before redaction; retain non-JSON error classification."""
    if not known:
        return body
    text = body.decode(errors="replace")
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return safe_image_text(text, known).encode()
    return json.dumps(diagnostic_copy(value, known)).encode()


def validate_image_budget(payload: dict[str, Any], *, plan: bool) -> None:
    """Plan-only documented total payload/image limits; no invented Codex cap."""
    if not plan:
        return
    count = 0

    def walk(obj: Any) -> None:
        nonlocal count
        if isinstance(obj, dict):
            if obj.get("type") == "input_image" and isinstance(
                obj.get("image_url"), str
            ):
                count += 1
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(payload)
    if count > PLAN_MAX_IMAGES:
        unsupported_image("plan request exceeds the documented 1,500-image limit.")
    if not count:
        return
    # Match HTTPX JSON encoding, including non-image fields. Iteration avoids an
    # additional full serialized-request allocation merely to check its length.
    encoder = json.JSONEncoder(
        ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    size = sum(len(chunk.encode("utf-8")) for chunk in encoder.iterencode(payload))
    if size > PLAN_MAX_PAYLOAD_BYTES:
        unsupported_image(
            "plan request exceeds the documented 512 MB total payload limit. "
            "Use explicit host history/attachment recovery; nothing was removed or resized."
        )
