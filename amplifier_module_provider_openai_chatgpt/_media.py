"""Codex-owned image positions, private witness and diagnostic copies.

No stores, hydration, Foundation imports, arbitrary JSON-string parsing, or
public metadata witnesses. Plan mode deliberately remains text/function only.
"""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import Any

from amplifier_core.llm_errors import InvalidRequestError

NAMESPACE = "experimental.media.v1"
NOT_RETAINED = "[Image transport not retained; no retrievable pixel reference]"
UNAVAILABLE = "[Image unavailable: selected route/model/position is not qualified; no pixels supplied]"
# First-party Codex input_modalities catalogue, immutable source retrieved
# 2026-11-14. This is payload support, not account availability/entitlement.
EVIDENCE = "https://github.com/openai/codex/blob/65a5410f72dac1f4b40862c34ef0a8b71b12834a/codex-rs/models-manager/models.json"
VISION_MODELS = frozenset(
    {
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-daybreak-blue-latest",
        "gpt-daybreak-red-latest",
        "codex-auto-review",
    }
)


def mapping(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def capability(coordinator: Any) -> Any:
    getter = getattr(coordinator, "get_capability", None)
    value = getter(NAMESPACE) if callable(getter) else None
    return value if getattr(value, "version", None) == 1 else None


def has_refs(request: Any) -> bool:
    return any(
        isinstance(getattr(m, "metadata", None), dict) and NAMESPACE in m.metadata
        for m in request.messages
    )


def has_images(content: Any) -> bool:
    if not isinstance(content, (list, tuple)):
        return False
    for item in content:
        block = mapping(item)
        if isinstance(block, dict):
            if block.get("type") == "image":
                return True
            if block.get("type") == "tool_result" and has_images(
                block.get("output", block.get("content"))
            ):
                return True
    return False


def protected(coordinator: Any, request: Any) -> bool:
    return (
        capability(coordinator) is not None
        or has_refs(request)
        or any(has_images(m.content) for m in request.messages)
    )


def validate(coordinator: Any, request: Any, options: dict[str, Any]) -> None:
    cap = capability(coordinator)
    if cap is None:
        if has_refs(request):
            raise InvalidRequestError(
                "Session image references require private preparation; request not sent.",
                provider="openai-chatgpt",
                retryable=False,
            )
        return
    try:
        result = cap.validate(
            request, purpose="transport", request_options=deepcopy(options)
        )
    except Exception:
        raise InvalidRequestError(
            "Private media projection changed, expired or was revoked; prepare and fit a new request. Request not sent.",
            provider="openai-chatgpt",
            retryable=False,
        ) from None
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise InvalidRequestError(
            "Private media validation refused; request not sent.",
            provider="openai-chatgpt",
            retryable=False,
        )


def image_part(block: dict[str, Any], *, allowed: bool) -> dict[str, Any]:
    if not allowed:
        return {"type": "input_text", "text": UNAVAILABLE}
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return {"type": "input_text", "text": UNAVAILABLE}
    mime, data = source.get("media_type"), source.get("data")
    if mime not in ("image/png", "image/jpeg") or not isinstance(data, str) or not data:
        raise InvalidRequestError(
            "Image requires verified PNG/JPEG base64 transport.",
            provider="openai-chatgpt",
            retryable=False,
        )
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, TypeError):
        raise InvalidRequestError(
            "Image base64 transport is malformed.",
            provider="openai-chatgpt",
            retryable=False,
        ) from None
    return {"type": "input_image", "image_url": f"data:{mime};base64,{data}"}


def tool_output(content: Any, *, allowed: bool) -> Any:
    if not has_images(content):
        return content if isinstance(content, str) else json.dumps(content)
    result = []
    for item in content:
        block = mapping(item)
        if isinstance(block, dict) and block.get("type") == "image":
            result.append(image_part(block, allowed=allowed))
        elif isinstance(block, dict) and block.get("type") == "text":
            result.append({"type": "input_text", "text": block.get("text", "")})
        else:
            raise InvalidRequestError(
                "Image-bearing tool output requires structured text/image parts.",
                provider="openai-chatgpt",
                retryable=False,
            )
    if not allowed:
        return "\n".join(part["text"] for part in result)
    return result


def diagnostic_copy(value: Any) -> Any:
    """Known wire/canonical schema omission; opaque strings remain opaque."""
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(v) for v in value]
    value = mapping(value)
    if not isinstance(value, dict):
        return deepcopy(value)
    kind = value.get("type")
    if isinstance(kind, str) and kind in {
        "image",
        "input_image",
        "image_url",
        "computer_screenshot",
    }:
        return {"type": "input_text", "text": NOT_RETAINED}
    copied = deepcopy(value)
    # These are schema-owned envelopes, not arbitrary arguments/metadata walks.
    for key in (
        "input",
        "output",
        "content",
        "messages",
        "events",
        "response",
        "item",
        "extra_body",
    ):
        if key in copied and isinstance(copied[key], (dict, list, tuple)):
            copied[key] = diagnostic_copy(copied[key])
    return copied


def transport_values(payload: Any) -> tuple[str, ...]:
    found = []

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for v in value:
                visit(v)
        elif isinstance(value, dict):
            kind = value.get("type")
            if isinstance(kind, str) and kind in {
                "input_image",
                "image_url",
                "computer_screenshot",
            }:
                url = value.get("image_url", value.get("url"))
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url:
                    found.append(url)
                    if ";base64," in url:
                        found.append(url.split(";base64,", 1)[1])
            for key in ("input", "output", "content", "messages", "extra_body"):
                if key in value:
                    visit(value[key])

    visit(payload)
    return tuple(sorted(set(found), key=len, reverse=True))


def safe_text(text: str, values: tuple[str, ...]) -> str:
    for value in values:
        text = text.replace(value, NOT_RETAINED)
    return text
