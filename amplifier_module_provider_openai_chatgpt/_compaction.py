"""Opt-in checkpoint codec. This module contains no history-selection policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from amplifier_core.message_models import Message, Usage

METADATA_KEY = "openai-chatgpt:compaction"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class CompactionCheckpoint:
    """One opaque checkpoint, never a canonical replacement history or count.

    The caller owns retained history, admission, persistence and installation.
    Sensitive state is excluded from the object's repr.
    """

    checkpoint: Message = field(repr=False)
    usage: Usage | None
    raw_usage: dict[str, Any] = field(repr=False)
    response_id: str
    terminal_event: str
    continuation_kind: str = "checkpoint_requires_retained_history"
    canonical_window: bool = False
    next_request_input_tokens: None = None


def validate_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or item.get("type") != "compaction":
        raise ValueError("Expected a compaction item")
    if (
        not isinstance(item.get("encrypted_content"), str)
        or not item["encrypted_content"]
    ):
        raise ValueError("Compaction item has no nonempty encrypted_content")
    # All other fields are vendor-owned, including absent/null IDs and future fields.
    return deepcopy(item)


def checkpoint_message(item: dict, provenance: dict[str, str]) -> Message:
    return Message(
        role="assistant",
        content="",
        metadata={
            METADATA_KEY: {
                "version": FORMAT_VERSION,
                "provenance": deepcopy(provenance),
                "item": validate_item(item),
            }
        },
    )


def replay_item(message: Message, provenance: dict[str, str]) -> dict | None:
    metadata = message.metadata or {}
    if METADATA_KEY not in metadata:
        return None
    envelope = metadata[METADATA_KEY]
    if (
        not isinstance(envelope, dict)
        or type(envelope.get("version")) is not int
        or envelope["version"] != FORMAT_VERSION
    ):
        raise ValueError("Unsupported ChatGPT compaction metadata version")
    if envelope.get("provenance") != provenance:
        raise ValueError(
            "ChatGPT checkpoint provider, route, account or model mismatch"
        )
    if message.role != "assistant" or message.content not in ("", []):
        raise ValueError(
            "ChatGPT checkpoint must be a dedicated empty assistant message"
        )
    return validate_item(envelope.get("item"))


def redact_opaque(value: Any) -> Any:
    """Keep raw diagnostic hooks useful without emitting encrypted state."""
    if isinstance(value, dict):
        return {
            k: "[redacted]" if k == "encrypted_content" else redact_opaque(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_opaque(v) for v in value]
    return value
