"""SSE parser — parse data lines from ChatGPT backend SSE streams into typed events.

The ChatGPT backend returns Server-Sent Events (SSE) even for non-streaming
requests. This module accumulates those events into a ParsedResponse, handling
text output, function calls, metadata, usage statistics, and error events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["SSEError", "ParsedResponse", "parse_sse_events"]

_ERROR_EVENT_TYPES = frozenset({"error", "response.failed", "response.incomplete"})


class SSEError(Exception):
    """Raised when the SSE stream contains an error, response.failed, or
    response.incomplete event inside an otherwise successful HTTP 200 response.

    Attributes:
        message:    Human-readable error description.
        code:       Machine-readable error code (may be None).
        event_type: The SSE event type that triggered the error.
    """

    def __init__(self, message: str, code: str | None, event_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.event_type = event_type


@dataclass
class ParsedResponse:
    """Accumulated result of an SSE stream from the ChatGPT backend."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    response_id: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw_events: list[dict] = field(default_factory=list)


def parse_sse_events(lines: list[str], collect_raw: bool = False) -> ParsedResponse:
    """Parse a list of raw SSE lines into a ParsedResponse.

    Args:
        lines:       Raw SSE lines (as returned by an HTTP response body iterator).
        collect_raw: When True, populate ``ParsedResponse.raw_events`` with every
                     successfully parsed JSON event.  Defaults to False to avoid
                     the memory overhead in normal production usage.

    Returns:
        ParsedResponse with accumulated content, tool calls, and metadata.

    Raises:
        SSEError: If the stream contains an error, response.failed, or
                  response.incomplete event.
    """
    result = ParsedResponse()

    for line in lines:
        # Only process data lines.
        if not line.startswith("data: "):
            continue

        data_str = line[6:]  # strip "data: " prefix

        # The [DONE] sentinel signals end of stream.
        if data_str == "[DONE]":
            break

        # Skip malformed JSON gracefully.
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # SSE records may contain valid JSON values that are not events.
        # Ignore those values rather than assuming a mapping below.
        if not isinstance(event, dict):
            continue

        if collect_raw:
            result.raw_events.append(event)

        raw_event_type = event.get("type", "")
        event_type = raw_event_type if isinstance(raw_event_type, str) else ""

        # ------------------------------------------------------------------
        # Error detection — raise immediately for error events.
        # ------------------------------------------------------------------
        if event_type in _ERROR_EVENT_TYPES:
            _raise_sse_error(event, event_type)

        # ------------------------------------------------------------------
        # Metadata extraction.
        # ------------------------------------------------------------------
        if event_type in ("response.created", "response.done"):
            resp = event.get("response")
            if not isinstance(resp, dict):
                resp = {}
            if not result.response_id:
                result.response_id = resp.get("id", "")
            if not result.model:
                result.model = resp.get("model", "")

        # ------------------------------------------------------------------
        # Usage extraction from response.done.
        # ------------------------------------------------------------------
        if event_type == "response.done":
            response = event.get("response")
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            if usage:
                result.input_tokens = usage.get("input_tokens", 0)
                result.output_tokens = usage.get("output_tokens", 0)

        # ------------------------------------------------------------------
        # Content accumulation from response.output_item.done (canonical).
        # ------------------------------------------------------------------
        if event_type == "response.output_item.done":
            item = event.get("item", {})
            item_type = item.get("type")

            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") in ("output_text", "text"):
                        result.content += part.get("text", "")

            elif item_type == "function_call":
                result.tool_calls.append(
                    {
                        "id": item.get("call_id") or item.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }
                )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _raise_sse_error(event: dict, event_type: str) -> None:
    """Extract error details from *event* and raise an :exc:`SSEError`."""
    if event_type == "error":
        error_obj = event.get("error")
    else:
        # response.failed / response.incomplete — error nested under "response"
        response = event.get("response")
        error_obj = response.get("error") if isinstance(response, dict) else None

    if isinstance(error_obj, str):
        message: str = error_obj
        code: str | None = None
    elif isinstance(error_obj, dict):
        raw_message = error_obj.get("message")
        message = (
            raw_message
            if isinstance(raw_message, str)
            else f"ChatGPT SSE {event_type} event"
        )
        raw_code = error_obj.get("code")
        code = raw_code if isinstance(raw_code, str) else None
    else:
        message = f"ChatGPT SSE {event_type} event"
        code = None

    raise SSEError(message=message, code=code, event_type=event_type)
