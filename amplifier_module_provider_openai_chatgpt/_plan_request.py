"""Public ChatGPT plan HTTP request contract, separate from Codex transport."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest

# The public plan preview is a restricted Responses route. Do not silently
# discard a requested capability or send it using another account/transport.
UNSUPPORTED_FIELDS = frozenset(
    {
        "background",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "moderation",
        "multi_agent",
        "prompt",
        "prompt_cache_retention",
        "safety_identifier",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "previous_response_id",
    }
)


def _unsupported(param: str, explanation: str = "") -> None:
    error = llm_errors.InvalidRequestError(
        f"ChatGPT plan mode does not support {param}. "
        + (
            explanation
            or "Remove this setting for this connection; the provider will not change connection modes."
        ),
        provider="openai-chatgpt",
        retryable=False,
    )
    error.error_code = "unsupported_plan_parameter"
    error.error_param = param
    raise error


def _function_tools(tools: Any, param: str) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        _unsupported(param, "Expected a list of function definitions.")
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            _unsupported(
                param,
                "This provider exposes local functions through additional_tools input items.",
            )
        if tool.get("defer_loading"):
            _unsupported(
                param, "Deferred tool_search is unavailable in ChatGPT plan mode."
            )
    return tools


def prepare_plan_payload(
    payload: dict[str, Any], request: ChatRequest
) -> dict[str, Any]:
    """Validate the preview surface and make local function tools available.

    additional_tools preserves the kernel's flat function names and call IDs.
    It precedes the history on every request, so tool-result continuations keep
    their definition available at the same point without namespace remapping.
    """
    payload = deepcopy(payload)
    for message in request.messages:
        if isinstance(message.content, list):
            for block in message.content:
                if block.type not in {"text", "thinking", "tool_call", "tool_result"}:
                    _unsupported(
                        f"messages.content.{block.type}",
                        "This provider's plan adapter currently supports text and function-call history; this content cannot be silently omitted.",
                    )
    for param in sorted(UNSUPPORTED_FIELDS):
        # ChatRequest.metadata is local provider/event control (for example,
        # stream=False suppresses UI events). It is never wire metadata.
        # An actual metadata field in the built payload remains unsupported.
        if param in payload or (
            param != "metadata" and getattr(request, param, None) is not None
        ):
            _unsupported(param)
    for param in ("conversation_id", "stop", "response_format"):
        if getattr(request, param, None) is not None:
            _unsupported(param)

    items = payload.get("input")
    if not isinstance(items, list):
        _unsupported("input", "Send complete history as an input array.")
    for item in items:
        if not isinstance(item, dict):
            _unsupported("input", "Each history item must be an object.")
        if item.get("type") == "additional_tools":
            if item.get("role") != "developer":
                _unsupported(
                    "input.additional_tools.role",
                    "Use the developer role for additional_tools.",
                )
            _function_tools(item.get("tools"), "input.additional_tools.tools")
        if item.get("role") == "system":
            # The first system message is instructions. Later system messages
            # retain their order/content as developer messages on this route.
            item["role"] = "developer"

    tools = payload.pop("tools", None)
    if tools is not None:
        if not isinstance(tools, list):
            _unsupported("tools", "Expected a list of tool definitions.")
        functions, hosted = [], []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "web_search":
                hosted.append(tool)
            else:
                functions.extend(_function_tools([tool], "tools"))
        if functions:
            payload["input"] = [
                {"type": "additional_tools", "role": "developer", "tools": functions},
                *items,
            ]
        if hosted:
            payload["tools"] = hosted

    # The route requires these regardless of the caller's streaming UI choice.
    payload["store"] = False
    payload["stream"] = True
    return payload
