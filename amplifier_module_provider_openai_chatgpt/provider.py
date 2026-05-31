"""ChatGPT subscription provider for Amplifier.

Implements the Amplifier Provider Protocol using raw httpx + manual SSE
against the ChatGPT backend API with OAuth authentication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import httpx

from amplifier_core import ModelInfo, ProviderInfo
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)
from amplifier_core.utils import redact_secrets

from ._sse import ParsedResponse, SSEError, parse_sse_events
from .models import (
    DEFAULT_CACHE_TTL_SECONDS,
    FALLBACK_MODELS,
    fetch_models,
    to_model_infos,
)
from .oauth import (
    CHATGPT_CODEX_BASE_URL,
    is_token_valid,
    load_tokens,
    refresh_tokens,
)

logger = logging.getLogger(__name__)

# Full endpoint for the ChatGPT Responses API
CHATGPT_CODEX_ENDPOINT = CHATGPT_CODEX_BASE_URL + "/responses"


# ---------------------------------------------------------------------------
# GPT-5.5-pro effort validator
# ---------------------------------------------------------------------------

_GPT_5_5_PRO_ALLOWED_EFFORTS = frozenset({"medium", "high", "xhigh"})


def _validate_gpt_5_5_pro_effort(model_id: str, reasoning_param: Any) -> None:
    """Pre-flight: reject effort below 'medium' for gpt-5.5-pro models."""
    if not model_id.startswith("gpt-5.5-pro"):
        return
    if reasoning_param is None:
        return
    # Handle both string ("low") and dict ({"effort": "low"}) forms
    if isinstance(reasoning_param, dict):
        effort = reasoning_param.get("effort")
    else:
        effort = reasoning_param
    if effort is None or effort in _GPT_5_5_PRO_ALLOWED_EFFORTS:
        return
    raise kernel_errors.InvalidRequestError(
        f"gpt-5.5-pro requires reasoning effort of 'medium' or above, "
        f"got '{effort}'. Allowed values: {['medium', 'high', 'xhigh']}",
        provider="openai-chatgpt",
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class ChatGPTProvider:
    """Amplifier provider for ChatGPT subscription API (OAuth-authenticated)."""

    name = "openai-chatgpt"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: Any = None,
        tokens: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        self._coordinator = coordinator
        self._tokens = tokens

        self.priority: int = int(self._config.get("priority", 100))
        self.raw: bool = bool(self._config.get("raw", False))
        self.default_model: str = self._config.get("default_model", "gpt-5.5")
        self.timeout: float = float(self._config.get("timeout", 300.0))
        self._token_file_path: str | None = self._config.get("token_file_path")

        # Model catalog cache: (monotonic_timestamp, models) or None when empty.
        self._models_cache_ttl: float = float(
            self._config.get("models_cache_ttl", DEFAULT_CACHE_TTL_SECONDS)
        )
        self._models_cache: tuple[float, list[ModelInfo]] | None = None
        self._models_lock = asyncio.Lock()

        # No persistent client — httpx.AsyncClient is created per-request in complete().
        # This is intentional: token refresh may change headers between calls.

    # ------------------------------------------------------------------
    # Provider Protocol
    # ------------------------------------------------------------------

    def get_info(self) -> ProviderInfo:
        """Return provider metadata."""
        return ProviderInfo(
            id="openai-chatgpt",
            display_name="OpenAI ChatGPT",
            capabilities=["streaming", "tools", "reasoning"],
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return ModelInfo objects for all available ChatGPT models.

        Delegates to :meth:`_get_catalog`, which fetches the live model
        catalog from the API and caches it for :attr:`_models_cache_ttl`
        seconds.  Falls back to a built-in list on any error.
        """
        return await self._get_catalog()

    async def _get_catalog(self) -> list[ModelInfo]:
        """Fetch and cache the live model catalog.

        Fast path (no lock): if a non-expired cache entry exists, return it.
        Slow path (under lock, with double-check): call
        :func:`~.models.fetch_models`, convert via
        :func:`~.models.to_model_infos`, and store in cache.

        On *any* exception (network error, auth failure, parse error, …)
        the fallback catalog is returned and the cache is **not** updated,
        so the next call will retry the live fetch.

        Returns:
            List of :class:`~amplifier_core.ModelInfo` objects.
        """
        now = time.monotonic()

        # Fast path: return cached catalog if still within TTL.
        if self._models_cache is not None:
            cached_at, models = self._models_cache
            if now - cached_at < self._models_cache_ttl:
                return models

        async with self._models_lock:
            # Double-check under lock to avoid redundant fetches.
            now = time.monotonic()
            if self._models_cache is not None:
                cached_at, models = self._models_cache
                if now - cached_at < self._models_cache_ttl:
                    return models

            try:
                await self._ensure_valid_tokens()
                entries = await fetch_models(
                    access_token=self._tokens["access_token"],  # type: ignore[index]
                    account_id=self._tokens["account_id"],  # type: ignore[index]
                )
                if not entries:
                    raise ValueError("Live model catalog returned 0 usable entries")
                models = to_model_infos(entries)
                self._models_cache = (time.monotonic(), models)
                return models
            except Exception as exc:
                logger.warning(
                    "Failed to fetch live model catalog, using fallback: %s",
                    exc,
                    exc_info=True,
                )
                # Do not cache the fallback — next call should retry.
                return to_model_infos(FALLBACK_MODELS)

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from a ChatResponse.

        Args:
            response: Typed chat response containing tool_calls.

        Returns:
            List of ToolCall objects from the response, or [] if none present.
        """
        if not response.tool_calls:
            return []
        return list(response.tool_calls)

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _convert_content(
        self, content: str | list[Any], role: str = "user"
    ) -> list[dict[str, Any]]:
        """Convert Amplifier message content to Responses API content format.

        - str → [{type: input_text|output_text, text}]
        - TextBlock → {type: input_text|output_text, text}
        - ThinkingBlock → {type: input_text|output_text, text: block.thinking}
        - Other block types (ToolCallBlock, ToolResultBlock) are skipped here
          and handled directly in _build_payload.

        Uses ``output_text`` when role is ``"assistant"``; ``input_text`` otherwise.
        """
        text_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, str):
            return [{"type": text_type, "text": content}]

        result: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextBlock):
                result.append({"type": text_type, "text": block.text})
            elif isinstance(block, ThinkingBlock):
                result.append({"type": text_type, "text": block.thinking})
            # ToolCallBlock and ToolResultBlock are handled in _build_payload
        return result

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Build Responses API payload from an Amplifier ChatRequest.

        Key rules enforced:
        - Uses ``input`` array (not ``messages``)
        - First system/developer message → top-level ``instructions``
        - ``stream: True`` and ``store: False`` are mandatory
        - Rejected params (max_output_tokens, temperature, truncation,
          parallel_tool_calls, include) are never included
        - ``-fast`` model suffix → strip suffix + ``service_tier: 'priority'``
        - request.model overrides default_model
        - Tools → {type, name, description, parameters} + tool_choice: 'auto'
        - ToolResultBlock → {type: function_call_output, call_id, output}
        - ToolCallBlock → {type: function_call, call_id, name, arguments}
        - Reasoning effort → {reasoning: {effort, summary: 'detailed'}}
        """
        # Resolve model (request overrides provider default)
        model: str = request.model or self.default_model

        # Handle -fast suffix → priority service tier
        service_tier: str | None = None
        if model.endswith("-fast"):
            model = model.removesuffix("-fast")
            service_tier = "priority"

        # Pre-flight: validate effort level for gpt-5.5-pro models
        _validate_gpt_5_5_pro_effort(model, request.reasoning_effort)

        # Build input array and extract instructions from system/developer message
        instructions: str | None = None
        input_items: list[dict[str, Any]] = []

        for message in request.messages:
            # First system/developer message becomes top-level instructions
            if message.role in ("system", "developer") and instructions is None:
                if isinstance(message.content, str):
                    instructions = message.content
                else:
                    texts = [
                        block.text
                        for block in message.content
                        if isinstance(block, TextBlock)
                    ]
                    instructions = " ".join(texts) if texts else ""
                continue  # Do not add to input array

            if message.role == "assistant":
                if isinstance(message.content, list):
                    # Split mixed content: text blocks go in role message,
                    # tool call blocks become standalone function_call items
                    text_parts: list[dict[str, Any]] = []
                    for block in message.content:
                        if isinstance(block, ToolCallBlock):
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": block.id,
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                }
                            )
                        elif isinstance(block, TextBlock):
                            text_parts.append(
                                {"type": "output_text", "text": block.text}
                            )
                        elif isinstance(block, ThinkingBlock):
                            text_parts.append(
                                {"type": "output_text", "text": block.thinking}
                            )
                    if text_parts:
                        input_items.append({"role": "assistant", "content": text_parts})
                else:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": self._convert_content(
                                message.content, role="assistant"
                            ),
                        }
                    )

            elif message.role == "tool":
                # Tool result messages → standalone function_call_output items
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            output = block.output
                            if not isinstance(output, str):
                                output = json.dumps(output)
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": block.tool_call_id,
                                    "output": output,
                                }
                            )
                else:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": getattr(message, "tool_call_id", "unknown"),
                            "output": str(message.content),
                        }
                    )

            else:
                # user, developer (additional after first), function
                input_items.append(
                    {
                        "role": message.role,
                        "content": self._convert_content(message.content),
                    }
                )

        # Assemble base payload (no rejected params)
        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
            "store": False,
        }

        # ChatGPT backend requires instructions even when no system message is present.
        payload["instructions"] = instructions or ""

        if service_tier is not None:
            payload["service_tier"] = service_tier

        # Tools → Responses API format
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"

        # Reasoning effort
        if request.reasoning_effort:
            payload["reasoning"] = {
                "effort": request.reasoning_effort,
                "summary": "detailed",
            }

        return payload

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Return the HTTP headers required for the ChatGPT Codex API.

        Reads ``access_token`` and ``account_id`` from ``self._tokens``.

        Returns:
            Dict with six required headers.

        Raises:
            kernel_errors.AuthenticationError: If ``access_token`` is absent
                or ``_tokens`` is None.
            kernel_errors.AuthenticationError: If ``account_id`` is absent.
        """
        if not self._tokens or not self._tokens.get("access_token"):
            raise kernel_errors.AuthenticationError(
                "No valid OAuth tokens available",
                provider=self.name,
                status_code=401,
                retryable=False,
            )

        account_id = self._tokens.get("account_id")
        if not account_id:
            raise kernel_errors.AuthenticationError(
                "No account_id in tokens — cannot build request headers",
                provider=self.name,
                status_code=401,
                retryable=False,
            )

        return {
            "Authorization": f"Bearer {self._tokens['access_token']}",
            "ChatGPT-Account-Id": account_id,
            "OpenAI-Beta": "responses=v1",
            "OpenAI-Originator": "codex",
            "Content-Type": "application/json",
            "accept": "text/event-stream",
        }

    async def close(self) -> None:
        """No-op — httpx clients are created per-request in complete()."""

    @staticmethod
    def _is_cloudflare_challenge(headers: httpx.Headers, body: bytes) -> bool:
        """Detect a Cloudflare browser-challenge HTML page in a 403 response.

        Args:
            headers: Response headers from the HTTP response.
            body: Raw response body bytes.

        Returns:
            True if the response looks like a Cloudflare challenge page.
        """
        # Primary signal: HTML content-type header.
        ct = headers.get("content-type", "").lower()
        if "text/html" in ct:
            return True

        # Fallback: scan body for known Cloudflare challenge markers.
        body_lower = body.decode(errors="replace").lower()
        cf_markers = (
            "just a moment",
            "cf-browser-verification",
            "checking if the site connection is secure",
        )
        return any(marker in body_lower for marker in cf_markers)

    @staticmethod
    def _raise_for_status(
        status: int,
        headers: httpx.Headers,
        body: bytes,
        provider_name: str,
    ) -> None:
        """Map a non-200, non-401 HTTP response to the correct kernel error.

        401 handling is intentionally excluded: it lives in the retry loop in
        ``complete()`` because it needs to control retry flow.

        Always raises; never returns normally.

        Args:
            status: HTTP status code.
            headers: Response headers.
            body: Raw response body bytes.
            provider_name: Provider name string for error context.

        Raises:
            kernel_errors.RateLimitError: 429.
            kernel_errors.ContextLengthError: 400 with context-length keywords.
            kernel_errors.ContentFilterError: 400 with content-filter keywords.
            kernel_errors.InvalidRequestError: 400 without special keywords.
            kernel_errors.ProviderUnavailableError: 403 Cloudflare challenge or 5xx.
            kernel_errors.AccessDeniedError: 403 non-Cloudflare.
            kernel_errors.NotFoundError: 404.
            kernel_errors.LLMError: Any other unexpected status code.
        """
        body_text = body.decode(errors="replace")

        if status == 429:
            retry_after: float | None = None
            ra_header = headers.get("retry-after")
            if ra_header is not None:
                try:
                    retry_after = float(ra_header)
                except ValueError:
                    pass
            raise kernel_errors.RateLimitError(
                f"ChatGPT API rate limit exceeded ({status}): {body_text}",
                provider=provider_name,
                status_code=status,
                retryable=True,
                retry_after=retry_after,
            )
        elif status == 400:
            body_lower = body_text.lower()
            if any(
                kw in body_lower
                for kw in (
                    "context length",
                    "too many tokens",
                    "maximum context",
                )
            ):
                raise kernel_errors.ContextLengthError(
                    f"ChatGPT API context length exceeded ({status}): {body_text}",
                    provider=provider_name,
                    status_code=status,
                    retryable=False,
                )
            elif any(
                kw in body_lower
                for kw in (
                    "content filter",
                    "safety",
                    "blocked",
                )
            ):
                raise kernel_errors.ContentFilterError(
                    f"ChatGPT API content filtered ({status}): {body_text}",
                    provider=provider_name,
                    status_code=status,
                    retryable=False,
                )
            else:
                raise kernel_errors.InvalidRequestError(
                    f"ChatGPT API invalid request ({status}): {body_text}",
                    provider=provider_name,
                    status_code=status,
                    retryable=False,
                )
        elif status == 403:
            if ChatGPTProvider._is_cloudflare_challenge(headers, body):
                raise kernel_errors.ProviderUnavailableError(
                    f"ChatGPT API blocked by Cloudflare challenge ({status})",
                    provider=provider_name,
                    status_code=status,
                    retryable=True,
                )
            else:
                raise kernel_errors.AccessDeniedError(
                    f"ChatGPT API access denied ({status}): {body_text}",
                    provider=provider_name,
                    status_code=status,
                    retryable=False,
                )
        elif status == 404:
            raise kernel_errors.NotFoundError(
                f"ChatGPT API endpoint not found ({status}): {body_text}",
                provider=provider_name,
                status_code=status,
                retryable=False,
            )
        elif status >= 500:
            raise kernel_errors.ProviderUnavailableError(
                f"ChatGPT API server error ({status}): {body_text}",
                provider=provider_name,
                status_code=status,
                retryable=True,
            )
        else:
            raise kernel_errors.LLMError(
                f"ChatGPT API error ({status}): {body_text}",
                provider=provider_name,
                status_code=status,
                retryable=False,
            )

    async def _ensure_valid_tokens(self) -> None:
        """Guarantee ``self._tokens`` holds a valid, unexpired access token.

        Resolution order:
        1. In-memory tokens pass ``is_token_valid()`` → done.
        2. Tokens loaded from disk pass ``is_token_valid()`` → update in-memory, done.
        3. Refresh using in-memory ``refresh_token`` → update in-memory, done.
        4. Refresh using disk ``refresh_token`` → update in-memory, done.
        5. None of the above succeeded → raise ``AuthenticationError``.

        Raises:
            kernel_errors.AuthenticationError: If no valid tokens can be
                obtained by any means.
        """
        # 1. In-memory tokens still valid.
        if is_token_valid(self._tokens):
            return

        # 2. Fresh load from disk.
        disk_tokens = load_tokens(path=self._token_file_path)
        if is_token_valid(disk_tokens):
            self._tokens = disk_tokens
            return

        # 3. Refresh using in-memory refresh_token.
        if self._tokens and self._tokens.get("refresh_token"):
            refreshed = await refresh_tokens(
                self._tokens["refresh_token"], path=self._token_file_path
            )
            if refreshed:
                self._tokens = refreshed
                return

        # 4. Refresh using disk refresh_token (different from in-memory).
        if disk_tokens and disk_tokens.get("refresh_token"):
            refreshed = await refresh_tokens(
                disk_tokens["refresh_token"], path=self._token_file_path
            )
            if refreshed:
                self._tokens = refreshed
                return

        raise kernel_errors.AuthenticationError(
            "No valid OAuth tokens available — please run the login flow again",
            provider=self.name,
            retryable=False,
        )

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Send a completion request to the ChatGPT Responses API.

        Flow:
        1. Ensure valid OAuth tokens.
        2. Build request payload.
        3. Emit ``llm:request`` event (once — NOT re-emitted on retry).
        4. POST to CHATGPT_CODEX_ENDPOINT with httpx streaming, collect SSE lines.
           On 401, attempt one token refresh and retry before propagating the error.
        5. Check HTTP status — map to kernel_errors subtypes.
        6. Parse SSE events.
        7. Emit ``llm:response`` event with usage and timing.
        8. Return ChatResponse via ``_to_chat_response()``.

        On any exception an ``llm:response`` event with ``status='error'`` is
        emitted before re-raising.  All exceptions that escape this method are
        :class:`~amplifier_core.llm_errors.LLMError` subtypes.
        """
        # 1. Ensure valid OAuth tokens.
        await self._ensure_valid_tokens()

        # 2. Build request payload.
        payload = self._build_payload(request)

        # Resolve effective model name (mirrors _build_payload logic) for events.
        model: str = request.model or self.default_model
        if model.endswith("-fast"):
            model = model.removesuffix("-fast")

        headers = self._build_headers()

        # 3. Emit llm:request event (NOT re-emitted on retry).
        _has_hooks = self._coordinator and hasattr(self._coordinator, "hooks")
        if _has_hooks:
            req_event: dict[str, Any] = {
                "provider": self.name,
                "model": model,
                "message_count": len(request.messages),
            }
            if self.raw:
                req_event["raw"] = redact_secrets(payload)
            await self._coordinator.hooks.emit("llm:request", req_event)

        start_time = time.monotonic()

        # Per-request streaming override (does NOT mutate config-level setting).
        # Callers pass metadata={"stream": False} to suppress llm:stream_* events.
        _use_streaming: bool = bool(self._config.get("use_streaming", True))
        _meta = getattr(request, "metadata", None)
        if isinstance(_meta, dict) and _meta.get("stream") is False:
            _use_streaming = False

        # Emit stream events only when coordinator is present AND streaming is on.
        emit_stream_events: bool = bool(_has_hooks and _use_streaming)

        # Streaming state — stable across the 401-retry loop.
        # request_id is generated once; seq/block_types reset per attempt (fresh stream).
        request_id: str = str(uuid.uuid4())
        seq: dict[int, int] = {}
        block_types: dict[int, str] = {}
        any_emitted: bool = False          # True once any delta/thinking event is emitted
        stream_aborted_emitted: bool = False

        # Local guard variable — concurrency-safe (no instance-level mutation).
        retry_attempted = False

        try:
            # 4. Two-attempt loop: first attempt + one optional retry on 401.
            lines: list[str] = []
            for _attempt in range(2):
                lines = []
                # Reset per-attempt stream state (fresh SSE stream from server).
                seq = {}
                block_types = {}
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        async with client.stream(
                            "POST",
                            CHATGPT_CODEX_ENDPOINT,
                            json=payload,
                            headers=headers,
                        ) as resp:
                            # 5. Check HTTP status — map to kernel error types.
                            if resp.status_code != 200:
                                error_body = await resp.aread()
                                status = resp.status_code

                                if status == 401 and not retry_attempted:
                                    # Mid-session expiry: refresh once, rebuild
                                    # headers, and retry.  The llm:request hook
                                    # is NOT re-emitted.
                                    retry_attempted = True
                                    await self._ensure_valid_tokens()
                                    headers = self._build_headers()
                                    continue  # re-enter the for loop
                                elif status == 401:
                                    body_text = error_body.decode(errors="replace")
                                    raise kernel_errors.AuthenticationError(
                                        f"ChatGPT API authentication failed ({status}): {body_text}",
                                        provider=self.name,
                                        status_code=status,
                                        retryable=False,
                                    )
                                else:
                                    # All other non-2xx codes — delegate to the
                                    # status-dispatch method (429, 400, 403, 404,
                                    # 5xx, other).
                                    self._raise_for_status(
                                        status, resp.headers, error_body, self.name
                                    )

                            # Dual-pass SSE loop: emit contract events inline (pass 1)
                            # and collect all lines for parse_sse_events below (pass 2).
                            # Each data: line is JSON-parsed twice — acceptable overhead.
                            async for line in resp.aiter_lines():
                                lines.append(line)

                                if not emit_stream_events:
                                    continue

                                if not line.startswith("data: "):
                                    continue

                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break

                                try:
                                    event = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                et = event.get("type", "")

                                if et == "response.output_item.added":
                                    idx: int = event.get("output_index", 0)
                                    item = event.get("item", {})
                                    item_type = item.get("type", "")
                                    block_type = {
                                        "message": "text",
                                        "reasoning": "thinking",
                                        "function_call": "tool_use",
                                    }.get(item_type, "text")
                                    block_types[idx] = block_type
                                    seq[idx] = 0
                                    bs_payload: dict[str, Any] = {
                                        "request_id": request_id,
                                        "block_index": idx,
                                        "block_type": block_type,
                                    }
                                    if block_type == "tool_use":
                                        tc_name = item.get("name")
                                        if tc_name:
                                            bs_payload["name"] = tc_name
                                    await self._coordinator.hooks.emit(
                                        "llm:stream_block_start", bs_payload
                                    )

                                elif et == "response.output_text.delta":
                                    delta_text = event.get("delta", "")
                                    if delta_text:
                                        idx = event.get("output_index", 0)
                                        await self._coordinator.hooks.emit(
                                            "llm:stream_block_delta",
                                            {
                                                "request_id": request_id,
                                                "block_index": idx,
                                                "block_type": "text",
                                                "sequence": seq.get(idx, 0),
                                                "text": delta_text,
                                            },
                                        )
                                        seq[idx] = seq.get(idx, 0) + 1
                                        any_emitted = True

                                elif et in (
                                    "response.reasoning_summary_text.delta",
                                    "response.reasoning_text.delta",
                                ):
                                    delta_text = event.get("delta", "")
                                    if delta_text:
                                        idx = event.get("output_index", 0)
                                        await self._coordinator.hooks.emit(
                                            "llm:stream_block_delta",
                                            {
                                                "request_id": request_id,
                                                "block_index": idx,
                                                "block_type": "thinking",
                                                "sequence": seq.get(idx, 0),
                                                "text": delta_text,
                                            },
                                        )
                                        seq[idx] = seq.get(idx, 0) + 1
                                        any_emitted = True

                                elif et == "response.output_item.done":
                                    idx = event.get("output_index", 0)
                                    if idx in block_types:
                                        await self._coordinator.hooks.emit(
                                            "llm:stream_block_end",
                                            {
                                                "request_id": request_id,
                                                "block_index": idx,
                                                "block_type": block_types[idx],
                                            },
                                        )

                                elif et == "error":
                                    # Emit stream_aborted now if we already sent deltas.
                                    # parse_sse_events will raise SSEError after the loop.
                                    if any_emitted and not stream_aborted_emitted:
                                        error_obj = event.get("error", {})
                                        err_msg = (
                                            error_obj.get("message", str(event))
                                            if isinstance(error_obj, dict)
                                            else str(error_obj)
                                        )
                                        await self._coordinator.hooks.emit(
                                            "llm:stream_aborted",
                                            {
                                                "request_id": request_id,
                                                "error": {
                                                    "type": "error",
                                                    "msg": err_msg,
                                                },
                                            },
                                        )
                                        stream_aborted_emitted = True

                                # response.function_call_arguments.delta: silently consumed

                    break  # request succeeded — exit the retry loop

                except kernel_errors.AuthenticationError:
                    # Belt-and-suspenders: catch any AuthenticationError that
                    # escaped the status-code block (e.g. _build_headers()
                    # raising during the mid-loop 401 recovery path). By the
                    # time we reach this handler retry_attempted is always
                    # True, so we simply propagate.
                    raise

            # 6. Parse SSE events.
            parsed = parse_sse_events(lines, collect_raw=self.raw)

            duration_ms = (time.monotonic() - start_time) * 1000

            # 7. Emit llm:response event (success).
            if _has_hooks:
                resp_event: dict[str, Any] = {
                    "provider": self.name,
                    "model": model,
                    "usage": {
                        "input_tokens": parsed.input_tokens,
                        "output_tokens": parsed.output_tokens,
                    },
                    "status": "ok",
                    "duration_ms": duration_ms,
                }
                if self.raw:
                    resp_event["raw"] = redact_secrets({"events": parsed.raw_events})
                await self._coordinator.hooks.emit("llm:response", resp_event)

            # 8. Return ChatResponse.
            return self._to_chat_response(parsed, model)

        except kernel_errors.LLMError as exc:
            # Already a typed kernel error — emit hook and re-raise unchanged.
            duration_ms = (time.monotonic() - start_time) * 1000
            if _has_hooks:
                await self._coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "status": "error",
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise

        except SSEError as exc:
            # Map SSE-layer errors to kernel types before escaping complete().
            # If a partial stream was emitted before the SSE error, signal abort.
            if any_emitted and _has_hooks and not stream_aborted_emitted:
                await self._coordinator.hooks.emit(
                    "llm:stream_aborted",
                    {
                        "request_id": request_id,
                        "error": {
                            "type": type(exc).__name__,
                            "msg": str(exc),
                        },
                    },
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            code = exc.code or ""
            msg = str(exc).lower()

            if "rate_limit" in code:
                mapped_exc: kernel_errors.LLMError = kernel_errors.RateLimitError(
                    str(exc), provider=self.name, retryable=True
                )
            elif any(kw in msg for kw in ("context length", "too many tokens")):
                mapped_exc = kernel_errors.ContextLengthError(
                    str(exc), provider=self.name, retryable=False
                )
            elif any(kw in msg for kw in ("content filter", "safety", "blocked")):
                mapped_exc = kernel_errors.ContentFilterError(
                    str(exc), provider=self.name, retryable=False
                )
            else:
                mapped_exc = kernel_errors.LLMError(
                    str(exc), provider=self.name, retryable=False
                )

            if _has_hooks:
                await self._coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "status": "error",
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise mapped_exc from exc

        except httpx.TimeoutException as exc:
            # Timeout → LLMTimeoutError (retryable).
            # If a partial stream was emitted before the timeout, signal abort.
            if any_emitted and _has_hooks and not stream_aborted_emitted:
                await self._coordinator.hooks.emit(
                    "llm:stream_aborted",
                    {
                        "request_id": request_id,
                        "error": {
                            "type": type(exc).__name__,
                            "msg": str(exc),
                        },
                    },
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            mapped_timeout = kernel_errors.LLMTimeoutError(
                f"ChatGPT API request timed out: {exc}",
                provider=self.name,
                retryable=True,
            )
            if _has_hooks:
                await self._coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "status": "error",
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise mapped_timeout from exc

        except httpx.TransportError as exc:
            # Transport / connection errors → ProviderUnavailableError (retryable).
            # Covers ConnectError, RemoteProtocolError, and other transport
            # failures. Note: TimeoutException is a separate hierarchy caught above.
            # If a partial stream was emitted before the transport error, signal abort.
            if any_emitted and _has_hooks and not stream_aborted_emitted:
                await self._coordinator.hooks.emit(
                    "llm:stream_aborted",
                    {
                        "request_id": request_id,
                        "error": {
                            "type": type(exc).__name__,
                            "msg": str(exc),
                        },
                    },
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            mapped_unavail = kernel_errors.ProviderUnavailableError(
                f"ChatGPT API connection error: {exc}",
                provider=self.name,
                retryable=True,
            )
            if _has_hooks:
                await self._coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "status": "error",
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise mapped_unavail from exc

        except Exception as exc:
            # Catch-all: wrap in LLMError so callers always receive a typed error.
            # If a partial stream was emitted, signal abort before translating error.
            if any_emitted and _has_hooks and not stream_aborted_emitted:
                await self._coordinator.hooks.emit(
                    "llm:stream_aborted",
                    {
                        "request_id": request_id,
                        "error": {
                            "type": type(exc).__name__,
                            "msg": str(exc),
                        },
                    },
                )
            duration_ms = (time.monotonic() - start_time) * 1000
            if _has_hooks:
                await self._coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "status": "error",
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise kernel_errors.LLMError(
                str(exc), provider=self.name, retryable=False
            ) from exc

    def _to_chat_response(self, parsed: ParsedResponse, model: str) -> ChatResponse:
        """Convert a ``ParsedResponse`` into an Amplifier ``ChatResponse``.

        Text content → ``TextBlock``.
        Tool calls → ``ToolCallBlock`` (in ``content``) + ``ToolCall`` (in
        ``tool_calls``).  JSON arguments are parsed; malformed JSON falls back
        to ``{"_raw": <original_string>}``.
        ``finish_reason`` is ``"tool_calls"`` when tool calls are present,
        otherwise ``"stop"``.
        """
        content_blocks: list[Any] = []
        tool_call_list: list[ToolCall] = []

        # Text content → TextBlock
        if parsed.content:
            content_blocks.append(TextBlock(text=parsed.content))

        # Tool calls → ToolCallBlock + ToolCall
        for tc in parsed.tool_calls:
            func = tc.get("function", {})
            name: str = func.get("name", "")
            call_id: str = tc.get("id", "")
            raw_args: str = func.get("arguments", "")

            # Parse JSON arguments; fall back to {"_raw": ...} on failure.
            try:
                arguments: dict[str, Any] = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, ValueError):
                arguments = {"_raw": raw_args}

            content_blocks.append(ToolCallBlock(id=call_id, name=name, input=arguments))
            tool_call_list.append(ToolCall(id=call_id, name=name, arguments=arguments))

        finish_reason = "tool_calls" if tool_call_list else "stop"

        usage = Usage(
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            total_tokens=parsed.input_tokens + parsed.output_tokens,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_call_list if tool_call_list else None,
            usage=usage,
            finish_reason=finish_reason,
        )
