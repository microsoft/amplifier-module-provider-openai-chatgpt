"""Amplifier ChatGPT subscription auth provider module.

Uses raw httpx + manual SSE against the ChatGPT backend API
(chatgpt.com/backend-api/codex/responses) with OAuth device code authentication.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .oauth import is_token_valid, load_tokens, login
from .provider import ChatGPTProvider

if TYPE_CHECKING:
    from amplifier_core import Coordinator

__amplifier_module_type__ = "provider"
__all__ = ["mount", "ChatGPTProvider"]

logger = logging.getLogger(__name__)


async def mount(
    coordinator: Coordinator,
    config: dict[str, Any] | None = None,
) -> Callable[[], Coroutine[Any, Any, None]] | None:
    """Mount the ChatGPT subscription provider.

    Loads OAuth tokens from disk, optionally initiating login when missing.
    On success, registers the provider with the coordinator and returns an
    async cleanup callable that closes the provider.

    Args:
        coordinator: The Amplifier module coordinator.
        config: Optional configuration dict with keys:
            - token_file_path: Path to the OAuth token file.
            - login_on_mount: If True (default), trigger login when tokens
              are absent or invalid.
            - raw: Pass raw payloads/events through provider hooks.
            - default_model: Default model name (default: 'gpt-5.5').
            - timeout: HTTP timeout in seconds (default: 300.0).

    Returns:
        Async cleanup callable on success, or None on failure.
    """
    if config is None:
        config = {}

    from .provider import _coerce_bool, _warn_unknown_config_keys

    _warn_unknown_config_keys(config)
    token_file_path: str | None = config.get("token_file_path")
    login_on_mount: bool = _coerce_bool(
        config.get("login_on_mount"), key="login_on_mount", default=True
    )

    # Load tokens from disk.
    tokens = load_tokens(token_file_path)

    # If tokens are not valid, try login when permitted.
    if not is_token_valid(tokens):
        if login_on_mount:
            try:
                tokens = await login(token_file_path=token_file_path)
            except Exception as exc:
                # Actionable, not a traceback: name the fix. exc_info is
                # deliberately omitted -- the provider is simply absent from
                # this session (see the return None below); a full traceback
                # here does not help the operator do anything differently.
                logger.warning(
                    "ChatGPT OAuth login failed during mount: %s. Run "
                    "`amplifier provider login openai-chatgpt` to authenticate, "
                    "then restart the session.",
                    exc,
                )
                return None
        else:
            return None

    # Guard: ensure tokens are valid before proceeding.
    if not is_token_valid(tokens):
        return None

    # Create and register the provider.
    provider = ChatGPTProvider(config, coordinator, tokens)
    await coordinator.mount("providers", provider, name="openai-chatgpt")

    # Return an async cleanup callable.
    async def cleanup() -> None:
        await provider.close()

    return cleanup
