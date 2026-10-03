"""Amplifier ChatGPT subscription auth provider module.

Uses the public Responses API for explicit ChatGPT plan sign-in, preserving
the legacy Codex-compatible OAuth/device transport for existing configurations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .oauth import is_token_valid, load_tokens, login
from .provider import ChatGPTProvider
from . import plan_auth

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

    With login disabled, registers the provider without loading or refreshing
    credentials. Authentication remains observable and is resolved on use.
    Otherwise loads OAuth tokens, optionally initiating login when missing.
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

    from .provider import (
        _coerce_bool, _warn_unknown_config_keys, normalize_auth_mode, validate_auth_paths,
    )

    _warn_unknown_config_keys(config)
    mode = normalize_auth_mode(config.get("auth_mode"))
    validate_auth_paths(config)
    token_file_path: str | None = config.get("token_file_path")
    login_on_mount: bool = _coerce_bool(
        config.get("login_on_mount"), key="login_on_mount", default=True
    )

    if not login_on_mount:
        # Availability is separate from the provider protocol. Neither OAuth
        # mode may read/refresh credentials or open consent during passive
        # preparation; auth_status and request-time authentication own that work.
        provider = ChatGPTProvider(config, coordinator)
        await coordinator.mount("providers", provider, name="openai-chatgpt")
        return provider.close

    if mode == plan_auth.MODE:
        if token_file_path is None:
            token_file_path = plan_auth.DEFAULT_TOKEN_FILE
        try:
            tokens = await plan_auth.ensure_tokens(token_file_path)
        except plan_auth.PlanAuthError as exc:
            # Interactive consent belongs to explicit account setup, never to a
            # worker mounting providers in the background.
            logger.warning("ChatGPT plan connection unavailable: %s", exc)
            return None
        provider = ChatGPTProvider(config, coordinator, tokens)
        await coordinator.mount("providers", provider, name="openai-chatgpt")
        return provider.close

    # Load legacy tokens only; never send plan credentials to Codex endpoints.
    tokens = load_tokens(token_file_path)
    if tokens and tokens.get("auth_mode") == plan_auth.MODE:
        raise ValueError("ChatGPT plan credentials require auth_mode=chatgpt_plan")

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
