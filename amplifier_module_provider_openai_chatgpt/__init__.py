"""Amplifier ChatGPT subscription auth provider module.

Uses the public Responses API for explicit ChatGPT plan sign-in, preserving
the legacy Codex-compatible OAuth/device transport for existing configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .provider import ChatGPTProvider

if TYPE_CHECKING:
    from amplifier_core import Coordinator

__amplifier_module_type__ = "provider"
__all__ = ["mount", "ChatGPTProvider"]


async def mount(
    coordinator: Coordinator,
    config: dict[str, Any] | None = None,
) -> Callable[[], Coroutine[Any, Any, None]] | None:
    """Register provider code without performing account authentication.

    Loading/validating a module must work offline, without credentials, and
    without browser consent. list_models/complete own automatic token refresh;
    explicit login() owns interactive sign-in. login_on_mount is retained as a
    deprecated configuration key but cannot trigger authentication here.
    """
    config = config or {}
    from .provider import _warn_unknown_config_keys, normalize_auth_mode, validate_auth_paths

    _warn_unknown_config_keys(config)
    normalize_auth_mode(config.get("auth_mode"))
    validate_auth_paths(config)
    # Do not read the token file here either: an unavailable account file must
    # not turn package validation into an account-readiness check.
    provider = ChatGPTProvider(config, coordinator)
    await coordinator.mount("providers", provider, name="openai-chatgpt")
    return provider.close
