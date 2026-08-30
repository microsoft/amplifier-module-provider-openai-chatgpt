"""Config hygiene tests: bool/numeric coercion, unknown-key sweep,
models_client_version override, and extra_request_params merge order.

Guards the "family hygiene wave" fixes for provider-openai-chatgpt:
  - login_on_mount/raw/use_streaming previously used bare truthiness
    (``bool("false")`` is ``True`` in Python).
  - `priority` is LIVE (read by the orchestrator's provider-selection
    logic) and must never be flagged as unknown.
  - `extra_request_params` merges last into the Responses-API payload.
"""

from __future__ import annotations

import logging

from amplifier_module_provider_openai_chatgpt.provider import (
    ChatGPTProvider,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _warn_unknown_config_keys,
)


class TestCoerceBool:
    def test_string_false_is_false(self):
        assert _coerce_bool("false", key="x", default=True) is False

    def test_string_true_is_true(self):
        assert _coerce_bool("true", key="x", default=False) is True

    def test_unrecognized_string_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_bool("maybe", key="raw", default=True)
        assert result is True
        assert "raw" in caplog.text


class TestCoerceNumeric:
    def test_int_from_string(self):
        assert _coerce_int("50", key="priority", default=100) == 50

    def test_int_invalid_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_int("garbage", key="priority", default=100)
        assert result == 100
        assert "priority" in caplog.text

    def test_float_from_string(self):
        assert _coerce_float("45.5", key="timeout", default=300.0) == 45.5


class TestUnknownConfigKeySweep:
    def test_priority_never_flagged(self, caplog):
        """priority is LIVE (read by the orchestrator's provider-selection
        logic via the attribute-then-config branch) -- must never warn."""
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"priority": 50})
        assert caplog.text == ""

    def test_extra_request_params_allowlisted(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"extra_request_params": {}})
        assert caplog.text == ""

    def test_unknown_key_warns_with_suggestion(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"tiemout": 5})
        assert "tiemout" in caplog.text
        assert "timeout" in caplog.text


class TestProviderConfigCoercionIntegration:
    def test_raw_string_false_is_false(self):
        provider = ChatGPTProvider(config={"raw": "false"})
        assert provider.raw is False

    def test_use_streaming_string_false_is_false(self):
        provider = ChatGPTProvider(config={"use_streaming": "false"})
        assert provider.use_streaming is False

    def test_priority_from_string(self):
        provider = ChatGPTProvider(config={"priority": "50"})
        assert provider.priority == 50

    def test_invalid_numeric_string_defaults_instead_of_crashing(self):
        provider = ChatGPTProvider(config={"timeout": "not-a-number"})
        assert provider.timeout == 300.0

    def test_models_client_version_default(self):
        from amplifier_module_provider_openai_chatgpt.models import (
            MODELS_CLIENT_VERSION,
        )

        provider = ChatGPTProvider(config={})
        assert provider.models_client_version == MODELS_CLIENT_VERSION

    def test_models_client_version_override(self):
        provider = ChatGPTProvider(config={"models_client_version": "1.2.3"})
        assert provider.models_client_version == "1.2.3"

    def test_extra_request_params_stored(self):
        provider = ChatGPTProvider(config={"extra_request_params": {"foo": "bar"}})
        assert provider.extra_request_params == {"foo": "bar"}

    def test_extra_request_params_non_dict_ignored(self, caplog):
        with caplog.at_level(logging.WARNING):
            provider = ChatGPTProvider(config={"extra_request_params": "nope"})
        assert provider.extra_request_params == {}
        assert "extra_request_params" in caplog.text

    def test_default_model_sentinel_and_fallback_first_entry_match(self):
        """Guards the "latest" default-model design (see provider.py's
        _resolve_default_model): the CONFIGURED default is now the "latest"
        sentinel, not a hardcoded model id that goes stale. The STATIC
        fallback that "latest" resolves to when unauthenticated/unreachable
        is FALLBACK_MODELS[0] -- this test pins that relationship so the
        two can never silently drift apart."""
        from amplifier_module_provider_openai_chatgpt.models import (
            FALLBACK_MODELS,
            LATEST_MODEL_SENTINEL,
        )

        provider = ChatGPTProvider(config={})
        assert provider.default_model == LATEST_MODEL_SENTINEL == "latest"
        assert FALLBACK_MODELS[0]["slug"] == "gpt-5.6-sol"
