"""Current fallback/payload behavior without authentication or network effects."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS, to_model_infos
from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-luna", "gpt-6.1-sol-fast"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_current_model_and_effort_reach_payload(model, effort):
    provider = ChatGPTProvider(config={"default_model": model})
    request = ChatRequest(messages=[Message(role="user", content="Hello")],
                          reasoning_effort=effort)
    payload = provider._build_payload(request)
    assert payload["model"] == model.removesuffix("-fast")
    assert payload["reasoning"]["effort"] == effort


def test_fallback_keeps_legacy_explicit_models_and_current_conservative_limits():
    infos = {m.id: m for m in to_model_infos(FALLBACK_MODELS)}
    assert FALLBACK_MODELS[0]["slug"] == "gpt-6.1-sol"
    assert {"gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-terra"} <= infos.keys()
    assert infos["gpt-6.1-sol"].context_window == 272_000
    assert infos["gpt-6.1-sol"].max_output_tokens == 128_000
    assert "vision" in infos["gpt-6.1-sol"].capabilities


def test_compatibility_matrix_has_only_current_clean_models():
    path = Path(__file__).resolve().parents[1] / "routing" / "openai-chatgpt.yaml"
    matrix = yaml.safe_load(path.read_text())
    assert len(matrix["roles"]) == 13
    for role in matrix["roles"].values():
        assert len(role["candidates"]) == 1
        assert role["candidates"][0]["model"] in {"gpt-6.1-sol", "gpt-6-luna"}


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("model", [
    "gpt-6.2-astra", "gpt-6.2-sol", "gpt-6.2-astra-fast", "gpt-6.2-sol-fast",
])
@pytest.mark.parametrize("effort", ["low", "medium"])
def test_62_exact_ids_efforts_and_config_omission_sentinel(mode, model, effort):
    provider = ChatGPTProvider({
        "auth_mode": mode, "default_model": model, "reasoning_effort": "none",
    })
    original = dict(provider.config)
    payload = provider._build_payload(ChatRequest(messages=[]))
    assert "reasoning" not in payload
    payload = provider._build_payload(ChatRequest(messages=[], reasoning_effort=effort))
    assert payload["reasoning"] == {"effort": effort, "summary": "detailed"}
    assert payload["model"] == (model.removesuffix("-fast") if mode == "chatgpt_codex" else model)
    assert payload.get("service_tier") == (
        "priority" if mode == "chatgpt_codex" and model.endswith("-fast") else None
    )
    assert payload["store"] is False and payload["stream"] is True
    assert provider.config == original
    assert not any(key in payload for key in ("temperature", "top_p", "top_logprobs", "include"))


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("extra,param", [
    ({"reasoning": {"effort": "none"}}, "reasoning.effort=none"),
    ({"reasoning": "none"}, "reasoning.effort=none"),
    ({"temperature": 0}, "temperature"),
    ({"top_p": 1}, "top_p"),
    ({"top_logprobs": 0}, "top_logprobs"),
    ({"logprobs": False}, "logprobs"),
    ({"include": ["message.output_text.logprobs"]}, "include.message.output_text.logprobs"),
])
def test_62_final_payload_rejects_incompatible_extra_fields(mode, extra, param):
    provider = ChatGPTProvider({
        "auth_mode": mode, "default_model": "gpt-6-astra",
        # Validate the FINAL wire model and parameters, not just initial config.
        "extra_request_params": {"model": "gpt-6.2-astra", **extra},
    })
    with pytest.raises(llm_errors.InvalidRequestError) as caught:
        provider._build_payload(ChatRequest(messages=[], reasoning_effort="low"))
    assert caught.value.error_param == param
    assert caught.value.retryable is False


@pytest.mark.parametrize("mode", ["chatgpt_codex", "chatgpt_plan"])
@pytest.mark.parametrize("model", ["gpt-6.2-astra", "gpt-6.2-sol-fast"])
def test_62_explicit_request_none_is_not_silently_omitted(mode, model):
    provider = ChatGPTProvider({"auth_mode": mode, "default_model": model})
    with pytest.raises(llm_errors.InvalidRequestError, match="reasoning.effort=none"):
        provider._build_payload(ChatRequest(messages=[], reasoning_effort="none"))


def test_62_native_sampling_preserves_each_channel_contract():
    request = ChatRequest(messages=[], temperature=0.3, top_p=0.9)
    codex = ChatGPTProvider({"default_model": "gpt-6.2-sol"})
    payload = codex._build_payload(request)
    assert "temperature" not in payload and "top_p" not in payload
    plan = ChatGPTProvider({"auth_mode": "chatgpt_plan", "default_model": "gpt-6.2-sol"})
    with pytest.raises(llm_errors.InvalidRequestError):
        plan._build_payload(request)


def test_62_gate_does_not_restrict_old_pins_or_unrelated_includes_and_history():
    extra = {
        "reasoning": {"effort": "none"},
        "include": ["reasoning.encrypted_content"],
        "input": [
            {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ],
    }
    for mode in ("chatgpt_codex", "chatgpt_plan"):
        provider = ChatGPTProvider({
            "auth_mode": mode, "default_model": "gpt-6-astra",
            "extra_request_params": deepcopy(extra),
        })
        assert provider._build_payload(ChatRequest(messages=[]))["reasoning"]["effort"] == "none"
        # Extra input remains ordered and unmodified, including opaque replay.
        provider.default_model = "gpt-6.2-astra"
        del provider.extra_request_params["reasoning"]
        payload = provider._build_payload(ChatRequest(messages=[]))
        assert payload["input"] == extra["input"]
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert extra["reasoning"]["effort"] == "none"