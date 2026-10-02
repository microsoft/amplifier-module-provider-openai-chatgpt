"""Current fallback/payload behavior without authentication or network effects."""

from pathlib import Path

import pytest
import yaml
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
    assert "vision" not in infos["gpt-6.1-sol"].capabilities


def test_compatibility_matrix_has_only_current_clean_models():
    path = Path(__file__).resolve().parents[1] / "routing" / "openai-chatgpt.yaml"
    matrix = yaml.safe_load(path.read_text())
    assert len(matrix["roles"]) == 13
    for role in matrix["roles"].values():
        assert len(role["candidates"]) == 1
        assert role["candidates"][0]["model"] in {"gpt-6.1-sol", "gpt-6-luna"}