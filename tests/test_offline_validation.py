"""Component qualification is independent of account readiness."""
import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest
from amplifier_core.validation import ProviderValidator
import amplifier_module_provider_openai_chatgpt as provider


@pytest.mark.asyncio
@pytest.mark.parametrize('auth_mode', ['chatgpt_codex', 'chatgpt_plan'])
@pytest.mark.parametrize('account', ['missing', 'expired'])
async def test_core_validation_does_not_authenticate(tmp_path, auth_mode, account):
    token_file = tmp_path / 'tokens.json'
    if account == 'expired':
        token_file.write_text(json.dumps({
            'access_token': 'synthetic-never-send',
            'refresh_token': 'synthetic-never-send',
            'expires_at': '2020-01-01T00:00:00+00:00',
        }))
    def forbidden(*args, **kwargs):
        raise AssertionError('Component validation must not access the network')
    with patch.object(socket.socket, 'connect', forbidden):
        result = await ProviderValidator().validate(Path(provider.__file__).parent, config={
            'auth_mode': auth_mode, 'token_file_path': str(token_file), 'login_on_mount': True,
        })
    assert result.passed, result.errors
