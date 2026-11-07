"""Portable atomic credential save and descriptor ownership regressions."""

import errno
import json
import os
from pathlib import Path

import pytest

from amplifier_module_provider_openai_chatgpt import oauth


@pytest.fixture
def temporary_descriptors(monkeypatch):
    original = oauth.tempfile.mkstemp
    created = []

    def record(**kwargs):
        fd, path = original(**kwargs)
        created.append((fd, Path(path)))
        return fd, path

    monkeypatch.setattr(oauth.tempfile, "mkstemp", record)
    yield created
    # Also close the predecessor's leaked descriptor after observing the failure.
    for fd, path in created:
        try:
            os.close(fd)
        except OSError as exc:
            assert exc.errno == errno.EBADF
        if path.exists():
            path.unlink()


def assert_no_leaks(created):
    assert len(created) == 1
    fd, path = created[0]
    with pytest.raises(OSError) as caught:
        os.fstat(fd)
    assert caught.value.errno == errno.EBADF
    assert not path.exists()


def test_atomic_writer_does_not_require_fchmod(monkeypatch, tmp_path, temporary_descriptors):
    path = tmp_path / "synthetic.json"
    path.write_text('{"old": true}')
    monkeypatch.delattr(oauth.os, "fchmod", raising=False)
    # Exercise the writer directly: the host's POSIX lock implementation is
    # not the Windows lock implementation, so do not simulate that unrelated API.
    oauth._write_tokens_atomic({"new": "synthetic"}, path)
    assert json.loads(path.read_text()) == {"new": "synthetic"}
    assert_no_leaks(temporary_descriptors)


@pytest.mark.parametrize("failure", ["permissions", "fdopen", "serialize", "replace"])
def test_failed_write_preserves_original_and_closes_descriptor(
    monkeypatch, tmp_path, temporary_descriptors, failure
):
    path = tmp_path / "synthetic.json"
    original = b'{"account": "original-synthetic"}\n'
    path.write_bytes(original)

    def fail(*args, **kwargs):
        raise OSError("synthetic failure")

    if failure == "permissions":
        monkeypatch.setattr(oauth.os, "chmod", fail)
        # The predecessor uses fchmod; fail that operation as well.
        monkeypatch.setattr(oauth.os, "fchmod", fail, raising=False)
    elif failure == "fdopen":
        monkeypatch.setattr(oauth.os, "fdopen", fail)
    elif failure == "serialize":
        monkeypatch.setattr(oauth.json, "dump", fail)
    else:
        monkeypatch.setattr(oauth.os, "replace", fail)

    with pytest.raises(OSError, match="synthetic failure"):
        oauth._write_tokens_atomic({"account": "new-synthetic"}, path)
    assert path.read_bytes() == original
    assert_no_leaks(temporary_descriptors)


def test_save_tokens_keeps_existing_public_roundtrip(tmp_path, temporary_descriptors):
    path = tmp_path / "nested" / "synthetic.json"
    oauth.save_tokens({"access_token": "synthetic-token"}, str(path))
    assert oauth.load_tokens(str(path)) == {"access_token": "synthetic-token"}
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert_no_leaks(temporary_descriptors)
