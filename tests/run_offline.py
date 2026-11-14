"""Run trusted tests with credentials removed and network denied before imports.

Use: .venv/bin/python -I -S tests/run_offline.py [pytest args]. The audit
boundary covers this Python process, not native code or spawned children.
"""

import atexit
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
attempts = []
loopback = []
allow_loopback = "--allow-loopback" in sys.argv
if allow_loopback:
    sys.argv.remove("--allow-loopback")
receipt = root / ".media-offline-receipt.json"
if "--receipt" in sys.argv:
    i = sys.argv.index("--receipt")
    receipt = Path(sys.argv[i + 1]).resolve()
    del sys.argv[i : i + 2]
for key in list(os.environ):
    if any(
        word in key.upper()
        for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    ):
        os.environ.pop(key, None)
for key in ("PYTEST_ADDOPTS", "PYTHONPATH", "PYTHONHOME"):
    os.environ.pop(key, None)
owned_home = tempfile.TemporaryDirectory(prefix="chatgpt-offline-")
for key in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
    os.environ[key] = owned_home.name
result = {
    "attempts": attempts,
    "permitted_loopback": loopback,
    "exit_code": None,
    "shutdown_accounted": False,
}


def guard(event, args):
    if allow_loopback and event == "socket.getaddrinfo" and args[0] == "127.0.0.1":
        loopback.append({"event": event})
        return
    if (
        allow_loopback
        and event == "socket.connect"
        and isinstance(args[1], tuple)
        and args[1][0] == "127.0.0.1"
    ):
        loopback.append({"event": event})
        return
    if (
        event == "socket.getaddrinfo"
        or event == "socket.sendto"
        or (event == "socket.connect" and getattr(args[0], "family", None) in (2, 10))
    ):
        attempts.append({"event": event})
        raise RuntimeError("Offline test network access denied")


sys.addaudithook(guard)
# -S prevented site/Core/pytest from loading before this guard.
import site  # noqa: E402 -- audit hook must precede site initialization

site.main()
sys.path.insert(0, str(root))


def finalize():
    owned_home.cleanup()
    result["shutdown_accounted"] = True
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n")


atexit.register(finalize)
import pytest  # noqa: E402 -- audit hook must precede pytest/plugin imports

try:
    result["exit_code"] = int(pytest.main(sys.argv[1:]))
finally:
    if attempts:
        result["exit_code"] = 1
raise SystemExit(result["exit_code"])
