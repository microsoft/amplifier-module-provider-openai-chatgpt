"""Run trusted offline tests with a loopback-only Python socket audit hook.

This is a test barrier, not a hostile-code sandbox. Counts are audit events,
not model requests. Dependency installation happens before this runner.
"""

import atexit
import json
import socket
import sys

counts = {"loopback_connect": 0, "loopback_lookup": 0, "denied": 0}


def audit(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        if args[0] not in {"127.0.0.1", "localhost", b"127.0.0.1", b"localhost"}:
            counts["denied"] += 1
            raise RuntimeError("Test runner denied a non-loopback name lookup")
        counts["loopback_lookup"] += 1
    elif event in {"socket.connect", "socket.sendto"}:
        sock, address = args[0], args[-1]
        if sock.family == getattr(socket, "AF_UNIX", None):
            return
        if not isinstance(address, tuple) or address[0] != "127.0.0.1":
            counts["denied"] += 1
            raise RuntimeError("Test runner denied a non-loopback socket operation")
        counts["loopback_connect"] += 1


if __name__ == "__main__":
    sys.addaudithook(audit)
    atexit.register(
        lambda: print("TEST_NETWORK_GUARD " + json.dumps(counts), file=sys.stderr)
    )
    import pytest

    result = pytest.main(sys.argv[1:])
    raise SystemExit(1 if counts["denied"] else result)
