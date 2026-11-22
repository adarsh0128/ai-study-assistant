"""Shared pytest configuration for the whole test suite.

The autouse fixture here blocks outbound network connections to anything
other than loopback. This keeps the suite fast, deterministic, free, and
runnable offline: no test may accidentally call the real Groq API, which
would be slow, flaky, and billable.

Loopback (127.0.0.1 / ::1) stays allowed because FastAPI's TestClient talks
to the app over an in-process socket pair — blocking it would break every
endpoint test without protecting anything.

If a test genuinely needs the real API (a manual smoke test you run by
hand, never in CI), opt out with the marker:

    @pytest.mark.allow_network
    def test_real_call(): ...

    pytest -m allow_network
"""

import socket

import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "", None}


def _is_loopback(address: object) -> bool:
    """True when the connect() target is local and therefore harmless."""
    host = None
    if isinstance(address, tuple) and address:
        host = address[0]
    elif isinstance(address, str):
        host = address
    if host in _LOOPBACK:
        return True
    # Anything in 127.0.0.0/8 counts as loopback.
    return isinstance(host, str) and host.startswith("127.")


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch, request):
    """Fail any test that connects to a non-loopback address."""
    if request.node.get_closest_marker("allow_network"):
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def guard(original):
        def wrapper(*args, **kwargs):
            # args[0] is `self` for the socket methods, the address otherwise.
            address = args[1] if len(args) > 1 else kwargs.get("address")
            if not _is_loopback(address):
                raise AssertionError(
                    f"Test attempted an external network connection to "
                    f"{address!r}. Mock the client instead, or mark the test "
                    f"with @pytest.mark.allow_network."
                )
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(socket.socket, "connect", guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(real_connect_ex))
    monkeypatch.setattr(
        socket, "create_connection", guard(real_create_connection)
    )
    yield


def pytest_configure(config):
    """Register custom markers so pytest does not warn about them."""
    config.addinivalue_line(
        "markers", "allow_network: test may open real network connections"
    )
    config.addinivalue_line("markers", "live: requires a real GROQ_API_KEY")
