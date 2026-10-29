"""The web UI's request handler called in this process, without a socket: automated checks open no listening
port. A headless browser reaches the handler through Playwright's request routing
(ui/web/walkthrough.py); the tests call it directly (tests/test_ui_web.py).

    handler = make_handler(config, mesh, ...)                  # ui/web/server.py
    status, headers, body = call(handler, "GET", "/api/config")
"""
import io

_CONN: dict[type, type] = {}


def _conn(handler: type) -> type:
    """handler with in-memory streams: the request bytes in, the response bytes kept after handle()."""
    if handler not in _CONN:
        class Conn(handler):
            def setup(self):
                self.rfile, self.wfile = io.BytesIO(self.request), io.BytesIO()

            def finish(self):                    # keep wfile open: call() reads it
                pass
        _CONN[handler] = Conn
    return _CONN[handler]


def call(handler: type, method: str, path: str, body: bytes = b"",
         headers: dict | None = None) -> tuple[int, dict[str, str], bytes]:
    """One request through the handler class: (status, response headers, body)."""
    head = [f"{method} {path} HTTP/1.1", "Host: meshom.local", "Connection: close"]
    head += [f"{k}: {v}" for k, v in (headers or {}).items()]
    if body or method == "POST":
        head.append(f"Content-Length: {len(body)}")
    raw = ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body
    out = _conn(handler)(raw, ("127.0.0.1", 0), None).wfile.getvalue()
    status, _, rest = out.partition(b"\r\n")
    hdr, _, payload = rest.partition(b"\r\n\r\n")
    hs = dict(line.decode("latin-1").split(":", 1) for line in hdr.split(b"\r\n") if b":" in line)
    return int(status.split()[1]), {k.strip(): v.strip() for k, v in hs.items()}, payload
