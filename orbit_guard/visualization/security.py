"""Hardening for the local replay server: security headers, a CSP matched to Dash's one inline script, host checks.

The app is read-only and meant for the operator's own machine. These controls keep a stray page (DNS rebinding,
framing, MIME sniffing) and an accidental `--host 0.0.0.0` from turning it into something else.
"""

from __future__ import annotations

import base64
import hashlib
import re

from dash import Dash
from flask import Response, request

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_REQUEST_BYTES = 1 << 20  # callback bodies are a few hundred bytes
_RENDERER_SCRIPT = re.compile(r'<script id="_dash-renderer"[^>]*>(.*?)</script>', re.DOTALL)

_STATIC_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}  # no HSTS: this is plain HTTP on loopback, and a preload on `localhost` would pin other local dev servers to HTTPS


class RemoteHostError(ValueError):
    """A non-loopback bind address was requested without `--allow-remote`."""


class CspUnavailableError(RuntimeError):
    """Dash's index page no longer carries the one inline renderer script the CSP is built from (fail closed)."""


def check_host(host: str, *, allow_remote: bool) -> None:
    if host not in LOOPBACK_HOSTS and not allow_remote:
        raise RemoteHostError(
            f"refusing to listen on {host!r}: the replay server has no authentication or TLS and serves the full episode data. "
            "Pass --allow-remote to do it anyway (e.g. behind a reverse proxy you control)."
        )


def content_security_policy(inline_script: str) -> str:
    """CSP for Dash 4.x: its single executed inline script is allowed by hash (Dash has no nonce support); no eval, no third party.

    `style-src 'unsafe-inline'` is needed because plotly.js injects <style> elements; `data:` images carry the icon masks.
    """
    digest = base64.b64encode(hashlib.sha256(inline_script.encode()).digest()).decode()
    return "; ".join([
        "default-src 'self'",
        f"script-src 'self' 'sha256-{digest}'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self'",
        "connect-src 'self'",
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


def install_security(app: Dash, *, allow_remote: bool = False) -> None:
    """Add the headers and limits to `app`. Call after the layout exists (the index page is rendered once to hash its inline script)."""
    server = app.server
    server.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    if not allow_remote:
        server.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1", "[::1]"]
    with server.test_request_context("/"):
        page = app.index()
    match = _RENDERER_SCRIPT.search(page)
    if match is None:  # fail closed: serving without a CSP would drop the script protection silently
        raise CspUnavailableError(
            "cannot build the Content-Security-Policy: the Dash index page has no '<script id=\"_dash-renderer\">' block "
            "to hash (Dash changed its renderer markup?). Refusing to serve the replay app without it."
        )
    policy = content_security_policy(match.group(1))

    @server.after_request
    def _headers(response: Response) -> Response:
        for name, value in _STATIC_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.path == "/":
            response.headers.setdefault("Content-Security-Policy", policy)
        return response
