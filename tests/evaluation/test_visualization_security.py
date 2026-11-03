"""Security hardening of the replay server: response headers, a CSP that matches Dash's one inline script, host checks."""

from __future__ import annotations

import base64
import hashlib
import re

import pytest

from orbit_guard.visualization.app import create_app
from orbit_guard.visualization.episodes import index_from_trace
from orbit_guard.visualization.replay import save_trace
from orbit_guard.visualization.replay_meta import ReplayMetaError, validate_replay_meta
from orbit_guard.visualization.security import RemoteHostError, check_host
from orbit_guard.visualization.summary import summary_totals
from tests.evaluation.replay_fixtures import shielded_trace


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("sec") / "episode.npz"
    save_trace(shielded_trace(), path)
    app = create_app(index_from_trace(path))
    return app.server.test_client()


def test_every_response_carries_the_standard_security_headers(client):
    for url in ("/", "/assets/10_og_core.js"):
        headers = client.get(url, base_url="http://127.0.0.1:8050").headers

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "camera=()" in headers["Permissions-Policy"]
        assert "Strict-Transport-Security" not in headers  # plain HTTP on loopback: HSTS would only pin other local servers


def test_csp_allows_only_dashs_one_inline_script_by_hash_and_never_eval(client):
    response = client.get("/", base_url="http://127.0.0.1:8050")
    csp = response.headers["Content-Security-Policy"]
    script_src = re.search(r"script-src ([^;]*)", csp).group(1)
    inline = re.search(r'<script id="_dash-renderer"[^>]*>(.*?)</script>', response.get_data(as_text=True), re.DOTALL).group(1)
    digest = "sha256-" + base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()

    assert f"'{digest}'" in script_src
    assert "'unsafe-eval'" not in csp and "'unsafe-inline'" not in script_src
    assert "default-src 'self'" in csp and "object-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert "http:" not in csp.replace("https:", "")  # nothing is loaded from a third party


def test_other_hosts_are_refused_to_stop_dns_rebinding(client):
    assert client.get("/", base_url="http://localhost:8050").status_code == 200
    assert client.get("/", base_url="http://evil.example:8050").status_code == 400


def test_request_bodies_are_capped(client):
    assert client.application.config["MAX_CONTENT_LENGTH"] <= 1 << 20


def test_a_non_loopback_host_needs_an_explicit_flag():
    check_host("127.0.0.1", allow_remote=False)
    check_host("localhost", allow_remote=False)
    check_host("::1", allow_remote=False)
    with pytest.raises(RemoteHostError, match="--allow-remote"):
        check_host("0.0.0.0", allow_remote=False)
    check_host("0.0.0.0", allow_remote=True)


@pytest.mark.parametrize(("field", "bad"), [("maneuver_count", float("inf")), ("maneuver_count", [1]), ("total_delta_v_mps", {"a": 1}), ("total_delta_v_mps", "x"), ("collision_count", float("nan"))])
def test_a_malformed_summary_value_is_a_meta_error_not_a_crash(field, bad):
    meta = validate_replay_meta(shielded_trace().meta)
    summary = {"total_delta_v_mps": 1.0, "maneuver_count": 1, "shield_intervention_count": 1, "collision_count": 0, "mean_mission_deviation": 0.1}

    with pytest.raises(ReplayMetaError, match=field):
        summary_totals({**summary, field: bad}, meta)


def test_a_non_finite_recorded_deviation_is_kept_as_no_value():
    meta = validate_replay_meta(shielded_trace().meta)
    summary = {"total_delta_v_mps": 1.0, "maneuver_count": 1, "shield_intervention_count": 1, "collision_count": 0, "mean_mission_deviation": float("nan")}

    assert summary_totals(summary, meta).mean_deviation != summary_totals(summary, meta).mean_deviation  # NaN: shown as a dash later


def test_install_security_refuses_to_build_an_app_whose_csp_cannot_be_derived(monkeypatch):
    """Dash changed its index page: no renderer script to hash. Serving without a CSP would be a silent downgrade."""
    from dash import Dash, html

    from orbit_guard.visualization.security import install_security

    app = Dash(__name__)
    app.layout = html.Div("replay")
    monkeypatch.setattr(app, "index", lambda: "<!DOCTYPE html><html><body>no renderer script here</body></html>")

    with pytest.raises(RuntimeError, match="Content-Security-Policy"):
        install_security(app)


def test_create_app_fails_before_serving_when_the_csp_cannot_be_derived(tmp_path, monkeypatch):
    from dash import Dash

    path = tmp_path / "episode.npz"
    save_trace(shielded_trace(), path)
    monkeypatch.setattr(Dash, "index", lambda self, *args, **kwargs: "<html><body>changed upstream</body></html>")

    with pytest.raises(RuntimeError, match="Content-Security-Policy"):
        create_app(index_from_trace(path))
