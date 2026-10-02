"""Guardrails on the replay app's static assets (redesign PR0, EC-30 / EC-33).

Dash serves EVERY file under `assets/`, so a stray copy of the design prototype's synthetic data or runtime would
silently ship to users. These tests pin what may and may not live there, and that the single plotly.js build the
page loads is the vendored, checksummed 2.35.2 (Dash's own bundle is 4.1.1 and would break the camera technique).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "orbit_guard" / "visualization" / "assets"
HANDOFF = ROOT / "design_handoff_orbitguard_replay" / "design"

PLOTLY_FILE = "00_plotly-2.35.2.min.js"
PLOTLY_SHA256 = "6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603"  # cdn.plot.ly == cdn.jsdelivr.net
FORBIDDEN_NAMES = ("orbitguard-data", "support.js", "synthetic", "dc.html")
ALLOWED_SUFFIXES = {".js", ".css", ".jpg", ".woff2", ".svg", ".md", ".txt"}  # .txt: font licences


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_files() -> list[Path]:
    return sorted(path for path in ASSETS.rglob("*") if path.is_file())


def test_assets_folder_exists():
    assert ASSETS.is_dir(), f"missing {ASSETS}"


def test_vendored_plotly_is_the_pinned_build():
    plotly = ASSETS / PLOTLY_FILE
    assert plotly.is_file()
    assert _sha256(plotly) == PLOTLY_SHA256
    assert "plotly.js v2.35.2" in plotly.read_text(encoding="utf8", errors="ignore")[:200]


def test_plotly_loads_before_every_other_script():
    scripts = [path.name for path in _asset_files() if path.suffix == ".js"]
    assert scripts[0] == PLOTLY_FILE  # Dash loads assets in sorted order


def test_no_prototype_runtime_or_synthetic_data_in_assets():
    offenders = [p.name for p in _asset_files() if any(token in p.name.lower() for token in FORBIDDEN_NAMES)]
    assert offenders == []


def test_assets_hold_only_expected_file_types():
    odd = [p.name for p in _asset_files() if p.suffix not in ALLOWED_SUFFIXES]
    assert odd == []


@pytest.mark.parametrize("name", ["earth-day-1k.jpg", "earth-night-1k.jpg"])
def test_earth_textures_are_byte_identical_to_the_handoff(name):
    assert _sha256(ASSETS / name) == _sha256(HANDOFF / "assets" / name)


def test_nocturne_base_css_is_the_handoff_file_minus_its_google_fonts_import():
    handoff = next((HANDOFF / "_ds").glob("*/styles.css")).read_text(encoding="utf8")
    without_import = "".join(line for line in handoff.splitlines(keepends=True) if not line.startswith("@import url('https://fonts.googleapis.com"))
    assert (ASSETS / "00_nocturne.css").read_text(encoding="utf8") == without_import and without_import != handoff


def test_token_file_defines_the_status_and_motion_tokens():
    css = (ASSETS / "01_orbitguard-tokens.css").read_text(encoding="utf8")
    for token in ("--og-collision", "--og-risk", "--og-dsafe", "--og-shield", "--og-maneuver", "--og-nominal",
                  "--og-dur-flash", "--og-dur-enter", "--og-ease-out", "--og-void", "--og-ice"):
        assert token in css, token
    assert "prefers-reduced-motion" in css


def test_css_loads_in_cascade_order():
    names = [path.name for path in _asset_files() if path.suffix == ".css"]
    assert names[:2] == ["00_nocturne.css", "01_orbitguard-tokens.css"]


def test_asset_licences_are_documented():
    notes = (ASSETS / "README_ASSETS.md").read_text(encoding="utf8")
    for needle in ("plotly.js", "MIT", "Earth", "Blue Marble"):
        assert needle in notes, needle


def test_no_stylesheet_or_stage_script_loads_anything_from_a_third_party():
    """Everything is self-hosted (D9): an `@import` or absolute url() would leak the viewer's IP and break offline use."""
    root = Path(__file__).parents[2] / "orbit_guard" / "visualization" / "assets"
    offenders = []
    for path in [*root.glob("*.css"), *root.glob("*_og_*.js")]:
        text = path.read_text()
        if "@import" in text or re.search(r"url\(\s*['\"]?(https?:)?//", text) or re.search(r"""['"`]https?://""", text):
            offenders.append(path.name)
    assert offenders == []


def test_font_licences_ship_with_the_fonts():
    fonts = Path(__file__).parents[2] / "orbit_guard" / "visualization" / "assets" / "fonts"
    assert (fonts / "OFL-InstrumentSans.txt").is_file() and (fonts / "OFL-MartianMono.txt").is_file()
