"""The status vocabulary (PRD §41: colour AND marker AND label, never colour alone) has ONE source, and its colours
are the design tokens (assets/01_orbitguard-tokens.css) so Python, CSS and the JS stage cannot drift apart."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orbit_guard.visualization.derived.status import (
    EVENT_NAMES,
    PAIR_STATUS_NAMES,
    SAT_STATUS_NAMES,
)
from orbit_guard.visualization.vocabulary import (
    EVENT_VOCAB,
    PAIR_VOCAB,
    SAT_VOCAB,
    vocabulary_payload,
)

TOKENS = Path(__file__).parents[2] / "orbit_guard" / "visualization" / "assets" / "01_orbitguard-tokens.css"
_HEX = re.compile(r"#[0-9a-f]{6}", re.IGNORECASE)


def _token(name: str) -> str:
    match = re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", TOKENS.read_text())
    assert match, f"token --{name} not found"
    return match.group(1).lower()


def test_vocabularies_are_indexed_by_the_status_and_event_codes():
    assert tuple(v.name for v in SAT_VOCAB) == SAT_STATUS_NAMES
    assert tuple(v.name for v in PAIR_VOCAB) == PAIR_STATUS_NAMES
    assert tuple(v.name for v in EVENT_VOCAB) == EVENT_NAMES


@pytest.mark.parametrize("vocab", [SAT_VOCAB, PAIR_VOCAB, EVENT_VOCAB[1:]])
def test_every_state_has_a_colour_a_glyph_and_a_label(vocab):
    for style in vocab:
        assert _HEX.fullmatch(style.color) and style.glyph.strip() and style.label.strip(), style.name


def test_no_two_states_share_a_glyph_within_a_vocabulary():
    for vocab in (SAT_VOCAB, PAIR_VOCAB, EVENT_VOCAB[1:]):
        glyphs = [s.glyph for s in vocab]
        assert len(glyphs) == len(set(glyphs))


def test_colours_are_the_design_tokens():
    assert SAT_VOCAB[3].color == _token("og-collision")
    assert SAT_VOCAB[2].color == _token("og-risk")
    assert SAT_VOCAB[1].color == _token("og-dsafe")
    assert EVENT_VOCAB[1].color == _token("og-maneuver")
    assert EVENT_VOCAB[2].color == _token("og-shield")
    assert SAT_VOCAB[0].color == "#9397ab"  # --og-nominal is var(--color-neutral-500) in the token file


def test_glyphs_follow_the_handoff_changes_for_plotly_3d():
    assert SAT_VOCAB[2].glyph == "◆" and SAT_VOCAB[1].glyph == "○" and SAT_VOCAB[3].glyph == "✕"
    assert EVENT_VOCAB[1].glyph == "■" and EVENT_VOCAB[2].glyph == "◇"


def test_payload_form_is_plain_json():
    payload = vocabulary_payload()

    assert [s["name"] for s in payload["sat_status"]] == list(SAT_STATUS_NAMES)
    assert payload["event"][0]["glyph"] == ""  # code 0 = none: nothing to draw
    assert {"color", "glyph", "label", "marker"} <= set(payload["sat_status"][3])
    assert {"dash", "width"} <= set(payload["pair_status"][2])


def test_the_stage_scripts_use_the_ice_token_colour():
    core = (TOKENS.parent / "10_og_core.js").read_text()

    assert f'const ICE = "{_token("og-ice")}"' in core
