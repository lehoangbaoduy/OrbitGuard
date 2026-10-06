"""The replay's status vocabulary: one source for colour, glyph, marker and label (PRD §41, handoff "Tokens").

Every state is told apart by colour AND a glyph/marker AND a text label, never colour alone. The colours are the
design tokens of `assets/01_orbitguard-tokens.css` (a test keeps them identical). Glyph changes from the handoff:
◆ for high risk and ○ for inside d_safe, because Plotly 3D has no triangle marker. Entries are indexed by the
status and event codes of `derived/status.py`; the whole table travels in the payload so the JS stage has no copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

COLLISION = "#ff5c7a"
RISK = "#ffb547"
DSAFE = "#4fd8e8"
SHIELD = "#bb9cf7"
MANEUVER = "#5fe08f"
NOMINAL = "#9397ab"


@dataclass(frozen=True)
class Style:
    name: str
    label: str
    glyph: str
    color: str
    marker: str = "circle"  # Plotly scatter3d marker symbol
    dash: str = "solid"  # conjunction-line dash (pair statuses)
    width: float = 1.0  # conjunction-line width (pair statuses)


SAT_VOCAB = (
    Style("nominal", "nominal", "●", NOMINAL, marker="circle"),
    Style("inside_d_safe", "INSIDE d_safe", "○", DSAFE, marker="circle-open"),
    Style("high_risk", "HIGH RISK", "◆", RISK, marker="diamond"),
    Style("collided", "COLLIDED", "✕", COLLISION, marker="x"),
)
PAIR_VOCAB = (
    Style("nominal", "nominal", "·", NOMINAL, marker="circle", width=1),
    Style("inside_d_safe", "INSIDE d_safe", "○", DSAFE, marker="circle-open", dash="dot", width=4),
    Style("high_risk", "HIGH RISK", "◆", RISK, marker="diamond", dash="dash", width=6),
    Style("collision", "COLLISION", "✕", COLLISION, marker="x", width=5),
)
EVENT_VOCAB = (
    Style("none", "", "", NOMINAL),
    Style("maneuver", "MANEUVER", "■", MANEUVER, marker="square"),
    Style("shield_intervention", "SHIELD INTERVENTION", "◇", SHIELD, marker="diamond-open"),
    Style("collision", "COLLISION", "✕", COLLISION, marker="x"),
)


# Thruster capability marks: neutral and dashed, never a status hue (the capability set itself comes from the trace).
CAPABILITY_VOCAB = {
    "offline": Style("offline", "THRUSTER OFFLINE", "⊘", NOMINAL),
    "limited": Style("limited", "LIMITED Δv", "◐", NOMINAL),
}


def vocabulary_payload() -> dict[str, list[dict]]:
    """Plain-JSON form of the vocabulary, aligned with the status and event code arrays in the payload."""
    return {
        "sat_status": [asdict(style) for style in SAT_VOCAB],
        "pair_status": [asdict(style) for style in PAIR_VOCAB],
        "event": [asdict(style) for style in EVENT_VOCAB],
        "capability": {key: asdict(style) for key, style in CAPABILITY_VOCAB.items()},
    }
