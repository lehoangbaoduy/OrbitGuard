"""The replay app must show only recorded data (redesign real-data rule 8, EC-30).

The design handoff's prototype runs on a synthetic generator; none of its numbers, episodes or modules may reach
`orbit_guard/visualization`. This scans every source file the app serves or runs (`.py`, `.js`, `.css`, `.html`)
for (a) imports/identifiers of synthetic or demo data modules and (b) literal episode numbers and labels copied
from the design's illustrations. The scanner is itself tested, so a green run is never vacuous.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "orbit_guard" / "visualization"
SCANNED_SUFFIXES = {".py", ".js", ".css", ".html"}
VENDORED_PREFIXES = ("00_plotly-",)  # third-party build, not our code

FORBIDDEN_IDENTIFIERS = ("SYNTHETIC", "orbitguard-data", "orbitguard_data", "OGData", "support.js", "targetsOff", "startStep")
# Specific numbers/labels that appear ONLY in the design's illustrations (never derive them from literals).
FORBIDDEN_LITERALS = (
    re.compile(r"\b111080\b"),
    re.compile(r"\b3,?728\b"),
    re.compile(r"\b3,?818\b"),
    re.compile(r"\b3,?872\b"),
    re.compile(r"\bseed\s+\d"),  # "seed 10": seeds come from trace.meta
    re.compile(r"\bT\+\d{2,}"),  # "T+250": the clock is formatted from the step ("(T+1, N)" shape notes are fine)
    re.compile(r"\b12\.20\b"),
    re.compile(r"\b\d+/10\b"),  # "5/10": reference counts come from an evaluation log
)


def scan_text(name: str, text: str) -> list[str]:
    findings = [f"{name}: identifier {token!r}" for token in FORBIDDEN_IDENTIFIERS if token in text]
    findings += [f"{name}: literal {pattern.pattern!r}" for pattern in FORBIDDEN_LITERALS if pattern.search(text)]
    return findings


def scan_python_imports(name: str, source: str) -> list[str]:
    findings = []
    for node in ast.walk(ast.parse(source)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        findings += [
            f"{name}: imports {module}" for module in modules if any(w in module.lower() for w in ("synthetic", "demo", "mock", "fake"))
        ]
    return findings


def _source_files() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.suffix in SCANNED_SUFFIXES and not path.name.startswith(VENDORED_PREFIXES)
    )


def test_scanner_flags_design_example_numbers():
    for bad in ("seed 111080", "3,728 interventions", "T+250", "const ref = '5/10'", "12.20 m/s", "OGData.source"):
        assert scan_text("x", bad), bad


def test_scanner_allows_data_driven_formatting():
    assert scan_text("x", "`T+${minutes}`; label = `seed ${meta.seed}`; text = f'{a}/{b}'") == []


def test_scanner_flags_synthetic_imports():
    assert scan_python_imports("x", "from orbit_guard.visualization.synthetic_data import episodes")
    assert scan_python_imports("x", "import demo_traces")
    assert scan_python_imports("x", "from orbit_guard.visualization.source import ReplaySource") == []


def test_package_is_scanned_and_nonempty():
    assert len(_source_files()) >= 3


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_synthetic_data_or_example_literals(path):
    text = path.read_text(encoding="utf8")
    findings = scan_text(str(path.relative_to(PACKAGE)), text)
    if path.suffix == ".py":
        findings += scan_python_imports(str(path.relative_to(PACKAGE)), text)
    assert findings == []
