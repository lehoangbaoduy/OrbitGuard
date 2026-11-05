"""Main experiments use no safety-based action masking (master-prompt checklist #16, PRD §28-§29).

Safety comes from the shield, which repairs proposed joint actions after the policy acts; masking the policy's
action space by safety would hide unsafe proposals from the shield-intervention diagnostics. Nothing in the
package may build or consume an action mask.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "orbit_guard"


def test_no_module_in_the_package_builds_or_reads_an_action_mask():
    offenders = [
        str(path.relative_to(_PACKAGE.parent))
        for path in sorted(_PACKAGE.rglob("*.py"))
        if "action_mask" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
