"""Comparative copy between the BEFORE and AFTER episodes, generated from data (real-data rule 7, EC-25).

Never claims an improvement the numbers do not show: it reports "more" as readily as "less", never rounds a
non-zero value down to "100% less", and returns None when a comparison is not meaningful (missing trace, zero
denominator, non-finite values, or a baseline policy that has no BEFORE/AFTER pair).
"""

from __future__ import annotations

import math

_RATIO_FROM = 2.0  # an "N× after" ratio only reads naturally from twice upward; below it a percentage is clearer
_RATIO_DECIMALS_BELOW = 10.0
_NEARLY_ALL = 99.5  # at or above this a reduction would round to "100%"


def _percent_less(value: float, other: float) -> str:
    percent = (1.0 - value / other) * 100.0
    if value == 0:
        return "100% less"
    return ">99% less" if percent >= _NEARLY_ALL else f"{percent:.0f}% less"


def comparison_note(policy: str, value: float | None, other: float | None) -> str | None:
    """Note under a summary cell: this tab's `value` against the other tab's `other`."""
    if value is None or other is None or not math.isfinite(value) or not math.isfinite(other):
        return None
    if other <= 0 or policy not in ("untrained", "trained"):
        return None
    versus = "after" if policy == "untrained" else "before"
    if math.isclose(value, other, rel_tol=1e-9):
        return f"same as {versus}"
    if value < other:
        return f"{_percent_less(value, other)} than {versus}"
    ratio = value / other
    if policy == "untrained" and ratio >= _RATIO_FROM:
        return f"{ratio:.0f}× after" if ratio >= _RATIO_DECIMALS_BELOW else f"{ratio:.1f}× after"
    return f"{(ratio - 1.0) * 100.0:.0f}% more than {versus}"
