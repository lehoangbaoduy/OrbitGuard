"""Candidate pair generation for conjunction screening (PRD §20)."""

from __future__ import annotations


def generate_candidate_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """Generate all unique unordered index pairs (i, j) with 0 <= i < j < n.

    Args:
        n: Total number of satellites in the constellation. Must be at least 2.

    Returns:
        Tuple of N(N-1)/2 index pairs (i, j) in canonical order (i < j).

    Raises:
        ValueError: If n < 2.
    """
    if n < 2:
        raise ValueError(f"Number of satellites n must be at least 2, got {n}.")
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))
