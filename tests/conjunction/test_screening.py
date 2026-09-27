"""Contract tests for orbit_guard.conjunction.screening (PRD §20).

At N=10-20, exhaustive pairwise screening is acceptable: N(N-1)/2 candidate
pairs (105 pairs at N=15). No spatial indexing is required for the MVP.
"""

from orbit_guard.conjunction.screening import generate_candidate_pairs


def test_generate_candidate_pairs_count_matches_n_choose_2():
    assert len(generate_candidate_pairs(15)) == 105  # 15*14/2
    assert len(generate_candidate_pairs(10)) == 45
    assert len(generate_candidate_pairs(20)) == 190


def test_generate_candidate_pairs_are_unique_unordered_index_pairs():
    pairs = generate_candidate_pairs(5)
    assert len(set(pairs)) == len(pairs)  # no duplicates
    for i, j in pairs:
        assert i < j  # canonical order, no (j, i) duplicate of (i, j)
        assert 0 <= i < 5
        assert 0 <= j < 5


def test_generate_candidate_pairs_covers_every_pair_exactly_once():
    pairs = set(generate_candidate_pairs(4))
    expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert pairs == expected


def test_generate_candidate_pairs_rejects_n_less_than_two():
    import pytest

    with pytest.raises(ValueError):
        generate_candidate_pairs(1)
