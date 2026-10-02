"""Pure derived-data functions over an `EpisodeTrace` (redesign real-data rule 1, PR1).

Every number the replay UI shows that is not a raw trace array is computed here, as a pure function with its own
tests. Nothing in this package imports Ray, torch, or anything outside NumPy and the replay module.
"""
