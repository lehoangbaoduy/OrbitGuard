"""Record the before/after replay episodes (3 scenarios x untrained/trained x shield on/off) to disk.

Output goes to a directory on disk (never /tmp, which is RAM) as `.npz` traces whose `meta` carries everything the
replay UI needs (see `visualization.replay_meta`): scenario, policy, shield flag, delta-v budget, the SHA-256 of the
trained weights and the init seed of the untrained network. Re-run it after any retrain: stale traces are then
detectable from their recorded weights hash. The untrained policy is a freshly initialised actor built exactly like
the trainer's (`ctde_module._mlp`, PyTorch default init) under a fixed seed.

A tuned overlay (`overlay_path`) applies to every episode of a recording: it reaches the environment and the
untrained actor's architecture, and the meta records it with `config_digest(config, overlay)`. A trained checkpoint
whose run_config.json sidecar records a different digest is refused (`CheckpointProvenanceError`) before anything
is recorded, exactly as the experiment refuses it; so is one whose sidecar records a margin or reward weights
other than the ones the item's model gets from config + overlay (training CLI overrides). A trained checkpoint
with no run_config.json sidecar is refused too (unlike a validation run, a replay is shown as a final model), and
with a freeze snapshot every replayed actor's sha256 must equal the frozen checkpoint of its model/training seed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gymnasium.spaces.utils import flatdim

from orbit_guard.config import config_digest, load_config
from orbit_guard.evaluation.baselines import build_env_factory
from orbit_guard.evaluation.experiment import (
    RUN_CONFIG_FILENAME,
    CheckpointProvenanceError,
    check_checkpoint_provenance,
)
from orbit_guard.evaluation.freeze import FrozenConfigError, read_snapshot
from orbit_guard.evaluation.shifts import condition_env_kwargs
from orbit_guard.rl.actor_policy import (
    ActorWeights,
    NumpyActorPolicy,
    load_actor_weights,
    save_actor_weights,
)
from orbit_guard.visualization.replay import record_episode, save_trace
from orbit_guard.visualization.replay_meta import build_replay_meta

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
# (condition, scenario seed): one sample episode per shift type shown in the replay app.
SCENARIOS: tuple[tuple[str, int], ...] = (
    ("normal", 111080),
    ("simultaneous_conjunctions", 10),
    ("capability_reduction", 10),
)
DEFAULT_INIT_SEED = 0
UNTRAINED_MODEL = "untrained_policy"
WEIGHTS_SUBDIR = "weights"


@dataclass(frozen=True)
class EpisodePlan:
    condition: str
    seed: int
    policy: str  # "untrained" | "trained"
    shield: bool
    model: str
    weights_path: Path | None
    training_seed: int | None = None  # which training run supplies a trained actor (None: untrained)


def build_plan(
    runs_dir: str | Path, scenarios: Sequence[tuple[str, int]] = SCENARIOS, trained_seed: int = 0
) -> list[EpisodePlan]:
    """12 episodes: for each scenario, untrained then trained, shield on then off."""
    runs = Path(runs_dir)
    plan = []
    for condition, seed in scenarios:
        for policy in ("untrained", "trained"):
            for shield in (True, False):
                if policy == "untrained":
                    model, weights = UNTRAINED_MODEL, None
                else:
                    variant = "adaptive" if shield else "noshield"
                    model = "adaptive_mappo" if shield else "adaptive_mappo_no_shield"
                    weights = runs / f"n15_{variant}_s{trained_seed}" / "actor_weights.npz"
                training_seed = None if policy == "untrained" else trained_seed
                plan.append(EpisodePlan(condition, seed, policy, shield, model, weights, training_seed))
    return plan


def _actor_hidden_layers(config_path: str | Path, overlay_path: str | Path | None) -> tuple[int, ...]:
    return tuple(load_config(config_path, overlay_path)["rl"]["actor_hidden_layers"])


def untrained_actor_weights(
    obs_dim: int, num_actions: int, init_seed: int = DEFAULT_INIT_SEED, hidden: Sequence[int] | None = None,
    config_path: str | Path = _DEFAULT_CONFIG, overlay_path: str | Path | None = None,
) -> ActorWeights:
    """A freshly initialised actor with the trainer's architecture; the global torch RNG is left untouched."""
    import torch

    from orbit_guard.rl.actor_policy import weights_from_actor
    from orbit_guard.rl.ctde_module import _mlp

    layers = tuple(hidden) if hidden is not None else _actor_hidden_layers(config_path, overlay_path)
    state = torch.get_rng_state()
    try:
        torch.manual_seed(init_seed)
        actor = _mlp(obs_dim, layers, num_actions)
    finally:
        torch.set_rng_state(state)
    return weights_from_actor(actor)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_hashes(freeze_snapshot: str | Path | None) -> dict[str, str] | None:
    if freeze_snapshot is None:
        return None
    try:
        snapshot = read_snapshot(freeze_snapshot)
    except FrozenConfigError as error:
        raise CheckpointProvenanceError(str(error)) from error
    return {name: entry["actor_weights_sha256"] for name, entry in snapshot["checkpoints"].items()}


def _check_frozen(item: EpisodePlan, frozen: Mapping[str, str]) -> None:
    """The replayed actor must be byte-identical to the frozen checkpoint of the same model and training seed."""
    name = f"{item.model}/{item.training_seed}"
    if name not in frozen:
        raise CheckpointProvenanceError(
            f"{item.weights_path} would be replayed as {name}, which is not in the freeze snapshot "
            f"(frozen: {sorted(frozen)})"
        )
    if _file_sha256(Path(item.weights_path)) != frozen[name]:
        raise CheckpointProvenanceError(
            f"{item.weights_path} is not the frozen {name} checkpoint (its sha256 differs from the one the freeze "
            "recorded): record replays from the frozen runs dir"
        )


def check_plan_provenance(
    plan: Sequence[EpisodePlan],
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    freeze_snapshot: str | Path | None = None,
) -> None:
    """Refuse the plan if any trained checkpoint has no run_config.json sidecar, or its sidecar records another
    config_digest, or a margin or reward weights other than the ones the item's model gets from config + overlay;
    with `freeze_snapshot`, also unless its weights are the frozen checkpoint of that model and training seed."""
    frozen = _frozen_hashes(freeze_snapshot)
    for item in plan:
        if item.policy != "trained" or item.weights_path is None:
            continue
        if not (Path(item.weights_path).parent / RUN_CONFIG_FILENAME).exists():
            raise CheckpointProvenanceError(
                f"{item.weights_path} has no {RUN_CONFIG_FILENAME} sidecar, so the configuration it was trained "
                "under cannot be shown (checkpoints trained before sidecars existed are retired): replay checkpoints "
                "from a runs dir written by rl/training.py"
            )
        check_checkpoint_provenance(item.weights_path, config_path=config_path, overlay_path=overlay_path, model=item.model)
        if frozen is not None:
            _check_frozen(item, frozen)


def record_plan_item(
    item: EpisodePlan,
    *,
    num_agents: int,
    out_dir: str | Path,
    init_seed: int = DEFAULT_INIT_SEED,
    env_kwargs: Mapping[str, Any] | None = None,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    freeze_snapshot: str | Path | None = None,
) -> Path:
    """Run one episode and write its trace; returns the path (the file name is informational only)."""
    check_plan_provenance([item], config_path=config_path, overlay_path=overlay_path, freeze_snapshot=freeze_snapshot)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env_model = "adaptive_mappo" if item.shield else "adaptive_mappo_no_shield"
    kwargs = {**condition_env_kwargs(item.condition, config_path), **(env_kwargs or {})}
    env = build_env_factory(env_model, num_agents, config_path=config_path, overlay_path=overlay_path, **kwargs)({})
    agent = env.possible_agents[0]
    if item.policy == "untrained":
        weights = untrained_actor_weights(
            flatdim(env.observation_space(agent)), int(env.action_space(agent).n), init_seed,
            config_path=config_path, overlay_path=overlay_path,
        )
        (out / WEIGHTS_SUBDIR).mkdir(exist_ok=True)  # weights live beside, not among, the traces the UI scans
        save_actor_weights(weights, out / WEIGHTS_SUBDIR / f"untrained_actor_init{init_seed}.npz")
        weights_sha, seed_for_meta = None, init_seed
    else:
        if item.weights_path is None:
            raise ValueError("a trained episode needs a weights file")
        weights, weights_sha, seed_for_meta = load_actor_weights(item.weights_path), _file_sha256(item.weights_path), None
    meta = build_replay_meta(
        condition=item.condition,
        seed=item.seed,
        policy=item.policy,
        shield=item.shield,
        model=item.model,
        n_agents=num_agents,
        delta_v_budget_mps=env.delta_v_budget_mps,
        max_delta_v_per_action_mps=env.max_delta_v_per_action_mps,
        weights_sha256=weights_sha,
        init_seed=seed_for_meta,
        config_digest=config_digest(config_path, overlay_path),
        overlay_path=None if overlay_path is None else str(overlay_path),
    )
    trace = record_episode(env, NumpyActorPolicy(weights), seed=item.seed, meta=meta)
    path = out / f"{item.condition}_{item.seed}__{item.policy}__shield-{'on' if item.shield else 'off'}.npz"
    save_trace(trace, path)
    return path


def record_all(
    plan: Sequence[EpisodePlan],
    out_dir: str | Path,
    *,
    num_agents: int,
    init_seed: int = DEFAULT_INIT_SEED,
    env_kwargs: Mapping[str, Any] | None = None,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    progress: Callable[[str], None] = print,
    freeze_snapshot: str | Path | None = None,
) -> list[Path]:
    """Record every plan item under one configuration; every checkpoint is checked before the first episode runs."""
    check_plan_provenance(plan, config_path=config_path, overlay_path=overlay_path, freeze_snapshot=freeze_snapshot)
    paths = []
    for number, item in enumerate(plan, start=1):
        progress(f"[{number}/{len(plan)}] {item.condition}:{item.seed} {item.policy} shield={'on' if item.shield else 'off'}")
        paths.append(record_plan_item(
            item, num_agents=num_agents, out_dir=out_dir, init_seed=init_seed, env_kwargs=env_kwargs,
            config_path=config_path, overlay_path=overlay_path, freeze_snapshot=freeze_snapshot,
        ))
    return paths
