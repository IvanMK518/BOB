"""Library-backed random search via Optuna RandomSampler."""

from __future__ import annotations

from .base import ObjectiveRecorder, SearchSpace
from .optuna_common import optimize_with_sampler


def optimize(
    objective: ObjectiveRecorder,
    search_space: SearchSpace,
    *,
    seed: int,
    n_evaluations: int,
    n_startup: int,
) -> None:
    """Run independent seeded sampling for the complete evaluation budget."""

    del n_startup  # RandomSampler remains random throughout the entire run.

    def sampler_factory(optuna: object) -> object:
        return optuna.samplers.RandomSampler(seed=seed)

    optimize_with_sampler(
        objective,
        search_space,
        sampler_factory,
        n_evaluations=n_evaluations,
    )
