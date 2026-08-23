"""Tree-structured Parzen estimation via Optuna TPESampler."""

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
    def sampler_factory(optuna: object) -> object:
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
        )

    optimize_with_sampler(
        objective,
        search_space,
        sampler_factory,
        n_evaluations=n_evaluations,
    )

