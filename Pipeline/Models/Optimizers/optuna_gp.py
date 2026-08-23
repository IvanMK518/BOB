"""Gaussian-process Bayesian optimization via Optuna GPSampler."""

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
        try:
            return optuna.samplers.GPSampler(
                seed=seed,
                n_startup_trials=n_startup,
                deterministic_objective=True,
            )
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "optuna_gp requires an Optuna version with GPSampler plus SciPy and PyTorch"
            ) from exc

    optimize_with_sampler(
        objective,
        search_space,
        sampler_factory,
        n_evaluations=n_evaluations,
    )

