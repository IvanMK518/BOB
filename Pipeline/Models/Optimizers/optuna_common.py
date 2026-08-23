"""Shared Optuna study plumbing used by both Optuna adapters."""

from __future__ import annotations

from collections.abc import Callable

from .base import ObjectiveRecorder, SearchSpace


def optimize_with_sampler(
    objective: ObjectiveRecorder,
    search_space: SearchSpace,
    sampler_factory: Callable[[object], object],
    *,
    n_evaluations: int,
) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna adapters require optuna; install the requirements") from exc

    sampler = sampler_factory(optuna)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def optuna_objective(trial: object) -> float:
        params: dict[str, int | float] = {}
        for dimension in search_space.dimensions:
            if dimension.kind == "integer":
                params[dimension.name] = trial.suggest_int(
                    dimension.name, int(dimension.low), int(dimension.high)
                )
            else:
                params[dimension.name] = trial.suggest_float(
                    dimension.name, float(dimension.low), float(dimension.high)
                )
        return objective(params)

    study.optimize(optuna_objective, n_trials=n_evaluations, n_jobs=1, show_progress_bar=False)

