"""Gaussian-process Bayesian optimization via scikit-optimize."""

from __future__ import annotations

from .base import ObjectiveRecorder, SearchSpace


def optimize(
    objective: ObjectiveRecorder,
    search_space: SearchSpace,
    *,
    seed: int,
    n_evaluations: int,
    n_startup: int,
) -> None:
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real
    except ImportError as exc:
        raise RuntimeError(
            "skopt_gp requires scikit-optimize; install the repository requirements"
        ) from exc

    dimensions = [
        (
            Integer(int(dimension.low), int(dimension.high), name=dimension.name)
            if dimension.kind == "integer"
            else Real(float(dimension.low), float(dimension.high), name=dimension.name)
        )
        for dimension in search_space.dimensions
    ]

    def skopt_objective(values: list[int | float]) -> float:
        return objective(search_space.from_sequence(values))

    gp_minimize(
        skopt_objective,
        dimensions,
        n_calls=n_evaluations,
        n_initial_points=n_startup,
        random_state=seed,
        verbose=False,
    )

