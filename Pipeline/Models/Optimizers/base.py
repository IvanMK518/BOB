"""Common optimizer interface and trial-history recording."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Literal, TypeAlias

import numpy as np


ParameterValue: TypeAlias = int | float
ParameterDict: TypeAlias = dict[str, ParameterValue]
Objective: TypeAlias = Callable[[ParameterDict], float]
DimensionKind = Literal["integer", "real"]

OPTIMIZER_NAMES = ("skopt_gp", "optuna_gp", "optuna_tpe", "random_search")


@dataclass(frozen=True, slots=True)
class ParameterDimension:
    name: str
    low: ParameterValue
    high: ParameterValue
    kind: DimensionKind

    def cast(self, value: ParameterValue) -> ParameterValue:
        return int(value) if self.kind == "integer" else float(value)


@dataclass(frozen=True, slots=True)
class SearchSpace:
    dimensions: tuple[ParameterDimension, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    def from_sequence(self, values: Sequence[ParameterValue]) -> ParameterDict:
        if len(values) != len(self.dimensions):
            raise ValueError(
                f"Expected {len(self.dimensions)} parameters, received {len(values)}"
            )
        return {
            dimension.name: dimension.cast(value)
            for dimension, value in zip(self.dimensions, values, strict=True)
        }

    def normalize(self, values: Mapping[str, ParameterValue]) -> ParameterDict:
        missing = set(self.names).difference(values)
        if missing:
            raise ValueError(f"Missing optimizer parameters: {sorted(missing)}")
        return {
            dimension.name: dimension.cast(values[dimension.name])
            for dimension in self.dimensions
        }

    def sample(self, rng: np.random.Generator) -> ParameterDict:
        sampled: ParameterDict = {}
        for dimension in self.dimensions:
            if dimension.kind == "integer":
                sampled[dimension.name] = int(
                    rng.integers(int(dimension.low), int(dimension.high) + 1)
                )
            else:
                sampled[dimension.name] = float(
                    rng.uniform(float(dimension.low), float(dimension.high))
                )
        return sampled

    def as_bounds(self) -> dict[str, list[ParameterValue]]:
        return {
            dimension.name: [dimension.low, dimension.high]
            for dimension in self.dimensions
        }


@dataclass(slots=True)
class TrialRecord:
    evaluation: int
    params: ParameterDict
    objective_mace_deg: float
    best_objective_mace_deg: float
    best_params: ParameterDict
    evaluation_seconds: float
    cumulative_seconds: float


@dataclass(slots=True)
class OptimizationRun:
    optimizer: str
    seed: int
    n_evaluations: int
    n_startup: int
    best_params: ParameterDict
    best_objective_mace_deg: float
    elapsed_seconds: float
    history: list[TrialRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer,
            "seed": self.seed,
            "n_evaluations": self.n_evaluations,
            "n_startup": self.n_startup,
            "best_params": self.best_params,
            "best_objective_mace_deg": self.best_objective_mace_deg,
            "elapsed_seconds": self.elapsed_seconds,
            "history": [asdict(record) for record in self.history],
        }


class ObjectiveRecorder:
    """Wrap an objective and retain every proposal plus its incumbent."""

    def __init__(self, objective: Objective, search_space: SearchSpace) -> None:
        self._objective = objective
        self._search_space = search_space
        self._started = perf_counter()
        self.history: list[TrialRecord] = []
        self.best_value = float("inf")
        self.best_params: ParameterDict | None = None

    def __call__(self, params: Mapping[str, ParameterValue]) -> float:
        normalized = self._search_space.normalize(params)
        evaluation_started = perf_counter()
        value = float(self._objective(normalized))
        duration = perf_counter() - evaluation_started
        if not np.isfinite(value):
            value = 90.0

        if value < self.best_value or self.best_params is None:
            self.best_value = value
            self.best_params = dict(normalized)

        self.history.append(
            TrialRecord(
                evaluation=len(self.history) + 1,
                params=dict(normalized),
                objective_mace_deg=value,
                best_objective_mace_deg=self.best_value,
                best_params=dict(self.best_params),
                evaluation_seconds=duration,
                cumulative_seconds=perf_counter() - self._started,
            )
        )
        return value


def _validate_budget(n_evaluations: int, n_startup: int) -> None:
    if n_evaluations <= 0:
        raise ValueError("n_evaluations must be positive")
    if n_startup < 0 or n_startup > n_evaluations:
        raise ValueError("n_startup must be between zero and n_evaluations")


def run_optimizer(
    optimizer: str,
    objective: Objective,
    search_space: SearchSpace,
    *,
    seed: int,
    n_evaluations: int,
    n_startup: int,
) -> OptimizationRun:
    """Run one adapter behind a uniform interface."""

    _validate_budget(n_evaluations, n_startup)
    if optimizer not in OPTIMIZER_NAMES:
        raise ValueError(f"Unknown optimizer {optimizer!r}; choose from {OPTIMIZER_NAMES}")

    if optimizer == "skopt_gp":
        from .skopt_gp import optimize
    elif optimizer == "optuna_gp":
        from .optuna_gp import optimize
    elif optimizer == "optuna_tpe":
        from .optuna_tpe import optimize
    else:
        from .random_search import optimize

    recorder = ObjectiveRecorder(objective, search_space)
    started = perf_counter()
    optimize(
        recorder,
        search_space,
        seed=seed,
        n_evaluations=n_evaluations,
        n_startup=n_startup,
    )
    elapsed = perf_counter() - started

    if len(recorder.history) != n_evaluations or recorder.best_params is None:
        raise RuntimeError(
            f"{optimizer} recorded {len(recorder.history)} of {n_evaluations} evaluations"
        )
    return OptimizationRun(
        optimizer=optimizer,
        seed=seed,
        n_evaluations=n_evaluations,
        n_startup=n_startup,
        best_params=recorder.best_params,
        best_objective_mace_deg=recorder.best_value,
        elapsed_seconds=elapsed,
        history=recorder.history,
    )
