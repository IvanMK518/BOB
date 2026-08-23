"""Dependency-light tests for shared metrics and optimizer plumbing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Pipeline.Models.Metrics.circular import (
    errors_to_metrics,
    phase_error_from_components,
)
from Pipeline.Models.Optimizers import (
    ParameterDimension,
    SearchSpace,
    run_optimizer,
)


class CircularMetricsTests(unittest.TestCase):
    def test_error_direction_is_estimated_minus_ground_truth(self) -> None:
        error = phase_error_from_components(1.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(float(error), np.pi / 2.0)

    def test_zero_errors_have_perfect_metrics(self) -> None:
        metrics = errors_to_metrics(np.zeros(10))
        self.assertEqual(metrics["mace"], 0.0)
        self.assertEqual(metrics["plv"], 1.0)
        self.assertEqual(metrics["acc"], 100.0)
        self.assertEqual(metrics["n_samples"], 10)

    def test_notebook_small_sample_sentinel_is_preserved(self) -> None:
        self.assertEqual(errors_to_metrics(np.zeros(9))["mace"], 90.0)
        self.assertEqual(errors_to_metrics(np.zeros(9))["n_samples"], 0)


class OptimizerInterfaceTests(unittest.TestCase):
    def test_optuna_random_search_records_every_evaluation_and_incumbent(self) -> None:
        space = SearchSpace(
            (
                ParameterDimension("integer", 0, 5, "integer"),
                ParameterDimension("real", -1.0, 1.0, "real"),
            )
        )
        sampler_seeds: list[int] = []

        class FakeRandomSampler:
            def __init__(self, *, seed: int) -> None:
                sampler_seeds.append(seed)

        class FakeTrial:
            def __init__(self, index: int) -> None:
                self.index = index

            def suggest_int(self, name: str, low: int, high: int) -> int:
                del name
                return low + self.index % (high - low + 1)

            def suggest_float(self, name: str, low: float, high: float) -> float:
                del name
                fraction = self.index / 10.0
                return low + fraction * (high - low)

        class FakeStudy:
            def optimize(
                study,
                objective: object,
                *,
                n_trials: int,
                n_jobs: int,
                show_progress_bar: bool,
            ) -> None:
                del study
                self.assertEqual(n_jobs, 1)
                self.assertFalse(show_progress_bar)
                for index in range(n_trials):
                    objective(FakeTrial(index))

        fake_optuna = SimpleNamespace(
            samplers=SimpleNamespace(RandomSampler=FakeRandomSampler),
            logging=SimpleNamespace(WARNING=30, set_verbosity=lambda level: None),
            create_study=lambda **kwargs: FakeStudy(),
        )
        with patch.dict("sys.modules", {"optuna": fake_optuna}):
            result = run_optimizer(
                "random_search",
                lambda params: float(
                    (params["integer"] - 2) ** 2 + params["real"] ** 2
                ),
                space,
                seed=42,
                n_evaluations=10,
                n_startup=2,
            )

        self.assertEqual(sampler_seeds, [42])
        self.assertEqual(len(result.history), 10)
        self.assertEqual(
            result.best_objective_mace_deg,
            min(record.objective_mace_deg for record in result.history),
        )
        self.assertEqual(result.history[-1].best_params, result.best_params)


if __name__ == "__main__":
    unittest.main()
