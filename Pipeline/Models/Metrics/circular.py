"""Circular phase-error metrics shared by every estimator.

The formulas and the fewer-than-ten-samples sentinel are intentionally the
same as the working notebooks so results remain directly comparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


EMPTY_METRICS: dict[str, float | int] = {
    "mace": 90.0,
    "plv": 0.0,
    "csd": 180.0,
    "acc": 0.0,
    "bias": 0.0,
    "n_samples": 0,
}


def as_error_array(errors: ArrayLike) -> NDArray[np.float64]:
    """Return phase errors as a flat float64 array in radians."""

    return np.asarray(errors, dtype=np.float64).reshape(-1)


def phase_error_from_components(
    estimated_sin: ArrayLike,
    estimated_cos: ArrayLike,
    ground_truth_sin: ArrayLike,
    ground_truth_cos: ArrayLike,
) -> NDArray[np.float64] | np.float64:
    """Wrapped estimated-minus-ground-truth phase error in radians.

    Accepts scalars or arrays and avoids explicitly reconstructing both phases.
    This is the common error direction used throughout the notebooks.
    """

    return np.arctan2(
        np.asarray(estimated_sin) * np.asarray(ground_truth_cos)
        - np.asarray(estimated_cos) * np.asarray(ground_truth_sin),
        np.asarray(estimated_cos) * np.asarray(ground_truth_cos)
        + np.asarray(estimated_sin) * np.asarray(ground_truth_sin),
    )


def mace_from_errors(errors: ArrayLike, *, min_samples: int = 10) -> float:
    """Mean absolute circular error in degrees."""

    values = as_error_array(errors)
    if values.size < min_samples:
        return 90.0
    return float(np.degrees(np.mean(np.abs(values))))


def errors_to_metrics(
    errors: ArrayLike,
    *,
    min_samples: int = 10,
) -> dict[str, float | int]:
    """Compute the notebook's pooled circular-error metrics.

    Parameters
    ----------
    errors:
        Wrapped estimated-minus-ground-truth phase errors in radians.
    min_samples:
        Match the notebook guard by default. Runs below this count receive the
        sentinel metrics and report zero valid samples.
    """

    values = as_error_array(errors)
    if values.size < min_samples:
        return dict(EMPTY_METRICS)

    mean_abs = float(np.mean(np.abs(values)))
    resultant = np.mean(np.exp(1j * values))
    plv = float(np.abs(resultant))
    return {
        "mace": float(np.degrees(mean_abs)),
        "plv": plv,
        "csd": float(np.degrees(np.sqrt(-2.0 * np.log(max(plv, 1e-12))))),
        "acc": float((1.0 - mean_abs / np.pi) * 100.0),
        "bias": float(np.degrees(np.angle(resultant))),
        "n_samples": int(values.size),
    }


def pooled_errors(
    errors_by_timepoint: Mapping[object, ArrayLike] | Iterable[ArrayLike],
) -> NDArray[np.float64]:
    """Concatenate nonempty per-timepoint errors into one array."""

    arrays = (
        errors_by_timepoint.values()
        if isinstance(errors_by_timepoint, Mapping)
        else errors_by_timepoint
    )
    prepared = [as_error_array(values) for values in arrays]
    nonempty = [values for values in prepared if values.size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)


def metrics_by_timepoint(
    errors_by_timepoint: Mapping[object, ArrayLike],
) -> dict[str, dict[str, float | int]]:
    """Compute metrics while preserving stable string timepoint keys."""

    return {
        str(timepoint): errors_to_metrics(errors)
        for timepoint, errors in errors_by_timepoint.items()
    }
