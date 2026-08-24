"""Plot full-resolution optimizer convergence from a BOB aggregate result.

The calibration curves use every objective evaluation stored in each run's
``history``. Held-out test performance is shown only at the experiment's
checkpoint evaluations so the figure preserves the calibration/test split.
Subject traces are averaged over seeds first; the plotted mean and 95% CI then
use subjects as the independent sampling unit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


OPTIMIZER_LABELS = {
    "skopt_gp": "Skopt GP",
    "optuna_gp": "Optuna GP",
    "optuna_tpe": "Optuna TPE",
    "random_search": "Random search",
}

# Okabe-Ito-derived colors plus distinct line styles remain readable in
# grayscale and for common forms of color-vision deficiency.
OPTIMIZER_STYLES = {
    "skopt_gp": ("#0072B2", "-"),
    "optuna_gp": ("#D55E00", "--"),
    "optuna_tpe": ("#009E73", "-."),
    "random_search": ("#7A5195", ":"),
}


@dataclass(frozen=True, slots=True)
class CurveSummary:
    """Mean curve and 95% confidence interval across subjects."""

    mean: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConvergenceSummary:
    """Plot-ready convergence data extracted from one aggregate JSON."""

    model: str
    band: str
    zero_phase: bool
    n_subjects: int
    n_seeds: int
    n_startup: int
    evaluations: NDArray[np.int64]
    checkpoints: NDArray[np.int64]
    default_test_mace_deg: float
    calibration: dict[str, CurveSummary]
    test_mace: dict[str, CurveSummary]
    cumulative_seconds: dict[str, CurveSummary]


def _curve_summary(subject_curves: list[NDArray[np.float64]]) -> CurveSummary:
    values = np.asarray(subject_curves, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"Expected a nonempty subject-by-evaluation matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Convergence data contain NaN or infinite values")

    mean = np.mean(values, axis=0)
    if values.shape[0] == 1:
        margin = np.zeros_like(mean)
    else:
        standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        margin = 1.96 * standard_error
    return CurveSummary(mean=mean, lower=mean - margin, upper=mean + margin)


def extract_convergence(payload: dict[str, Any]) -> ConvergenceSummary:
    """Validate and aggregate every optimizer iteration in ``payload``."""

    signature = payload.get("run_signature")
    subjects = payload.get("subjects")
    if not isinstance(signature, dict) or not isinstance(subjects, list) or not subjects:
        raise ValueError("Input is not a populated BOB aggregate result JSON")

    optimizers = tuple(signature.get("optimizers", ()))
    seeds = tuple(int(seed) for seed in signature.get("seeds", ()))
    n_evaluations = int(signature.get("n_evaluations_total", 0))
    checkpoints = tuple(int(value) for value in signature.get("checkpoint_evaluations", ()))
    if not optimizers or not seeds or n_evaluations <= 0 or not checkpoints:
        raise ValueError("Run signature is missing optimizers, seeds, or evaluation counts")

    calibration: dict[str, CurveSummary] = {}
    test_mace: dict[str, CurveSummary] = {}
    cumulative_seconds: dict[str, CurveSummary] = {}

    for optimizer in optimizers:
        subject_calibration: list[NDArray[np.float64]] = []
        subject_test: list[NDArray[np.float64]] = []
        subject_seconds: list[NDArray[np.float64]] = []

        for subject in subjects:
            subject_name = subject.get("subject", {}).get("name", "unknown subject")
            runs = subject.get("optimizer_runs", {}).get(optimizer)
            if not isinstance(runs, dict):
                raise ValueError(f"{subject_name} is missing optimizer {optimizer}")

            seed_calibration: list[NDArray[np.float64]] = []
            seed_test: list[NDArray[np.float64]] = []
            seed_seconds: list[NDArray[np.float64]] = []
            for seed in seeds:
                run = runs.get(str(seed))
                if not isinstance(run, dict):
                    raise ValueError(f"{subject_name}, {optimizer} is missing seed {seed}")

                history = run.get("history", [])
                if len(history) != n_evaluations:
                    raise ValueError(
                        f"{subject_name}, {optimizer}, seed {seed}: expected "
                        f"{n_evaluations} history rows, found {len(history)}"
                    )
                recorded_evaluations = [int(row["evaluation"]) for row in history]
                if recorded_evaluations != list(range(1, n_evaluations + 1)):
                    raise ValueError(
                        f"{subject_name}, {optimizer}, seed {seed}: history is not contiguous"
                    )

                seed_calibration.append(
                    np.asarray(
                        [row["best_objective_mace_deg"] for row in history],
                        dtype=np.float64,
                    )
                )
                seed_seconds.append(
                    np.asarray(
                        [row["cumulative_seconds"] for row in history],
                        dtype=np.float64,
                    )
                )

                checkpoint_rows = {
                    int(row["evaluation"]): row for row in run.get("checkpoints", [])
                }
                missing_checkpoints = set(checkpoints).difference(checkpoint_rows)
                if missing_checkpoints:
                    raise ValueError(
                        f"{subject_name}, {optimizer}, seed {seed}: missing checkpoints "
                        f"{sorted(missing_checkpoints)}"
                    )
                seed_test.append(
                    np.asarray(
                        [
                            checkpoint_rows[evaluation]["test_metrics_pooled"]["mace"]
                            for evaluation in checkpoints
                        ],
                        dtype=np.float64,
                    )
                )

            # Average repeated optimizer seeds within a subject before computing
            # the group interval, preserving subjects as the independent unit.
            subject_calibration.append(np.mean(seed_calibration, axis=0))
            subject_test.append(np.mean(seed_test, axis=0))
            subject_seconds.append(np.mean(seed_seconds, axis=0))

        calibration[optimizer] = _curve_summary(subject_calibration)
        test_mace[optimizer] = _curve_summary(subject_test)
        cumulative_seconds[optimizer] = _curve_summary(subject_seconds)

    default_values = np.asarray(
        [subject["default"]["test_metrics_pooled"]["mace"] for subject in subjects],
        dtype=np.float64,
    )
    if not np.isfinite(default_values).all():
        raise ValueError("Default test MACE contains NaN or infinite values")

    return ConvergenceSummary(
        model=str(signature.get("model", "BOB")),
        band=str(signature.get("band", "unknown")),
        zero_phase=bool(signature.get("zero_phase", False)),
        n_subjects=len(subjects),
        n_seeds=len(seeds),
        n_startup=int(signature.get("n_startup_evaluations", 0)),
        evaluations=np.arange(1, n_evaluations + 1, dtype=np.int64),
        checkpoints=np.asarray(checkpoints, dtype=np.int64),
        default_test_mace_deg=float(np.mean(default_values)),
        calibration=calibration,
        test_mace=test_mace,
        cumulative_seconds=cumulative_seconds,
    )


def load_convergence(path: str | Path) -> ConvergenceSummary:
    """Load and aggregate a BOB result JSON."""

    input_path = Path(path).expanduser().resolve()
    with input_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return extract_convergence(payload)


def plot_convergence(summary: ConvergenceSummary, output_path: str | Path, *, dpi: int) -> Path:
    """Render calibration, held-out, and timing convergence panels."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Plotting requires matplotlib; install the repository requirements"
        ) from exc

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(3, 1, figsize=(11.0, 11.5), sharex=True)
    phase_label = "zero phase" if summary.zero_phase else "forward pass"
    figure.suptitle(
        f"{summary.model} {summary.band} optimizer convergence ({phase_label})\n"
        f"{summary.n_subjects} subjects × {summary.n_seeds} seeds; shaded bands are 95% CIs across subjects",
        y=0.985,
    )

    for axis in axes:
        axis.axvspan(1, summary.n_startup, color="#B8B8B8", alpha=0.16, linewidth=0)
        axis.axvline(summary.n_startup, color="#777777", linewidth=1.0, linestyle="--")
        axis.grid(axis="y", color="#D6D6D6", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)

    for optimizer, curve in summary.calibration.items():
        color, linestyle = OPTIMIZER_STYLES.get(optimizer, ("#333333", "-"))
        label = OPTIMIZER_LABELS.get(optimizer, optimizer)
        axes[0].fill_between(
            summary.evaluations, curve.lower, curve.upper, color=color, alpha=0.10
        )
        axes[0].plot(
            summary.evaluations,
            curve.mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    axes[0].set_title("A. Calibration incumbent at every optimizer evaluation", loc="left")
    axes[0].set_ylabel("Best calibration MACE (°)")
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    axes[0].text(
        max(1, summary.n_startup / 2),
        0.035,
        "startup",
        color="#666666",
        ha="center",
        va="bottom",
        transform=axes[0].get_xaxis_transform(),
    )

    for optimizer, curve in summary.test_mace.items():
        color, linestyle = OPTIMIZER_STYLES.get(optimizer, ("#333333", "-"))
        axes[1].fill_between(
            summary.checkpoints, curve.lower, curve.upper, color=color, alpha=0.10
        )
        axes[1].plot(
            summary.checkpoints,
            curve.mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )
    axes[1].axhline(
        summary.default_test_mace_deg,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=1.4,
    )
    axes[1].text(
        summary.evaluations[-1],
        summary.default_test_mace_deg,
        f"  default {summary.default_test_mace_deg:.2f}°",
        color="#555555",
        ha="right",
        va="bottom",
    )
    axes[1].set_title("B. Held-out test performance at 10-evaluation checkpoints", loc="left")
    axes[1].set_ylabel("Test MACE (°)")

    for optimizer, curve in summary.cumulative_seconds.items():
        color, linestyle = OPTIMIZER_STYLES.get(optimizer, ("#333333", "-"))
        axes[2].fill_between(
            summary.evaluations, curve.lower, curve.upper, color=color, alpha=0.10
        )
        axes[2].plot(
            summary.evaluations,
            curve.mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
    axes[2].set_title("C. Cumulative optimizer time at every evaluation", loc="left")
    axes[2].set_ylabel("Cumulative time (s)")
    axes[2].set_xlabel("Objective evaluations")
    axes[2].set_xlim(1, int(summary.evaluations[-1]))
    tick_step = 10 if summary.evaluations[-1] <= 120 else 20
    axes[2].set_xticks(
        np.unique(
            np.concatenate(
                (
                    np.asarray([1], dtype=np.int64),
                    np.arange(tick_step, summary.evaluations[-1] + 1, tick_step),
                )
            )
        )
    )

    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.95))
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot full-resolution optimizer convergence from a BOB aggregate JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Aggregate BOB JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG, PDF, or SVG")
    parser.add_argument("--dpi", type=int, default=180, help="Raster output resolution")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    summary = load_convergence(args.input)
    output = plot_convergence(summary, args.output, dpi=args.dpi)
    print(f"Saved convergence figure -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
