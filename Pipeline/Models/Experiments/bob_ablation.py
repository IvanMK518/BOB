"""Run the four-way BOB optimizer ablation locally.

Example
-------
Run all four optimizers for theta in the original forward-only mode::

    python -m Pipeline.Models.Experiments.bob_ablation \
        --band theta --zero-phase false

The default evaluation budget is 100 total objective calls, of which the first
20 are exploratory/startup calls, matching the supplied notebook. Every run is
repeated for seeds 42--46 and held-out performance is evaluated every 10 calls.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

# Optuna GPSampler uses PyTorch internally. This must be set before PyTorch is
# imported and prevents each process from creating a large OpenMP thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from Pipeline.data_loader import load_subject, subject_paths
from Pipeline.Models.bob import BOBModel, bob_config_for_band, timepoint_key
from Pipeline.Models.Metrics.circular import errors_to_metrics, pooled_errors
from Pipeline.Models.Metrics.descriptors import (
    band_snr,
    ground_truth_itpc,
    spectral_peak,
)
from Pipeline.Models.Optimizers import OPTIMIZER_NAMES, run_optimizer


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "Results" / "Raw" / "BOB"


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    band: str
    zero_phase: bool
    optimizers: tuple[str, ...]
    seeds: tuple[int, ...]
    n_evaluations: int
    n_startup: int
    checkpoint_interval: int
    horizon_ms: float

    @property
    def checkpoints(self) -> tuple[int, ...]:
        regular = list(
            range(self.checkpoint_interval, self.n_evaluations + 1, self.checkpoint_interval)
        )
        if not regular or regular[-1] != self.n_evaluations:
            regular.append(self.n_evaluations)
        return tuple(regular)

    def signature(self) -> dict[str, Any]:
        model_config = bob_config_for_band(self.band)
        return {
            "schema_version": 1,
            "model": "BOB",
            "band": self.band,
            "zero_phase": self.zero_phase,
            "horizon_ms": self.horizon_ms,
            "optimizers": list(self.optimizers),
            "seeds": list(self.seeds),
            "n_evaluations_total": self.n_evaluations,
            "n_startup_evaluations": self.n_startup,
            "checkpoint_evaluations": list(self.checkpoints),
            "model_config": model_config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SubjectJob:
    source_path: str
    settings: ExperimentSettings
    condition_dir: str


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "scipy", "scikit-optimize", "optuna", "torch")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _subject_signature(path: Path, settings: ExperimentSettings) -> dict[str, Any]:
    stat = path.stat()
    signature = settings.signature()
    signature["source_file"] = {
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }
    return signature


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _params_cache_key(params: dict[str, int | float]) -> tuple[int | float, ...]:
    ordered_names = (
        "ar_order",
        "context_s",
        "filt_low",
        "filt_high",
        "filt_order",
        "hilbert_pad",
    )
    return tuple(params[name] for name in ordered_names)


def _run_subject(job: SubjectJob) -> dict[str, Any]:
    source_path = Path(job.source_path)
    settings = job.settings
    condition_dir = Path(job.condition_dir)
    subject = load_subject(source_path, settings.band)
    band_config = bob_config_for_band(settings.band)
    model = BOBModel(
        band_config,
        fs=subject.fs,
        onset_sample=subject.onset_sample,
        zero_phase=settings.zero_phase,
        horizon_ms=settings.horizon_ms,
    )

    invalid_samples = [
        sample
        for sample in model.evaluation_samples
        if sample + model.horizon_samples >= subject.n_samples
    ]
    if invalid_samples:
        raise ValueError(
            f"{subject.name}: evaluation samples {invalid_samples} exceed epoch length "
            f"{subject.n_samples}"
        )

    subject_started = perf_counter()
    print(f"[{subject.name}] preparing ground truth", flush=True)
    ground_truth = model.prepare_ground_truth(subject.epochs)

    raw_arrays: dict[str, np.ndarray] = {}
    raw_manifest: dict[str, dict[str, Any]] = {}

    def store_errors(
        prefix: str,
        errors_by_timepoint: dict[float, np.ndarray],
        *,
        split: str,
        evaluation: int | None,
    ) -> list[str]:
        keys: list[str] = []
        for timepoint, errors in errors_by_timepoint.items():
            key = f"{prefix}__t_{timepoint_key(timepoint)}"
            raw_arrays[key] = np.asarray(errors, dtype=np.float64)
            raw_manifest[key] = {
                "split": split,
                "evaluation": evaluation,
                "timepoint_s": timepoint,
                "units": "radians",
                "error_direction": "estimated_minus_ground_truth_wrapped",
                "n_samples": int(len(errors)),
            }
            keys.append(key)
        return keys

    default_errors = model.errors_by_timepoint(
        subject.epochs,
        ground_truth,
        subject.test_indices,
        band_config.default_parameters,
    )
    default_raw_keys = store_errors(
        "default",
        default_errors,
        split="test",
        evaluation=None,
    )
    default_metrics_by_timepoint = {
        str(timepoint): errors_to_metrics(errors)
        for timepoint, errors in default_errors.items()
    }

    test_cache: dict[tuple[int | float, ...], dict[float, np.ndarray]] = {}

    def test_errors(params: dict[str, int | float]) -> dict[float, np.ndarray]:
        cache_key = _params_cache_key(params)
        if cache_key not in test_cache:
            test_cache[cache_key] = model.errors_by_timepoint(
                subject.epochs,
                ground_truth,
                subject.test_indices,
                params,
            )
        return test_cache[cache_key]

    optimizer_runs: dict[str, dict[str, Any]] = {}
    for optimizer in settings.optimizers:
        seed_runs: dict[str, Any] = {}
        for seed in settings.seeds:
            print(f"[{subject.name}] {optimizer}, seed {seed}", flush=True)

            def objective(params: dict[str, int | float]) -> float:
                return model.objective_mace(
                    subject.epochs,
                    ground_truth,
                    subject.cal_indices,
                    params,
                )

            optimization = run_optimizer(
                optimizer,
                objective,
                band_config.search_space,
                seed=seed,
                n_evaluations=settings.n_evaluations,
                n_startup=settings.n_startup,
            )
            run_payload = optimization.to_dict()
            checkpoint_payloads: list[dict[str, Any]] = []
            for checkpoint in settings.checkpoints:
                incumbent = optimization.history[checkpoint - 1]
                errors = test_errors(incumbent.best_params)
                raw_prefix = f"{optimizer}__seed_{seed}__eval_{checkpoint:03d}"
                raw_keys = store_errors(
                    raw_prefix,
                    errors,
                    split="test",
                    evaluation=checkpoint,
                )
                checkpoint_payloads.append(
                    {
                        "evaluation": checkpoint,
                        "best_params": incumbent.best_params,
                        "calibration_incumbent_mace_deg": (
                            incumbent.best_objective_mace_deg
                        ),
                        "optimizer_cumulative_seconds": incumbent.cumulative_seconds,
                        "test_metrics_pooled": errors_to_metrics(pooled_errors(errors)),
                        "test_metrics_per_timepoint": {
                            str(timepoint): errors_to_metrics(values)
                            for timepoint, values in errors.items()
                        },
                        "raw_error_keys": raw_keys,
                    }
                )

            final_checkpoint = checkpoint_payloads[-1]
            run_payload.update(
                {
                    "checkpoints": checkpoint_payloads,
                    "final_test_metrics_pooled": final_checkpoint[
                        "test_metrics_pooled"
                    ],
                    "final_test_metrics_per_timepoint": final_checkpoint[
                        "test_metrics_per_timepoint"
                    ],
                    "final_raw_error_keys": final_checkpoint["raw_error_keys"],
                }
            )
            seed_runs[str(seed)] = run_payload
        optimizer_runs[optimizer] = seed_runs

    itpc_by_timepoint = ground_truth_itpc(
        ground_truth.sin,
        ground_truth.cos,
        subject.test_indices,
        model.evaluation_samples,
        band_config.evaluation_times_s,
    )
    descriptor_snr = (
        subject.snr
        if subject.snr is not None
        else band_snr(
            subject.epochs,
            subject.fs,
            band_config.ground_truth_band_hz,
        )
    )
    peak_hz = spectral_peak(
        subject.epochs,
        subject.fs,
        band_config.peak_search_band_hz,
        start_sample=subject.onset_sample,
    )

    raw_path = condition_dir / "raw_errors" / f"{subject.name}.npz"
    subject_json_path = condition_dir / "subjects" / f"{subject.name}.json"
    result = {
        "run_signature": _subject_signature(source_path, settings),
        "subject": {
            "name": subject.name,
            "source_path": str(subject.source_path),
            "band": subject.band,
            "fs": subject.fs,
            "channel_name": subject.channel_name,
            "channel_index": subject.channel_index,
            "n_channels": subject.n_channels,
            "n_trials": subject.n_trials,
            "n_cal": subject.n_cal,
            "n_test": subject.n_test,
            "onset_sample": subject.onset_sample,
        },
        "descriptors": {
            f"{settings.band}_snr": descriptor_snr,
            f"{settings.band}_peak_hz": peak_hz,
            f"{settings.band}_itpc_per_timepoint": itpc_by_timepoint,
            f"{settings.band}_itpc_mean": float(
                np.mean(list(itpc_by_timepoint.values()))
            ),
        },
        "default": {
            "params": band_config.default_parameters.to_dict(),
            "test_metrics_pooled": errors_to_metrics(pooled_errors(default_errors)),
            "test_metrics_per_timepoint": default_metrics_by_timepoint,
            "raw_error_keys": default_raw_keys,
        },
        "optimizer_runs": optimizer_runs,
        "raw_errors": {
            "file": str(raw_path.relative_to(condition_dir).as_posix()),
            "format": "numpy_compressed_npz",
            "arrays": raw_manifest,
        },
        "elapsed_seconds": perf_counter() - subject_started,
    }
    _write_npz_atomic(raw_path, raw_arrays)
    _write_json_atomic(subject_json_path, result)
    return result


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not finite.size:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "n": int(finite.size),
    }


def _aggregate_results(
    results: list[dict[str, Any]],
    settings: ExperimentSettings,
) -> dict[str, Any]:
    convergence: dict[str, dict[str, Any]] = {}
    variability: dict[str, dict[str, Any]] = {}
    for optimizer in settings.optimizers:
        optimizer_convergence: dict[str, Any] = {}
        for checkpoint in settings.checkpoints:
            calibration: list[float] = []
            optimizer_seconds: list[float] = []
            metrics: dict[str, list[float]] = {
                "mace": [],
                "plv": [],
                "csd": [],
                "acc": [],
                "bias": [],
            }
            for result in results:
                for seed in settings.seeds:
                    run = result["optimizer_runs"][optimizer][str(seed)]
                    checkpoint_result = next(
                        row
                        for row in run["checkpoints"]
                        if row["evaluation"] == checkpoint
                    )
                    calibration.append(
                        checkpoint_result["calibration_incumbent_mace_deg"]
                    )
                    optimizer_seconds.append(
                        checkpoint_result["optimizer_cumulative_seconds"]
                    )
                    test_metrics = checkpoint_result["test_metrics_pooled"]
                    for metric_name in metrics:
                        metrics[metric_name].append(test_metrics[metric_name])
            optimizer_convergence[str(checkpoint)] = {
                "calibration_incumbent_mace_deg": _summary(calibration),
                "optimizer_cumulative_seconds": _summary(optimizer_seconds),
                "test_metrics_pooled": {
                    metric_name: _summary(values)
                    for metric_name, values in metrics.items()
                },
            }
        convergence[optimizer] = optimizer_convergence

        per_subject_seed_stds: list[float] = []
        final_maces: list[float] = []
        per_seed_group_means: list[float] = []
        for result in results:
            subject_maces = [
                result["optimizer_runs"][optimizer][str(seed)][
                    "final_test_metrics_pooled"
                ]["mace"]
                for seed in settings.seeds
            ]
            final_maces.extend(subject_maces)
            per_subject_seed_stds.append(
                float(np.std(subject_maces, ddof=1)) if len(subject_maces) > 1 else 0.0
            )
        for seed in settings.seeds:
            per_seed_group_means.append(
                float(
                    np.mean(
                        [
                            result["optimizer_runs"][optimizer][str(seed)][
                                "final_test_metrics_pooled"
                            ]["mace"]
                            for result in results
                        ]
                    )
                )
            )
        variability[optimizer] = {
            "final_test_mace_deg_all_subject_seed_runs": _summary(final_maces),
            "within_subject_seed_std_deg": _summary(per_subject_seed_stds),
            "per_seed_group_mean_mace_deg": {
                str(seed): mean
                for seed, mean in zip(
                    settings.seeds, per_seed_group_means, strict=True
                )
            },
            "between_seed_group_mean_std_deg": (
                float(np.std(per_seed_group_means, ddof=1))
                if len(per_seed_group_means) > 1
                else 0.0
            ),
        }

    default_maces = [result["default"]["test_metrics_pooled"]["mace"] for result in results]
    return {
        "n_subjects": len(results),
        "default_test_mace_deg": _summary(default_maces),
        "optimizer_convergence": convergence,
        "optimizer_seed_variability": variability,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _run_condition(
    paths: tuple[Path, ...],
    settings: ExperimentSettings,
    output_root: Path,
    *,
    workers: int,
    overwrite: bool,
) -> Path:
    phase_label = str(settings.zero_phase).lower()
    condition_dir = output_root / settings.band / f"zero_phase_{phase_label}"
    subject_dir = condition_dir / "subjects"
    raw_dir = condition_dir / "raw_errors"
    subject_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict[str, Any]] = {}
    jobs: list[SubjectJob] = []
    for path in paths:
        json_path = subject_dir / f"{path.stem}.json"
        raw_path = raw_dir / f"{path.stem}.npz"
        expected_signature = _subject_signature(path, settings)
        if json_path.exists() and raw_path.exists() and not overwrite:
            prior = _read_json(json_path)
            if prior.get("run_signature") != expected_signature:
                raise RuntimeError(
                    f"Existing result has a different configuration: {json_path}\n"
                    "Use --overwrite to replace the selected subject result."
                )
            completed[path.stem] = prior
            print(f"[{path.stem}] resumed from completed result", flush=True)
        else:
            jobs.append(SubjectJob(str(path), settings, str(condition_dir)))

    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            result = _run_subject(job)
            completed[result["subject"]["name"]] = result
            print(
                f"Completed {index}/{len(jobs)} pending subjects: "
                f"{result['subject']['name']}",
                flush=True,
            )
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_subject, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                completed[result["subject"]["name"]] = result
                print(
                    f"Completed {index}/{len(jobs)} pending subjects: "
                    f"{result['subject']['name']}",
                    flush=True,
                )

    ordered_results = [completed[path.stem] for path in paths]
    aggregate_path = condition_dir / f"bob_{settings.band}_zero_phase_{phase_label}.json"
    aggregate = {
        "run_signature": settings.signature(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "summary": _aggregate_results(ordered_results, settings),
        "subjects": ordered_results,
    }
    _write_json_atomic(aggregate_path, aggregate)
    return aggregate_path


def _select_subjects(
    band: str,
    data_root: Path | None,
    requested: list[str] | None,
    limit: int | None,
) -> tuple[Path, ...]:
    available = subject_paths(band, data_root)
    if requested:
        lookup = {path.stem.lower(): path for path in available}
        missing = [name for name in requested if name.lower() not in lookup]
        if missing:
            raise ValueError(
                f"Unknown {band} subject(s): {missing}. "
                f"Available examples: {[path.stem for path in available[:5]]}"
            )
        selected = tuple(lookup[name.lower()] for name in requested)
    else:
        selected = available
    if limit is not None:
        if limit <= 0:
            raise ValueError("--subject-limit must be positive")
        selected = selected[:limit]
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the modular four-optimizer BOB ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--band", choices=("theta", "alpha", "both"), required=True)
    parser.add_argument(
        "--zero-phase",
        choices=("true", "false", "both"),
        required=True,
        help="Run the AR-padded backward filter, the original forward pass, or both.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=OPTIMIZER_NAMES,
        default=list(OPTIMIZER_NAMES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=100,
        help="Total objective evaluations per optimizer/seed, including startup.",
    )
    parser.add_argument("--n-startup", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--horizon-ms", type=float, default=0.0)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Processed Data root or a final onset/LEMON directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject stems; omit to use every subject.",
    )
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Subject-level worker processes; start with one for Optuna GP memory safety.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace selected subject results instead of resuming matching files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments/data discovery and print the planned work only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.n_evaluations <= 0:
        raise ValueError("--n-evaluations must be positive")
    if args.n_startup < 0 or args.n_startup > args.n_evaluations:
        raise ValueError("--n-startup must be between zero and --n-evaluations")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if len(set(args.optimizers)) != len(args.optimizers):
        raise ValueError("--optimizers contains duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")

    bands = ("theta", "alpha") if args.band == "both" else (args.band,)
    zero_phase_modes = (
        (False, True)
        if args.zero_phase == "both"
        else (args.zero_phase == "true",)
    )
    output_root = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None

    selections = {
        band: _select_subjects(band, data_root, args.subjects, args.subject_limit)
        for band in bands
    }
    n_runs = sum(len(paths) for paths in selections.values()) * len(zero_phase_modes)
    optimizer_runs_per_subject = len(args.optimizers) * len(args.seeds)
    print(
        f"Plan: {n_runs} subject/phase jobs; {optimizer_runs_per_subject} optimizer-seed "
        f"runs per job; {args.n_evaluations} total evaluations per run "
        f"({args.n_startup} startup).",
        flush=True,
    )
    if args.dry_run:
        for band, paths in selections.items():
            print(f"{band}: {len(paths)} subjects from {paths[0].parent}")
        print(f"Output root: {output_root}")
        return 0

    aggregate_paths: list[Path] = []
    for band in bands:
        for zero_phase in zero_phase_modes:
            settings = ExperimentSettings(
                band=band,
                zero_phase=zero_phase,
                optimizers=tuple(args.optimizers),
                seeds=tuple(args.seeds),
                n_evaluations=args.n_evaluations,
                n_startup=args.n_startup,
                checkpoint_interval=args.checkpoint_interval,
                horizon_ms=args.horizon_ms,
            )
            print(
                f"Starting {band}, zero_phase={zero_phase}, "
                f"{len(selections[band])} subjects",
                flush=True,
            )
            aggregate_paths.append(
                _run_condition(
                    selections[band],
                    settings,
                    output_root,
                    workers=args.workers,
                    overwrite=args.overwrite,
                )
            )

    print("Completed aggregate result files:", flush=True)
    for path in aggregate_paths:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
