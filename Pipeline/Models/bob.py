"""Reusable BOB (Burg Optimization-Based) phase-estimation architecture.

This module contains the signal-processing model only. Optimizer selection,
experiment loops, data loading, and serialization live elsewhere so the exact
same BOB objective can be passed to every optimizer in the ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, filtfilt, firwin, hilbert, lfilter, lfilter_zi

from .Metrics.circular import (
    mace_from_errors,
    phase_error_from_components,
    pooled_errors,
)
from .Optimizers.base import ParameterDimension, ParameterValue, SearchSpace


@dataclass(frozen=True, slots=True)
class BOBParameters:
    ar_order: int
    context_s: float
    filt_low: float
    filt_high: float
    filt_order: int
    hilbert_pad: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, ParameterValue]) -> "BOBParameters":
        return cls(
            ar_order=int(values["ar_order"]),
            context_s=float(values["context_s"]),
            filt_low=float(values["filt_low"]),
            filt_high=float(values["filt_high"]),
            filt_order=int(values["filt_order"]),
            hilbert_pad=int(values["hilbert_pad"]),
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BOBBandConfig:
    band: str
    evaluation_times_s: tuple[float, ...]
    ground_truth_band_hz: tuple[float, float]
    ground_truth_fir_numtaps: int
    peak_search_band_hz: tuple[float, float]
    default_parameters: BOBParameters
    search_space: SearchSpace
    ar_context_divisor: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "evaluation_times_s": list(self.evaluation_times_s),
            "ground_truth_band_hz": list(self.ground_truth_band_hz),
            "ground_truth_fir_numtaps": self.ground_truth_fir_numtaps,
            "peak_search_band_hz": list(self.peak_search_band_hz),
            "default_parameters": self.default_parameters.to_dict(),
            "search_bounds": self.search_space.as_bounds(),
            "ar_context_divisor": self.ar_context_divisor,
        }


@dataclass(frozen=True, slots=True)
class GroundTruthPhase:
    sin: NDArray[np.float32]
    cos: NDArray[np.float32]


def bob_config_for_band(band: str) -> BOBBandConfig:
    """Return the exact theta or alpha notebook defaults and search bounds."""

    normalized = band.strip().lower()
    if normalized in {"theta", "onset"}:
        return BOBBandConfig(
            band="theta",
            evaluation_times_s=(0.3, 0.4, 0.5, 0.6, 0.7),
            ground_truth_band_hz=(4.0, 8.0),
            ground_truth_fir_numtaps=175,
            peak_search_band_hz=(4.0, 8.0),
            default_parameters=BOBParameters(
                ar_order=16,
                context_s=1.0,
                filt_low=4.0,
                filt_high=8.0,
                filt_order=2,
                hilbert_pad=60,
            ),
            search_space=SearchSpace(
                (
                    ParameterDimension("ar_order", 24, 100, "integer"),
                    ParameterDimension("context_s", 0.8, 1.6, "real"),
                    ParameterDimension("filt_low", 4.0, 5.5, "real"),
                    ParameterDimension("filt_high", 6.0, 8.0, "real"),
                    ParameterDimension("filt_order", 1, 4, "integer"),
                    ParameterDimension("hilbert_pad", 30, 100, "integer"),
                )
            ),
        )
    if normalized in {"alpha", "lemon", "mu"}:
        return BOBBandConfig(
            band="alpha",
            evaluation_times_s=(1.5, 2.0, 2.5, 3.0, 3.5),
            ground_truth_band_hz=(9.0, 13.0),
            ground_truth_fir_numtaps=231,
            peak_search_band_hz=(9.0, 13.0),
            default_parameters=BOBParameters(
                ar_order=96,
                context_s=1.5,
                filt_low=9.0,
                filt_high=13.0,
                filt_order=2,
                hilbert_pad=120,
            ),
            search_space=SearchSpace(
                (
                    ParameterDimension("ar_order", 48, 180, "integer"),
                    ParameterDimension("context_s", 0.8, 2.0, "real"),
                    ParameterDimension("filt_low", 9.0, 10.5, "real"),
                    ParameterDimension("filt_high", 11.0, 13.0, "real"),
                    ParameterDimension("filt_order", 1, 4, "integer"),
                    ParameterDimension("hilbert_pad", 60, 180, "integer"),
                )
            ),
        )
    raise ValueError(f"Unknown BOB band {band!r}; expected theta or alpha")


class BOBModel:
    """BOB phase estimator bound to one subject's sampling geometry."""

    def __init__(
        self,
        config: BOBBandConfig,
        *,
        fs: float,
        onset_sample: int,
        zero_phase: bool,
        horizon_ms: float = 0.0,
    ) -> None:
        self.config = config
        self.fs = float(fs)
        self.onset_sample = int(onset_sample)
        self.zero_phase = bool(zero_phase)
        self.horizon_ms = float(horizon_ms)
        self.horizon_samples = int(self.horizon_ms * self.fs / 1000.0)
        self.evaluation_samples = tuple(
            self.onset_sample + int(timepoint * self.fs)
            for timepoint in config.evaluation_times_s
        )

    def prepare_ground_truth(self, epochs: ArrayLike) -> GroundTruthPhase:
        """Compute the notebook's noncausal FIR/filtfilt reference phase once."""

        raw = np.asarray(epochs)
        taps = firwin(
            self.config.ground_truth_fir_numtaps,
            list(self.config.ground_truth_band_hz),
            pass_zero=False,
            fs=self.fs,
        )
        gt_sin = np.empty(raw.shape, dtype=np.float32)
        gt_cos = np.empty(raw.shape, dtype=np.float32)
        for trial, epoch in enumerate(raw):
            filtered = filtfilt(taps, [1.0], np.asarray(epoch, dtype=np.float64))
            phase = np.angle(hilbert(filtered))
            gt_sin[trial] = np.sin(phase).astype(np.float32)
            gt_cos[trial] = np.cos(phase).astype(np.float32)
        return GroundTruthPhase(gt_sin, gt_cos)

    def errors_by_timepoint(
        self,
        epochs: ArrayLike,
        ground_truth: GroundTruthPhase,
        trials: ArrayLike,
        parameters: BOBParameters | Mapping[str, ParameterValue],
    ) -> dict[float, NDArray[np.float64]]:
        """Return wrapped estimated-minus-reference errors in radians."""

        raw = np.asarray(epochs)
        params = (
            parameters
            if isinstance(parameters, BOBParameters)
            else BOBParameters.from_mapping(parameters)
        )
        empty = {
            timepoint: np.empty(0, dtype=np.float64)
            for timepoint in self.config.evaluation_times_s
        }
        try:
            filter_b, filter_a = make_causal_filter(
                params.filt_low,
                params.filt_high,
                params.filt_order,
                self.fs,
            )
        except (TypeError, ValueError):
            return empty

        errors: dict[float, list[float]] = {
            timepoint: [] for timepoint in self.config.evaluation_times_s
        }
        epoch_length = raw.shape[1]
        for trial in np.asarray(trials, dtype=np.int64):
            causal_signal = causal_filter_signal(raw[trial], filter_b, filter_a)
            for timepoint, evaluation_sample in zip(
                self.config.evaluation_times_s,
                self.evaluation_samples,
                strict=True,
            ):
                ground_truth_sample = evaluation_sample + self.horizon_samples
                if evaluation_sample >= epoch_length or ground_truth_sample >= epoch_length:
                    continue
                estimated_sin, estimated_cos = self.predict_phase_components(
                    causal_signal,
                    evaluation_sample,
                    params,
                    filter_b,
                    filter_a,
                )
                gt_sin = ground_truth.sin[trial, ground_truth_sample]
                gt_cos = ground_truth.cos[trial, ground_truth_sample]
                errors[timepoint].append(
                    float(
                        phase_error_from_components(
                            estimated_sin,
                            estimated_cos,
                            gt_sin,
                            gt_cos,
                        )
                    )
                )
        return {
            timepoint: np.asarray(values, dtype=np.float64)
            for timepoint, values in errors.items()
        }

    def objective_mace(
        self,
        epochs: ArrayLike,
        ground_truth: GroundTruthPhase,
        calibration_trials: ArrayLike,
        parameters: BOBParameters | Mapping[str, ParameterValue],
    ) -> float:
        """Shared calibration objective supplied unchanged to every optimizer."""

        errors = self.errors_by_timepoint(
            epochs,
            ground_truth,
            calibration_trials,
            parameters,
        )
        return mace_from_errors(pooled_errors(errors))

    def predict_phase_components(
        self,
        causal_signal: NDArray[np.floating],
        predict_at: int,
        parameters: BOBParameters,
        filter_b: NDArray[np.floating],
        filter_a: NDArray[np.floating],
    ) -> tuple[float, float]:
        """Run Burg extrapolation and return sine/cosine phase components."""

        context_samples = int(parameters.context_s * self.fs)
        start = max(0, predict_at - context_samples + 1)
        context = causal_signal[start : predict_at + 1].astype(np.float64)
        ar_order = min(
            int(parameters.ar_order),
            len(context) // self.config.ar_context_divisor,
        )
        if len(context) < ar_order + 5 or np.std(context) < 1e-10:
            return 0.0, 1.0

        extrapolated = ar_extrapolate(
            context,
            burg_ar_coefficients(context, ar_order),
            self.horizon_samples + int(parameters.hilbert_pad),
        )

        # The optional backward half uses only the AR-generated look-ahead. It
        # therefore reproduces the alpha notebook's zero-phase mode without
        # introducing real future samples, and is now available to theta too.
        if self.zero_phase:
            initial_state = lfilter_zi(filter_b, filter_a) * extrapolated[-1]
            reversed_signal, _ = lfilter(
                filter_b,
                filter_a,
                extrapolated[::-1],
                zi=initial_state,
            )
            extrapolated = reversed_signal[::-1]

        target = len(context) - 1 + self.horizon_samples
        if target >= len(extrapolated):
            return 0.0, 1.0
        phase = np.angle(hilbert(extrapolated)[target])
        return float(np.sin(phase)), float(np.cos(phase))


def make_causal_filter(
    low_hz: float,
    high_hz: float,
    order: int,
    fs: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the notebook's clamped Butterworth band-pass filter."""

    nyquist = fs / 2.0
    low = max(0.5, min(float(low_hz), nyquist - 1.0))
    high = max(low + 0.5, min(float(high_hz), nyquist - 0.5))
    return butter(int(order), [low / nyquist, high / nyquist], btype="band")


def causal_filter_signal(
    signal: ArrayLike,
    filter_b: NDArray[np.floating],
    filter_a: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Single forward causal pass with the notebook's warm start."""

    values = np.asarray(signal)
    initial_state = lfilter_zi(filter_b, filter_a) * values[0]
    filtered, _ = lfilter(filter_b, filter_a, values, zi=initial_state)
    return filtered.astype(np.float32)


def burg_ar_coefficients(values: ArrayLike, order: int) -> NDArray[np.float64]:
    """Notebook-equivalent Burg recursion returning prediction coefficients."""

    signal = np.asarray(values, dtype=np.float64)
    n_values = len(signal)
    order = int(order)
    if n_values <= order + 1:
        return np.zeros(order)

    forward_error = signal.copy()
    backward_error = signal.copy()
    coefficients = np.zeros(order)
    for iteration in range(order):
        forward_slice = forward_error[iteration + 1 : n_values]
        backward_slice = backward_error[iteration : n_values - 1]
        length = min(len(forward_slice), len(backward_slice))
        if length <= 0:
            break
        forward_slice = forward_slice[:length].copy()
        backward_slice = backward_slice[:length].copy()
        numerator = -2.0 * np.dot(forward_slice, backward_slice)
        denominator = np.dot(forward_slice, forward_slice) + np.dot(
            backward_slice, backward_slice
        )
        if denominator < 1e-12:
            break
        reflection = np.clip(numerator / denominator, -0.9999, 0.9999)
        updated = np.zeros(iteration + 1)
        updated[iteration] = reflection
        if iteration > 0:
            updated[:iteration] = (
                coefficients[:iteration]
                + reflection * coefficients[iteration - 1 :: -1]
            )
        coefficients[: iteration + 1] = updated
        forward_error[iteration + 1 : iteration + 1 + length] = (
            forward_slice + reflection * backward_slice
        )
        backward_error[iteration + 1 : iteration + 1 + length] = (
            backward_slice + reflection * forward_slice
        )
        n_values = iteration + 1 + length
    return -coefficients


def ar_extrapolate(
    values: ArrayLike,
    coefficients: ArrayLike,
    n_steps: int,
) -> NDArray[np.float64]:
    """Append autoregressive predictions to the observed context."""

    signal = np.asarray(values, dtype=np.float64)
    ar_coefficients = np.asarray(coefficients, dtype=np.float64)
    order = len(ar_coefficients)
    output = np.empty(len(signal) + int(n_steps), dtype=np.float64)
    output[: len(signal)] = signal
    for index in range(len(signal), len(output)):
        output[index] = (
            output[index - 1]
            if index < order
            else np.dot(ar_coefficients, output[index - order : index][::-1])
        )
    return output


def timepoint_key(seconds: float) -> str:
    """Stable filesystem/NPZ label such as ``300ms`` or ``1500ms``."""

    return f"{int(round(seconds * 1000.0))}ms"
