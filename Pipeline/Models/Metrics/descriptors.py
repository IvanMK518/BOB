"""Dataset descriptors shared by model experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import welch


def band_snr(
    epochs: ArrayLike,
    fs: float,
    band: tuple[float, float],
    *,
    broadband: tuple[float, float] = (1.0, 40.0),
) -> float:
    """Mean per-epoch FFT band-power to broadband-power ratio."""

    raw = np.asarray(epochs)
    ratios: list[float] = []
    for epoch in raw:
        values = np.asarray(epoch, dtype=np.float64)
        psd = np.abs(np.fft.rfft(values)) ** 2
        frequencies = np.fft.rfftfreq(values.size, 1.0 / fs)
        selected = (frequencies >= band[0]) & (frequencies <= band[1])
        broad = (frequencies >= broadband[0]) & (frequencies <= broadband[1])
        numerator = float(np.mean(psd[selected]))
        denominator = float(np.mean(psd[broad]))
        ratios.append(numerator / denominator if denominator > 0 else 0.0)
    return float(np.mean(ratios))


def ground_truth_itpc(
    gt_sin: NDArray[np.floating],
    gt_cos: NDArray[np.floating],
    trials: ArrayLike,
    evaluation_samples: Sequence[int],
    evaluation_times: Sequence[float],
) -> dict[str, float]:
    """Ground-truth phase ITPC for the selected trials at each timepoint."""

    indices = np.asarray(trials, dtype=np.int64)
    values: dict[str, float] = {}
    for timepoint, sample in zip(evaluation_times, evaluation_samples, strict=True):
        phase = np.arctan2(gt_sin[indices, sample], gt_cos[indices, sample])
        values[str(timepoint)] = float(np.abs(np.mean(np.exp(1j * phase))))
    return values


def spectral_peak(
    epochs: ArrayLike,
    fs: float,
    band: tuple[float, float],
    *,
    start_sample: int = 0,
) -> float | None:
    """Notebook-equivalent mean-Welch spectral peak within ``band``."""

    raw = np.asarray(epochs)
    segment = raw[:, start_sample:]
    if segment.shape[1] < 16:
        return None

    nfft = int(2 ** np.ceil(np.log2(segment.shape[1] * 4)))
    mean_psd: NDArray[np.float64] | None = None
    frequencies: NDArray[np.float64] | None = None
    for epoch in segment:
        frequencies, power = welch(
            np.asarray(epoch, dtype=np.float64),
            fs=fs,
            nperseg=segment.shape[1],
            nfft=nfft,
            detrend="linear",
        )
        mean_psd = power if mean_psd is None else mean_psd + power

    assert mean_psd is not None and frequencies is not None
    mean_psd /= segment.shape[0]
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    if not np.any(selected):
        return None
    return float(frequencies[selected][np.argmax(mean_psd[selected])])

