"""Shared data loading for every BOB phase-estimation model.

The notebooks use two processed datasets with different on-disk layouts:

* ``theta`` / ``onset``: multi-channel, onset-aligned epochs.  Models use the
  subject-specific channel selected on calibration data.
* ``alpha`` / ``LEMON``: single-channel, five-second C3-Hjorth epochs.

This module normalizes both layouts into :class:`SubjectData`.  In both cases,
``SubjectData.epochs`` is the exact model-ready, single-channel input with shape
``(n_trials, n_samples)``.  Calibration and test indices are also normalized so
model scripts do not need dataset-specific loading branches.

Typical use::

    from Pipeline.data_loader import iter_subjects

    for subject in iter_subjects("theta"):
        raw = subject.epochs
        cal = subject.cal_indices
        test = subject.test_indices

The iterator intentionally loads one subject at a time, matching the notebooks'
memory-efficient execution pattern.  Use :func:`load_dataset` when an eager list
of all subjects is more convenient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import numpy as np
from numpy.typing import NDArray


Band = Literal["theta", "alpha"]

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "Data" / "Processed Data"

_BAND_ALIASES: dict[str, Band] = {
    "theta": "theta",
    "onset": "theta",
    "alpha": "alpha",
    "lemon": "alpha",
    "mu": "alpha",
}

_DATASET_FOLDER: dict[Band, str] = {
    "theta": "onset",
    "alpha": "LEMON",
}

_THETA_REQUIRED_KEYS = {
    "epochs",
    "split",
    "fs",
    "best_ch_idx",
    "best_ch_name",
    "ch_names",
    "ch_theta_snr",
    "onset_in_epoch",
    "pre_samp",
    "post_samp",
}

_ALPHA_REQUIRED_KEYS = {
    "epochs",
    "fs",
    "n_epochs",
    "cal_n",
    "test_n",
    "montage",
    "epoch_len",
}


@dataclass(slots=True)
class SubjectData:
    """Normalized input and metadata for one subject.

    Attributes
    ----------
    epochs:
        Model-ready single-channel epochs, always shaped
        ``(n_trials, n_samples)``.
    cal_indices, test_indices:
        Integer row indices into ``epochs``.
    onset_sample:
        Onset location within each epoch.  This is ``0`` for resting-state
        LEMON epochs.
    snr:
        Precomputed selected-channel theta SNR for onset data.  LEMON stores no
        SNR, so alpha models compute their shared alpha-SNR descriptor later.
    metadata:
        Small dataset-specific arrays and scalar values.  The large original
        multi-channel theta tensor is deliberately not duplicated here.
    """

    name: str
    band: Band
    epochs: NDArray[np.floating[Any]]
    fs: float
    cal_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    onset_sample: int
    channel_name: str
    channel_index: int
    n_channels: int
    snr: float | None
    source_path: Path
    metadata: Mapping[str, Any]

    @property
    def n_trials(self) -> int:
        return int(self.epochs.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.epochs.shape[1])

    @property
    def n_cal(self) -> int:
        return int(self.cal_indices.size)

    @property
    def n_test(self) -> int:
        return int(self.test_indices.size)


def normalize_band(band: str) -> Band:
    """Return the canonical band name, accepting notebook-friendly aliases."""

    try:
        return _BAND_ALIASES[band.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(_BAND_ALIASES))
        raise ValueError(f"Unknown dataset/band {band!r}; expected one of: {choices}") from exc


def resolve_dataset_dir(
    band: str,
    data_root: str | Path | None = None,
) -> Path:
    """Locate the directory containing the requested dataset's ``.npz`` files.

    ``data_root`` may be the shared ``Processed Data`` directory or the final
    ``onset``/``LEMON`` directory itself.  The default discovery supports the
    timestamped Google Drive download folders currently in this repository.
    Ambiguous matches raise an error instead of silently selecting a dataset.
    """

    canonical = normalize_band(band)
    expected_name = _DATASET_FOLDER[canonical]
    root = Path(data_root).expanduser().resolve() if data_root else DEFAULT_DATA_ROOT

    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    if root.is_dir() and root.name.lower() == expected_name.lower() and any(root.glob("*.npz")):
        return root

    candidates = sorted(
        {
            path.parent.resolve()
            for path in root.rglob("*.npz")
            if path.parent.name.lower() == expected_name.lower()
        },
        key=lambda path: str(path).lower(),
    )

    if not candidates:
        raise FileNotFoundError(
            f"No {expected_name!r} dataset containing .npz files was found under {root}"
        )
    if len(candidates) > 1:
        joined = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            f"Multiple {expected_name!r} datasets were found. Pass data_root explicitly:\n  {joined}"
        )
    return candidates[0]


def subject_paths(
    band: str,
    data_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return sorted subject files for ``band`` without loading their arrays."""

    dataset_dir = resolve_dataset_dir(band, data_root)
    paths = tuple(sorted(dataset_dir.glob("*.npz"), key=lambda path: path.name.lower()))
    if not paths:
        raise FileNotFoundError(f"No .npz subject files found in {dataset_dir}")
    return paths


def load_subject(
    path: str | Path,
    band: str,
    *,
    validate: bool = True,
) -> SubjectData:
    """Load and normalize one theta/onset or alpha/LEMON subject file."""

    canonical = normalize_band(band)
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Subject file does not exist: {source_path}")

    with np.load(source_path, allow_pickle=False) as data:
        if canonical == "theta":
            subject = _load_theta_subject(data, source_path)
        else:
            subject = _load_alpha_subject(data, source_path)

    if validate:
        _validate_subject(subject)
    return subject


def iter_subjects(
    band: str,
    data_root: str | Path | None = None,
    *,
    validate: bool = True,
) -> Iterator[SubjectData]:
    """Yield normalized subjects one at a time in filename-sorted order."""

    canonical = normalize_band(band)
    for path in subject_paths(canonical, data_root):
        yield load_subject(path, canonical, validate=validate)


def load_dataset(
    band: str,
    data_root: str | Path | None = None,
    *,
    validate: bool = True,
) -> list[SubjectData]:
    """Eagerly load every subject in a dataset.

    Prefer :func:`iter_subjects` for model sweeps so only one subject is held in
    memory at a time.
    """

    return list(iter_subjects(band, data_root, validate=validate))


def _missing_keys(data: np.lib.npyio.NpzFile, required: set[str]) -> set[str]:
    return required.difference(data.files)


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    return np.asarray(data[key]).item()


def _load_theta_subject(
    data: np.lib.npyio.NpzFile,
    source_path: Path,
) -> SubjectData:
    missing = _missing_keys(data, _THETA_REQUIRED_KEYS)
    if missing:
        raise ValueError(f"{source_path.name} is missing theta keys: {sorted(missing)}")

    all_epochs = np.asarray(data["epochs"])
    if all_epochs.ndim != 3:
        raise ValueError(
            f"{source_path.name}: theta epochs must be 3-D; got {all_epochs.shape}"
        )

    channel_index = int(_scalar(data, "best_ch_idx"))
    if not 0 <= channel_index < all_epochs.shape[1]:
        raise ValueError(
            f"{source_path.name}: best_ch_idx={channel_index} is outside "
            f"0..{all_epochs.shape[1] - 1}"
        )

    split = np.asarray(data["split"]).astype(str)
    if split.shape != (all_epochs.shape[0],):
        raise ValueError(
            f"{source_path.name}: split shape {split.shape} does not match "
            f"{all_epochs.shape[0]} trials"
        )

    cal_indices = np.flatnonzero(split == "cal").astype(np.int64)
    test_indices = np.flatnonzero(split == "test").astype(np.int64)
    unknown = sorted(set(np.unique(split)).difference({"cal", "test"}))
    if unknown:
        raise ValueError(f"{source_path.name}: unknown split labels: {unknown}")

    channel_snrs = np.asarray(data["ch_theta_snr"], dtype=np.float32)
    channel_names = np.asarray(data["ch_names"]).astype(str)
    metadata: dict[str, Any] = {
        "split_labels": split,
        "codes": np.asarray(data["codes"]) if "codes" in data.files else None,
        "event_samples": (
            np.asarray(data["event_sample"]) if "event_sample" in data.files else None
        ),
        "channel_names": channel_names,
        "channel_theta_snr": channel_snrs,
        "roi_names": np.asarray(data["roi_names"]).astype(str) if "roi_names" in data.files else None,
        "roi_theta_snr": (
            np.asarray(data["roi_theta_snr"], dtype=np.float32)
            if "roi_theta_snr" in data.files
            else None
        ),
        "selection_basis": (
            str(_scalar(data, "selection_basis")) if "selection_basis" in data.files else None
        ),
        "pre_samples": int(_scalar(data, "pre_samp")),
        "post_samples": int(_scalar(data, "post_samp")),
        "cal_split_sample": (
            int(_scalar(data, "cal_split_sample"))
            if "cal_split_sample" in data.files
            else None
        ),
        "cal_fraction": (
            float(_scalar(data, "cal_fraction")) if "cal_fraction" in data.files else None
        ),
        "theta_band_hz": (
            np.asarray(data["theta_band"], dtype=float)
            if "theta_band" in data.files
            else None
        ),
    }

    return SubjectData(
        name=source_path.stem,
        band="theta",
        epochs=np.ascontiguousarray(all_epochs[:, channel_index, :]),
        fs=float(_scalar(data, "fs")),
        cal_indices=cal_indices,
        test_indices=test_indices,
        onset_sample=int(_scalar(data, "onset_in_epoch")),
        channel_name=str(_scalar(data, "best_ch_name")),
        channel_index=channel_index,
        n_channels=int(all_epochs.shape[1]),
        snr=float(channel_snrs[channel_index]),
        source_path=source_path,
        metadata=metadata,
    )


def _load_alpha_subject(
    data: np.lib.npyio.NpzFile,
    source_path: Path,
) -> SubjectData:
    missing = _missing_keys(data, _ALPHA_REQUIRED_KEYS)
    if missing:
        raise ValueError(f"{source_path.name} is missing alpha keys: {sorted(missing)}")

    epochs = np.asarray(data["epochs"])
    if epochs.ndim != 2:
        raise ValueError(
            f"{source_path.name}: LEMON epochs must be 2-D; got {epochs.shape}"
        )

    n_epochs = int(_scalar(data, "n_epochs"))
    cal_n = int(_scalar(data, "cal_n"))
    test_n = int(_scalar(data, "test_n"))
    if n_epochs != epochs.shape[0]:
        raise ValueError(
            f"{source_path.name}: n_epochs={n_epochs} but epochs has {epochs.shape[0]} rows"
        )
    if cal_n + test_n != n_epochs:
        raise ValueError(
            f"{source_path.name}: cal_n + test_n ({cal_n + test_n}) != n_epochs ({n_epochs})"
        )

    metadata: dict[str, Any] = {
        "montage": str(_scalar(data, "montage")),
        "epoch_length": int(_scalar(data, "epoch_len")),
        "n_epochs": n_epochs,
        "cal_n": cal_n,
        "test_n": test_n,
    }

    return SubjectData(
        name=source_path.stem,
        band="alpha",
        epochs=np.ascontiguousarray(epochs),
        fs=float(_scalar(data, "fs")),
        cal_indices=np.arange(cal_n, dtype=np.int64),
        test_indices=np.arange(cal_n, n_epochs, dtype=np.int64),
        onset_sample=0,
        channel_name=metadata["montage"],
        channel_index=0,
        n_channels=1,
        snr=None,
        source_path=source_path,
        metadata=metadata,
    )


def _validate_subject(subject: SubjectData) -> None:
    if subject.epochs.ndim != 2:
        raise ValueError(f"{subject.name}: normalized epochs must be 2-D")
    if subject.n_trials == 0 or subject.n_samples == 0:
        raise ValueError(f"{subject.name}: dataset contains empty epochs")
    if not np.issubdtype(subject.epochs.dtype, np.floating):
        raise TypeError(f"{subject.name}: epochs must be floating point, got {subject.epochs.dtype}")
    if not np.isfinite(subject.epochs).all():
        raise ValueError(f"{subject.name}: epochs contain NaN or infinite values")
    if not np.isfinite(subject.fs) or subject.fs <= 0:
        raise ValueError(f"{subject.name}: invalid sampling rate {subject.fs}")
    if not 0 <= subject.onset_sample < subject.n_samples:
        raise ValueError(
            f"{subject.name}: onset_sample={subject.onset_sample} outside epoch length {subject.n_samples}"
        )
    if subject.n_cal == 0 or subject.n_test == 0:
        raise ValueError(f"{subject.name}: calibration and test splits must both be non-empty")

    all_indices = np.concatenate((subject.cal_indices, subject.test_indices))
    if np.any(all_indices < 0) or np.any(all_indices >= subject.n_trials):
        raise ValueError(f"{subject.name}: split indices fall outside the epoch array")
    if np.intersect1d(subject.cal_indices, subject.test_indices).size:
        raise ValueError(f"{subject.name}: calibration and test splits overlap")
    if np.unique(all_indices).size != subject.n_trials:
        raise ValueError(f"{subject.name}: calibration/test splits do not cover every trial")


def _print_summary(band: Band) -> None:
    paths = subject_paths(band)
    first = load_subject(paths[0], band)
    print(
        f"{band}: {len(paths)} subjects | {first.fs:g} Hz | "
        f"model input {first.epochs.shape} for {first.name} | "
        f"cal/test {first.n_cal}/{first.n_test}"
    )


if __name__ == "__main__":
    for _band in ("theta", "alpha"):
        _print_summary(_band)
