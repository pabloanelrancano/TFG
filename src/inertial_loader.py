# src/inertial_loader.py
# Pablo Anel Rancaño – TFG HAR
"""
Loads the 9-channel raw inertial signals from UCI HAR's Inertial Signals/ folders.
Each file has shape (n_windows, 128) — that's 2.56s at 50 Hz.
Also validates that window counts match labels and subjects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CHANNEL_NAMES: List[str] = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

WINDOW_SIZE = 128  # 2.56s @ 50Hz


def _inertial_file(dataset_path: Path, split: str, channel: str) -> Path:
    """Build the path to one inertial signal file."""
    filename = f"{channel}_{split}.txt"
    return dataset_path / split / "Inertial Signals" / filename


def load_inertial_signals(
    split: str,
    dataset_path: Path,
    channels: List[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Load raw inertial signals for a split.

    Returns a dict mapping channel_name -> array(n_windows, 128).
    """
    channels = channels or CHANNEL_NAMES
    signals: Dict[str, np.ndarray] = {}

    for ch in channels:
        fpath = _inertial_file(dataset_path, split, ch)
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Inertial signal file not found: {fpath.resolve()}\n"
                f"Ensure the UCI HAR 'Inertial Signals' folder is present."
            )
        data = np.loadtxt(fpath)
        if data.ndim != 2 or data.shape[1] != WINDOW_SIZE:
            raise ValueError(
                f"Unexpected shape for {fpath.name}: {data.shape}. "
                f"Expected (n_windows, {WINDOW_SIZE})."
            )
        signals[ch] = data

    # All channels must have the same number of windows
    n_windows_set = {arr.shape[0] for arr in signals.values()}
    if len(n_windows_set) > 1:
        raise ValueError(
            f"Inconsistent window counts across channels: {n_windows_set}"
        )

    return signals


def load_inertial_split(
    split: str,
    dataset_path: Path,
) -> Tuple[np.ndarray, List[str]]:
    """Load all 9 channels concatenated into one 2D matrix (n_windows, 1152)."""
    signals = load_inertial_signals(split, dataset_path)
    arrays = [signals[ch] for ch in CHANNEL_NAMES]
    X_raw = np.hstack(arrays)
    return X_raw, CHANNEL_NAMES


def load_inertial_3d(
    split: str,
    dataset_path: Path,
) -> np.ndarray:
    """Load signals as a 3D array (n_windows, 9, 128). Handy for per-channel work."""
    signals = load_inertial_signals(split, dataset_path)
    arrays = [signals[ch] for ch in CHANNEL_NAMES]
    return np.stack(arrays, axis=1)


def validate_alignment(
    signals: Dict[str, np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    split: str,
) -> None:
    """Check that inertial windows, labels, and subjects all have the same length."""
    n_sig = next(iter(signals.values())).shape[0]
    if n_sig != len(y):
        raise ValueError(
            f"[{split}] Inertial windows ({n_sig}) ≠ labels ({len(y)})"
        )
    if n_sig != len(subjects):
        raise ValueError(
            f"[{split}] Inertial windows ({n_sig}) ≠ subjects ({len(subjects)})"
        )
    print(f"  [{split}] Alignment OK: {n_sig} windows, "
          f"{len(np.unique(subjects))} subjects, "
          f"{len(np.unique(y))} activities")
