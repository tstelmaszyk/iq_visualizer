"""Computations on IQ samples (numpy only)."""

from typing import Optional, Tuple

import numpy as np

WINDOWS = {"hann": np.hanning, "rectangular": np.ones}
SCALES = ("db", "linear")
DB_FLOOR = 1e-12  # avoids log10(0)


def time_axis(n_samples: int,
              sample_rate: Optional[float] = None) -> np.ndarray:
    """Seconds if sample_rate is given, otherwise sample indices."""
    indices = np.arange(n_samples)
    if sample_rate is None:
        return indices
    return indices / sample_rate


def compute_spectrum(
    iq: np.ndarray,
    sample_rate: Optional[float] = None,
    nfft: Optional[int] = None,
    window: str = "hann",
    scale: str = "db",
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the spectrum (frequencies, values), centered on 0.

    The spectrum is normalized: a complex sinusoid of amplitude 1
    gives a peak of 1 in linear scale, i.e. 0 dB.
    """
    if iq.size == 0:
        raise ValueError("The signal is empty.")
    if window not in WINDOWS:
        raise ValueError(
            f"Unknown window {window!r}, choices: {sorted(WINDOWS)}."
        )
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}, choices: {SCALES}.")

    n_fft = iq.size if nfft is None else nfft
    segment = iq[:n_fft]
    weights = WINDOWS[window](segment.size)
    if weights.sum() == 0:
        raise ValueError(
            f"Signal too short ({segment.size} samples) "
            f"for the {window!r} window."
        )

    spectrum = np.fft.fftshift(np.fft.fft(segment * weights, n=n_fft))
    magnitude = np.abs(spectrum) / weights.sum()

    spacing = 1.0 if sample_rate is None else 1.0 / sample_rate
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_fft, d=spacing))

    if scale == "db":
        return frequencies, 20 * np.log10(np.maximum(magnitude, DB_FLOOR))
    return frequencies, magnitude
