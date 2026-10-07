"""Calculs sur les échantillons IQ (numpy uniquement)."""

from typing import Optional, Tuple

import numpy as np

WINDOWS = {"hann": np.hanning, "rectangular": np.ones}
SCALES = ("db", "linear")
DB_FLOOR = 1e-12  # évite log10(0)


def time_axis(n_samples: int,
              sample_rate: Optional[float] = None) -> np.ndarray:
    """Secondes si sample_rate est fourni, sinon indices d'échantillons."""
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
    """Renvoie (fréquences, valeurs) du spectre, centrés sur 0.

    Le spectre est normalisé : une sinusoïde complexe d'amplitude 1
    donne un pic à 1 en linéaire, soit 0 dB.
    """
    if iq.size == 0:
        raise ValueError("Le signal est vide.")
    if window not in WINDOWS:
        raise ValueError(
            f"Fenêtre inconnue {window!r}, choix : {sorted(WINDOWS)}."
        )
    if scale not in SCALES:
        raise ValueError(f"Échelle inconnue {scale!r}, choix : {SCALES}.")

    n_fft = iq.size if nfft is None else nfft
    segment = iq[:n_fft]
    weights = WINDOWS[window](segment.size)
    if weights.sum() == 0:
        raise ValueError(
            f"Signal trop court ({segment.size} échantillons) "
            f"pour la fenêtre {window!r}."
        )

    spectrum = np.fft.fftshift(np.fft.fft(segment * weights, n=n_fft))
    magnitude = np.abs(spectrum) / weights.sum()

    spacing = 1.0 if sample_rate is None else 1.0 / sample_rate
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_fft, d=spacing))

    if scale == "db":
        return frequencies, 20 * np.log10(np.maximum(magnitude, DB_FLOOR))
    return frequencies, magnitude
