"""Gap detection in a reference tone (numpy only).

A clean complex tone advances by the same phase step at every sample and
keeps a constant amplitude. A sample that breaks either rule marks a gap:
lost samples, zeros, or unrelated data.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Gap:
    """Run of faulty samples, iq[start:stop]."""

    start: int
    stop: int   # exclusive

    @property
    def length(self) -> int:
        return self.stop - self.start


class ToneChecker:
    """Finds the gaps in a tone.

    The expected phase step and amplitude are the medians over the whole
    signal, so most of it must be a clean tone.
    """

    def __init__(self, phase_tolerance_deg: float = 5.0,
                 amplitude_tolerance: float = 0.1) -> None:
        self.phase_tolerance = np.radians(phase_tolerance_deg)
        self.amplitude_tolerance = amplitude_tolerance   # relative

    def find_gaps(self, iq: np.ndarray) -> List[Gap]:
        if iq.size < 2:
            raise ValueError("The tone check needs at least 2 samples.")
        return _runs(self._faulty_samples(iq))

    def _faulty_samples(self, iq: np.ndarray) -> np.ndarray:
        amplitude = np.abs(iq)
        steps = np.angle(iq[1:] * np.conj(iq[:-1]))

        # A tone always rotates: constant data (zeros, stuck values) is
        # left out of the reference estimation. Real captures can end with
        # such data, sometimes longer than the tone itself: a median over
        # every sample would then take the parasitic data as the reference
        # and flag the actual tone as one big gap. Constant data is still
        # checked below, and reported as a gap.
        rotating = steps != 0
        if not rotating.any():
            raise ValueError("No tone found: the signal does not rotate.")
        expected_step = np.median(steps[rotating])
        expected_amplitude = np.median(amplitude[1:][rotating])

        faulty = (np.abs(amplitude - expected_amplitude)
                  > self.amplitude_tolerance * expected_amplitude)
        # Wrapped difference, so a step near ±pi is compared correctly.
        error = np.angle(np.exp(1j * (steps - expected_step)))
        bad_step = np.abs(error) > self.phase_tolerance

        # A bad step between n-1 and n is blamed on sample n.
        faulty[1:] |= bad_step
        return faulty


def _runs(flags: np.ndarray) -> List[Gap]:
    """Groups consecutive True values into gaps."""
    edges = np.diff(np.concatenate(([0], flags.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [Gap(int(start), int(stop)) for start, stop in zip(starts, stops)]


def format_report(gaps: List[Gap],
                  sample_rate: Optional[float] = None) -> str:
    """Human-readable summary of the gaps."""
    if not gaps:
        return "Tone check: no gap detected."
    lines = [f"Tone check: {len(gaps)} gap(s) detected."]
    for gap in gaps:
        line = (f"  samples {gap.start}-{gap.stop - 1} "
                f"({gap.length} samples)")
        if sample_rate is not None:
            line += f", t = {gap.start / sample_rate:g} s"
        lines.append(line)
    return "\n".join(lines)
