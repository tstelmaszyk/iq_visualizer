"""CSV reading and visualization settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class CsvConfig:
    """Settings for reading a CSV file containing I and Q columns."""

    path: Path
    separator: str = ","      # ",", ";", "\t", r"\s+" (whitespace)
    decimal: str = "."        # "," for European-style CSV files
    has_header: bool = False
    i_column: int = 0         # index of the I column
    q_column: int = 1         # index of the Q column

    def __post_init__(self) -> None:
        if self.i_column < 0 or self.q_column < 0:
            raise ValueError("Column indices must be non-negative.")
        if self.i_column == self.q_column:
            raise ValueError("The I and Q columns must be different.")


@dataclass
class VisualizerConfig:
    """Analysis and display settings."""

    sample_rate: Optional[float] = None   # Hz; None → samples
    nfft: Optional[int] = None            # None → whole signal
    window: Literal["hann", "rectangular"] = "hann"
    scale: Literal["db", "linear"] = "db"
    max_plot_samples: int = 10_000        # plotted points (time, IQ)
    output_html: Optional[Path] = None    # optional output file
    check_tone: bool = False              # look for gaps in a tone

    def __post_init__(self) -> None:
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("The sample rate must be > 0.")
        if self.nfft is not None and self.nfft <= 0:
            raise ValueError("nfft must be > 0.")
        if self.max_plot_samples <= 0:
            raise ValueError("max_plot_samples must be > 0.")
