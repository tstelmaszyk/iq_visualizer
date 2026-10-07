"""Paramètres de lecture CSV et de visualisation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class CsvConfig:
    """Paramètres de lecture d'un fichier CSV contenant des colonnes I et Q."""

    path: Path
    separator: str = ","      # ",", ";", "\t", r"\s+" (espaces)
    decimal: str = "."        # "," pour les CSV "à la française"
    has_header: bool = False
    i_column: int = 0         # index de la colonne I
    q_column: int = 1         # index de la colonne Q

    def __post_init__(self) -> None:
        if self.i_column < 0 or self.q_column < 0:
            raise ValueError("Les index de colonnes doivent être positifs.")
        if self.i_column == self.q_column:
            raise ValueError("Les colonnes I et Q doivent être différentes.")


@dataclass
class VisualizerConfig:
    """Paramètres d'analyse et d'affichage."""

    sample_rate: Optional[float] = None   # Hz ; None → échantillons
    nfft: Optional[int] = None            # None → tout le signal
    window: Literal["hann", "rectangular"] = "hann"
    scale: Literal["db", "linear"] = "db"
    max_plot_samples: int = 10_000        # points tracés (temps, IQ)
    output_html: Optional[Path] = None    # sauvegarde optionnelle

    def __post_init__(self) -> None:
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("La fréquence d'échantillonnage doit être > 0.")
        if self.nfft is not None and self.nfft <= 0:
            raise ValueError("nfft doit être > 0.")
        if self.max_plot_samples <= 0:
            raise ValueError("max_plot_samples doit être > 0.")
