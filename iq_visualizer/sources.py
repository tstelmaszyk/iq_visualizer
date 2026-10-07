"""Sources d'échantillons IQ.

Pour ajouter une source (fichier binaire, SDR...), il suffit d'hériter de
IQSource et d'implémenter read().
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from iq_visualizer.config import CsvConfig


class IQSource(ABC):
    """Interface commune à toutes les sources d'IQ."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Renvoie les échantillons sous forme de tableau numpy 1D complexe."""


class CsvIQSource(IQSource):
    """Lit les échantillons IQ depuis deux colonnes d'un fichier CSV."""

    def __init__(self, config: CsvConfig) -> None:
        self.config = config

    def read(self) -> np.ndarray:
        cfg = self.config
        try:
            table = pd.read_csv(
                cfg.path,
                sep=cfg.separator,
                decimal=cfg.decimal,
                header=0 if cfg.has_header else None,
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"Le fichier {cfg.path} est vide.") from None

        if table.shape[1] <= max(cfg.i_column, cfg.q_column):
            raise ValueError(
                f"{cfg.path} contient {table.shape[1]} colonne(s), "
                f"impossible de lire les colonnes {cfg.i_column} et "
                f"{cfg.q_column}. Le séparateur {cfg.separator!r} "
                f"est-il correct ?"
            )

        columns = table.iloc[:, [cfg.i_column, cfg.q_column]]
        if columns.empty:
            raise ValueError(f"Aucun échantillon dans {cfg.path}.")
        if not all(is_numeric_dtype(dtype) for dtype in columns.dtypes):
            raise ValueError(
                f"Les colonnes I/Q de {cfg.path} ne sont pas numériques. "
                f"Vérifier le séparateur, la décimale et l'option d'en-tête."
            )
        values = columns.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Valeurs manquantes ou infinies dans {cfg.path}.")

        return values[:, 0] + 1j * values[:, 1]
