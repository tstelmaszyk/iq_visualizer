"""IQ sample sources.

To add a source (binary file, SDR...), inherit from IQSource and implement
read().
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from iq_visualizer.config import CsvConfig


class IQSource(ABC):
    """Common interface for all IQ sources."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Returns the samples as a 1D complex numpy array."""


class CsvIQSource(IQSource):
    """Reads IQ samples from two columns of a CSV file."""

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
            raise ValueError(f"The file {cfg.path} is empty.") from None

        if table.shape[1] <= max(cfg.i_column, cfg.q_column):
            raise ValueError(
                f"{cfg.path} has {table.shape[1]} column(s), "
                f"cannot read columns {cfg.i_column} and "
                f"{cfg.q_column}. Is the separator {cfg.separator!r} "
                f"correct?"
            )

        columns = table.iloc[:, [cfg.i_column, cfg.q_column]]
        if columns.empty:
            raise ValueError(f"No samples in {cfg.path}.")
        if not all(is_numeric_dtype(dtype) for dtype in columns.dtypes):
            raise ValueError(
                f"The I/Q columns of {cfg.path} are not numeric. "
                f"Check the separator, decimal and header options."
            )
        values = columns.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Missing or infinite values in {cfg.path}.")

        return values[:, 0] + 1j * values[:, 1]
