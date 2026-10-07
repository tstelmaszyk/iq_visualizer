from pathlib import Path

import pytest

from iq_visualizer.config import CsvConfig, VisualizerConfig


def test_visualizer_config_defaults():
    config = VisualizerConfig()
    assert config.sample_rate is None
    assert config.nfft is None
    assert config.window == "hann"
    assert config.scale == "db"
    assert config.max_plot_samples == 10_000
    assert config.output_html is None


@pytest.mark.parametrize("kwargs", [
    {"sample_rate": 0},
    {"sample_rate": -1e6},
    {"nfft": 0},
    {"max_plot_samples": 0},
])
def test_visualizer_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        VisualizerConfig(**kwargs)


def test_csv_config_defaults():
    config = CsvConfig(path=Path("data.csv"))
    assert config.separator == ","
    assert config.decimal == "."
    assert config.has_header is False
    assert (config.i_column, config.q_column) == (0, 1)


@pytest.mark.parametrize("i_column, q_column", [(0, 0), (-1, 1)])
def test_csv_config_rejects_invalid_columns(i_column, q_column):
    with pytest.raises(ValueError):
        CsvConfig(path=Path("data.csv"), i_column=i_column,
                  q_column=q_column)
