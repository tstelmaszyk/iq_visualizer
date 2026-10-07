from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from iq_visualizer.config import CsvConfig, VisualizerConfig
from iq_visualizer.main import main, parse_args, run
from iq_visualizer.sources import IQSource


class FakeSource(IQSource):
    def read(self):
        return np.exp(2j * np.pi * 0.1 * np.arange(64))


@pytest.fixture
def no_browser(monkeypatch):
    """Empêche fig.show() d'ouvrir un navigateur pendant les tests."""
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))
    return shown


def test_parse_args_defaults_come_from_dataclasses():
    csv_config, vis_config = parse_args(["data.csv"])
    assert csv_config == CsvConfig(path=Path("data.csv"))
    assert vis_config == VisualizerConfig()


def test_parse_args_fills_all_options():
    csv_config, vis_config = parse_args([
        "data.csv", "--sep", ";", "--decimal", ",", "--header",
        "--i-col", "2", "--q-col", "3", "--fs", "1e6", "--nfft", "1024",
        "--window", "rectangular", "--scale", "linear",
        "--max-points", "500", "--save", "out.html",
    ])
    assert csv_config == CsvConfig(
        path=Path("data.csv"), separator=";", decimal=",", has_header=True,
        i_column=2, q_column=3)
    assert vis_config == VisualizerConfig(
        sample_rate=1e6, nfft=1024, window="rectangular", scale="linear",
        max_plot_samples=500, output_html=Path("out.html"))


@pytest.mark.parametrize("alias, separator", [
    ("comma", ","), ("semicolon", ";"), ("tab", "\t"), ("space", r"\s+"),
])
def test_parse_args_resolves_separator_aliases(alias, separator):
    csv_config, _ = parse_args(["data.csv", "--sep", alias])
    assert csv_config.separator == separator


def test_run_shows_figure_and_writes_html(tmp_path, no_browser):
    output = tmp_path / "out.html"
    run(FakeSource(), VisualizerConfig(output_html=output))
    assert len(no_browser) == 1
    assert "<html>" in output.read_text()


def test_main_runs_on_a_csv_file(tmp_path, no_browser):
    path = tmp_path / "iq.csv"
    path.write_text("1,0\n0,1\n-1,0\n0,-1\n")
    assert main([str(path)]) == 0
    assert len(no_browser) == 1


def test_main_reports_missing_file_without_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "absent.csv")]) == 1
    assert "Erreur" in capsys.readouterr().err


def test_main_reports_invalid_option_value(tmp_path, capsys):
    assert main([str(tmp_path / "iq.csv"), "--fs", "0"]) == 1
    assert "Erreur" in capsys.readouterr().err
