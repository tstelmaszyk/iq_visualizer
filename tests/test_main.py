from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from iq_visualizer.config import CsvConfig, VisualizerConfig
from iq_visualizer.main import main, parse_args, run
from iq_visualizer.plotting import PAGE_BACKGROUND_SCRIPT
from iq_visualizer.sources import IQSource


class FakeSource(IQSource):
    def read(self):
        return np.exp(2j * np.pi * 0.1 * np.arange(64))


@pytest.fixture
def no_browser(monkeypatch):
    """Prevents fig.show() from opening a browser during tests.

    Records the keyword arguments of each call.
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show",
                        lambda self, **kwargs: shown.append(kwargs))
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
        "--max-points", "500", "--save", "out.html", "--check-tone",
    ])
    assert csv_config == CsvConfig(
        path=Path("data.csv"), separator=";", decimal=",", has_header=True,
        i_column=2, q_column=3)
    assert vis_config == VisualizerConfig(
        sample_rate=1e6, nfft=1024, window="rectangular", scale="linear",
        max_plot_samples=500, output_html=Path("out.html"), check_tone=True)


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


def test_run_paints_page_background(tmp_path, no_browser):
    output = tmp_path / "out.html"
    run(FakeSource(), VisualizerConfig(output_html=output))
    assert no_browser[0]["post_script"] == PAGE_BACKGROUND_SCRIPT
    assert PAGE_BACKGROUND_SCRIPT in output.read_text()


def test_main_runs_on_a_csv_file(tmp_path, no_browser):
    path = tmp_path / "iq.csv"
    path.write_text("1,0\n0,1\n-1,0\n0,-1\n")
    assert main([str(path)]) == 0
    assert len(no_browser) == 1


def test_main_reports_missing_file_without_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "absent.csv")]) == 1
    assert "Error" in capsys.readouterr().err


def test_main_reports_invalid_option_value(tmp_path, capsys):
    assert main([str(tmp_path / "iq.csv"), "--fs", "0"]) == 1
    assert "Error" in capsys.readouterr().err


def test_main_reports_os_errors_without_traceback(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1  # a directory instead of a file
    assert "Error" in capsys.readouterr().err


def test_parse_args_enables_tone_check():
    _, vis_config = parse_args(["data.csv", "--check-tone"])
    assert vis_config.check_tone


def write_tone(path, n_samples):
    iq = 1000 * np.exp(2j * np.pi * np.arange(n_samples) / 16)
    np.savetxt(path, np.column_stack((iq.real, iq.imag)), delimiter=",")


def test_main_tone_check_passes_on_a_clean_tone(tmp_path, capsys,
                                                no_browser):
    path = tmp_path / "iq.csv"
    write_tone(path, 200)
    assert main([str(path), "--check-tone"]) == 0
    assert "no gap detected" in capsys.readouterr().out
    assert len(no_browser) == 1


def test_main_tone_check_fails_on_gaps(tmp_path, capsys, no_browser):
    path = tmp_path / "iq.csv"
    write_tone(path, 200)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:50] + lines[53:]) + "\n")
    assert main([str(path), "--check-tone"]) == 1
    assert "1 gap(s) detected" in capsys.readouterr().out
    assert len(no_browser) == 1
