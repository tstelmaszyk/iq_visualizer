import numpy as np

from iq_visualizer.config import VisualizerConfig
from iq_visualizer.plotting import PAPER_COLOR, PLOT_COLOR, build_figure


def test_figure_has_time_constellation_and_spectrum_traces():
    iq = np.exp(2j * np.pi * 0.1 * np.arange(64))
    figure = build_figure(iq, VisualizerConfig())
    names = [trace.name for trace in figure.data]
    assert names == ["I", "Q", "Constellation", "Spectrum"]


def test_time_and_constellation_respect_max_plot_samples():
    iq = np.ones(500, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(max_plot_samples=100))
    i_trace, q_trace, constellation, spectrum = figure.data
    assert len(i_trace.x) == len(q_trace.x) == len(constellation.x) == 100
    assert len(spectrum.x) == 500  # spectrum over the whole signal


def test_signal_shorter_than_max_plot_samples_is_fully_plotted():
    iq = np.ones(10, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(max_plot_samples=100))
    assert len(figure.data[0].x) == 10


def test_axis_labels_follow_configuration():
    iq = np.ones(16, dtype=complex)
    with_fs = build_figure(iq, VisualizerConfig(sample_rate=1e3,
                                                scale="linear"))
    assert with_fs.layout.xaxis.title.text == "Time (s)"
    assert with_fs.layout.xaxis3.title.text == "Frequency (Hz)"
    assert with_fs.layout.yaxis3.title.text == "Magnitude"

    without_fs = build_figure(iq, VisualizerConfig())
    assert without_fs.layout.xaxis.title.text == "Sample"
    assert without_fs.layout.xaxis3.title.text == "Normalized frequency"
    assert without_fs.layout.yaxis3.title.text == "Magnitude (dB)"


def test_figure_uses_dark_blue_theme():
    figure = build_figure(np.ones(16, dtype=complex), VisualizerConfig())
    assert figure.layout.paper_bgcolor == PAPER_COLOR
    assert figure.layout.plot_bgcolor == PLOT_COLOR


def test_constellation_markers_are_enlarged():
    figure = build_figure(np.ones(16, dtype=complex), VisualizerConfig())
    constellation = figure.data[2]
    assert constellation.marker.size == 6
    assert constellation.marker.opacity == 0.6
