import numpy as np

from iq_visualizer.config import VisualizerConfig
from iq_visualizer.plotting import build_figure
from iq_visualizer.tone_check import Gap


def test_gap_view_plots_the_whole_signal():
    iq = np.ones(500, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(max_plot_samples=100),
                          gaps=[])
    names = [trace.name for trace in figure.data]
    assert names[-2:] == ["I (gap view)", "Q (gap view)"]
    assert len(figure.data[-1].x) == 500


def test_each_gap_is_highlighted_in_samples():
    iq = np.ones(500, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(),
                          gaps=[Gap(10, 12), Gap(300, 301)])
    spans = [(shape.x0, shape.x1) for shape in figure.layout.shapes]
    assert spans == [(10, 12), (300, 301)]


def test_gap_highlight_has_a_visible_border():
    iq = np.ones(500, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(), gaps=[Gap(10, 11)])
    assert figure.layout.shapes[0].line.width >= 3


def test_gaps_are_highlighted_in_seconds_when_sample_rate_is_known():
    iq = np.ones(500, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(sample_rate=100.0),
                          gaps=[Gap(10, 20)])
    shape = figure.layout.shapes[0]
    assert (shape.x0, shape.x1) == (0.1, 0.2)
    assert figure.layout.xaxis4.title.text == "Time (s)"


def test_no_gap_view_without_gaps():
    figure = build_figure(np.ones(16, dtype=complex), VisualizerConfig())
    assert "xaxis4" not in figure.layout


def test_gap_view_leaves_a_vertical_margin_around_the_signal():
    iq = np.array([-1.0, 1.0, 0.5j, -0.5j])
    figure = build_figure(iq, VisualizerConfig(), gaps=[Gap(1, 2)])
    assert tuple(figure.layout.yaxis4.range) == (-1.2, 1.2)


def test_gap_view_margin_survives_a_constant_signal():
    figure = build_figure(np.ones(16, dtype=complex), VisualizerConfig(),
                          gaps=[Gap(1, 2)])
    low, high = figure.layout.yaxis4.range
    assert low < 1.0 < high
