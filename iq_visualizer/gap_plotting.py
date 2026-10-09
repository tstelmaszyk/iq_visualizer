"""Tone gap view: I/Q over the whole signal with the gaps highlighted.

Self-contained: removing this view only takes this module and the few
lines that call it in plotting.build_figure().
"""

from typing import List

import numpy as np
import plotly.graph_objects as go

from iq_visualizer.config import VisualizerConfig
from iq_visualizer.processing import time_axis
from iq_visualizer.tone_check import Gap

TITLE = "Tone gaps"
# Border in pixels: keeps a short gap visible when the whole signal is shown
GAP_LINE_WIDTH = 3
# Headroom above and below the signal, as a fraction of its span: the curve
# no longer touches the plot edges, which eases box-zoom selections
Y_MARGIN = 0.1


def add_gap_view(figure: go.Figure, iq: np.ndarray, gaps: List[Gap],
                 config: VisualizerConfig, row: int) -> None:
    """Adds the view on `row` (whole signal: a gap may be anywhere)."""
    times = time_axis(iq.size, config.sample_rate)
    figure.add_trace(go.Scattergl(x=times, y=iq.real, name="I (gap view)"),
                     row=row, col=1)
    figure.add_trace(go.Scattergl(x=times, y=iq.imag, name="Q (gap view)"),
                     row=row, col=1)

    scale = 1.0 if config.sample_rate is None else 1.0 / config.sample_rate
    for gap in gaps:
        figure.add_vrect(x0=gap.start * scale, x1=gap.stop * scale,
                         fillcolor="red", opacity=0.5, line_color="red",
                         line_width=GAP_LINE_WIDTH, row=row, col=1)

    x_label = "Sample" if config.sample_rate is None else "Time (s)"
    figure.update_xaxes(title_text=x_label, row=row, col=1)
    figure.update_yaxes(title_text="Amplitude", range=_y_range(iq),
                         row=row, col=1)


def _y_range(iq: np.ndarray) -> List[float]:
    """Signal extent on I and Q, widened by Y_MARGIN on each side."""
    low = float(min(iq.real.min(), iq.imag.min()))
    high = float(max(iq.real.max(), iq.imag.max()))
    # A constant signal has no span: fall back to its level (or 1)
    margin = Y_MARGIN * ((high - low) or max(abs(high), 1.0))
    return [low - margin, high + margin]
