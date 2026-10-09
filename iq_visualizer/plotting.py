"""Plotly figure construction: time, constellation, spectrum."""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iq_visualizer import gap_plotting
from iq_visualizer.config import VisualizerConfig
from iq_visualizer.processing import compute_spectrum, time_axis
from iq_visualizer.tone_check import Gap

# Dark theme: very dark blue background.
PAPER_COLOR = "#0b1426"   # around the plots
PLOT_COLOR = "#101d36"    # plot areas, slightly lighter
GRID_COLOR = "#22324f"

# --- Cosmetic only: page background -------------------------------------
# The HTML page around the figure stays white; this JavaScript, run after
# the figure is drawn (`post_script` of show()/write_html()), paints it in
# the figure color. Not needed for the IQ analysis: removing it only takes
# this constant and its two uses in main.run().
PAGE_BACKGROUND_SCRIPT = (
    "document.body.style.backgroundColor = '%s';" % PAPER_COLOR
)
# -------------------------------------------------------------------------


def build_figure(iq: np.ndarray, config: VisualizerConfig,
                 gaps: Optional[List[Gap]] = None) -> go.Figure:
    """Builds the figure (without showing it).

    With `gaps` (tone check result), a gap view is added as a third row.
    """
    specs = [[{}, {}], [{"colspan": 2}, None]]
    titles = ["I / Q", "Constellation", "Spectrum"]
    if gaps is not None:
        specs.append([{"colspan": 2}, None])
        titles.append(gap_plotting.TITLE)

    figure = make_subplots(
        rows=len(specs),
        cols=2,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.12 * 2 / len(specs),
    )
    shown = iq[:config.max_plot_samples]
    _add_time_traces(figure, shown, config)
    _add_constellation(figure, shown)
    _add_spectrum(figure, iq, config)
    if gaps is not None:
        gap_plotting.add_gap_view(figure, iq, gaps, config, row=3)
    figure.update_layout(height=400 * len(specs), template="plotly_dark",
                         paper_bgcolor=PAPER_COLOR, plot_bgcolor=PLOT_COLOR,
                         showlegend=False)
    figure.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    figure.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return figure


def _add_time_traces(figure: go.Figure, iq: np.ndarray,
                     config: VisualizerConfig) -> None:
    times = time_axis(iq.size, config.sample_rate)
    # Scattergl (WebGL) stays smooth with many points.
    figure.add_trace(go.Scattergl(x=times, y=iq.real, name="I"),
                     row=1, col=1)
    figure.add_trace(go.Scattergl(x=times, y=iq.imag, name="Q"),
                     row=1, col=1)
    x_label = "Sample" if config.sample_rate is None else "Time (s)"
    figure.update_xaxes(title_text=x_label, row=1, col=1)
    figure.update_yaxes(title_text="Amplitude", row=1, col=1)


def _add_constellation(figure: go.Figure, iq: np.ndarray) -> None:
    figure.add_trace(
        go.Scattergl(x=iq.real, y=iq.imag, mode="markers",
                     marker={"size": 6, "opacity": 0.6}, name="Constellation"),
        row=1, col=2,
    )
    figure.update_xaxes(title_text="I", row=1, col=2)
    # 1:1 aspect ratio so the constellation is not distorted.
    figure.update_yaxes(title_text="Q", scaleanchor="x2", scaleratio=1,
                        row=1, col=2)


def _add_spectrum(figure: go.Figure, iq: np.ndarray,
                  config: VisualizerConfig) -> None:
    frequencies, values = compute_spectrum(
        iq,
        sample_rate=config.sample_rate,
        nfft=config.nfft,
        window=config.window,
        scale=config.scale,
    )
    figure.add_trace(go.Scattergl(x=frequencies, y=values, name="Spectrum"),
                     row=2, col=1)
    x_label = ("Normalized frequency" if config.sample_rate is None
               else "Frequency (Hz)")
    y_label = "Magnitude (dB)" if config.scale == "db" else "Magnitude"
    figure.update_xaxes(title_text=x_label, row=2, col=1)
    figure.update_yaxes(title_text=y_label, row=2, col=1)
