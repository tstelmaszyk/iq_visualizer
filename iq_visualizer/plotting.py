"""Plotly figure construction: time, constellation, spectrum."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iq_visualizer.config import VisualizerConfig
from iq_visualizer.processing import compute_spectrum, time_axis


def build_figure(iq: np.ndarray, config: VisualizerConfig) -> go.Figure:
    """Builds the figure (without showing it)."""
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("I / Q", "Constellation", "Spectrum"),
        vertical_spacing=0.12,
    )
    shown = iq[:config.max_plot_samples]
    _add_time_traces(figure, shown, config)
    _add_constellation(figure, shown)
    _add_spectrum(figure, iq, config)
    figure.update_layout(height=800, template="plotly_white")
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
                     marker={"size": 3}, name="Constellation"),
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
