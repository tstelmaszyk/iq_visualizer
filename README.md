# IQ Visualizer

Visualizes IQ samples read from a CSV file: I/Q over time, constellation and
spectrum (dB or linear magnitude), in an interactive Plotly figure opened in
the browser.

Compatible with Python 3.8+.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # usage
.venv/bin/pip install -r requirements-dev.txt    # development (tests, flake8)
```

## Usage

```bash
.venv/bin/python -m iq_visualizer.main examples/sample.csv --fs 1e6
```

The CSV file contains one row per sample, with an I column and a Q column.

| Option | Purpose | Default |
|---|---|---|
| `--sep` | separator: `comma`, `semicolon`, `tab`, `space` or a raw character | `,` |
| `--decimal` | decimal separator | `.` |
| `--header` | the first row is a header | no |
| `--i-col`, `--q-col` | indices of the I and Q columns | `0`, `1` |
| `--fs` | sample rate (Hz) | none (samples, normalized frequency) |
| `--nfft` | FFT size | whole signal |
| `--window` | `hann` or `rectangular` | `hann` |
| `--scale` | `db` or `linear` | `db` |
| `--max-points` | points plotted in the time and constellation views | `10000` |
| `--save` | save the figure as HTML | none |

Example for a European-style CSV (semicolon separator, decimal comma) with a
header:

```bash
.venv/bin/python -m iq_visualizer.main measurement.csv --sep semicolon --decimal "," --header --fs 2.4e6
```

Default values can be changed in `iq_visualizer/config.py`.

## Spectrum scale

The spectrum is normalized: a complex sinusoid of amplitude 1 gives a peak of
1 in linear scale, i.e. 0 dB.

## Adding an IQ source

Inherit from `IQSource` (`iq_visualizer/sources.py`) and implement `read()`,
which returns a 1D complex numpy array, then pass it to
`iq_visualizer.main.run()`:

```python
class BinaryIQSource(IQSource):
    def __init__(self, path):
        self.path = path

    def read(self):
        return np.fromfile(self.path, dtype=np.complex64)


run(BinaryIQSource("capture.bin"), VisualizerConfig(sample_rate=2.4e6))
```

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/flake8 iq_visualizer tests
```
