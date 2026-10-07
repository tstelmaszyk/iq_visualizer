# IQ Visualizer — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** outil en ligne de commande qui lit des échantillons IQ dans un CSV et affiche, dans une figure Plotly, I/Q en fonction du temps, la constellation et le spectre (dB ou linéaire).

**Architecture :** petit package à plat, un module par responsabilité : `config` (dataclasses), `sources` (interface `IQSource` + `CsvIQSource`), `processing` (fonctions numpy pures), `plotting` (figure Plotly), `main` (argparse et orchestration). La seule abstraction est `IQSource`, point de remplacement de la source de données.

**Tech Stack :** Python 3.9+, numpy, pandas, plotly ; pytest et flake8 pour le développement.

**Spec :** `docs/superpowers/specs/2026-10-07-iq-visualizer-design.md`

## Global Constraints

- Python 3.9 minimum : `Optional`, `Union`, `List`, `Tuple`, `Dict` de `typing` ; jamais `X | None`, `list[int]` en annotation, ni `match`.
- PEP 8 vérifié par `flake8 iq_visualizer tests` (79 caractères par ligne, configuration par défaut).
- Dépendances d'exécution : numpy, pandas, plotly uniquement. Développement : pytest, flake8.
- Pas d'usine à gaz : pas de registre, de fabrique ni de sous-package ; `IQSource` est la seule classe abstraite.
- Messages d'erreur et docstrings en français ; noms de code en anglais.
- Toujours lancer les tests avec `.venv/bin/python -m pytest` depuis la racine du dépôt (le `-m` ajoute la racine au `sys.path`, donc pas besoin d'installer le package).
- Les valeurs par défaut n'existent qu'à un seul endroit : les dataclasses de `config.py`.

## Review Focus

1. **Mauvais séparateur** (CSV `1;2` lu avec `,`) : erreur claire qui évoque le séparateur, pas une `IndexError` de pandas. Testé dans la Task 2 (`test_wrong_separator_gives_explicit_error`).
2. **En-tête présent mais non déclaré, ou valeurs manquantes** : erreur claire, pas un graphique vide ou rempli de NaN. Testé dans la Task 2 (`test_undeclared_header_gives_explicit_error`, `test_missing_value_gives_explicit_error`).
3. **Colonnes I/Q non contiguës ou inversées** (`--i-col 3 --q-col 1`) : l'ordre demandé est respecté. `usecols` de pandas renvoie l'ordre du fichier, d'où la sélection par `iloc`. Testé dans la Task 2 (`test_reads_non_contiguous_columns_in_requested_order`).
4. **Signal très court** (1 ou 2 échantillons) ou silence total : pas de NaN, pas de `-inf`, pas de division par zéro (`np.hanning(2) == [0, 0]`). Testé dans la Task 3 (`test_single_sample_signal_is_supported`, `test_signal_too_short_for_hann_is_rejected`, `test_db_scale_has_no_infinite_values_on_silence`).
5. **Valeurs de paramètres absurdes depuis la CLI** (`--fs 0`, `--nfft -1`, `--max-points 0`, `--i-col` égal à `--q-col`) : message d'erreur et code de sortie 1, sans traceback. Testé dans les Tasks 1 (`test_*_rejects_*`) et 5 (`test_main_reports_invalid_option_value`).

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `iq_visualizer/__init__.py` | marque le package |
| `iq_visualizer/config.py` | `CsvConfig`, `VisualizerConfig` (défauts et validation) |
| `iq_visualizer/sources.py` | `IQSource` (ABC), `CsvIQSource` |
| `iq_visualizer/processing.py` | `WINDOWS`, `SCALES`, `time_axis()`, `compute_spectrum()` |
| `iq_visualizer/plotting.py` | `build_figure()` et trois fonctions privées |
| `iq_visualizer/main.py` | `SEPARATOR_ALIASES`, `parse_args()`, `run()`, `main()` |
| `tests/test_*.py` | un fichier de tests par module |
| `examples/sample.csv` | signal de démonstration (2 tons + bruit, Fs = 1 MHz) |
| `requirements.txt`, `requirements-dev.txt` | dépendances |
| `README.md`, `TODO.md`, `.gitignore` | documentation, pistes futures |

---

### Task 1 : squelette du projet et configuration

**Files :**
- Create : `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `iq_visualizer/__init__.py`, `iq_visualizer/config.py`
- Test : `tests/test_config.py`

**Interfaces :**
- Consumes : rien.
- Produces :
  - `CsvConfig(path: Path, separator: str = ",", decimal: str = ".", has_header: bool = False, i_column: int = 0, q_column: int = 1)`
  - `VisualizerConfig(sample_rate: Optional[float] = None, nfft: Optional[int] = None, window: Literal["hann", "rectangular"] = "hann", scale: Literal["db", "linear"] = "db", max_plot_samples: int = 10_000, output_html: Optional[Path] = None)`
  - Les deux lèvent `ValueError` dans `__post_init__` si une valeur est invalide.

- [ ] **Step 1 : Créer les fichiers de dépendances et le `.gitignore`**

`requirements.txt` :

```text
numpy>=1.21
pandas>=1.3
plotly>=5.0
```

`requirements-dev.txt` :

```text
-r requirements.txt
pytest>=7.0
flake8>=5.0
```

`.gitignore` :

```text
.venv/
__pycache__/
.pytest_cache/
*.html
```

`iq_visualizer/__init__.py` :

```python
"""Visualisation d'échantillons IQ : temps, constellation et spectre."""
```

- [ ] **Step 2 : Créer l'environnement virtuel**

Run : `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`
Expected : installation réussie de numpy, pandas, plotly, pytest, flake8.

- [ ] **Step 3: Écrire les tests (ils doivent échouer)**

Créer `tests/test_config.py` :

```python
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
```

- [ ] **Step 4: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL (erreur de collecte) avec `ModuleNotFoundError: No module named 'iq_visualizer.config'`

- [ ] **Step 5: Écrire l'implémentation**

Créer `iq_visualizer/config.py` :

```python
"""Paramètres de lecture CSV et de visualisation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class CsvConfig:
    """Paramètres de lecture d'un fichier CSV contenant des colonnes I et Q."""

    path: Path
    separator: str = ","      # ",", ";", "\t", r"\s+" (espaces)
    decimal: str = "."        # "," pour les CSV "à la française"
    has_header: bool = False
    i_column: int = 0         # index de la colonne I
    q_column: int = 1         # index de la colonne Q

    def __post_init__(self) -> None:
        if self.i_column < 0 or self.q_column < 0:
            raise ValueError("Les index de colonnes doivent être positifs.")
        if self.i_column == self.q_column:
            raise ValueError("Les colonnes I et Q doivent être différentes.")


@dataclass
class VisualizerConfig:
    """Paramètres d'analyse et d'affichage."""

    sample_rate: Optional[float] = None   # Hz ; None → échantillons
    nfft: Optional[int] = None            # None → tout le signal
    window: Literal["hann", "rectangular"] = "hann"
    scale: Literal["db", "linear"] = "db"
    max_plot_samples: int = 10_000        # points tracés (temps, IQ)
    output_html: Optional[Path] = None    # sauvegarde optionnelle

    def __post_init__(self) -> None:
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("La fréquence d'échantillonnage doit être > 0.")
        if self.nfft is not None and self.nfft <= 0:
            raise ValueError("nfft doit être > 0.")
        if self.max_plot_samples <= 0:
            raise ValueError("max_plot_samples doit être > 0.")
```

- [ ] **Step 6: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, `8 passed`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt .gitignore iq_visualizer/__init__.py iq_visualizer/config.py tests/test_config.py
git commit -m "feat: add project skeleton and configuration dataclasses"
```

---

### Task 2 : sources d'IQ (`IQSource`, `CsvIQSource`)

**Files :**
- Create : `iq_visualizer/sources.py`
- Test : `tests/test_sources.py`

**Interfaces :**
- Consumes : `CsvConfig` (Task 1).
- Produces :
  - `class IQSource(ABC)` avec `read(self) -> np.ndarray` abstraite (tableau 1D complexe).
  - `class CsvIQSource(IQSource)` avec `__init__(self, config: CsvConfig)` et `read(self) -> np.ndarray`.
  - Erreurs : `FileNotFoundError` (natif) ; `ValueError` si fichier vide (message contenant « vide »), pas assez de colonnes (« séparateur »), colonnes non numériques (« numériques »), aucun échantillon (« Aucun échantillon »), valeurs manquantes (« manquantes »).

Point d'attention : ne pas utiliser `usecols`, car pandas renvoie alors les colonnes dans l'ordre du fichier et non dans l'ordre demandé. On lit tout puis on sélectionne avec `iloc[:, [i_column, q_column]]`.

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

Créer `tests/test_sources.py` :

```python
import numpy as np
import pytest

from iq_visualizer.config import CsvConfig
from iq_visualizer.sources import CsvIQSource, IQSource


def read_csv_text(tmp_path, text, **csv_options):
    path = tmp_path / "iq.csv"
    path.write_text(text)
    return CsvIQSource(CsvConfig(path=path, **csv_options)).read()


def test_csv_source_is_an_iq_source(tmp_path):
    assert isinstance(CsvIQSource(CsvConfig(path=tmp_path)), IQSource)


def test_reads_comma_separated_values(tmp_path):
    iq = read_csv_text(tmp_path, "1,2\n3,-4\n")
    np.testing.assert_array_equal(iq, [1 + 2j, 3 - 4j])
    assert np.iscomplexobj(iq)


def test_reads_semicolon_and_decimal_comma(tmp_path):
    iq = read_csv_text(tmp_path, "1,5;2\n3;-4,25\n",
                       separator=";", decimal=",")
    np.testing.assert_array_equal(iq, [1.5 + 2j, 3 - 4.25j])


def test_reads_whitespace_separated_values(tmp_path):
    iq = read_csv_text(tmp_path, "1 2\n3    4\n", separator=r"\s+")
    np.testing.assert_array_equal(iq, [1 + 2j, 3 + 4j])


def test_reads_file_with_header(tmp_path):
    iq = read_csv_text(tmp_path, "I,Q\n1,2\n", has_header=True)
    np.testing.assert_array_equal(iq, [1 + 2j])


def test_reads_non_contiguous_columns_in_requested_order(tmp_path):
    iq = read_csv_text(tmp_path, "0,10,99,20\n1,11,99,21\n",
                       i_column=3, q_column=1)
    np.testing.assert_array_equal(iq, [20 + 10j, 21 + 11j])


def test_undeclared_header_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="numériques"):
        read_csv_text(tmp_path, "I,Q\n1,2\n")


def test_wrong_separator_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="séparateur"):
        read_csv_text(tmp_path, "1;2\n3;4\n")


def test_empty_file_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="vide"):
        read_csv_text(tmp_path, "")


def test_header_only_file_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="Aucun échantillon"):
        read_csv_text(tmp_path, "I,Q\n", has_header=True)


def test_missing_value_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="manquantes"):
        read_csv_text(tmp_path, "1,2\n3,\n")


def test_missing_file_raises_file_not_found(tmp_path):
    source = CsvIQSource(CsvConfig(path=tmp_path / "absent.csv"))
    with pytest.raises(FileNotFoundError):
        source.read()
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_sources.py -v`
Expected: FAIL (erreur de collecte) avec `ModuleNotFoundError: No module named 'iq_visualizer.sources'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `iq_visualizer/sources.py` :

```python
"""Sources d'échantillons IQ.

Pour ajouter une source (fichier binaire, SDR...), il suffit d'hériter de
IQSource et d'implémenter read().
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from iq_visualizer.config import CsvConfig


class IQSource(ABC):
    """Interface commune à toutes les sources d'IQ."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Renvoie les échantillons sous forme de tableau numpy 1D complexe."""


class CsvIQSource(IQSource):
    """Lit les échantillons IQ depuis deux colonnes d'un fichier CSV."""

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
            raise ValueError(f"Le fichier {cfg.path} est vide.") from None

        if table.shape[1] <= max(cfg.i_column, cfg.q_column):
            raise ValueError(
                f"{cfg.path} contient {table.shape[1]} colonne(s), "
                f"impossible de lire les colonnes {cfg.i_column} et "
                f"{cfg.q_column}. Le séparateur {cfg.separator!r} "
                f"est-il correct ?"
            )

        columns = table.iloc[:, [cfg.i_column, cfg.q_column]]
        if columns.empty:
            raise ValueError(f"Aucun échantillon dans {cfg.path}.")
        if not all(is_numeric_dtype(dtype) for dtype in columns.dtypes):
            raise ValueError(
                f"Les colonnes I/Q de {cfg.path} ne sont pas numériques. "
                f"Vérifier le séparateur, la décimale et l'option d'en-tête."
            )
        if columns.isna().to_numpy().any():
            raise ValueError(f"Valeurs manquantes dans {cfg.path}.")

        i_values = columns.iloc[:, 0].to_numpy(dtype=float)
        q_values = columns.iloc[:, 1].to_numpy(dtype=float)
        return i_values + 1j * q_values
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_sources.py -v`
Expected: PASS, `12 passed`

- [ ] **Step 5: Commit**

```bash
git add iq_visualizer/sources.py tests/test_sources.py
git commit -m "feat: add IQSource interface and CSV source"
```

---

### Task 3 : traitement du signal (`time_axis`, `compute_spectrum`)

**Files :**
- Create : `iq_visualizer/processing.py`
- Test : `tests/test_processing.py`

**Interfaces :**
- Consumes : rien (numpy uniquement, ne doit importer ni pandas ni plotly).
- Produces :
  - `WINDOWS: Dict[str, Callable[[int], np.ndarray]] = {"hann": np.hanning, "rectangular": np.ones}`
  - `SCALES = ("db", "linear")`
  - `time_axis(n_samples: int, sample_rate: Optional[float] = None) -> np.ndarray`
  - `compute_spectrum(iq: np.ndarray, sample_rate: Optional[float] = None, nfft: Optional[int] = None, window: str = "hann", scale: str = "db") -> Tuple[np.ndarray, np.ndarray]` qui renvoie `(fréquences, valeurs)` triées de −Fs/2 à +Fs/2.
  - Normalisation : `|X| / sum(fenêtre)`, donc une sinusoïde complexe d'amplitude 1 donne un pic à 1 en linéaire, soit 0 dB.

Rappel du test clé : une sinusoïde de 100 Hz échantillonnée à 1000 Hz sur 1000 points tombe exactement sur un bin, son pic vaut donc exactement `sum(fenêtre) / sum(fenêtre) = 1`.

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

Créer `tests/test_processing.py` :

```python
import numpy as np
import pytest

from iq_visualizer.processing import compute_spectrum, time_axis


def complex_tone(frequency, sample_rate, n_samples):
    times = np.arange(n_samples) / sample_rate
    return np.exp(2j * np.pi * frequency * times)


def test_time_axis_in_seconds_with_sample_rate():
    np.testing.assert_allclose(time_axis(4, 2.0), [0.0, 0.5, 1.0, 1.5])


def test_time_axis_in_samples_without_sample_rate():
    np.testing.assert_array_equal(time_axis(3), [0, 1, 2])


@pytest.mark.parametrize("window", ["hann", "rectangular"])
def test_tone_peak_is_at_its_frequency_and_0_db(window):
    iq = complex_tone(100.0, 1000.0, 1000)
    frequencies, values = compute_spectrum(iq, sample_rate=1000.0,
                                           window=window, scale="db")
    peak = np.argmax(values)
    assert frequencies[peak] == pytest.approx(100.0)
    assert values[peak] == pytest.approx(0.0, abs=1e-9)


def test_tone_peak_is_1_in_linear_scale():
    iq = complex_tone(-250.0, 1000.0, 1000)
    frequencies, values = compute_spectrum(iq, sample_rate=1000.0,
                                           scale="linear")
    peak = np.argmax(values)
    assert frequencies[peak] == pytest.approx(-250.0)
    assert values[peak] == pytest.approx(1.0)


def test_frequencies_are_centered_and_sorted():
    frequencies, _ = compute_spectrum(np.ones(8), sample_rate=8.0)
    np.testing.assert_allclose(frequencies, np.arange(-4, 4))


def test_normalized_frequencies_without_sample_rate():
    frequencies, _ = compute_spectrum(np.ones(8))
    assert frequencies[0] == pytest.approx(-0.5)
    assert frequencies[-1] < 0.5


def test_nfft_smaller_than_signal_truncates():
    frequencies, values = compute_spectrum(np.ones(100), nfft=16)
    assert frequencies.size == values.size == 16


def test_nfft_larger_than_signal_zero_pads():
    iq = complex_tone(100.0, 1000.0, 1000)
    frequencies, values = compute_spectrum(iq, sample_rate=1000.0,
                                           nfft=4000, scale="linear")
    assert frequencies.size == 4000
    peak = np.argmax(values)
    assert frequencies[peak] == pytest.approx(100.0)
    assert values[peak] == pytest.approx(1.0)


def test_db_scale_has_no_infinite_values_on_silence():
    _, values = compute_spectrum(np.zeros(16, dtype=complex),
                                 window="rectangular")
    assert np.all(np.isfinite(values))


def test_single_sample_signal_is_supported():
    frequencies, values = compute_spectrum(np.array([1 + 1j]))
    assert frequencies.size == values.size == 1


@pytest.mark.parametrize("kwargs, message", [
    ({"window": "blackman"}, "Fenêtre"),
    ({"scale": "power"}, "Échelle"),
])
def test_unknown_option_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_spectrum(np.ones(8), **kwargs)


def test_empty_signal_is_rejected():
    with pytest.raises(ValueError, match="vide"):
        compute_spectrum(np.array([], dtype=complex))


def test_signal_too_short_for_hann_is_rejected():
    with pytest.raises(ValueError, match="trop court"):
        compute_spectrum(np.ones(2, dtype=complex), window="hann")
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_processing.py -v`
Expected: FAIL (erreur de collecte) avec `ModuleNotFoundError: No module named 'iq_visualizer.processing'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `iq_visualizer/processing.py` :

```python
"""Calculs sur les échantillons IQ (numpy uniquement)."""

from typing import Optional, Tuple

import numpy as np

WINDOWS = {"hann": np.hanning, "rectangular": np.ones}
SCALES = ("db", "linear")
DB_FLOOR = 1e-12  # évite log10(0)


def time_axis(n_samples: int,
              sample_rate: Optional[float] = None) -> np.ndarray:
    """Secondes si sample_rate est fourni, sinon indices d'échantillons."""
    indices = np.arange(n_samples)
    if sample_rate is None:
        return indices
    return indices / sample_rate


def compute_spectrum(
    iq: np.ndarray,
    sample_rate: Optional[float] = None,
    nfft: Optional[int] = None,
    window: str = "hann",
    scale: str = "db",
) -> Tuple[np.ndarray, np.ndarray]:
    """Renvoie (fréquences, valeurs) du spectre, centrés sur 0.

    Le spectre est normalisé : une sinusoïde complexe d'amplitude 1
    donne un pic à 1 en linéaire, soit 0 dB.
    """
    if iq.size == 0:
        raise ValueError("Le signal est vide.")
    if window not in WINDOWS:
        raise ValueError(
            f"Fenêtre inconnue {window!r}, choix : {sorted(WINDOWS)}."
        )
    if scale not in SCALES:
        raise ValueError(f"Échelle inconnue {scale!r}, choix : {SCALES}.")

    n_fft = iq.size if nfft is None else nfft
    segment = iq[:n_fft]
    weights = WINDOWS[window](segment.size)
    if weights.sum() == 0:
        raise ValueError(
            f"Signal trop court ({segment.size} échantillons) "
            f"pour la fenêtre {window!r}."
        )

    spectrum = np.fft.fftshift(np.fft.fft(segment * weights, n=n_fft))
    magnitude = np.abs(spectrum) / weights.sum()

    spacing = 1.0 if sample_rate is None else 1.0 / sample_rate
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_fft, d=spacing))

    if scale == "db":
        return frequencies, 20 * np.log10(np.maximum(magnitude, DB_FLOOR))
    return frequencies, magnitude
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_processing.py -v`
Expected: PASS, `15 passed`

- [ ] **Step 5: Commit**

```bash
git add iq_visualizer/processing.py tests/test_processing.py
git commit -m "feat: add time axis and normalized spectrum computation"
```

---

### Task 4 : figure Plotly (`build_figure`)

**Files :**
- Create : `iq_visualizer/plotting.py`
- Test : `tests/test_plotting.py`

**Interfaces :**
- Consumes : `VisualizerConfig` (Task 1), `time_axis`, `compute_spectrum` (Task 3).
- Produces : `build_figure(iq: np.ndarray, config: VisualizerConfig) -> go.Figure`, qui construit la figure sans l'afficher. Traces dans cet ordre, avec ces noms : `"I"`, `"Q"`, `"Constellation"`, `"Spectre"`. Axes Plotly : `xaxis`/`yaxis` pour le temps, `xaxis2`/`yaxis2` pour la constellation, `xaxis3`/`yaxis3` pour le spectre.

Notes :
- `go.Scattergl` (WebGL) plutôt que `go.Scatter`, pour rester fluide quand le spectre porte sur un long signal. L'API est identique.
- La constellation garde un ratio 1:1 grâce à `scaleanchor="x2"`.

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

Créer `tests/test_plotting.py` :

```python
import numpy as np

from iq_visualizer.config import VisualizerConfig
from iq_visualizer.plotting import build_figure


def test_figure_has_time_constellation_and_spectrum_traces():
    iq = np.exp(2j * np.pi * 0.1 * np.arange(64))
    figure = build_figure(iq, VisualizerConfig())
    names = [trace.name for trace in figure.data]
    assert names == ["I", "Q", "Constellation", "Spectre"]


def test_time_and_constellation_respect_max_plot_samples():
    iq = np.ones(500, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(max_plot_samples=100))
    i_trace, q_trace, constellation, spectrum = figure.data
    assert len(i_trace.x) == len(q_trace.x) == len(constellation.x) == 100
    assert len(spectrum.x) == 500  # spectre sur tout le signal


def test_signal_shorter_than_max_plot_samples_is_fully_plotted():
    iq = np.ones(10, dtype=complex)
    figure = build_figure(iq, VisualizerConfig(max_plot_samples=100))
    assert len(figure.data[0].x) == 10


def test_axis_labels_follow_configuration():
    iq = np.ones(16, dtype=complex)
    with_fs = build_figure(iq, VisualizerConfig(sample_rate=1e3,
                                                scale="linear"))
    assert with_fs.layout.xaxis.title.text == "Temps (s)"
    assert with_fs.layout.xaxis3.title.text == "Fréquence (Hz)"
    assert with_fs.layout.yaxis3.title.text == "Magnitude"

    without_fs = build_figure(iq, VisualizerConfig())
    assert without_fs.layout.xaxis.title.text == "Échantillon"
    assert without_fs.layout.xaxis3.title.text == "Fréquence normalisée"
    assert without_fs.layout.yaxis3.title.text == "Magnitude (dB)"
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: FAIL (erreur de collecte) avec `ModuleNotFoundError: No module named 'iq_visualizer.plotting'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `iq_visualizer/plotting.py` :

```python
"""Construction de la figure Plotly : temps, constellation, spectre."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iq_visualizer.config import VisualizerConfig
from iq_visualizer.processing import compute_spectrum, time_axis


def build_figure(iq: np.ndarray, config: VisualizerConfig) -> go.Figure:
    """Construit la figure (sans l'afficher)."""
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("I / Q", "Constellation", "Spectre"),
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
    # Scattergl (WebGL) reste fluide avec beaucoup de points.
    figure.add_trace(go.Scattergl(x=times, y=iq.real, name="I"),
                     row=1, col=1)
    figure.add_trace(go.Scattergl(x=times, y=iq.imag, name="Q"),
                     row=1, col=1)
    x_label = "Échantillon" if config.sample_rate is None else "Temps (s)"
    figure.update_xaxes(title_text=x_label, row=1, col=1)
    figure.update_yaxes(title_text="Amplitude", row=1, col=1)


def _add_constellation(figure: go.Figure, iq: np.ndarray) -> None:
    figure.add_trace(
        go.Scattergl(x=iq.real, y=iq.imag, mode="markers",
                     marker={"size": 3}, name="Constellation"),
        row=1, col=2,
    )
    figure.update_xaxes(title_text="I", row=1, col=2)
    # Ratio 1:1 pour ne pas déformer la constellation.
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
    figure.add_trace(go.Scattergl(x=frequencies, y=values, name="Spectre"),
                     row=2, col=1)
    x_label = ("Fréquence normalisée" if config.sample_rate is None
               else "Fréquence (Hz)")
    y_label = "Magnitude (dB)" if config.scale == "db" else "Magnitude"
    figure.update_xaxes(title_text=x_label, row=2, col=1)
    figure.update_yaxes(title_text=y_label, row=2, col=1)
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_plotting.py -v`
Expected: PASS, `4 passed`

- [ ] **Step 5: Commit**

```bash
git add iq_visualizer/plotting.py tests/test_plotting.py
git commit -m "feat: add Plotly figure with time, constellation and spectrum"
```

---

### Task 5 : ligne de commande et orchestration (`main.py`)

**Files :**
- Create : `iq_visualizer/main.py`
- Test : `tests/test_main.py`

**Interfaces :**
- Consumes : `CsvConfig`, `VisualizerConfig` (Task 1) ; `IQSource`, `CsvIQSource` (Task 2) ; `WINDOWS`, `SCALES` (Task 3) ; `build_figure` (Task 4).
- Produces :
  - `SEPARATOR_ALIASES = {"comma": ",", "semicolon": ";", "tab": "\t", "space": r"\s+"}`
  - `parse_args(argv: Optional[List[str]] = None) -> Tuple[CsvConfig, VisualizerConfig]`
  - `run(source: IQSource, config: VisualizerConfig) -> None`
  - `main(argv: Optional[List[str]] = None) -> int` (0 en cas de succès, 1 en cas d'erreur connue)

Notes :
- Les défauts argparse sont lus dans les dataclasses via `_defaults()` (`dataclasses.fields`), ils ne sont jamais recopiés.
- `--header` utilise `argparse.BooleanOptionalAction` (Python 3.9+), qui donne `--header` et `--no-header`.
- `parse_args` est appelé **dans** le `try` de `main`, pour que les `ValueError` de validation des dataclasses produisent un message propre.
- Les tests remplacent `go.Figure.show` avec `monkeypatch` pour ne jamais ouvrir de navigateur.

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

Créer `tests/test_main.py` :

```python
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
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_main.py -v`
Expected: FAIL (erreur de collecte) avec `ModuleNotFoundError: No module named 'iq_visualizer.main'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `iq_visualizer/main.py` :

```python
"""Point d'entrée en ligne de commande.

Exemple :
    python -m iq_visualizer.main data.csv --sep semicolon --fs 1e6
"""

import argparse
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iq_visualizer.config import CsvConfig, VisualizerConfig
from iq_visualizer.plotting import build_figure
from iq_visualizer.processing import SCALES, WINDOWS
from iq_visualizer.sources import CsvIQSource, IQSource

SEPARATOR_ALIASES = {
    "comma": ",",
    "semicolon": ";",
    "tab": "\t",
    "space": r"\s+",
}


def _defaults(config_class: type) -> Dict[str, Any]:
    """Valeurs par défaut d'une dataclass (source unique des défauts)."""
    return {field.name: field.default for field in fields(config_class)
            if field.default is not MISSING}


def parse_args(
    argv: Optional[List[str]] = None,
) -> Tuple[CsvConfig, VisualizerConfig]:
    """Construit les configurations à partir de la ligne de commande."""
    csv_defaults = _defaults(CsvConfig)
    vis_defaults = _defaults(VisualizerConfig)

    parser = argparse.ArgumentParser(
        description="Visualise des échantillons IQ lus dans un fichier CSV.")
    parser.add_argument("path", type=Path,
                        help="fichier CSV contenant les colonnes I et Q")
    parser.add_argument("--sep", default=csv_defaults["separator"],
                        help="séparateur : comma, semicolon, tab, space "
                             "ou caractère brut (défaut : %(default)r)")
    parser.add_argument("--decimal", default=csv_defaults["decimal"],
                        help="séparateur décimal (défaut : %(default)r)")
    parser.add_argument("--header", action=argparse.BooleanOptionalAction,
                        default=csv_defaults["has_header"],
                        help="la première ligne est un en-tête")
    parser.add_argument("--i-col", type=int,
                        default=csv_defaults["i_column"],
                        help="index de la colonne I (défaut : %(default)s)")
    parser.add_argument("--q-col", type=int,
                        default=csv_defaults["q_column"],
                        help="index de la colonne Q (défaut : %(default)s)")
    parser.add_argument("--fs", type=float,
                        default=vis_defaults["sample_rate"],
                        help="fréquence d'échantillonnage en Hz")
    parser.add_argument("--nfft", type=int, default=vis_defaults["nfft"],
                        help="taille de la FFT (défaut : tout le signal)")
    parser.add_argument("--window", choices=sorted(WINDOWS),
                        default=vis_defaults["window"])
    parser.add_argument("--scale", choices=SCALES,
                        default=vis_defaults["scale"])
    parser.add_argument("--max-points", type=int,
                        default=vis_defaults["max_plot_samples"],
                        help="points tracés en temps et constellation "
                             "(défaut : %(default)s)")
    parser.add_argument("--save", type=Path,
                        default=vis_defaults["output_html"],
                        help="sauvegarde la figure dans ce fichier HTML")
    args = parser.parse_args(argv)

    csv_config = CsvConfig(
        path=args.path,
        separator=SEPARATOR_ALIASES.get(args.sep, args.sep),
        decimal=args.decimal,
        has_header=args.header,
        i_column=args.i_col,
        q_column=args.q_col,
    )
    vis_config = VisualizerConfig(
        sample_rate=args.fs,
        nfft=args.nfft,
        window=args.window,
        scale=args.scale,
        max_plot_samples=args.max_points,
        output_html=args.save,
    )
    return csv_config, vis_config


def run(source: IQSource, config: VisualizerConfig) -> None:
    """Lit la source, construit la figure, la sauvegarde et l'affiche."""
    iq = source.read()
    figure = build_figure(iq, config)
    if config.output_html is not None:
        figure.write_html(str(config.output_html))
    figure.show()


def main(argv: Optional[List[str]] = None) -> int:
    """Renvoie le code de sortie du programme."""
    try:
        csv_config, vis_config = parse_args(argv)
        run(CsvIQSource(csv_config), vis_config)
    except (FileNotFoundError, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_main.py -v`
Expected: PASS, `10 passed`

- [ ] **Step 5: Commit**

```bash
git add iq_visualizer/main.py tests/test_main.py
git commit -m "feat: add command line interface"
```

---

### Task 6 : exemple, documentation et vérification de bout en bout

**Files :**
- Create : `examples/sample.csv` (généré), `README.md`, `TODO.md`

**Interfaces :**
- Consumes : tout le package (Tasks 1 à 5).
- Produces : documentation utilisateur et fichier de démonstration.

- [ ] **Step 1 : Générer `examples/sample.csv`**

Run : `mkdir -p examples && .venv/bin/python -c '<script ci-dessous>'`, ou enregistrer le script dans le scratchpad et l'exécuter. Il ne fait pas partie du dépôt.

```python
import numpy as np

rng = np.random.default_rng(seed=0)
fs, n = 1e6, 4096
t = np.arange(n) / fs
iq = (np.exp(2j * np.pi * 100e3 * t)
      + 0.1 * np.exp(-2j * np.pi * 250e3 * t)
      + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
np.savetxt("examples/sample.csv", np.column_stack([iq.real, iq.imag]),
           delimiter=",", fmt="%.6f")
```

Expected : `examples/sample.csv` contient 4096 lignes, la première étant `1.106287,-0.098238`.

Run : `wc -l examples/sample.csv && head -1 examples/sample.csv`

- [ ] **Step 2 : Écrire `README.md`**

````markdown
# IQ Visualizer

Visualise des échantillons IQ lus dans un fichier CSV : I/Q en fonction du
temps, constellation et spectre (dB ou magnitude linéaire), dans une figure
Plotly interactive ouverte dans le navigateur.

Compatible Python 3.9+.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # utilisation
.venv/bin/pip install -r requirements-dev.txt    # développement (tests, flake8)
```

## Utilisation

```bash
.venv/bin/python -m iq_visualizer.main examples/sample.csv --fs 1e6
```

Le fichier CSV contient une ligne par échantillon, avec une colonne I et une
colonne Q.

| Option | Rôle | Défaut |
|---|---|---|
| `--sep` | séparateur : `comma`, `semicolon`, `tab`, `space` ou caractère brut | `,` |
| `--decimal` | séparateur décimal | `.` |
| `--header` / `--no-header` | la première ligne est un en-tête | non |
| `--i-col`, `--q-col` | index des colonnes I et Q | `0`, `1` |
| `--fs` | fréquence d'échantillonnage (Hz) | aucune (échantillons, fréquence normalisée) |
| `--nfft` | taille de la FFT | tout le signal |
| `--window` | `hann` ou `rectangular` | `hann` |
| `--scale` | `db` ou `linear` | `db` |
| `--max-points` | points tracés en temps et constellation | `10000` |
| `--save` | sauvegarde la figure en HTML | aucune |

Exemple pour un CSV « à la française » avec en-tête :

```bash
.venv/bin/python -m iq_visualizer.main mesure.csv --sep semicolon --decimal "," --header --fs 2.4e6
```

Les valeurs par défaut se changent dans `iq_visualizer/config.py`.

## Échelle du spectre

Le spectre est normalisé : une sinusoïde complexe d'amplitude 1 donne un pic
à 1 en linéaire, soit 0 dB.

## Ajouter une source d'IQ

Hériter de `IQSource` (`iq_visualizer/sources.py`) et implémenter `read()`,
qui renvoie un tableau numpy 1D complexe, puis la passer à
`iq_visualizer.main.run()` :

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
````

- [ ] **Step 3 : Écrire `TODO.md`**

```markdown
# TODO

## Spectre moyenné (méthode de Welch)

Découper le signal en segments (avec recouvrement), calculer la FFT de chacun
et moyenner les magnitudes, pour obtenir un spectre plus lisible sur un signal
bruité. Seule `compute_spectrum()` dans `iq_visualizer/processing.py` est
concernée (paramètres supplémentaires : taille de segment, recouvrement).

## Acquisition en temps réel (streaming)

Lire les IQ en continu (SDR, socket...) et rafraîchir les graphiques. Une
nouvelle classe héritant de `IQSource` fournirait les blocs d'échantillons ;
l'affichage devra passer d'une figure statique à une mise à jour périodique
(par exemple Plotly Dash ou `FigureWidget`).
```

- [ ] **Step 4 : Suite complète et PEP 8**

Run : `.venv/bin/python -m pytest -q && .venv/bin/flake8 iq_visualizer tests`
Expected : `49 passed`, et aucune sortie de flake8.

- [ ] **Step 5 : Lancement réel**

Run : `.venv/bin/python -m iq_visualizer.main examples/sample.csv --fs 1e6 --save /tmp/iq_demo.html; echo "exit=$?"`
Expected : `exit=0`, le navigateur s'ouvre sur la figure : pic à +100 kHz vers 0 dB, second pic à −250 kHz vers −20 dB, constellation en anneau autour du rayon 1. `/tmp/iq_demo.html` existe.

Run : `.venv/bin/python -m iq_visualizer.main absent.csv; echo "exit=$?"`
Expected : `Erreur : ...` sur stderr, `exit=1`, aucun traceback.

- [ ] **Step 6 : Commit**

```bash
git add examples/sample.csv README.md TODO.md
git commit -m "docs: add README, TODO and sample IQ file"
```

---

### Task 7 : vérification sous Python 3.9

**Files :** aucun (vérification seulement, sauf correctif éventuel).

Contexte : la machine a Python 3.14 ; ni `python3.9` ni `uv` ne sont installés.

- [ ] **Step 1 : Obtenir un interpréteur 3.9**

Si `uv` est absent, **demander à l'utilisateur** l'autorisation de l'installer (`brew install uv`) ou un autre moyen d'obtenir Python 3.9. Ne rien installer sans accord.

- [ ] **Step 2 : Lancer la suite sous 3.9**

Run : `uv run --python 3.9 --with-requirements requirements-dev.txt python -m pytest -q`
Expected : `49 passed`.

- [ ] **Step 3 : En cas d'échec**

Corriger la cause (syntaxe ou API non disponible en 3.9) dans le module concerné, relancer les étapes 2 de cette tâche et 4 de la Task 6, puis faire un commit `fix: restore Python 3.9 compatibility`. Si l'utilisateur refuse l'installation, le signaler explicitement dans le compte rendu final : la compatibilité 3.9 n'est alors **pas** vérifiée.
