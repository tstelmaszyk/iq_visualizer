# IQ Visualizer — Design

Date : 2026-10-07

## 1. Objectif

Programme Python permettant de visualiser des échantillons IQ chargés depuis
un fichier CSV, sous trois vues :

- I(t) et Q(t) dans le domaine temporel ;
- constellation (Q en fonction de I) ;
- spectre (FFT), affichable en dB ou en magnitude linéaire.

### Contraintes

- Python 3.9 minimum (pas de syntaxe `X | None`, pas de `match`).
- PEP 8, vérifié par `flake8`.
- Bibliothèques reconnues uniquement : numpy, pandas, plotly (pytest et
  flake8 pour le développement).
- Code clair et paramétrable, avec classes et fonctions, **sans usine à gaz** :
  une seule abstraction (`IQSource`), le reste en fonctions simples.

### Critères de réussite

- Changer le fichier, le séparateur, Fs ou l'échelle (dB / linéaire) se fait
  par un argument de ligne de commande ou une valeur par défaut de dataclass.
- Ajouter une nouvelle source d'IQ se limite à écrire une classe implémentant
  `IQSource.read()`, sans modifier le reste du code.

### Hors périmètre (voir `TODO.md`)

- Spectre moyenné (méthode de Welch).
- Acquisition en temps réel / streaming.

## 2. Architecture

```
iq_visualizer/
├── __init__.py
├── config.py        # dataclasses CsvConfig et VisualizerConfig
├── sources.py       # IQSource (ABC) + CsvIQSource
├── processing.py    # time_axis(), compute_spectrum() — numpy pur
├── plotting.py      # build_figure() — Plotly
└── main.py          # argparse → config → source → figure → affichage
tests/
├── test_sources.py
├── test_processing.py
├── test_plotting.py
└── test_main.py
examples/sample.csv  # sinusoïde complexe + bruit
requirements.txt      # numpy, pandas, plotly
requirements-dev.txt  # pytest, flake8
README.md
TODO.md
```

Flux de données :

```
CLI (argparse) ──► CsvConfig, VisualizerConfig
CsvConfig ──► CsvIQSource.read() ──► np.ndarray complexe 1D
iq + VisualizerConfig ──► build_figure() ──► go.Figure ──► show() / write_html()
                              │
                              └─► processing.time_axis(), compute_spectrum()
```

Dépendances entre modules : `main` dépend de tous ; `plotting` dépend de
`config` et `processing` ; `sources` dépend de `config` ; `processing` ne
dépend que de numpy.

## 3. Composants

### 3.1 `config.py`

```python
@dataclass
class CsvConfig:
    path: Path
    separator: str = ","      # ",", ";", "\t", r"\s+" (espaces)
    decimal: str = "."        # "," pour les CSV "à la française"
    has_header: bool = False
    i_column: int = 0         # index de la colonne I
    q_column: int = 1         # index de la colonne Q


@dataclass
class VisualizerConfig:
    sample_rate: Optional[float] = None
    nfft: Optional[int] = None
    window: Literal["hann", "rectangular"] = "hann"
    scale: Literal["db", "linear"] = "db"
    max_plot_samples: int = 10_000
    output_html: Optional[Path] = None
```

- `sample_rate = None` : axe temporel en indices d'échantillons, axe
  fréquentiel normalisé (−0,5 à +0,5 cycle/échantillon).
- `nfft = None` : FFT sur tout le signal.
- `max_plot_samples` : seuls les N premiers échantillons sont tracés dans les
  vues temporelle et constellation (performance Plotly). Le spectre est
  toujours calculé sur le signal complet ou sur `nfft`.
- `output_html` : si renseigné, la figure est aussi sauvegardée en HTML.
- Les colonnes sont désignées par index (fonctionne avec ou sans en-tête).
- Validation dans `__post_init__` (`ValueError`) : `sample_rate`, `nfft` et
  `max_plot_samples` strictement positifs ; index de colonnes positifs et
  I ≠ Q.

### 3.2 `sources.py`

```python
class IQSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        """Renvoie un tableau numpy 1D complexe."""


class CsvIQSource(IQSource):
    def __init__(self, config: CsvConfig) -> None: ...
    def read(self) -> np.ndarray: ...
```

`CsvIQSource.read()` utilise `pd.read_csv` avec `sep`, `decimal` et
`header=0 if has_header else None`, sélectionne les colonnes par position
(`iloc[:, [i_column, q_column]]`, car `usecols` ne respecte pas l'ordre
demandé), puis renvoie `I + 1j * Q`.

Erreurs :

- fichier absent : `FileNotFoundError` natif ;
- fichier vide : `ValueError` explicite ;
- pas assez de colonnes : `ValueError` suggérant de vérifier le séparateur ;
- colonnes non numériques : `ValueError` explicite, avec une indication sur
  les causes probables (mauvais séparateur, en-tête non déclaré) ;
- en-tête seul (aucun échantillon), valeurs manquantes ou infinies : `ValueError`.

### 3.3 `processing.py`

```python
WINDOWS = {"hann": np.hanning, "rectangular": np.ones}

def time_axis(n_samples: int, sample_rate: Optional[float]) -> np.ndarray: ...

def compute_spectrum(
    iq: np.ndarray,
    sample_rate: Optional[float] = None,
    nfft: Optional[int] = None,
    window: str = "hann",
    scale: str = "db",
) -> Tuple[np.ndarray, np.ndarray]: ...
```

`time_axis` : `np.arange(n) / sample_rate` si Fs est fournie, sinon
`np.arange(n)`.

`compute_spectrum` :

1. `n = nfft or len(iq)` ; le signal est tronqué à `n` échantillons s'il est
   plus long, et complété par des zéros (zero-padding) s'il est plus court. La
   fenêtre est appliquée sur `min(n, len(iq))` échantillons.
2. Fenêtrage avec `WINDOWS[window]`.
3. `np.fft.fft` puis `np.fft.fftshift` ; axe
   `fftshift(fftfreq(n, d=1/sample_rate))`, ou `d=1` sans Fs.
4. Normalisation : `|X| / sum(window)`. Une sinusoïde complexe d'amplitude 1
   donne un pic à 1 (linéaire), soit 0 dB.
5. Échelle : `"linear"` → `|X|` ; `"db"` → `20·log10(max(|X|, 1e-12))`.

Erreurs (`ValueError`) : signal vide, fenêtre inconnue, échelle inconnue,
signal trop court pour la fenêtre (somme des poids nulle, ex. Hann sur 2
échantillons).

### 3.4 `plotting.py`

```python
def build_figure(iq: np.ndarray, config: VisualizerConfig) -> go.Figure: ...
```

Grille `make_subplots` 2×2 :

- ligne 1, colonne 1 : I(t) et Q(t) (2 traces) ;
- ligne 1, colonne 2 : constellation, ratio d'axes 1:1 ;
- ligne 2, sur les deux colonnes : spectre.

Trois fonctions privées (`_add_time_traces`, `_add_constellation`,
`_add_spectrum`) gardent `build_figure` courte. Libellés d'axes selon la
configuration : « Temps (s) » / « Échantillon », « Fréquence (Hz) » /
« Fréquence normalisée », « Magnitude (dB) » / « Magnitude ».
`build_figure` construit la figure sans l'afficher. Les traces utilisent
`go.Scattergl` (WebGL) pour rester fluides quand le spectre porte sur un
signal long.

### 3.5 `main.py`

```python
def parse_args(argv: Optional[List[str]] = None) -> Tuple[CsvConfig, VisualizerConfig]: ...
def run(source: IQSource, config: VisualizerConfig) -> None: ...
def main(argv: Optional[List[str]] = None) -> int: ...
```

Utilisation :

```
python -m iq_visualizer.main FICHIER [--sep SEP] [--decimal DEC] [--header]
    [--i-col N] [--q-col N] [--fs HZ] [--nfft N]
    [--window {hann,rectangular}] [--scale {db,linear}]
    [--max-points N] [--save FICHIER.html]
```

- Les valeurs par défaut d'argparse sont lues dans les dataclasses (pas de
  duplication).
- `--header/--no-header` (`argparse.BooleanOptionalAction`, Python 3.9+).
- `--sep` accepte les alias `comma`, `semicolon`, `tab`, `space` (→ `r"\s+"`)
  ou un séparateur brut.
- `run()` : `source.read()` → `build_figure()` → `fig.show()` et, si
  `output_html` est renseigné, `fig.write_html()`.
- `main()` intercepte `OSError` (fichier absent, dossier, droits…) et `ValueError` (y compris ceux de
  la validation des dataclasses) : message sur stderr, code de sortie 1, sans
  traceback. Il renvoie 0 en cas de succès.

## 4. Tests (pytest)

- `test_sources.py` : CSV avec virgule ; point-virgule et décimale « , » ;
  espaces ; avec et sans en-tête ; colonnes I/Q non contiguës ; erreur sur
  colonne non numérique ; erreur sur fichier vide.
- `test_processing.py` : sinusoïde complexe à fréquence connue → pic à la
  bonne fréquence, valeur 0 dB (dB) et 1 (linéaire) ; troncature et
  zero-padding via `nfft` ; axe normalisé sans Fs ; `time_axis` avec et sans
  Fs ; erreurs sur fenêtre, échelle inconnues et signal vide.
- `test_plotting.py` : la figure contient 4 traces ; le nombre de points
  tracés en temporel respecte `max_plot_samples`.
- `test_main.py` : `parse_args` remplit correctement les dataclasses, y
  compris les alias de séparateur ; `run()` avec une fausse source écrit le
  HTML demandé (`go.Figure.show` remplacé par `monkeypatch` pour ne pas
  ouvrir de navigateur).

Vérifications finales : `pytest` et `flake8` passent ; la suite de tests est
exécutée sous Python 3.9 (via `uv` si disponible).
