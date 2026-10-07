# IQ Visualizer

Visualise des échantillons IQ lus dans un fichier CSV : I/Q en fonction du
temps, constellation et spectre (dB ou magnitude linéaire), dans une figure
Plotly interactive ouverte dans le navigateur.

Compatible Python 3.8+.

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
| `--header` | la première ligne est un en-tête | non |
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
