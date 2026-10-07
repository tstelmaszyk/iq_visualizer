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
