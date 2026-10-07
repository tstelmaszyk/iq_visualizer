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
    except (OSError, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
