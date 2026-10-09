"""Command line entry point.

Example:
    python -m iq_visualizer.main data.csv --sep semicolon --fs 1e6
"""

import argparse
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iq_visualizer.config import CsvConfig, VisualizerConfig
from iq_visualizer.plotting import PAGE_BACKGROUND_SCRIPT, build_figure
from iq_visualizer.processing import SCALES, WINDOWS
from iq_visualizer.sources import CsvIQSource, IQSource
from iq_visualizer.tone_check import Gap, ToneChecker, format_report

SEPARATOR_ALIASES = {
    "comma": ",",
    "semicolon": ";",
    "tab": "\t",
    "space": r"\s+",
}


def _defaults(config_class: type) -> Dict[str, Any]:
    """Default values of a dataclass (single source of defaults)."""
    return {field.name: field.default for field in fields(config_class)
            if field.default is not MISSING}


def parse_args(
    argv: Optional[List[str]] = None,
) -> Tuple[CsvConfig, VisualizerConfig]:
    """Builds the configurations from the command line."""
    csv_defaults = _defaults(CsvConfig)
    vis_defaults = _defaults(VisualizerConfig)

    parser = argparse.ArgumentParser(
        description="Visualizes IQ samples read from a CSV file.")
    parser.add_argument("path", type=Path,
                        help="CSV file containing the I and Q columns")
    parser.add_argument("--sep", default=csv_defaults["separator"],
                        help="separator: comma, semicolon, tab, space "
                             "or a raw character (default: %(default)r)")
    parser.add_argument("--decimal", default=csv_defaults["decimal"],
                        help="decimal separator (default: %(default)r)")
    parser.add_argument("--header", action="store_true",
                        default=csv_defaults["has_header"],
                        help="the first row is a header")
    parser.add_argument("--i-col", type=int,
                        default=csv_defaults["i_column"],
                        help="index of the I column (default: %(default)s)")
    parser.add_argument("--q-col", type=int,
                        default=csv_defaults["q_column"],
                        help="index of the Q column (default: %(default)s)")
    parser.add_argument("--fs", type=float,
                        default=vis_defaults["sample_rate"],
                        help="sample rate in Hz")
    parser.add_argument("--nfft", type=int, default=vis_defaults["nfft"],
                        help="FFT size (default: whole signal)")
    parser.add_argument("--window", choices=sorted(WINDOWS),
                        default=vis_defaults["window"])
    parser.add_argument("--scale", choices=SCALES,
                        default=vis_defaults["scale"])
    parser.add_argument("--max-points", type=int,
                        default=vis_defaults["max_plot_samples"],
                        help="points plotted in the time and constellation "
                             "views (default: %(default)s)")
    parser.add_argument("--save", type=Path,
                        default=vis_defaults["output_html"],
                        help="save the figure to this HTML file")
    parser.add_argument("--check-tone", action="store_true",
                        default=vis_defaults["check_tone"],
                        help="the signal is a reference tone: report the "
                             "gaps (exit code 1 if any)")
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
        check_tone=args.check_tone,
    )
    return csv_config, vis_config


def run(source: IQSource, config: VisualizerConfig) -> List[Gap]:
    """Reads the source, builds the figure, saves and shows it.

    Returns the tone gaps (always empty without config.check_tone).
    """
    iq = source.read()
    gaps = None
    if config.check_tone:
        gaps = ToneChecker().find_gaps(iq)
        print(format_report(gaps, config.sample_rate))
    figure = build_figure(iq, config, gaps)
    # post_script: cosmetic page background (plotting.PAGE_BACKGROUND_SCRIPT)
    if config.output_html is not None:
        figure.write_html(str(config.output_html),
                          post_script=PAGE_BACKGROUND_SCRIPT)
    figure.show(post_script=PAGE_BACKGROUND_SCRIPT)
    return gaps or []


def main(argv: Optional[List[str]] = None) -> int:
    """Returns the program exit code."""
    try:
        csv_config, vis_config = parse_args(argv)
        gaps = run(CsvIQSource(csv_config), vis_config)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
