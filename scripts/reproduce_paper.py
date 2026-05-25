#!/usr/bin/env python3
"""Regenerate the CSV data and figures used in the SDIEP paper."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nla_application_experiments import run_application_experiments
from sdiep_schur.experiments import (
    computational_cost,
    convergence_study,
    sharpness_curve,
    success_rate_grid,
    systematic_coherence,
    threshold_landscape,
)
from sdiep_schur.plotting import (
    plot_coherence,
    plot_convergence,
    plot_sharpness,
    plot_success_heatmaps,
    plot_threshold_landscape,
    plot_timing,
)


def copy_csv_outputs(source_dir: Path, target_dir: Path) -> None:
    """Copy generated CSV outputs into the manuscript data directory.

    Parameters
    ----------
    source_dir:
        Directory containing generated CSV files.
    target_dir:
        Destination directory used by the manuscript archive.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(source_dir.glob("*.csv")):
        shutil.copy2(csv_path, target_dir / csv_path.name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Directory for CSV outputs.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory for figure PDFs.",
    )
    parser.add_argument(
        "--paper-data-dir",
        type=Path,
        default=Path("paper/data"),
        help="Directory for manuscript CSV copies.",
    )
    return parser.parse_args()


def main() -> None:
    """Regenerate all paper data and figures."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    df1 = systematic_coherence(args.out_dir)
    plot_coherence(df1, args.figure_dir)

    df2 = success_rate_grid(args.out_dir)
    plot_success_heatmaps(df2, args.figure_dir)

    df3 = sharpness_curve(args.out_dir)
    plot_sharpness(df3, args.figure_dir)

    df4 = computational_cost(args.out_dir)
    plot_timing(df4, args.figure_dir)

    df5 = convergence_study(args.out_dir)
    plot_convergence(df5, args.figure_dir)

    df_supp = threshold_landscape(args.out_dir)
    plot_threshold_landscape(df_supp, args.figure_dir)

    run_application_experiments(args.out_dir, args.figure_dir)
    copy_csv_outputs(args.out_dir, args.paper_data_dir)


if __name__ == "__main__":
    main()
