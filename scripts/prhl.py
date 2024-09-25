#!/usr/bin/env python
from pathlib import Path
from sys import argv
import argparse
import json
import logging
import os


import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_validation import prhl_validation
from pps_mw_validation.baltrad import Baltrad
from pps_mw_validation.prhl import PrHL
from pps_mw_validation.logging import LOG_LEVEL, set_log_level


STATS_DIR = os.environ.get("STATS_DIR", ".")
MIN_RATE_RATE = 0.1  # mm / h


logger = logging.getLogger(__name__)


def collect_stats(
    prhl_files: list[Path],
    stats_dir: Path = Path(STATS_DIR),
):
    """Collect and write stats files."""
    for prhl_file in prhl_files:
        prhl = PrHL.get_precipitation(prhl_file)
        try:
            baltrad = Baltrad.get_precipitation(prhl)
        except Exception:
            logger.warning(
                f"Failed load BALTRAD data matching {prhl_file.as_posix()}"
            )
        stats = prhl_validation.get_stats(prhl, baltrad, MIN_RATE_RATE)
        if not stats_dir.exists():
            stats_dir.mkdir(parents=True, exist_ok=True)
        outfile = stats_dir / f"{prhl_file.stem}_stats.nc"
        stats.to_netcdf(outfile)
        logger.info(f"Wrote stats to file: {outfile.as_posix()}")


def summarize_stats(
    stats_files: list[Path],
    stats_dir: Path = Path(STATS_DIR),
):
    """Collect stats files and summarise the comparison."""
    records = [xr.load_dataset(f) for f in stats_files]

    baltrad = np.concatenate([r["baltrad"] for r in records])
    prhl = np.concatenate([r["prhl"] for r in records])
    condition = np.concatenate([r["condition"] for r in records])

    stats = prhl_validation.get_detection_performance(
        [r.attrs for r in records]
    )

    for surf_type, value in prhl_validation.SURFACE_TYPE.items():
        filt = np.bitwise_and(condition, value) == value
        summary_stats = prhl_validation.get_summary_stats(
            baltrad[filt],
            prhl[filt],
        )
        stats[surf_type] = stats[surf_type] | summary_stats
        prhl_validation.make_plots(
            baltrad[filt],
            prhl[filt],
            surf_type,
            stats_dir,
        )
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)
    outfile = stats_dir / "baltrad_comparison_summary.json"
    with open(outfile, "w") as f:
        f.write(json.dumps(stats, indent=4))
    logger.info(f"Wrote summary file: {outfile.as_posix()}")


def cli(args_list: list[str] = argv[1:]):
    """Compare PRHL aginst BALTRAD precipitation data."""
    parser = argparse.ArgumentParser(
        description="Compare PRHL aginst BALTRAD precipitation.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="operation_type",
        default="collect",
        type=str,
        help="Type of operation.",
        choices=["collect", "summarize"],
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        type=str,
        nargs="+",
        required=True,
        help="PR-HL file(s) to process.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        default="warning",
        type=str,
        help="Provide logging level.",
        choices=[level for level in LOG_LEVEL],
    )

    args = parser.parse_args(args_list)

    log_level = args.log_level
    operation_type = args.operation_type
    files = [Path(f) for f in args.files]

    set_log_level(LOG_LEVEL[log_level])

    if operation_type == "collect":
        collect_stats(files)

    if operation_type == "summarize":
        summarize_stats(files)


if __name__ == "__main__":
    cli(argv[1:])
