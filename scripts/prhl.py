#!/usr/bin/env python
from pathlib import Path
from sys import argv
import argparse
import json
import logging
import os

from pyresample.geometry import AreaDefinition  # type: ignore

from pps_mw_validation import baltrad, prhl, prhl_validation
from pps_mw_validation.logging import LOG_LEVEL, set_log_level


BALTRAD_CONFIG_FILE = os.environ.get("BALTRAD_CONFIG_FILE", "baltrad.yaml")
PPSMW_STATS_DIR = os.environ.get("PPSMW_STATS_DIR", ".")


logger = logging.getLogger(__name__)


def collect_stats(
    prx_files: list[Path],
    resampler: baltrad.BaltradResampler,
    make_plots: bool,
    stats_dir: Path,
):
    """Compare PR-HL or PR-S files against BALTRAD and write stats to file."""

    stats = prhl_validation.Stats.zeros(resampler.area.shape)

    for prx_file in prx_files:

        try:
            prx_dataset = prhl.resample(
                prx_file, resampler.area, resampler.radius_of_influence
            )
        except Exception:
            logger.warning(f"Failed loading file: {prx_file.as_posix()}.")
            continue

        try:
            baltrad_file = resampler.get_matching_file(prx_dataset["central_time"])
            baltrad_dataset = resampler.resample(baltrad_file)
            assert baltrad_dataset is not None
        except Exception:
            logger.warning(f"Failed loading file: {baltrad_file.as_posix()}.")
            continue

        rr_prx = prx_dataset["rainfall_rate"].values
        rr_baltrad = baltrad_dataset["rainfall_rate"].values
        stats.add(rr_prx, rr_baltrad)

        if make_plots:
            outfile = stats_dir / f"{prx_file.name}.png"
            prhl_validation.plot_scene(rr_prx, rr_baltrad, resampler.area, outfile)

        logger.info(f"Processed file: {prx_file.as_posix()}.")

    stats_dir.mkdir(parents=True, exist_ok=True)
    outfile = stats_dir / f"{stats.get_tag(prx_files[0])}_stats.nc"
    stats.as_dataset.to_netcdf(outfile)
    logger.info(f"Wrote stats to file: {outfile.as_posix()}")


def summarize_stats(
    stats_files: list[Path],
    make_plots: bool,
    area: AreaDefinition,
    stats_dir: Path,
):
    """Summarize the comparison(s) and write the validation score to a json file."""

    validation_score = {}
    for stats_file in stats_files:
        stats = prhl_validation.Stats.from_file(stats_file)
        validation_score[stats.product_tag] = stats.validation_score
        if make_plots:
            stats.plot_stats(area, stats_dir)

    outfile = stats_dir / "baltrad_comparison_summary.json"
    with open(outfile, "w") as f:
        f.write(json.dumps(validation_score, indent=4))

    logger.info(f"Wrote summary stats file: {outfile.as_posix()}.")


def cli(args_list: list[str] = argv[1:]):
    """Run the command line interface for the PR-HL validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Compare PR-HL or PR-S level2 data aginst BALTRAD precipitation.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="operation_type",
        default="collect",
        type=str,
        help="type of operation",
        choices=["collect", "summarize"],
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        type=str,
        nargs="+",
        required=True,
        help="level2 file(s) to process",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="flag for plotting result",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        default="warning",
        type=str,
        help="provide logging level",
        choices=[level for level in LOG_LEVEL],
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="baltrad_config_file",
        default=BALTRAD_CONFIG_FILE,
        type=str,
        help="BALTRAD resamplig config file",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        dest="stats_dir",
        default=PPSMW_STATS_DIR,
        type=str,
        help="Directory where to write stats files.",
    )

    args = parser.parse_args(args_list)

    log_level = args.log_level
    operation_type = args.operation_type
    plot = args.plot
    baltrad_config_file = Path(args.baltrad_config_file)
    stats_dir = Path(args.stats_dir)
    files = [Path(f) for f in args.files]

    set_log_level(LOG_LEVEL[log_level])

    resampler = baltrad.BaltradResampler.from_config_file(baltrad_config_file)

    if operation_type == "collect":
        collect_stats(files, resampler, plot, stats_dir)

    if operation_type == "summarize":
        summarize_stats(files, plot, resampler.area, stats_dir)


if __name__ == "__main__":
    cli(argv[1:])
