#!/usr/bin/env python
from pathlib import Path
from sys import argv
import argparse
import datetime as dt
import json
import logging
import os

import numpy as np  # type: ignore
import xarray as xr  # type: ignore
from pyresample.geometry import AreaDefinition  # type: ignore

from pps_mw_validation import baltrad, prhl, prhl_validation
from pps_mw_validation.logging import LOG_LEVEL, set_log_level


BALTRAD_CONFIG_FILE = os.environ.get("BALTRAD_CONFIG_FILE", "baltrad.yaml")
STATS_DIR = os.environ.get("PPSMW_STATS_DIR", ".")
THRESHOLDS = [0, 0.01, 0.1, 1.0]  # precipitation rates in mm/h


logger = logging.getLogger(__name__)


def collect_stats(
    prx_files: list[Path],
    resampler: baltrad.BaltradResampler,
    make_plots: bool,
    stats_dir: Path,
):
    """Collect and write stats files."""

    shape = (len(THRESHOLDS), resampler.area.shape[0], resampler.area.shape[1])

    count_rain = np.zeros(shape)
    count_clear = np.zeros(shape)
    sum_baltrad = np.zeros(shape)
    sum_prx = np.zeros(shape)
    sum_square_baltrad = np.zeros(shape)
    sum_square_prx = np.zeros(shape)
    sum_product = np.zeros(shape)
    sum_diff_square = np.zeros(shape)
    true_detection = np.zeros(shape)
    false_detection = np.zeros(shape)

    for prx_file in prx_files:

        try:
            prx_dataset = prhl.load_prx_dataset(
                prx_file, resampler.area, resampler.radius_of_influence
            )
        except Exception:
            logger.warning(f"Failed loading file: {prx_file.as_posix()}.")
            continue

        rr_prx = prx_dataset["rainfall_rate"].values
        t0 = dt.datetime.fromisoformat(prx_dataset.attrs["time_coverage_start"])
        t1 = dt.datetime.fromisoformat(prx_dataset.attrs["time_coverage_end"])

        try:
            baltrad_file = baltrad.get_matching_baltrad_file(t0 + (t1 - t0) / 2)
            baltrad_dataset = resampler.resample(baltrad_file)
            assert baltrad_dataset is not None
        except Exception:
            logger.warning(f"Failed loading file: {baltrad_file.as_posix()}.")
            continue

        rr_baltrad = baltrad_dataset["rainfall_rate"].values

        if make_plots:
            outfile = stats_dir / f"{prx_file.name}.png"
            prhl_validation.plot_scene(rr_prx, rr_baltrad, resampler.area, outfile)

        for idx, threshold in enumerate(THRESHOLDS):

            filt_rain = (rr_baltrad >= threshold) & np.isfinite(rr_prx)
            filt_clear = (rr_baltrad < threshold) & np.isfinite(rr_prx)

            count_rain[idx] = np.where(filt_rain, count_rain[idx] + 1, count_rain[idx])
            count_clear[idx] = np.where(filt_clear, count_clear[idx] + 1, count_clear[idx])

            sum_baltrad[idx] = np.where(filt_rain, sum_baltrad[idx] + rr_baltrad, sum_baltrad[idx])
            sum_prx[idx] = np.where(filt_rain, sum_prx[idx] + rr_prx, sum_prx[idx])

            sum_square_baltrad[idx] = np.where(
                filt_rain, sum_square_baltrad[idx] + rr_baltrad ** 2, sum_square_baltrad[idx]
            )
            sum_square_prx[idx] = np.where(
                filt_rain, sum_square_prx[idx] + rr_prx ** 2, sum_square_prx[idx]
            )

            sum_product[idx] = np.where(
                filt_rain, sum_product[idx] + rr_prx * rr_baltrad, sum_product[idx]
            )
            sum_diff_square[idx] = np.where(
                filt_rain, sum_diff_square[idx] + (rr_baltrad - rr_prx) ** 2, sum_diff_square[idx]
            )
            true_detection[idx] = np.where(
                (rr_baltrad >= threshold) & (rr_prx >= threshold),
                true_detection[idx] + 1,
                true_detection[idx]
            )
            false_detection[idx] = np.where(
                (rr_baltrad < threshold) & (rr_prx >= threshold),
                false_detection[idx] + 1,
                false_detection[idx]
            )

        logger.info(f"Processed file: {prx_file.as_posix()}.")

    data = xr.Dataset(
        {
            "sum_baltrad": (("t", "y", "x"), sum_baltrad),
            "sum_square_baltrad": (("t", "y", "x"), sum_square_baltrad),
            "sum_prx": (("t", "y", "x"), sum_prx),
            "sum_square_prx": (("t", "y", "x"), sum_square_prx),
            "sum_diff_square": (("t", "y", "x"), sum_diff_square),
            "sum_product": (("t", "y", "x"), sum_product),
            "count_rain": (("t", "y", "x"), count_rain),
            "count_clear": (("t", "y", "x"), count_clear),
            "true_detection": (("t", "y", "x"), true_detection),
            "false_detection": (("t", "y", "x"), false_detection),
        }
    )

    stats_dir.mkdir(parents=True, exist_ok=True)
    _, product, sensor, _ = prx_files[0].name.lower().split("_")
    outfile = stats_dir / f"{product}-{sensor}_stats.nc"
    data.to_netcdf(outfile)
    logger.info(f"Wrote stats to file: {outfile.as_posix()}")


def summarize_stats(
    stats_files: list[Path],
    make_plots: bool,
    area: AreaDefinition,
    stats_dir: Path,
):
    """Collect stats files and summarise the comparison."""

    stats = {}
    for stat_file in stats_files:

        _, _product, _sensor, _ = stat_file.name.lower().split("_")
        product = f"{_product}-{_sensor}"

        data = xr.load_dataset(stat_file)

        count_rain = data["count_rain"].values
        count_clear = data["count_clear"].values
        sum_prx = data["sum_prx"].values
        sum_baltrad = data["sum_baltrad"].values
        sum_square_prx = data["sum_square_prx"].values
        sum_square_baltrad = data["sum_square_baltrad"].values
        sum_diff_square = data["sum_diff_square"].values
        sum_product = data["sum_product"].values
        true_detection = data["true_detection"].values
        false_detection = data["false_detection"].values

        prx_mean = sum_prx / count_rain
        prx_square_mean = sum_square_prx / count_rain
        prx_std = np.sqrt(prx_square_mean - prx_mean ** 2)

        baltrad_mean = sum_baltrad / count_rain
        baltrad_square_mean = sum_square_baltrad / count_rain
        baltrad_std = np.sqrt(baltrad_square_mean - baltrad_mean ** 2)

        covariance = sum_product / count_rain - prx_mean * baltrad_mean

        metrics = {
            "bias": prx_mean - baltrad_mean,
            "fse": np.sqrt(sum_diff_square / count_rain) / baltrad_mean,
            "pod": true_detection / count_rain,
            "pofd": false_detection / count_clear,
            "corr":  covariance / (prx_std * baltrad_std)
        }

        stats[product] = {
            metric: {
                threshold: [_mean, _std] for threshold, _mean, _std in zip(
                    THRESHOLDS,
                    np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=(1, 2)),
                    np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=(1, 2)),
                )
            } for metric, values in metrics.items()
        }

        if make_plots:
            prhl_validation.plot_stats(product, metrics, area, stats_dir)

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
        "-p",
        "--plot",
        action="store_true",
        help="Flag for plotting result",
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
    plot = args.plot
    files = [Path(f) for f in args.files]

    set_log_level(LOG_LEVEL[log_level])

    config = baltrad.load_config(Path(BALTRAD_CONFIG_FILE))
    resampler = baltrad.BaltradResampler.from_config(config)
    stats_dir = Path(STATS_DIR)

    if operation_type == "collect":
        collect_stats(files, resampler, plot, stats_dir)

    if operation_type == "summarize":
        summarize_stats(files, plot, resampler.area, stats_dir)


if __name__ == "__main__":
    cli(argv[1:])
