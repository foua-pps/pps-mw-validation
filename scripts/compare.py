#!/usr/bin/env python
from pathlib import Path
from sys import argv
from typing import Dict, List, Optional
import argparse
import datetime as dt
import os

import matplotlib.colors as colors  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_validation.cloudnet import CloudnetSite
from pps_mw_validation.data_model import DatasetType, RegionOfInterest
from pps_mw_validation.utils import load_netcdf_data, get_files, get_stats

CLOUDNET_PATH = Path(os.environ.get("CLOUDNET_RESAMPLED_PATH", os.getcwd()))
CMIC_PATH = Path(os.environ.get("CMIC_PATH", os.getcwd()))
DARDAR_PATH = Path(os.environ.get("DARDAR_RESAMPLED_PATH", os.getcwd()))
PLATFORMS = ["noaa20", "npp"]
DATE_FORMAT = mdates.DateFormatter('%Y-%m-%d')
COLORS = [f"C{i}" for i in range(10)] + list(colors._colors_full_map.values())
START = dt.date(2023, 4, 1)
END = dt.date(2023, 5, 1)
MIN_IWP = 1e-6
MAX_IWP = 10.
THRESHOLD_IWP = 1e-2
N_BINS = 70
MAX_TIME_DIFF = 900  # [seconds]
ACCURACY = {   # [kg/m2]
    "interquartile_range": {
        "threshold": 0.27,
        "target": 0.14,
        "optimum": 0.06,
    },
    "median_bias": {
        "threshold": 0.04,
        "target": 0.02,
        "optimum": 0.01,
    },
    "mean_absolute_error": {
        "threshold": 0.17,
        "target": 0.09,
        "optimum": 0.04,
    },
}


def load_cmic_data(
    start: dt.date,
    end: dt.date,
    sites: List[CloudnetSite] = list(CloudnetSite),
) -> Dict[CloudnetSite, xr.DataArray]:
    """Load cmic data."""
    data: Dict[CloudnetSite, xr.DataArray] = {}
    for site in sites:
        files = get_files(
            CMIC_PATH,
            f"*{site.lower_case_name}*",
            (start, end, "%Y%m%d")
        )
        times = []
        iwps = []
        for f in files:
            data_by_site = load_netcdf_data(f)
            iwp = np.nanmean(data_by_site.ice_water_path)
            if ~np.isnan(iwp):
                times.append(np.datetime64(data_by_site.attrs["time"]))
                iwps.append(iwp)
            idxs = np.argsort(times)
        data[site] = xr.DataArray(
            [iwps[idx] for idx in idxs],
            dims="time",
            coords={"time": ("time", [times[idx] for idx in idxs])},
        )
    return data


def load_cloudnet_data(
    start: dt.date,
    end: dt.date,
    sites: List[CloudnetSite] = list(CloudnetSite),
) -> Dict[CloudnetSite, xr.Dataset]:
    """Load cloudnet data."""
    data: Dict[CloudnetSite, xr.Dataset] = {}
    for site in sites:
        files = get_files(
            CLOUDNET_PATH,
            f"*{site.lower_case_name}*",
            (start, end, "%Y%m%d")
        )
        if len(files) > 0:
            data[site] = xr.concat(
                [load_netcdf_data(f) for f in files],
                dim="time",
            )
    return data


def load_dardar_distribution(
    start: dt.date,
    end: dt.date,
    edges: np.ndarray,
    rois: List[RegionOfInterest] = list(RegionOfInterest),
) -> Dict[RegionOfInterest, xr.Dataset]:
    """Load dardar data."""
    data: Dict[RegionOfInterest, xr.DataArray] = {}
    for roi in rois:
        files = get_files(
            DARDAR_PATH,
            f"*{DatasetType.DARDAR.name}*{roi.value}*",
            (start, end, "%Y-%m-%d")
        )
        if len(files) > 0:
            dataset = xr.concat(
                [load_netcdf_data(f) for f in files],
                dim="time",
            )
            data[roi] = get_stats(
                dataset.ice_water_path, edges, 2 * MIN_IWP,
            )
    return data


def load_cmic_distribution(
    start: dt.date,
    end: dt.date,
    edges: np.ndarray,
    rois: List[RegionOfInterest] = list(RegionOfInterest),
) -> Dict[RegionOfInterest, xr.Dataset]:
    """Load dardar data."""
    data: Dict[RegionOfInterest, xr.DataArray] = {}
    for roi in rois:
        files = get_files(
            CMIC_PATH,
            f"*{DatasetType.CMIC.name}*{roi.value}*",
            (start, end, "%Y-%m-%d")
        )
        if len(files) > 0:
            dataset = xr.concat(
                [load_netcdf_data(f) for f in files],
                dim="time",
            )
            data[roi] = get_stats(
                dataset.ice_water_path, edges, 2 * MIN_IWP,
            )
    return data


def compare(
    cloudnet_data: xr.Dataset,
    cmic_data: xr.DataArray,
    min_iwp: float = THRESHOLD_IWP,
) -> Optional[xr.Dataset]:
    """Match and compare cloudnet and cmic data."""
    # first interpolate cloudnet data on cmic time using nearest neighbour
    idxs = []
    for idx_cmic, t in enumerate(cmic_data.time):
        time_diff = np.abs(t - cloudnet_data.time) / np.timedelta64(1, "s")
        idx_cloudnet = np.argmin(time_diff.values)
        if time_diff[idx_cloudnet] < MAX_TIME_DIFF:
            idxs.append((idx_cmic, idx_cloudnet))
    matching_data = xr.Dataset(
        data_vars={
            "ice_water_path_cmic": (
                "time",
                [cmic_data.values[idx] for idx, _ in idxs],
            ),
            "ice_water_path_cloudnet": (
                "time",
                [
                    cloudnet_data["ice_water_path"].values[idx]
                    for _, idx in idxs
                ],
            ),
        },
        coords={
            "time": (
                "time",
                [cmic_data.time.values[idx] for idx, _ in idxs]
            )
        },
    )
    filt = matching_data.ice_water_path_cloudnet >= min_iwp
    diff = (
        matching_data.ice_water_path_cmic[filt]
        - matching_data.ice_water_path_cloudnet[filt]
    )
    try:
        q1, q2, q3 = np.percentile(diff, [25, 50, 75])
    except IndexError:
        return None
    matching_data.attrs = {
        "median_bias": q2,
        "interquartile_range": q3 - q1,
        "mean_absolute_error": np.mean(np.abs(diff)),
    }
    return matching_data


def show_cloudnet_distribution(
    start: dt.date,
    end: dt.date,
) -> None:
    """Show cloudnet IWP distribution."""
    edges = np.logspace(np.log10(MIN_IWP), np.log10(MAX_IWP), N_BINS)
    data = load_cloudnet_data(start, end)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    offset = 0.01
    for idx, (site, data_by_site) in enumerate(data.items()):
        stats = get_stats(data_by_site.ice_water_path, edges, MIN_IWP * 2)
        axs[0].loglog(stats.x, stats.pdf, '-', color=COLORS[idx])
        axs[0].grid(True)
        axs[0].set_xlim([MIN_IWP * 5, MAX_IWP])
        axs[0].set_xlabel("IWP [kg/m2]")
        axs[0].set_ylabel("PDF [1/(kg/m2)]")
        axs[1].semilogx(
            stats.x, stats.dist, '-', color=COLORS[idx], label=site.value,
        )
        axs[1].grid(True)
        axs[1].set_xlim([MIN_IWP * 5, MAX_IWP])
        axs[1].set_ylim([-offset, 1 + offset])
        axs[1].set_xlabel("IWP [kg/m2]")
        axs[1].set_ylabel("Distribution [-]")
    plt.legend()
    plt.savefig("cloudnet_stat.png", bbox_inches='tight')
    plt.show()


def validate_by_region(
    start: dt.date,
    end: dt.date,
) -> None:
    """Comare dardar and cmic distribution."""
    edges = np.logspace(np.log10(MIN_IWP), np.log10(MAX_IWP), N_BINS)
    dardar = load_dardar_distribution(start, end, edges)
    cmic = load_cmic_distribution(start, end, edges)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for idx, (roi, stats) in enumerate(dardar.items()):
        color = COLORS[idx]
        axs[0].loglog(stats.x, stats.pdf, '-', color=color)
        axs[0].grid(True)
        axs[0].set_xlim([MIN_IWP * 10, MAX_IWP])
        axs[0].set_xlabel("IWP [kg/m2]")
        axs[0].set_ylabel("PDF [1/(kg/m2)]")
        axs[1].semilogx(
            stats.x,
            stats.dist,
            '-',
            color=color,
            label=f"DARDAR: {roi.value.replace('_', ' ')}",
        )
        axs[1].grid(True)
        axs[1].set_xlim([MIN_IWP * 10, MAX_IWP])
        axs[1].set_ylim([0, 1.01])
        axs[1].set_xlabel("IWP [kg/m2]")
        axs[1].set_ylabel("Distribution [-]")
        if roi in cmic:
            print(
                f'{roi.name}:'
                f' DARDAR median: {stats.attrs["median"]}'
                f' CMIC median: {cmic[roi].attrs["median"]}'
                f' DARDAR iqr: {stats.attrs["interquartile_range"]}'
                f' CMIC iqr: {cmic[roi].attrs["interquartile_range"]}'
            )
            axs[0].loglog(cmic[roi].x, cmic[roi].pdf, '--', color=color)
            axs[1].semilogx(
                cmic[roi].x,
                cmic[roi].dist,
                '--',
                color=color,
                label=f"CMIC: {roi.value.replace('_', ' ')}",
            )
    plt.legend()
    plt.savefig("dardar_cmic_stat.png", bbox_inches='tight')
    plt.show()


def show_time_series(
    start: dt.date,
    end: dt.date,
) -> None:
    """Show time series of cmic and cloudnet data."""
    cmic_data = load_cmic_data(start, end)
    cloudnet_data = load_cloudnet_data(start, end)
    fig = plt.figure(figsize=(18, 10))
    nrows = 3
    ncols = 5
    idx = 0
    for site in CloudnetSite:
        if site not in cmic_data or site not in cloudnet_data:
            continue
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        line_cloudnet = ax.semilogy(
            cloudnet_data[site].time, cloudnet_data[site].ice_water_path,  'C0.'
        )
        line_cmic = ax.semilogy(
            cmic_data[site].time, cmic_data[site],  'C1.'
        )
        if idx == ncols - 1:
            line_cloudnet[0].set_label("cloudnet")
            line_cmic[0].set_label("cmic")
            ax.legend()
        if idx % ncols == 0:
            ax.set_ylabel("IWP [kg/m2]")
        if idx // ncols == nrows - 1:
            ax.xaxis.set_major_formatter(DATE_FORMAT)
            ax.xaxis.set_tick_params(rotation=-70)
        else:
            ax.xaxis.set_ticklabels([])
        ax.set_xlim([start, end])
        ax.set_ylim([1e-6, 10])
        ax.set_title(site.value)
        idx += 1
    plt.savefig("cloudnet_cmic_comp.png", bbox_inches='tight')
    plt.show()


def validate_by_site(
    start: dt.date,
    end: dt.date,
) -> None:
    """Compare cmic and cloudnet data."""
    cmic_data = load_cmic_data(start, end)
    cloudnet_data = load_cloudnet_data(start, end)
    summary: Dict[CloudnetSite, Dict[str, float]] = {}
    for site in CloudnetSite:
        if site not in cmic_data or site not in cloudnet_data:
            continue
        stats = compare(cloudnet_data[site], cmic_data[site])
        if stats is not None:
            summary[site] = stats.attrs
    sites = [s.value for s in summary]
    ones = np.ones(len(sites))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 7))
    offset = 0.005
    for idx, param in enumerate(
        ["median_bias", "mean_absolute_error", "interquartile_range"]
    ):
        for idx_p, (p, value) in enumerate(ACCURACY[param].items()):
            axs[idx].plot(
                sites, value * ones, f'C{idx_p}-', label=p, linewidth=3,
            )
            if param == "median_bias":
                axs[idx].plot(sites, -value * ones, f'C{idx_p}-', linewidth=3)
        axs[idx].plot(sites, [data[param] for data in summary.values()], 'ko')
        axs[idx].set_ylabel(f'{param.replace("_", " ")} [kg/m2]')
        axs[idx].xaxis.set_tick_params(rotation=-80)
        axs[idx].grid(True)
        if idx == 2:
            axs[idx].legend(loc="upper right")
        if param != "median_bias":
            axs[idx].set_ylim([-offset, ACCURACY[param]["threshold"] + offset])
        else:
            value = ACCURACY[param]["threshold"] + offset
            axs[idx].set_ylim([-value, value])
    plt.savefig("cloudnet_cmic_validation.png", bbox_inches='tight')
    plt.show()


def add_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    description: str,
    start: dt.date = START,
    end: dt.date = END,
) -> None:
    """Add parser."""
    parser = subparsers.add_parser(
        command,
        description=description,
        help=description,
    )
    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        type=str,
        default=start.isoformat(),
        help=(
            "Start date (format: YYYY-MM-DD),"
            f" default is {start}."
        ),
    )
    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        type=str,
        default=end.isoformat(),
        help=(
            "End date (format: YYYY-MM-DD),"
            f" default is {end}."
        ),
    )


def cli(args_list: List[str] = argv[1:]) -> None:
    parser = argparse.ArgumentParser(
        description="Run the ppsmw data comparison app."
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    add_parser(
        subparsers,
        "validate-by-region",
        "Compare CMIC and DARDAR IWP distributions.",
    )
    add_parser(
        subparsers,
        "cloudnet-distribution",
        "Show CLOUDNET IWP distribution.",
    )
    add_parser(
        subparsers,
        "time-series",
        "Show time series of CLOUDNET and CMIC IWP data.",
    )
    add_parser(
        subparsers,
        "validate-by-site",
        "Compare CLOUDNET and CMIC IWP data.",
    )
    args = parser.parse_args(args_list)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    comparison_type = args.command
    if comparison_type == "cloudnet-distribution":
        show_cloudnet_distribution(start, end)
    elif comparison_type == "validate-by-region":
        validate_by_region(start, end)
    elif comparison_type == "time-series":
        show_time_series(start, end)
    elif comparison_type == "validate-by-site":
        validate_by_site(start, end)


if __name__ == "__main__":
    cli(argv[1:])
