#!/usr/bin/env python
from sys import argv
from typing import Dict, List, Optional
import argparse
import datetime as dt
import json

import matplotlib.colors as colors  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from pps_mw_validation.cloudnet import CloudnetSite
from pps_mw_validation.comparison import (
    compare,
    load_cloudnet_data,
    load_dataset,
    load_dataset_distribution,
)
from pps_mw_validation.data_model import DatasetType
from pps_mw_validation.utils import DatasetLoader, get_stats


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
        counts = DatasetLoader.get_counts(
            data_by_site.ice_water_path,
            edges,
        )
        stats = get_stats(counts.to_dataset(name="ice_water_path_count"))
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
    dataset: DatasetType,
    start: dt.date,
    end: dt.date,
) -> None:
    """Compare dataset to DARDAR IWP distribution."""
    dardar = load_dataset_distribution(DatasetType.DARDAR, start, end)
    other = load_dataset_distribution(dataset, start, end)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    summary: Dict[str, Dict[str, float]] = {}
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
        if roi in other:
            summary[roi.name] = {
                "DARDAR": stats.attrs,
                dataset.name: other[roi].attrs,
            }
            axs[0].loglog(other[roi].x, other[roi].pdf, '--', color=color)
            axs[1].semilogx(
                other[roi].x,
                other[roi].dist,
                '--',
                color=color,
                label=f"{dataset.name}: {roi.value.replace('_', ' ')}",
            )
    plt.legend()
    outfile_stem = f"dardar_{dataset.value}_comp"
    plt.savefig(f"{outfile_stem}.png", bbox_inches='tight')
    with open(f"{outfile_stem}.json", "w") as outfile:
        outfile.write(json.dumps(summary, indent=4))
    plt.show()


def show_time_series(
    dataset: DatasetType,
    start: dt.date,
    end: dt.date,
) -> None:
    """Show time series of cloudnet and other dataset."""
    cloudnet_data = load_cloudnet_data(start, end)
    other_data = load_dataset(dataset, start, end)
    fig = plt.figure(figsize=(18, 10))
    nrows = 3
    ncols = 5
    idx = 0
    for site in CloudnetSite:
        if site not in other_data or site not in cloudnet_data:
            continue
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        line_cloudnet = ax.semilogy(
            cloudnet_data[site].time, cloudnet_data[site].ice_water_path,  'C0.'
        )
        line_other = ax.semilogy(
            other_data[site].time, other_data[site],  'C1.'
        )
        if idx == ncols - 1:
            line_cloudnet[0].set_label("cloudnet")
            line_other[0].set_label(dataset.value)
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
    plt.savefig(f"cloudnet_{dataset.value}_comp.png", bbox_inches='tight')
    plt.show()


def validate_by_site(
    dataset: DatasetType,
    start: dt.date,
    end: dt.date,
) -> None:
    """Compare cloudnet and another dataset."""
    cloudnet_data = load_cloudnet_data(start, end)
    other_data = load_dataset(dataset, start, end)
    summary: Dict[str, Dict[str, float]] = {}
    for site in CloudnetSite:
        if site not in other_data or site not in cloudnet_data:
            continue
        stats = compare(
            cloudnet_data[site], other_data[site], THRESHOLD_IWP, MAX_TIME_DIFF,
        )
        if stats is not None:
            summary[site.value] = stats.attrs
    outfile_stem = f"cloudnet_{dataset.name}_validation"
    with open(f"{outfile_stem}.json", "w") as outfile:
        outfile.write(json.dumps(summary, indent=4))
    sites = [s for s in summary]
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
    plt.savefig(f"{outfile_stem}.png", bbox_inches='tight')
    plt.show()


def add_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    description: str,
    start: dt.date = START,
    end: dt.date = END,
    dataset: Optional[DatasetType] = None,
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
    if dataset is not None:
        parser.add_argument(
            "-d",
            "--dataset",
            dest="dataset",
            type=str,
            help=(
                "Dataset to use for comparison, "
                f"default is {dataset.name}."
            ),
            choices=[d.name for d in [DatasetType.CMIC, DatasetType.IWP_ICI]],
            default=dataset.name,
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
        "Compare CMIC or ICI data to DARDAR IWP distributions.",
        dataset=DatasetType.CMIC,
    )
    add_parser(
        subparsers,
        "cloudnet-distribution",
        "Show CLOUDNET IWP distribution.",
    )
    add_parser(
        subparsers,
        "time-series",
        "Show time series of CMIC or ICI and CLOUDNET IWP data.",
        dataset=DatasetType.CMIC,
    )
    add_parser(
        subparsers,
        "validate-by-site",
        "Compare CMIC or ICI to CLOUDNET IWP data.",
        dataset=DatasetType.CMIC,
    )
    args = parser.parse_args(args_list)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    comparison_type = args.command
    if comparison_type == "cloudnet-distribution":
        show_cloudnet_distribution(start, end)
    else:
        dataset_type = DatasetType(args.dataset.lower())
        if comparison_type == "validate-by-region":
            validate_by_region(dataset_type, start, end)
        elif comparison_type == "time-series":
            show_time_series(dataset_type, start, end)
        elif comparison_type == "validate-by-site":
            validate_by_site(dataset_type, start, end)


if __name__ == "__main__":
    cli(argv[1:])
