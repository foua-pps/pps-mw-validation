#!/usr/bin/env python
from pathlib import Path
from sys import argv
from typing import List, Dict, Optional, Union
import argparse
import datetime as dt
import os

from pps_mw_validation.cloudnet import CLOUDNET_LOCATION, CloudnetSite
from pps_mw_validation.cmic import CmicLoader
from pps_mw_validation.dardar import DardarLoader
from pps_mw_validation.ici import IwpIciLoader
from pps_mw_validation.data_model import (
    REGION_OF_INTEREST, RegionOfInterest, DatasetType,
)


OUTDIR = Path(os.environ.get("PPSMW_OUTDIR_PATH", os.getcwd()))
DARDAR_PATH = Path(os.environ.get("DARDAR_RESAMPLED_PATH", os.getcwd()))
ICI_PATH = Path(os.environ.get("ICI_PATH", os.getcwd()))
CMIC_PATH = Path("/data/lang/satellit2/polar/pps")
LOADER_TYPE = Union[CmicLoader, DardarLoader, IwpIciLoader]
DATASET_LOADER: Dict[DatasetType, LOADER_TYPE] = {
    DatasetType.CMIC: CmicLoader(
        base_path=CMIC_PATH,
        file_pattern="*{product}*{platform}*",
    ),
    DatasetType.DARDAR: DardarLoader(
        base_path=DARDAR_PATH,
        file_pattern="DARDAR-CLOUD_{year}{doy}*_resampled.nc",
    ),
    DatasetType.IWP_ICI: IwpIciLoader(
        base_path=ICI_PATH,
        file_pattern="W_XX-EUMETSAT-Darmstadt,SAT,SGB1-MSP-02-LIW_C_EUMT_(?P<created>\d+)_G_D_(?P<start>\d+)_(?P<end>\d+)_",  # noqa
    ),
}
PLATFORMS = ["eos1", "eos2", "metopb", "metopc", "noaa20", "npp"]
MAX_DISTANCE = 8e3  # [m]
START = dt.datetime.utcnow().date()
END = START + dt.timedelta(days=1)


def add_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    description: str,
    start: dt.date,
    end: dt.date,
    outdir: Path,
    datasets: List[DatasetType] = [d for d in DATASET_LOADER],
    location: Optional[CloudnetSite] = None,
    max_distance: Optional[float] = None,
    roi: Optional[RegionOfInterest] = None,
    platform: Optional[str] = None,
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
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="outdir",
        type=str,
        help=f"directory where to write data, default is {outdir}",
        default=outdir.as_posix(),
    )
    if location is not None:
        parser.add_argument(
            "-l",
            "--location",
            dest="locations",
            type=str,
            nargs='+',
            help="Cloudnet location(s), default is Norunda.",
            choices=[loc.value for loc in CloudnetSite],
            default=[location.value],
        )

    if max_distance is not None:
        parser.add_argument(
            "-m",
            "--max-distance",
            dest="max_distance",
            type=float,
            help=f"Max distance, default is {max_distance} m.",
            default=max_distance,
        )
    if platform is not None:
        parser.add_argument(
            "-p",
            "--platform",
            dest="platforms",
            type=str,
            nargs='+',
            help=f"Associated platform(s) of CMIC data, default is {platform}.",
            choices=PLATFORMS,
            default=["noaa20"],
        )
    if roi is not None:
        parser.add_argument(
            "-r",
            "--region",
            dest="region_of_interests",
            type=str,
            nargs='+',
            help="Region of interest(s), default is tropics.",
            choices=[roi.value for roi in RegionOfInterest],
            default=[roi.value],
        )
    if dataset is not None:
        parser.add_argument(
            "-d",
            "--dataset",
            dest="dataset",
            type=str,
            help=(
                "Dataset from which to extract data, "
                f"default is {dataset.name}."
            ),
            choices=[d.name for d in datasets],
            default=dataset.name,
        )


def cli(args_list: List[str] = argv[1:]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ppsmw validation data collection app. "
        )
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    add_parser(
        subparsers,
        "site",
        "Extract data around given Cloudnet radar station.",
        location=CloudnetSite.NORUNDA,
        start=START,
        end=END,
        dataset=DatasetType.CMIC,
        datasets=[DatasetType.CMIC, DatasetType.IWP_ICI],
        max_distance=MAX_DISTANCE,
        outdir=OUTDIR,
        platform="noaa20",
    )
    add_parser(
        subparsers,
        "roi",
        "Extract stats within given region of interest.",
        roi=RegionOfInterest.TROPICS,
        start=START,
        end=END,
        dataset=DatasetType.DARDAR,
        outdir=OUTDIR,
        platform="noaa20",
    )
    args = parser.parse_args(args_list)
    start = dt.datetime.fromisoformat(args.start).date()
    end = dt.datetime.fromisoformat(args.end).date()
    dataset_type = DatasetType(args.dataset.lower())
    outdir = Path(args.outdir)
    loader = DATASET_LOADER[dataset_type]
    if args.command == "site":
        max_distance = args.max_distance
        platforms = args.platforms
        location = {
            CloudnetSite(loc): CLOUDNET_LOCATION[CloudnetSite(loc)]
            for loc in args.locations
        }
        if isinstance(loader, CmicLoader):
            platforms = args.platforms
            loader.collect_site_stats(
                start, end, platforms, location, max_distance, outdir,
            )
        elif isinstance(loader, IwpIciLoader):
            loader.collect_site_stats(
                start, end, location, max_distance, outdir,
            )
    elif args.command == "roi":
        roi = {
            RegionOfInterest(r): REGION_OF_INTEREST[RegionOfInterest(r)]
            for r in args.region_of_interests
        }
        if isinstance(loader, CmicLoader):
            platforms = args.platforms
            loader.collect_roi_stats(start, end, platforms, roi, outdir)
        else:
            loader.collect_roi_stats(start, end, roi, outdir)


if __name__ == "__main__":
    cli(argv[1:])
