#!/usr/bin/env python
from pathlib import Path
from sys import argv
from typing import List, Optional, Tuple
import argparse
import os

from cartopy.mpl.gridliner import (  # type: ignore
    LONGITUDE_FORMATTER, LATITUDE_FORMATTER
)
import cartopy.crs as ccrs  # type: ignore
import matplotlib  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_validation.cloudnet import CloudnetResampler, match_files
from pps_mw_validation.dardar import DardarResampler
from pps_mw_validation.data_model import LandWaterMask
from pps_mw_validation.utils import (
    INCIDENCE_ANGLE,
    FOOTPRINT_SIZE,
    REF_HEIGHT,
    SAMPLING_INTERVAL,
    load_netcdf_data,
)


DARDAR_PATH = Path(os.environ.get("DARDAR_PATH", os.getcwd()))
DARDAR_FILE = DARDAR_PATH / "DARDAR-CLOUD_2009335054454_19119_V3-10.nc"
# Data files can be obtained from https://cloudnet.fmi.fi/
CLOUDNET_PATH = Path(os.environ.get("CLOUDNET_PATH", os.getcwd()))
IWC_FILE = CLOUDNET_PATH / "20230401_norunda_iwc-Z-T-method.nc"
NWP_FILE = CLOUDNET_PATH / "20230401_norunda_ecmwf.nc"


def plot_dardar_stats(resampled: xr.Dataset) -> None:
    """Plot datdar stats."""
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    axs[0, 0].remove()
    axs[1, 0].remove()
    axs[2, 0].remove()
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    c = ax.scatter(
        resampled["longitude"],
        resampled["latitude"],
        s=5,
        c=resampled["land_water_mask"],
        cmap="Set2",
        vmin=0,
        vmax=8,
    )
    cbar = fig.colorbar(
        c,
        location="bottom",
        shrink=0.8,
        ticks=[v.value + 0.5 for v in LandWaterMask],
    )
    cbar.ax.set_xticklabels(
        [v.name.replace("_", " ").lower() for v in LandWaterMask],
        rotation=-80,
    )
    ax.set_xlabel("longitude [degrees]")
    ax.set_ylabel("latitude [degrees]")
    ax = axs[0, 1]
    ax.semilogy(resampled["time"], resampled["ice_water_path"])
    ax.set_ylabel("ice water path [kg/m2]")
    ax.set_ylim([1e-4, 10.])
    ax = axs[1, 1]
    ax.plot(resampled["time"], resampled["ice_mass_height"] / 1e3)
    ax.set_ylabel("mean cloud ice mass height [km]")
    ax = axs[2, 1]
    ax.semilogy(resampled["time"], resampled["ice_mass_size"])
    ax.set_ylim([10, 1e4])
    ax.set_ylabel("mean cloud ice mass size [microns]")
    ax.set_xlabel("time")
    plt.savefig('dardar.png', bbox_inches='tight')
    plt.show()


def plot_cloudnet_stats(resampled: xr.Dataset, orig: xr.Dataset) -> None:
    """Plot cloudnet stats."""
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    axs[0, 0].remove()
    axs[1, 0].remove()
    axs[2, 0].remove()
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    c = ax.plot(
        resampled.attrs["longitude"],
        resampled.attrs["latitude"],
        'o'
    )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color='gray',
        alpha=0.5,
        linestyle='--',
    )
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_title(resampled.attrs["location"])
    ax = axs[0, 1]
    t, h = np.meshgrid(orig.time, orig.height / 1e3)
    filt = ~np.isfinite(orig.iwc.values)
    orig.iwc.values[filt] = 1e-6
    c = ax.scatter(
        t,
        h,
        s=0.1,
        c=orig.iwc.values.T,
        norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-3),
        cmap="magma",
    )
    cbar = fig.colorbar(
        c,
        location="bottom",
        shrink=0.8,
    )
    cbar.ax.set_xlabel("ice water content [kg/m3]")
    ax.set_xlim([resampled.time[0], resampled.time[-1]])
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("altutude [km]")
    ax = axs[1, 1]
    fmt = mdates.DateFormatter('%Y-%m-%dT%H:%M')
    ax.semilogy(resampled["time"], resampled["ice_water_path"])
    ax.set_ylabel("ice water path [kg/m2]")
    ax.set_ylim([1e-4, 10.])
    ax.set_xlim([resampled.time[0], resampled.time[-1]])
    ax.grid(visible=True, which='both')
    ax.xaxis.set_ticklabels([])
    ax = axs[2, 1]
    ax.plot(resampled["time"], resampled["ice_mass_height"] / 1e3)
    ax.set_ylabel("mean cloud ice mass height [km]")
    ax.set_xlabel("time")
    ax.set_xlim([resampled.time[0], resampled.time[-1]])
    ax.grid(visible=True, which='both')
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_tick_params(rotation=-20)
    plt.savefig('cloudnet.png', bbox_inches='tight')
    plt.show()


def resample_dardar_files(
    dardar_files: List[Path],
    sampling_interval: int,
    incidence_angle: float,
    footprint_size: float,
    outdir: Path,
    plot: bool = False
) -> None:
    """Resample dardar files."""
    for dardar_file in dardar_files:
        print(f"Start processing file: {dardar_file}.")
        dardar_resampler = DardarResampler(dardar_file)
        resampled = dardar_resampler.resample(
            sampling_interval,
            incidence_angle,
            footprint_size,
        )
        outfile = outdir / "resampled" / f"{dardar_file.stem}_resampled.nc"
        resampled.to_netcdf(outfile)
        if plot:
            plot_dardar_stats(resampled)


def resample_cloudnet_files(
    cloudnet_files: List[Tuple[Path, Path]],
    sampling_interval: int,
    incidence_angle: float,
    footprint_size: float,
    ref_height: float,
    outdir: Path,
    plot: bool = False
) -> None:
    """Resample cloudnet files."""
    for iwc_file, nwp_file in cloudnet_files:
        print(f"Start processing file: {iwc_file}.")
        cloudnet_resampler = CloudnetResampler(iwc_file, nwp_file)
        resampled = cloudnet_resampler.resample(
            sampling_interval,
            incidence_angle,
            footprint_size,
            ref_height,
        )
        outfile = outdir / "resampled" / f"{iwc_file.stem}_resampled.nc"
        resampled.to_netcdf(outfile)
        if plot:
            plot_cloudnet_stats(resampled, orig=load_netcdf_data(iwc_file))


def add_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    description: str,
    footprint_size: float,
    incidence_angle: float,
    sampling_interval: int,
    ref_height: Optional[float] = None,
    iwc_file: Optional[Path] = None,
    nwp_file: Optional[Path] = None,
    outdir: Optional[Path] = None,
) -> None:
    """Add parser."""
    parser = subparsers.add_parser(
        command,
        description=description,
        help=description,
    )
    parser.add_argument(
        "-f",
        "--footprint-size",
        dest="footprint_size",
        type=float,
        help=f"Footprint size [m], default is {footprint_size}",
        default=footprint_size,
    )
    parser.add_argument(
        "-i",
        "--incidence-angle",
        dest="incidence_angle",
        type=float,
        help=f"Incidence angle [degrees], default is {incidence_angle}",
        default=incidence_angle,
    )
    parser.add_argument(
        "-s",
        "--sampling-interval",
        dest="sampling_interval",
        type=int,
        help=f"Sampling interval [-], default is {sampling_interval}",
        default=sampling_interval,
    )
    parser.add_argument(
        "-p",
        "--plotdata",
        action="store_true",
        help="Flag for plotting result",
    )
    if iwc_file is not None:
        parser.add_argument(
            "-c",
            "--iwcfile",
            dest="iwc_files",
            type=str,
            nargs='+',
            help=(
                "Product file(s) containing ice water content data, "
                f"default is {iwc_file}"
            ),
            default=[iwc_file.as_posix()],
        )
    if nwp_file is not None:
        parser.add_argument(
            "-n",
            "--nwp-file",
            dest="nwp_files",
            type=str,
            nargs='+',
            help=f"NWP file(s), default is {nwp_file}",
            default=[nwp_file.as_posix()],
        )
    if outdir is not None:
        parser.add_argument(
            "-o",
            "--out-dir",
            dest="outdir",
            type=str,
            help=f"directory where to write data, default is {outdir}",
            default=outdir.as_posix(),
        )
    if ref_height is not None:
        parser.add_argument(
            "-r",
            "--ref-height",
            dest="ref_height",
            type=float,
            help=f"Reference height [m], default is {ref_height}",
            default=ref_height,
        )


def cli(args_list: List[str] = argv[1:]) -> None:
    parser = argparse.ArgumentParser(
        description="Run the ppsmw data resampler app."
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    add_parser(
        subparsers,
        "cloudnet",
        "Resample cloudnet data as observed by a conical scanner.",
        FOOTPRINT_SIZE,
        INCIDENCE_ANGLE,
        SAMPLING_INTERVAL,
        ref_height=REF_HEIGHT,
        iwc_file=IWC_FILE,
        nwp_file=NWP_FILE,
        outdir=CLOUDNET_PATH,
    )
    add_parser(
        subparsers,
        "dardar",
        "Resample dardar data as observed by a conical scanner.",
        FOOTPRINT_SIZE,
        INCIDENCE_ANGLE,
        SAMPLING_INTERVAL,
        iwc_file=DARDAR_FILE,
        outdir=DARDAR_PATH,
    )
    args = parser.parse_args(args_list)
    if args.command == "cloudnet":
        matched_files = match_files(
            iwc_files=[Path(f) for f in args.iwc_files],
            nwp_files=[Path(f) for f in args.nwp_files],
        )
        resample_cloudnet_files(
            cloudnet_files=matched_files,
            sampling_interval=args.sampling_interval,
            incidence_angle=args.incidence_angle,
            footprint_size=args.footprint_size,
            ref_height=args.ref_height,
            outdir=Path(args.outdir),
            plot=args.plotdata,
        )
    else:
        resample_dardar_files(
            dardar_files=[Path(f) for f in args.iwc_files],
            sampling_interval=args.sampling_interval,
            incidence_angle=args.incidence_angle,
            footprint_size=args.footprint_size,
            outdir=Path(args.outdir),
            plot=args.plotdata,
        )


if __name__ == "__main__":
    cli(argv[1:])
