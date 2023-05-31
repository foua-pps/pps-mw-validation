from pathlib import Path
from typing import Dict, List, Optional
import datetime as dt
import os

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_validation.cloudnet import CloudnetSite
from pps_mw_validation.data_model import DatasetType, RegionOfInterest
from pps_mw_validation.utils import load_netcdf_data, get_files, get_stats


CLOUDNET_PATH = Path(os.environ.get("CLOUDNET_RESAMPLED_PATH", os.getcwd()))
DARDAR_PATH = Path(os.environ.get("DARDAR_STAT_PATH", os.getcwd()))
CMIC_PATH = Path(os.environ.get("CMIC_STAT_PATH", os.getcwd()))
ICI_PATH = Path(os.environ.get("ICI_STAT_PATH", os.getcwd()))
DATASET_PATH = {
    DatasetType.CMIC: CMIC_PATH,
    DatasetType.DARDAR: DARDAR_PATH,
    DatasetType.IWP_ICI: ICI_PATH,
}


def load_dataset(
    dataset: DatasetType,
    start: dt.date,
    end: dt.date,
    sites: List[CloudnetSite] = list(CloudnetSite),
) -> Dict[CloudnetSite, xr.DataArray]:
    """Load dataset data."""
    data: Dict[CloudnetSite, xr.DataArray] = {}
    for site in sites:
        files = get_files(
            DATASET_PATH[dataset],
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


def load_dataset_distribution(
    dataset_type: DatasetType,
    start: dt.date,
    end: dt.date,
    rois: List[RegionOfInterest] = list(RegionOfInterest),
) -> Dict[RegionOfInterest, xr.Dataset]:
    """Load dataset distribution data."""
    data: Dict[RegionOfInterest, xr.DataArray] = {}
    for roi in rois:
        files = get_files(
            DATASET_PATH[dataset_type],
            f"*{dataset_type.name}*{roi.value}*",
            (start, end, "%Y-%m-%d")
        )
        if len(files) > 0:
            dataset = sum([load_netcdf_data(f) for f in files])
            data[roi] = get_stats(dataset)
    return data


def compare(
    cloudnet_data: xr.Dataset,
    other_data: xr.DataArray,
    min_iwp: float,
    max_time_diff: float,
) -> Optional[xr.Dataset]:
    """Match and compare cloudnet to another dataset."""
    # interpolate cloudnet data to time of other dataset using nearest neighbour
    idxs = []
    for idx_other, t in enumerate(other_data.time):
        time_diff = np.abs(t - cloudnet_data.time) / np.timedelta64(1, "s")
        idx_cloudnet = np.argmin(time_diff.values)
        if time_diff[idx_cloudnet] < max_time_diff:
            idxs.append((idx_other, idx_cloudnet))
    matching_data = xr.Dataset(
        data_vars={
            "ice_water_path_other": (
                "time",
                [other_data.values[idx] for idx, _ in idxs],
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
                [other_data.time.values[idx] for idx, _ in idxs]
            )
        },
    )
    filt = (
        (matching_data.ice_water_path_cloudnet >= min_iwp)
        | (matching_data.ice_water_path_other >= min_iwp)
    )
    diff = (
        matching_data.ice_water_path_other[filt]
        - matching_data.ice_water_path_cloudnet[filt]
    )
    try:
        q1, q2, q3 = np.percentile(diff, [25, 50, 75])
    except IndexError:
        return None
    matching_data.attrs = {
        "counts": diff.size,
        "median_bias": q2,
        "interquartile_range": q3 - q1,
        "mean_absolute_error": float(np.mean(np.abs(diff))),
    }
    return matching_data
