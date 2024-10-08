from dataclasses import dataclass
from math import nan
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import datetime as dt

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .data_model import (
    BoundingBox,
    CloudnetSite,
    LandWaterMask,
    Location,
    RegionOfInterest,
)


INCIDENCE_ANGLE = 51.
FOOTPRINT_SIZE = 16e3
REF_HEIGHT = 6e3
SAMPLING_INTERVAL = 10

N0_STAR_K = 5.65 * 1e14
N0_STAR_L = -2.57

FILL_VALUE = {"iwc": 0.}

SITE_TYPE = Dict[CloudnetSite, Location]
ROI_TYPE = Dict[RegionOfInterest, BoundingBox]
TARGET_TYPE = Union[SITE_TYPE, ROI_TYPE]


@dataclass
class DatasetLoader:
    base_path: Path
    file_pattern: str

    @staticmethod
    def _get_files(
        base_path: Path,
        file_pattern: str,
    ) -> List[Path]:
        """Get files."""
        return get_files(base_path, file_pattern)

    def get_data(
        self,
        datafile: Path,
        time_idx: Optional[int] = None,
        group: Optional[str] = None,
    ) -> xr.Dataset:
        """Get data."""
        data = load_netcdf_data(datafile, group=group)
        if time_idx is not None:
            data = data.isel(time=time_idx, drop=True)
        return data

    @staticmethod
    def get_hits_by_site(
        geoloc: xr.Dataset,
        location: SITE_TYPE,
        max_distance: float,
    ) -> Dict[CloudnetSite, np.ndarray]:
        """Get hits by site."""
        return {
            loc: coords.is_inside(geoloc, max_distance)
            for loc, coords in location.items()
        }

    @staticmethod
    def get_hits_by_roi(
        geoloc: xr.Dataset,
        roi: ROI_TYPE,
    ) -> Dict[RegionOfInterest, np.ndarray]:
        """Get hits by roi."""
        return {roi: bbox.is_inside(geoloc) for roi, bbox in roi.items()}

    def get_hits(
        self,
        geoloc: xr.Dataset,
        target: TARGET_TYPE,
        max_distance: Optional[float] = None,
    ) -> Dict[Any, np.ndarray]:
        """Get hits."""
        if isinstance(list(target.keys())[0], CloudnetSite):
            assert max_distance is not None
            location = cast(SITE_TYPE, target)
            return self.get_hits_by_site(geoloc, location, max_distance)
        roi = cast(ROI_TYPE, target)
        return self.get_hits_by_roi(geoloc, roi)

    @staticmethod
    def get_counts(
        data: xr.DataArray,
        edges: np.ndarray,
    ) -> xr.DataArray:
        """Get counts."""
        offset = (edges[0] + edges[1]) / 2
        counts, _ = np.histogram(data.values + offset, edges)
        return xr.DataArray(
            counts,
            dims="bin",
            coords={
                "lower": ("bin", edges[0:-1]),
                "upper": ("bin", edges[1::]),
            },
        )


def get_files(
    base_path: Path,
    file_pattern: str,
    date_range: Optional[Tuple[dt.date, dt.date, str]] = None,
) -> List[Path]:
    """Get files."""
    files = [f for f in base_path.glob(file_pattern) if f.is_file()]
    if date_range is not None:
        filtered_files: List[Path] = []
        start, end, date_fmt = date_range
        while start <= end:
            date_str = start.strftime(date_fmt)
            filtered_files += [f for f in files if date_str in f.stem]
            start += dt.timedelta(days=1)
        return np.unique(filtered_files).tolist()
    return files


def load_netcdf_data(
    netcdf_file: Path,
    fill_values: Optional[Dict[str, float]] = None,
    group: Optional[str] = None,
) -> xr.Dataset:
    """Load Cloudnet dataset."""
    with xr.open_dataset(netcdf_file, group=group) as ds:
        if fill_values is not None:
            for param, fill_value in fill_values.items():
                if param in ds:
                    ds[param] = set_non_finites(ds[param], fill_value)
        return ds


def set_non_finites(
    data: xr.DataArray,
    fill_value: float,
) -> xr.Dataset:
    """Set non finites."""
    filt = ~np.isfinite(data.values)
    data.values[filt] = fill_value
    return data


def data_array_to_netcdf(
    data_array: xr.DataArray,
    name: str,
    outfile: Path,
) -> None:
    """Write data array to netcdf file."""
    dataset = data_array.to_dataset(name=name, promote_attrs=True)
    dataset[name].attrs = {}
    dataset.to_netcdf(outfile)


def get_tilted_data(
    data: xr.Dataset,
    params: List[str],
    incidence_angle: float,
    get_distance_caller: Callable[[xr.Dataset], xr.DataArray],
    aux_params: List[str] = ["height", "land_water_mask"],
) -> xr.Dataset:
    """Get data in a tilted geometry."""
    distance = get_distance_caller(data)
    dataset = xr.Dataset(
        data_vars={
            param: (("time", "height"), np.zeros_like(data[param].values))
            for param in params
        }
    )
    dataset["distance"] = distance
    for param in aux_params:
        if param in data:
            dataset[param] = data[param]
    for idx, height in enumerate(data.height.values):
        for param in params:
            dataset[param].values[:, idx] = np.interp(
                distance.values + height * np.tan(np.radians(incidence_angle)),
                distance.values,
                data[param].values[:, idx],
            )
    return dataset


def get_cloud_ice_prop(
    data: xr.Dataset,
    target_distance: float,
    footprint_size: float,
) -> Tuple[float, float, float, Optional[int]]:
    """Get cloud ice properties."""
    distance = np.abs(data.distance.values - target_distance)
    filt = distance <= footprint_size / 2
    n = np.count_nonzero(filt)
    ice_water_paths = np.trapz(data.iwc[filt], data.height)
    ice_mass_heights = np.trapz(
        data.iwc[filt] * np.tile(data.height, (n, 1)),
        data.height,
    ) / np.where(ice_water_paths > 0, ice_water_paths, 1.)
    if "dm" in data:
        ice_mass_sizes = np.trapz(
            data.iwc[filt] * data.dm[filt],
            data.height,
        ) / np.where(ice_water_paths > 0, ice_water_paths, 1.)
    ice_water_path = np.mean(ice_water_paths)
    if ice_water_path > 0:
        ice_mass_height = np.sum(
            ice_mass_heights * ice_water_paths
        ) / np.sum(ice_water_paths)
    else:
        ice_mass_height = nan
    if ice_water_path > 0 and "dm" in data:
        ice_mass_size = np.sum(
            ice_mass_sizes * ice_water_paths
        ) / np.sum(ice_water_paths)
    else:
        ice_mass_size = nan
    if "land_water_mask" in data:
        mask = LandWaterMask.get_mask(
            data.land_water_mask.values[filt]
        )
    else:
        mask = None
    return ice_water_path, ice_mass_height, ice_mass_size, mask


def get_stats(
    data: xr.Dataset,
) -> xr.Dataset:
    """Get pdf and distribution."""
    center = (data.lower + data.upper).values / 2.
    bin_size = (data.upper - data.lower).values
    count = data.ice_water_path_count.values
    pdf = count / bin_size / np.sum(count)
    dist = np.cumsum(pdf * bin_size)
    quartile = np.interp([0.25, 0.5, 0.75], dist, center)
    return xr.Dataset(
        data_vars={
            "pdf": ("x", pdf),
            "dist": ("x", dist)
        },
        coords={
            "x": ("x", center)
        },
        attrs={
            "mean": np.trapz(center * pdf, center),
            "quartile1": quartile[0],
            "quartile2": quartile[1],
            "quartile3": quartile[2],
            "interquartile_range": quartile[2] - quartile[0],
        }
    )
