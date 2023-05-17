from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import datetime as dt

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .data_model import (
    BoundingBox,
    DatasetType,
    LandWaterMask,
    RegionOfInterest,
)
from .distance import get_distance
from .utils import (
    DatasetLoader,
    get_cloud_ice_prop,
    get_tilted_data,
    set_non_finites,
)


N0_STAR_K = 5.65 * 1e14
N0_STAR_L = -2.57
FILL_VALUE = {"iwc": 0.}
OUTFILE = "{dataset}_{date}_{roi}.nc"


@dataclass
class DardarLoader(DatasetLoader):
    """Class for collecting stats from dardar files."""

    def get_files(
        self,
        date: dt.date,
    ) -> List[Path]:
        """Get files for given date."""
        return self._get_files(
            self.base_path,
            self.file_pattern.format(year=date.year, doy=date.strftime('%j')),
        )

    def collect_roi_stats(
        self,
        start: dt.date,
        end: dt.date,
        region_of_interest: Dict[RegionOfInterest, BoundingBox],
        outdir: Path,
    ) -> None:
        """Collect stats within given period."""
        while start < end:
            for roi, bbox in region_of_interest.items():
                stats = self.get_stats_by_date_and_bounding_box(start, bbox)
                if stats is not None:
                    outfile = outdir / OUTFILE.format(
                        dataset=DatasetType.DARDAR.name,
                        date=start.isoformat(),
                        roi=roi.value,
                    )
                    stats.to_netcdf(outfile)
            start += dt.timedelta(days=1)

    def get_stats_by_date_and_bounding_box(
        self,
        date: dt.date,
        bbox: BoundingBox,
    ) -> Optional[xr.Dataset]:
        """Get stats by date and bounding box."""
        datasets: List[xr.Dataset] = []
        for dardar_file in self.get_files(date):
            dataset = self.get_data(dardar_file)
            filt = bbox.is_inside(dataset)
            if filt.any():
                datasets.append(dataset.isel(time=filt, drop=True))
        return xr.concat(datasets, dim="time") if len(datasets) > 0 else None


@dataclass
class DardarResampler:
    """Class for resample dardar data."""

    dardar_file: Path

    def get_data(self) -> xr.Dataset:
        """Get the Dardar dataset."""
        with xr.open_dataset(self.dardar_file) as ds:
            return xr.Dataset(
                data_vars={
                    "latitude": (
                        "time", ds.latitude.values
                    ),
                    "longitude": (
                        "time", ds.longitude.values
                    ),
                    "land_water_mask": (
                        "time", ds.land_water_mask.values.astype(int)
                    ),
                    "iwc": (
                        ("time", "height"), np.flip(ds.iwc.values, axis=1)
                    ),
                    "n0_star": (
                        ("time", "height"), np.flip(ds.N0star.values, axis=1)
                    ),
                },
                coords={
                    "height": ("height", np.flip(ds.height.values)),
                    "time":  ("time", ds.time.values),
                }
            )

    @staticmethod
    def get_cumulative_distance_from_coordinates(
        data: xr.Dataset
    ) -> xr.DataArray:
        """Get cumulative distance."""
        distances = [
            get_distance(
                data.latitude.values[idx],
                data.longitude.values[idx],
                data.latitude.values[idx + 1],
                data.longitude.values[idx + 1],
            ) for idx in range(data.latitude.values.size - 1)
        ]
        return xr.DataArray(
            np.cumsum(np.concatenate([[0.], distances])),
            dims="time",
        )

    @staticmethod
    def n0_star_to_ice_mass_size(
        n0_star: xr.DataArray,
    ) -> xr.DataArray:
        """
        Convert from n0_star to ice_mass_size.

        Following eq. 23 in:
        https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JD020700
        """
        ice_mass_size = (n0_star / N0_STAR_K) ** (1 / N0_STAR_L)
        return set_non_finites(ice_mass_size, 0.)

    def resample(
        self,
        sampling_interval: int,
        incidence_angle: float,
        footprint_size: float,
    ) -> xr.DataArray:
        """
        Resample the DARDAR file.

        DARDAR dataset is resampled to be similar to what is observed
        by a conical scanner, taking into account given incidence
        angle and footprint size, and the following data are extracted

          * ice water path
          * mean cloud ice mass height
          * mean cloud ice particle size
          * land water mask
        """
        data = self.get_data()
        tilted_data = get_tilted_data(
            data,
            ["iwc", "n0_star"],
            incidence_angle,
            self.get_cumulative_distance_from_coordinates,
        )
        tilted_data["dm"] = self.n0_star_to_ice_mass_size(
            tilted_data["n0_star"]
        )
        target_times = data.time.values[0::sampling_interval]
        target_lats = data.latitude.values[0::sampling_interval]
        target_lons = data.longitude.values[0::sampling_interval]
        target_distances = tilted_data.distance.values[0::sampling_interval]
        ice_water_paths = np.full(target_times.size, np.nan)
        ice_mass_heights = np.full(target_times.size, np.nan)
        ice_mass_sizes = np.full(target_times.size, np.nan)
        land_water_mask = LandWaterMask.get_prefilled_array(target_times)
        for idx in range(target_times.size):
            (
                ice_water_path,
                ice_mass_height,
                ice_mass_size,
                surface_type,
            ) = get_cloud_ice_prop(
                tilted_data,
                target_distances[idx],
                footprint_size,
            )
            ice_water_paths[idx] = ice_water_path
            ice_mass_heights[idx] = ice_mass_height
            ice_mass_sizes[idx] = ice_mass_size
            land_water_mask.values[idx] = surface_type
        return xr.Dataset(
            data_vars={
                "ice_water_path": (
                    "time", ice_water_paths.astype(np.float32)
                ),
                "ice_mass_height": (
                    "time", ice_mass_heights.astype(np.float32)
                ),
                "ice_mass_size": (
                    "time", ice_mass_sizes.astype(np.float32)
                ),
                "land_water_mask": land_water_mask,
            },
            coords={
                "longitude": ("time", target_lons.astype(np.float32)),
                "latitude": ("time", target_lats.astype(np.float32)),
                "time": ("time", target_times),
            },
            attrs={
                "incidence_angle": incidence_angle,
                "footprint_size": footprint_size,
                "sampling_interval": sampling_interval,
            }
        )
