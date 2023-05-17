from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .data_model import CloudnetSite, Location
from .utils import (
    FILL_VALUE,
    get_cloud_ice_prop,
    get_tilted_data,
    load_netcdf_data,
)

CLOUDNET_LOCATION: Dict[CloudnetSite, Location] = {
    CloudnetSite.BUCHAREST: Location(44.348, 26.029),
    CloudnetSite.CHILBOLTON: Location(51.144, -1.439),
    CloudnetSite.GALATI: Location(45.435, 28.037),
    CloudnetSite.HYYTIALA: Location(61.844, 24.288),
    CloudnetSite.JUELICH: Location(50.906, 6.407),
    CloudnetSite.KENTTAROVA: Location(67.988, 24.243),
    CloudnetSite.LEIPZIG: Location(51.353, 12.435),
    CloudnetSite.LINDENBERG: Location(52.208, 14.118),
    CloudnetSite.MUNICH: Location(48.148, 11.573),
    CloudnetSite.NORUNDA: Location(60.086, 17.479),
    CloudnetSite.NY_ALESUND: Location(78.924, 11.870),
    CloudnetSite.PALAISEAU: Location(48.716, 2.212),
    CloudnetSite.SCHNEEFERNERHAUS: Location(47.417, 10.977),
}


@dataclass
class CloudnetResampler:
    """Class for resampling cloudnet data."""

    iwc_file: Path
    nwp_file: Path

    @staticmethod
    def get_wind_speed(
        data: xr.Dataset,
        time: np.ndarray,
        ref_height: float,
    ) -> xr.DataArray:
        """Get wind speed."""
        idx = np.argmin(np.abs(data.height[0] - ref_height).values)
        return xr.DataArray(
            np.interp(
                time,
                data.time,
                np.sqrt(data.uwind[:, idx] ** 2 + data.vwind[:, idx] ** 2),
            ),
            dims="time",
        )

    @staticmethod
    def get_cumulative_distance_from_wind_speed(
        data: xr.Dataset,
    ) -> xr.DataArray:
        """Get cumulative distance."""
        time_delta = (data.time - data.time[0]).astype(float) / 1e9
        distance = np.array([
            np.trapz(data.wind_speed[0:idx], time_delta[0:idx])
            for idx in range(1, time_delta.size + 1)
        ])
        return xr.DataArray(distance, dims="time")

    def resample(
        self,
        sampling_interval: int,
        incidence_angle: float,
        footprint_size: float,
        ref_height: float,
    ) -> xr.Dataset:
        """
        Resample cloudnet files.

        Cloudnet dataset is resampled to be similar to what is observed
        by a conical scanner, taking into account given incidence
        angle and footprint size, and the following data are extracted

          * ice water path
          * mean cloud ice mass height
        """
        radar_data = load_netcdf_data(self.iwc_file, fill_values=FILL_VALUE)
        nwp_data = load_netcdf_data(self.nwp_file)
        radar_data["wind_speed"] = self.get_wind_speed(
            nwp_data,
            radar_data.time,
            ref_height,
        )
        tilted_data = get_tilted_data(
            radar_data,
            ["iwc"],
            incidence_angle,
            self.get_cumulative_distance_from_wind_speed,
        )
        target_times = radar_data.time.values[0::sampling_interval]
        target_distances = tilted_data.distance.values[0::sampling_interval]
        ice_water_paths = np.full(target_times.size, np.nan)
        ice_mass_heights = np.full(target_times.size, np.nan)
        for idx in range(target_times.size):
            ice_water_path, ice_mass_height, _, _ = get_cloud_ice_prop(
                tilted_data,
                target_distances[idx],
                footprint_size,
            )
            ice_water_paths[idx] = ice_water_path
            ice_mass_heights[idx] = ice_mass_height
        return xr.Dataset(
            data_vars={
                "ice_water_path": (
                    "time", ice_water_paths.astype(np.float32)
                ),
                "ice_mass_height": (
                    "time", ice_mass_heights.astype(np.float32)
                ),
            },
            coords={
                "time": ("time", target_times),
            },
            attrs={
                "location": radar_data.attrs["location"],
                "longitude": radar_data.longitude.data,
                "latitude": radar_data.latitude.data,
                "altitude": radar_data.altitude.data,
                "incidence_angle": incidence_angle,
                "footprint_size": footprint_size,
                "sampling_interval": sampling_interval,
                "ref_height": ref_height,
            }
        )


def get_file_info(
    cloudnet_file: Path,
) -> Optional[Dict[str, str]]:
    """Get info from file name."""
    m = re.match(r"(?P<date>\d+)_(?P<location>[a-z\-]+)_", cloudnet_file.stem)
    if m is not None:
        return m.groupdict()
    return None


def match_files(
    iwc_files: List[Path],
    nwp_files: List[Path],
) -> List[Tuple[Path, Path]]:
    """Get matched files."""
    matched_files: List[Tuple[Path, Path]] = []
    nwp_file_info = [get_file_info(f) for f in nwp_files]
    for iwc_file in iwc_files:
        iwc_file_info = get_file_info(iwc_file)
        if iwc_file_info is None:
            continue
        index = nwp_file_info.index(iwc_file_info)
        matched_files.append((iwc_file, nwp_files[index]))
    return matched_files
