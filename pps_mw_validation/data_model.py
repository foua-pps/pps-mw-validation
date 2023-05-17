from dataclasses import dataclass
from enum import Enum
from statistics import mode
from typing import Dict

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .distance import get_distance


class DatasetType(Enum):
    CLOUDNET = "cloudnet"
    CMIC = "cmic"
    DARDAR = "dardar"


class CloudnetSite(Enum):
    BUCHAREST = 'Bucharest'
    CHILBOLTON = 'Chilbolton'
    GALATI = 'Galați'
    HYYTIALA = 'Hyytiälä'
    JUELICH = 'Jülich'
    KENTTAROVA = 'Kenttärova'
    LEIPZIG = 'Leipzig'
    LINDENBERG = 'Lindenberg'
    MUNICH = 'Munich'
    NORUNDA = 'Norunda'
    NY_ALESUND = 'Ny-Ålesund'
    PALAISEAU = 'Palaiseau'
    SCHNEEFERNERHAUS = 'Schneefernerhaus'

    @property
    def lower_case_name(self) -> str:
        """Get last part of lower case name."""
        lower = self.name.lower()
        if "_" in lower:
            return lower.split("_")[-1]
        return lower


class RegionOfInterest(Enum):
    ARCTIC = 'arctic'
    CENTRAL_ANTARCTICA = 'central_antarctica'
    MID_LATITUDE_NORTH = 'mid_latitude_north'
    MID_LATITUDE_SOUTH = 'mid_latitude_south'
    SOUTHERN_OCEAN = 'southern_ocean'
    TROPICS = 'tropics'


@dataclass
class BoundingBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def is_inside(
        self,
        data: xr.Dataset,
    ) -> xr.DataArray:
        """Check if coordinates are inside bbox."""
        lon = "lon" if "lon" in data else "longitude"
        lat = "lat" if "lat" in data else "latitude"
        return (
            (data[lon] >= self.lon_min)
            & (data[lon] <= self.lon_max)
            & (data[lat] >= self.lat_min)
            & (data[lat] <= self.lat_max)
        )


@dataclass
class Location:
    lat: float
    lon: float

    def is_inside(
        self,
        data: xr.Dataset,
        max_distance: float
    ) -> xr.DataArray:
        distance = get_distance(self.lat, self.lon, data.lat, data.lon)
        return distance <= max_distance


class LandWaterMask(Enum):
    SHALLOW_OCEAN = 0
    LAND = 1
    COASTLINES = 2
    SHALLOW_INLAND_WATER = 3
    INTERMITTENT_WATER = 4
    DEEP_INLAND_WATER = 5
    CONTINENTAL_OCEAN = 6
    DEEP_OCEAN = 7
    MISSING = -9

    @classmethod
    def get_mask(cls, mask: np.ndarray) -> int:
        """Get a single value for the land water wask"""
        if np.any(mask == cls.COASTLINES.value):
            return cls.COASTLINES.value
        elif np.any(mask == cls.LAND.value) and np.any(mask != cls.LAND.value):
            return cls.COASTLINES.value
        return cls(mode(mask)).value

    @classmethod
    def get_prefilled_array(cls, time: np.ndarray) -> xr.DataArray:
        """Get prefilled land water mask."""
        return xr.DataArray(
            data=np.zeros(time.size, dtype=np.int8),
            dims="time",
            attrs={
                "flag_values": [np.int8(mask.value) for mask in cls],
                "flag_meanings": " ".join(mask.name.lower() for mask in cls),
            }
        )


REGION_OF_INTEREST: Dict[RegionOfInterest, BoundingBox] = {
    RegionOfInterest.ARCTIC: BoundingBox(
        lat_min=75.,
        lat_max=90.,
        lon_min=-180.,
        lon_max=180.,
    ),
    RegionOfInterest.CENTRAL_ANTARCTICA: BoundingBox(
        lat_min=-90.,
        lat_max=-80.,
        lon_min=-180.,
        lon_max=180.,
    ),
    RegionOfInterest.MID_LATITUDE_NORTH: BoundingBox(
        lat_min=45.,
        lat_max=60.,
        lon_min=-180.,
        lon_max=180.,
    ),
    RegionOfInterest.MID_LATITUDE_SOUTH: BoundingBox(
        lat_min=-60.,
        lat_max=-45.,
        lon_min=-180.,
        lon_max=180.,
    ),
    RegionOfInterest.SOUTHERN_OCEAN: BoundingBox(
        lat_min=-65.,
        lat_max=-55.,
        lon_min=-180.,
        lon_max=180.,
    ),
    RegionOfInterest.TROPICS: BoundingBox(
        lat_min=-25.,
        lat_max=25.,
        lon_min=-180.,
        lon_max=180.,
    ),
}
