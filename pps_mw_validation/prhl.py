from dataclasses import dataclass
from pathlib import Path
import datetime as dt

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


RESOLUTION = 4  # [pixel]


@dataclass
class PrHL:
    """Class for loading PR-HL data."""

    @staticmethod
    def _load_dataset(prhlfile: Path) -> xr.Dataset:
        """Load dataset."""
        with xr.load_dataset(prhlfile) as data:
            return data

    @staticmethod
    def _get_time(data: xr.Dataset) -> dt.datetime:
        """Get time associated to the data"""
        start, end = (
            dt.datetime.fromisoformat(data.attrs[time])
            for time in ["time_coverage_start", "time_coverage_end"]
        )
        return start + (end - start) / 2

    @classmethod
    def get_precipitation(
        cls,
        prhlfile: Path,
        resolution: int = RESOLUTION,
    ) -> xr.Dataset:
        """Get precipitation at specfied resolution."""
        data = cls._load_dataset(prhlfile)
        s0, s1 = data["rainfall_rate"].values.shape
        s0 = int(s0 // resolution)
        s1 = int(s1 // resolution)
        y = np.arange(0, s0 * resolution, resolution)
        x = np.arange(0, s1 * resolution, resolution)
        return xr.Dataset(
            {
                "rainfall_rate": (
                    ["y", "x"],
                    data["rainfall_rate"].values[y][:, x],
                ),
                "rainfall_rate_uncertainty": (
                    ["y", "x"],
                    data["rainfall_rate_uncertainty"].values[y][:, x],
                ),
                "condition": (
                    ["y", "x"],
                    data["condition"].values[y][:, x].astype(int),
                )
            },
            coords={"y": 2 * y + 1, "x": 2 * x + 1},
            attrs={"time": cls._get_time(data)}
        )
