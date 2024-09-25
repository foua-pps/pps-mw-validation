from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import datetime as dt
import os

import h5py  # type: ignore
import numpy as np  # type: ignore
import xarray as xr  # type: ignore


BALTRAD_DIR = os.environ.get(
    "BALTRAD_DIR",
    "BALTRAD/{year}/{month}/{day}/{hour}/{minute}",
)
BALTRAD_FILENAME_FMT = "comp_pcappi_blt2km_pn150_{year}{month}{day}T{hour}{minute}00Z_0x40000000001.h5"  # noqa: E501
MIN_BALTRAD_QUALITY = 0.8  # [-]
FWHM = 8  # [pixel]


class PrecipitationCalculator(Enum):
    """Precipitation calculator enum."""
    # http://www.borenv.net/BER/archive/pdfs/ber7/ber7-253.pdf

    RAIN = (200., 1.5)
    SNOW = (400., 2.)

    def __init__(
        self,
        a: float,
        b: float,
    ):
        self.a = a
        self.b = b

    def get_rate(
        self,
        dbz: np.ndarray,
    ) -> np.ndarray:
        """Get precipitation rate from radar reflectivity."""
        return 10 ** ((dbz - 10 * np.log10(self.a)) / (10 * self.b))


@dataclass
class Baltrad:
    """Class for loading BALTRAD data."""

    @staticmethod
    def get_matching_file(t0: dt.datetime) -> Path:
        """Get BALTRAD file path matching given time."""

        def fill_date(filename: str, t0: dt.datetime) -> str:
            """Fill in date"""
            return filename.format(
                year=t0.year,
                month=f"{t0.month:02}",
                day=f"{t0.day:02}",
                hour=f"{t0.hour:02}",
                minute=f"{t0.minute:02}",
            )

        nearest_quarter = dt.datetime(
            t0.year, t0.month, t0.day, t0.hour,
        ) + dt.timedelta(minutes=15 * np.round(t0.minute / 15))
        path = fill_date(BALTRAD_DIR, nearest_quarter)
        name = fill_date(BALTRAD_FILENAME_FMT, nearest_quarter)
        return Path(path) / name

    @staticmethod
    def _smooth(
        data: np.ndarray,
        coords: xr.DataArray,
        fwhm: float,
    ) -> np.ndarray:
        """Smooth data with Gaussian kernel."""
        n = int(fwhm * 2)
        s0, s1 = data.shape
        x, y = np.meshgrid(np.arange(s1), np.arange(s0))
        out = np.full((coords.y.size, coords.x.size), np.nan)
        for idx_i, i in enumerate(coords.y.values):
            for idx_j, j in enumerate(coords.x.values):
                i0 = max(i - n, 0)
                i1 = min(i + n + 1, coords.y.values[-1])
                j0 = max(j - n, 0)
                j1 = min(j + n + 1, coords.x.values[-1])
                w = np.exp(
                    -(
                        (
                            (i - y[i0:i1, j0:j1]) ** 2
                            + (j - x[i0:i1, j0:j1]) ** 2
                        ) / (2 * (fwhm / 2.335) ** 2)
                    )
                )
                out[idx_i, idx_j] = (
                    np.nanmean(w * data[i0:i1, j0:j1])
                    / np.nanmean(w)
                )
        return out

    @classmethod
    def get_reflectivity(cls, t0: dt.datetime) -> np.ndarray:
        """Load BALTRAD reflectivity."""
        with h5py.File(cls.get_matching_file(t0), "r") as data:

            dataset = data["dataset1"]["data3"]["quality4"]
            quality = dataset["data"][:]
            gain = dataset["what"].attrs["gain"]
            offset = dataset["what"].attrs["offset"]

            quality_filt = (quality * gain + offset) < MIN_BALTRAD_QUALITY

            dataset = data["dataset1"]["data3"]
            dbz = dataset["data"][:]
            gain = dataset["what"].attrs["gain"]
            offset = dataset["what"].attrs["offset"]
            no_data = dataset["what"].attrs["nodata"]

            no_data_filt = dbz == no_data

            dbz = (dbz * gain + offset).astype(np.float32)
            dbz[quality_filt | no_data_filt] = np.nan
            return dbz

    @classmethod
    def get_precipitation(
        cls,
        prhl: xr.DataArray,
        fwhm: Optional[float] = FWHM,
    ):
        """Get precipitation matching given PRHL data."""
        reflectivity = cls.get_reflectivity(prhl.attrs["time"])
        rain_rate = PrecipitationCalculator.RAIN.get_rate(reflectivity)
        if fwhm is not None:
            return cls._smooth(rain_rate, prhl, fwhm)
        return rain_rate
