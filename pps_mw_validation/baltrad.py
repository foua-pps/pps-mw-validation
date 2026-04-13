from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import datetime as dt
import logging

import numpy as np  # type: ignore
import xarray as xr  # type: ignore
import yaml

from h5py import File  # type: ignore
from pyproj import Transformer  # type: ignore
from pyresample import AreaDefinition, load_area  # type: ignore
from pyresample.geometry import SwathDefinition  # type: ignore
from pyresample.kd_tree import resample_gauss  # type: ignore
from pyresample.utils import fwhm2sigma  # type: ignore


BALTRAD_FILE_FMT = "/data/lang/radar/baltrad/%Y/%m/%d/%H/%M/comp_pcappi_blt2km_pn150_%Y%m%dT%H%M00Z_0x40000000001.h5"  # noqa: E501
DATASET = {
    "reflectivity": "dataset1/data3",
    "distance": "dataset1/data3/quality3",
    "qi": "dataset1/data3/quality4",
    "site": "dataset1/data3/quality5",
    "pod": "dataset1/data3/quality6",
}
DIMS = ("y", "x")


logger = logging.getLogger(__name__)


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
class BaltradReader:
    """Reader of BALTRAD composite file."""

    compositefile: Path

    def _get_reflectivity(self, fh: File) -> Optional[xr.DataArray]:
        """Get reflectivity."""
        try:
            dataset = fh[DATASET["reflectivity"]]
        except KeyError:
            logger.warning("Found no reflectivity dataset in file.")
            return None

        data = dataset["data"][:].astype(np.float32)
        gain = dataset["what"].attrs["gain"]
        offset = dataset["what"].attrs["offset"]
        no_data = dataset["what"].attrs["nodata"]
        quantity = dataset["what"].attrs["quantity"]
        return xr.DataArray(
            np.where(data == no_data, np.nan, data * gain + offset),
            name="reflectivity",
            dims=DIMS,
            attrs={
                "scale_factor": gain,
                "add_offset": offset,
                "missing": no_data,
                "quantity": quantity.decode(),
                "area": self._get_area(fh),
            }
        )

    @staticmethod
    def _get_quantity(fh: File, quantity: str) -> Optional[xr.DataArray]:
        """Get desired quantity."""
        try:
            dataset = fh[DATASET[quantity]]
        except KeyError:
            logger.warning(f"Found no {quantity} dataset in file.")
            return None

        data = dataset["data"][:].astype(np.float32)
        gain = dataset["what"].attrs["gain"]
        offset = dataset["what"].attrs["offset"]
        return xr.DataArray(
            data * gain + offset,
            name=quantity,
            dims=DIMS,
            attrs={
                "scale_factor": gain,
                "add_offset": offset,
            }
        )

    @staticmethod
    def _get_site(fh: File) -> xr.DataArray:
        """Get radar site."""
        site_ids = fh["dataset1"]["how"].attrs["nodes"].decode().replace("'", "").split(",")
        return xr.DataArray(
            fh[DATASET["site"]]["data"][:],
            name="site",
            dims=DIMS,
            attrs={"site": {site: idx for idx, site in enumerate(["N/A"] + site_ids)}},
        )

    @staticmethod
    def _get_area(fh: File) -> AreaDefinition:
        """Get projection info from the Baltrad file."""
        area = fh["where"]
        projdef = f"{area.attrs["projdef"].decode()} +units=m"
        transformer = Transformer.from_crs(
            crs_from="+proj=longlat +ellps=bessel +datum=WGS84 +units=m",
            crs_to=projdef,
            always_xy=True
        )
        x_min, y_min = transformer.transform(area.attrs["LL_lon"], area.attrs["LL_lat"])
        x_max, y_max = transformer.transform(area.attrs["UR_lon"], area.attrs["UR_lat"])
        return AreaDefinition(
            "BALTRAD",
            "BALTRAD region of interest over the nordic countries.",
            "BALTRAD",
            projdef,
            area.attrs["xsize"],
            area.attrs["ysize"],
            (x_min, y_min, x_max, y_max),
        )

    def get_data(self) -> Optional[xr.Dataset]:
        """Get data for given given file."""
        with File(self.compositefile, mode="r") as fh:
            reflectivity = self._get_reflectivity(fh)
            if reflectivity is None:
                return None
            site = self._get_site(fh)
            qi, pod, distance = [self._get_quantity(fh, q) for q in ["qi", "pod", "distance"]]
        return xr.merge(
            [q for q in [reflectivity, site, qi, pod, distance] if q is not None]
        )


@dataclass
class BaltradResampler:
    """Baltrad data resampler."""

    baltradfile_format: str
    outfile_format: str
    area: AreaDefinition
    radius_of_influence: float
    full_width_half_maximum: float
    neighbours: int
    timedelta: dt.timedelta
    elevation: Optional[xr.DataArray]
    distance_max: Optional[float] = None
    poo_max: Optional[float] = None
    qi_min: Optional[float] = None
    elevation_max: Optional[float] = None
    blacklist: Optional[list[str]] = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaltradResampler":
        """Get resampler from config dict."""
        try:
            if "elevationfile" in config:
                data = xr.load_dataset(config["elevationfile"])
                elevation = data["elevation"]
            else:
                elevation = None

            return cls(
                baltradfile_format=config["baltradfile_format"],
                outfile_format=config["outfile_format"],
                area=load_area(config["areafile"], config["area"]),
                radius_of_influence=config["radius_of_influence"],
                full_width_half_maximum=config["full_width_half_maximum"],
                neighbours=config["neighbours"],
                timedelta=dt.timedelta(minutes=config["timedelta_minutes"]),
                elevation=elevation,
                distance_max=config.get("distance_max", None),
                poo_max=config.get("poo_max", None),
                qi_min=config.get("qi_min", None),
                elevation_max=config.get("elevation_max", None),
                blacklist=config.get("blacklist", None),
            )
        except KeyError as err:
            raise ValueError(f"Missing settings in config file: {str(err)}")

    def _get_quality(self, dataset: xr.Dataset) -> xr.DataArray:
        """Get custom quality, or set quality to 0.0 for certain conditions."""

        qi = dataset["qi"].copy()

        if self.qi_min is not None:
            ok = qi.values >= self.qi_min
        else:
            ok = np.ones_like(qi.values, dtype=bool)

        if self.elevation is not None and self.elevation_max is not None:
            ok = ok & (self.elevation.values <= self.elevation_max)

        if "pod" in dataset and self.poo_max is not None:
            # pod is 1 - poo
            ok = ok & (dataset["pod"].values >= self.poo_max)

        if "distance" in dataset and self.distance_max is not None:
            ok = ok & (dataset["distance"].values <= self.distance_max)

        if self.blacklist is not None:
            for site in self.blacklist:
                if site in dataset["site"].attrs["site"]:
                    site_value = dataset["site"].attrs["site"][site]
                    ok = ok & (dataset["site"].values != site_value)

        qi.values[~ok] = 0.0

        return qi

    def _resample(self, reflectivity: xr.DataArray, quality: xr.DataArray) -> xr.Dataset:
        """Resample to desired area."""

        rainfall_rate = np.where(
            quality.values >= 0.9,
            PrecipitationCalculator.RAIN.get_rate(reflectivity.values),
            np.nan,
        )

        lons, lats = reflectivity.attrs["area"].get_lonlats()
        filt = np.isfinite(rainfall_rate)

        resampled = resample_gauss(
            SwathDefinition(lons=lons[filt], lats=lats[filt]),
            rainfall_rate[filt],
            self.area,
            fill_value=np.nan,
            neighbours=self.neighbours,
            radius_of_influence=self.radius_of_influence,
            sigmas=fwhm2sigma(self.full_width_half_maximum),
        )
        return xr.Dataset(
            {
                "rainfall_rate": xr.DataArray(
                    resampled,
                    dims=DIMS,
                    attrs=reflectivity.attrs,
                ),
                self.area.area_id: xr.DataArray(
                    data=0,
                    attrs=self.area.crs.to_cf(),
                ),
                DIMS[0]: xr.DataArray(
                    self.area.projection_y_coords,
                    dims=DIMS[0],
                    attrs={
                        "standard_name": "projection_y_coordinate",
                        "units": "m",
                    }
                ),
                DIMS[1]: xr.DataArray(
                    self.area.projection_x_coords,
                    dims=DIMS[1],
                    attrs={
                        "standard_name": "projection_x_coordinate",
                        "units": "m",
                    }
                )
            }
        )

    def resample(self, baltradfile: Path) -> Optional[xr.Dataset]:
        """Process given time slot."""
        if not baltradfile.is_file():
            logger.warning(f"BALTRAD file {baltradfile.as_posix()} is not available.")
            return None

        reader = BaltradReader(baltradfile)
        dataset = reader.get_data()
        if dataset is None:
            logger.warning(f"Failed resample BALTRAD file {baltradfile.as_posix()}.")
            return None

        quality = self._get_quality(dataset)
        dataset = self._resample(dataset["reflectivity"], quality)
        return dataset


def get_matching_file(t0: dt.datetime) -> Path:
    """Get the nearest in time BALTRAD file."""
    nearest_quarter = (
        dt.datetime(t0.year, t0.month, t0.day, t0.hour)
        + dt.timedelta(minutes=np.round(t0.minute / 15) * 15)
    )
    return Path(nearest_quarter.strftime(BALTRAD_FILE_FMT))


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load config file."""
    try:
        with open(config_path, mode="rt", encoding="utf-8") as fgr:
            return yaml.safe_load(fgr)
    except Exception:
        raise ValueError(
            f"Failed loading config file: {config_path.as_posix()}"
        )
