from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import datetime as dt

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .data_model import DatasetType
from .utils import (
    ROI_TYPE,
    SITE_TYPE,
    TARGET_TYPE,
    DatasetLoader,
    data_array_to_netcdf
)


PRODUCT_CMA = "_CMA_"
PRODUCT_CMIC = "_CMIC_"
OUTFILE = "{orig_file}_{target}.nc"
OUTFILE_WITH_DATE = "{dataset}_{date}_{platform}_{target}.nc"


@dataclass
class CmicLoader(DatasetLoader):
    """Class for extracting cmic data around given coordinates."""

    def get_files(
        self,
        date: dt.date,
        product: str,
        platform: str,
    ) -> List[Path]:
        """Get dataset files for given date."""
        return self._get_files(
            (
                self.base_path
                / f"{date.year}"
                / f"{date.month:02}"
                / f"{date.day:02}"
            ),
            self.file_pattern.format(product=product, platform=platform),
        )

    def collect_site_stats(
        self,
        start: dt.date,
        end: dt.date,
        platforms: List[str],
        location: SITE_TYPE,
        max_distance: float,
        outdir: Path,
    ) -> None:
        """Collect stats around given period and locations."""
        while start < end:
            for platform in platforms:
                stats = self.get_stats_by_date_and_target(
                    start,
                    location,
                    platform,
                    max_distance=max_distance,
                )
                for stat in stats:
                    outfile = outdir / OUTFILE.format(
                        orig_file=stat.attrs["cmic_file"],
                        target=stat.attrs["target"],
                    )
                    data_array_to_netcdf(stat, "ice_water_path", outfile)
            start += dt.timedelta(days=1)

    def collect_roi_stats(
        self,
        start: dt.date,
        end: dt.date,
        platforms: List[str],
        region_of_interest: ROI_TYPE,
        edges: np.ndarray,
        outdir: Path,
    ) -> None:
        """Collect stats around given period and regions of interest."""
        while start < end:
            for platform in platforms:
                stats = self.get_stats_by_date_and_target(
                    start,
                    region_of_interest,
                    platform,
                )
                for target in set([stat.attrs["target"] for stat in stats]):
                    stat = xr.concat(
                        [s for s in stats if s.attrs["target"] == target],
                        dim="pos",
                    )
                    counts = self.get_counts(stat, edges)
                    outfile = outdir / OUTFILE_WITH_DATE.format(
                        dataset=DatasetType.CMIC.name,
                        date=start.isoformat(),
                        platform=platform,
                        target=target,
                    )
                    data_array_to_netcdf(
                        counts,
                        "ice_water_path_count",
                        outfile,
                    )
            start += dt.timedelta(days=1)

    def get_stats_by_date_and_target(
        self,
        date: dt.date,
        target: TARGET_TYPE,
        platform: str,
        max_distance: Optional[float] = None,
    ) -> List[xr.DataArray]:
        """Get stats by date and location."""
        data: List[xr.DataArray] = []
        for cmic_file in self.get_files(date, PRODUCT_CMIC, platform):
            print(cmic_file)
            data += self.get_stats_by_file(cmic_file, target, max_distance)
        return data

    def get_geoloc(
        self,
        cmic_file: Path,
    ) -> xr.Dataset:
        """Get geolocation associated to cmic file."""
        return self.get_data(
            Path(cmic_file.as_posix().replace(PRODUCT_CMIC, PRODUCT_CMA))
        )

    def get_stats_by_file(
        self,
        cmic_file: Path,
        target: TARGET_TYPE,
        max_distance: Optional[float] = None,
    ) -> List[xr.DataArray]:
        """Get cmic stats by file."""
        geoloc = self.get_geoloc(cmic_file)
        hits = self.get_hits(geoloc, target, max_distance)
        data_arrays: List[xr.DataArray] = []
        if any([filt.any() for filt in hits.values()]):
            cmic_data = self.get_data(cmic_file)
            for targ, filt in hits.items():
                if filt.any():
                    data_array = xr.DataArray(
                        cmic_data.cmic_iwp[0].values[filt],
                        dims="pos",
                        coords={
                            "latitude": ("pos", geoloc.lat.values[filt]),
                            "longitude": ("pos", geoloc.lon.values[filt]),
                        },
                        attrs={
                            "time": np.datetime_as_string(
                                cmic_data.time.values[0], timezone='UTC'
                            ),
                            "target": targ.value,
                            "cmic_file": cmic_file.stem,
                        },
                    )
                    if max_distance is not None:
                        data_array.attrs["max_distance"] = max_distance
                    data_arrays.append(data_array)
        return data_arrays
