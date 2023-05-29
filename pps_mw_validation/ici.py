from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import datetime as dt
import re

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .data_model import CloudnetSite, DatasetType
from .utils import (
    ROI_TYPE,
    SITE_TYPE,
    TARGET_TYPE,
    DatasetLoader,
    data_array_to_netcdf,
)


DATE_FMT = "%Y%m%d%H%M%S"
OUTFILE = "{orig_file}_{target}.nc"
OUTFILE_WITH_DATE = "{dataset}_{date}_{target}.nc"


@dataclass
class IwpIciLoader(DatasetLoader):
    """Class for extracting iwp data around given coordinates."""

    def get_files(
        self,
        date: dt.date,
    ) -> List[Path]:
        """Get dataset files for given date."""
        return [
            f for f in self.base_path.glob("*")
            if f.is_file() and cover_date(f, self.file_pattern, date)
        ]

    def get_iwp_data(
        self,
        ici_file: Path,
    ) -> xr.Dataset:
        """Get ice water path data."""
        data = self.get_data(ici_file, group="data/iwp")
        data = data.rename({
            "ici_latitude": "lat",
            "ici_longitude": "lon",
            "ici_time_start_scan_utc": "time",
        })
        return data

    def collect_site_stats(
        self,
        start: dt.date,
        end: dt.date,
        location: SITE_TYPE,
        max_distance: float,
        outdir: Path,
    ) -> None:
        """Collect stats around given period and locations."""
        while start < end:
            stats = self.get_stats_by_date_and_target(
                start,
                location,
                max_distance=max_distance,
            )
            for stat in stats:
                outfile = outdir / OUTFILE.format(
                    orig_file=stat.attrs["ici_file"],
                    target=stat.attrs["target"],
                )
                data_array_to_netcdf(stat, "ice_water_path", outfile)
            start += dt.timedelta(days=1)

    def collect_roi_stats(
        self,
        start: dt.date,
        end: dt.date,
        region_of_interest: ROI_TYPE,
        outdir: Path,
    ) -> None:
        """Collect stats around given period and regions of interest."""
        while start < end:
            stats = self.get_stats_by_date_and_target(
                start,
                region_of_interest,
            )
            for target in set([stat.attrs["target"] for stat in stats]):
                stat = xr.concat(
                    [s for s in stats if s.attrs["target"] == target],
                    dim="pos",
                )
                outfile = outdir / OUTFILE_WITH_DATE.format(
                    dataset=DatasetType.IWP_ICI.name,
                    date=start.isoformat(),
                    target=target,
                )
                data_array_to_netcdf(stat, "ice_water_path", outfile)
            start += dt.timedelta(days=1)

    def get_stats_by_date_and_target(
        self,
        date: dt.date,
        target: TARGET_TYPE,
        max_distance: Optional[float] = None,
    ) -> List[xr.DataArray]:
        """Get stats by date and location."""
        data: List[xr.DataArray] = []
        for ici_file in self.get_files(date):
            data += self.get_stats_by_file(ici_file, target, max_distance)
        return data

    def get_stats_by_file(
        self,
        ici_file: Path,
        target: TARGET_TYPE,
        max_distance: Optional[float] = None,
    ) -> List[xr.DataArray]:
        """Get stats by file."""
        data = self.get_iwp_data(ici_file)
        hits: Dict[Any, np.ndarray] = {}
        if isinstance(list(target.keys())[0], CloudnetSite):
            assert max_distance is not None
            location = cast(SITE_TYPE, target)
            hits = self.get_hits_by_site(data, location, max_distance)
        else:
            roi = cast(ROI_TYPE, target)
            hits = self.get_hits_by_roi(data, roi)
        data_arrays: List[xr.DataArray] = []
        if any([filt.any() for filt in hits.values()]):
            for targ, filt in hits.items():
                if filt.any():
                    idxs_scan, _ = np.where(filt)
                    data_array = xr.DataArray(
                        data.ice_water_path.values[filt],
                        dims="pos",
                        coords={
                            "latitude": ("pos", data.lat.values[filt]),
                            "longitude": ("pos", data.lon.values[filt]),
                        },
                        attrs={
                            "time": np.datetime_as_string(
                                data.time.values[idxs_scan[0]],
                                timezone='UTC',
                            ),
                            "target": targ.name.lower(),
                            "ici_file": ici_file.stem,
                        },
                    )
                    if max_distance is not None:
                        data_array.attrs["max_distance"] = max_distance
                    data_arrays.append(data_array)
        return data_arrays


def cover_date(
    ici_file: Path,
    file_pattern: str,
    target_date: dt.date,
) -> bool:
    """Check if file contains measurements for given date."""
    m = re.match(file_pattern, ici_file.stem)
    if m is not None:
        data = m.groupdict()
        start_date = dt.datetime.strptime(data["start"], DATE_FMT).date()
        end_date = dt.datetime.strptime(data["end"], DATE_FMT).date()
        return start_date == target_date or end_date == target_date
    return False
