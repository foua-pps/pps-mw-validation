from pathlib import Path
import numpy as np  # type: ignore
import xarray as xr  # type: ignore
from pyresample.geometry import AreaDefinition, SwathDefinition  # type: ignore
from pyresample.kd_tree import resample_nearest  # type: ignore


def load_prx_dataset(
    prx_file: Path,
    area: AreaDefinition,
    radius_of_influence: float,
) -> xr.Dataset:
    """Load and resample PR-HL or PR-S file."""
    dataset = xr.load_dataset(prx_file)
    resampled = resample_nearest(
        SwathDefinition(dataset["longitude"].values, dataset["latitude"].values),
        dataset["rainfall_rate"].values,
        area,
        radius_of_influence=radius_of_influence,
        fill_value=None,
    )
    return xr.Dataset(
        {
            "rainfall_rate": (("y", "x"), resampled.filled(np.nan))
        },
        attrs=dataset.attrs,
    )
