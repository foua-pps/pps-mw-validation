import pytest  # type: ignore

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


from pps_mw_validation import utils
from pps_mw_validation.dardar import DardarResampler as Resampler


@pytest.fixture
def fake_iwc_data() -> xr.Dataset:
    ntimes = 3
    nheights = 5
    iwp = np.zeros((ntimes, nheights))
    dm = np.zeros((ntimes, nheights))
    iwp[1, 3: 6] = 1.
    dm[1, 3: 6] = 100.
    return xr.Dataset(
        data_vars={
            "iwc": (("time", "height"), iwp),
            "dm": (("time", "height"), dm),
            "land_water_mask": ("time", np.zeros(ntimes)),
        },
        coords={
            "time": ("time", np.arange(ntimes)),
            "height": ("height", np.linspace(0, 10e3, nheights)),
            "latitude": ("time", np.linspace(0., 1., ntimes)),
            "longitude": ("time", np.zeros(ntimes)),
        },
    )


def test_set_non_finites():
    """Test set not finites."""
    fill_value = 2
    data_with_nans = xr.DataArray([0, 1, np.nan])
    data = utils.set_non_finites(data_with_nans, fill_value)
    np.testing.assert_array_equal(data, [0, 1, 2])


def test_get_tilted_data(
    fake_iwc_data: xr.Dataset,
):
    """Test get tilted data."""
    tilted_data = utils.get_tilted_data(
        data=fake_iwc_data,
        params=["iwc"],
        incidence_angle=45.,
        get_distance_caller=Resampler.get_cumulative_distance_from_coordinates,
    )
    np.testing.assert_array_almost_equal(
        tilted_data["iwc"].values,
        [
            [0., 0., 0., 0.1347473, 0.17966307],
            [0., 0., 0., 0.86525263, 0.82033684],
            [0., 0., 0., 0., 0.],
        ],
        decimal=3,
    )


def test_get_cloud_ice_prop(
    fake_iwc_data: xr.Dataset,
):
    """Test get cloud ice prop."""
    fake_iwc_data["distance"] = ("time", 10e3 * np.arange(3))
    cloud_ice_prop = utils.get_cloud_ice_prop(
        data=fake_iwc_data,
        target_distance=10e3,
        footprint_size=1e3,
    )
    np.testing.assert_array_almost_equal(
        cloud_ice_prop,
        (3750.0, 8333.333333333334, 100., 0)
    )


@pytest.mark.parametrize("param,expect", (
    ("median", 0.5),
    ("interquartile_range", 0.6),
))
def test_get_stats(
    param: str,
    expect: float
):
    """Test get stats."""
    edges = np.linspace(0, 1, 101)
    counts, _ = np.histogram(
        [0.1, 0.2, 0.5, 0.8, 0.9],
        edges,
    )
    data = xr.Dataset(
        data_vars={
            "ice_water_path_count": ("bin", counts),
        },
        coords={
            "lower": ("bin", edges[0:-1]),
            "upper": ("bin", edges[1::]),
        },
    )
    stats = utils.get_stats(data)
    assert stats.attrs[param] == pytest.approx(expect, abs=1e-2)
