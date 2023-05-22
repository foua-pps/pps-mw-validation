from pps_mw_validation.dardar import DardarResampler

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


class TestDardarResampler:
    """Test class for DardarResampler."""

    def test_get_cumulative_distance_from_coordinates(self):
        """Test get cumulative distance from coordinates."""
        data = xr.Dataset(
            data_vars={
                "latitude": ("pos", [0., .25, 0.5, 0.75, 1.0]),
                "longitude": ("pos", [0., 0., 0., 0., 0.]),
            },
            coords={"pos": ("pos", [0, 1, 2, 3, 4])},
        )
        dist = DardarResampler.get_cumulative_distance_from_coordinates(
            data
        )
        np.testing.assert_array_almost_equal(
            dist,
            [0.,  27829.9,  55659.7,  83489.6, 111319.5],
            decimal=1,
        )
