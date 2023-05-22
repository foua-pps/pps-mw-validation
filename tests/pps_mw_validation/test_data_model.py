import pytest  # type: ignore

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_validation.data_model import (
    BoundingBox,
    CloudnetSite,
    LandWaterMask,
    Location,
)


@pytest.fixture
def bbox() -> BoundingBox:
    return BoundingBox(
        lat_min=10.,
        lat_max=20.,
        lon_min=30.,
        lon_max=40.,
    )


@pytest.fixture
def location() -> Location:
    return Location(lat=10., lon=30.)


class TestCloudnetSite:
    """Test class for CloudnetSite."""

    @pytest.mark.parametrize("site,expect", (
        (CloudnetSite.BUCHAREST, "bucharest"),
        (CloudnetSite.NY_ALESUND, "alesund"),
    ))
    def test_lower_case_name(
        self,
        site: CloudnetSite,
        expect: str,
    ):
        """Test lower case name."""
        assert site.lower_case_name == expect


class TestBoudingBox:
    """Test class for BoundingBox."""

    @pytest.mark.parametrize("lat,lon,expect", (
        (15., 35., True),
        (10., 30., True),
        (5., 30., False),
        (10., 41., False),
    ))
    def test_is_inside(
        self,
        bbox: BoundingBox,
        lat: float,
        lon: float,
        expect: bool,
    ):
        """Test is inside."""
        data = xr.Dataset(
            {
                "lat": ("pos", np.array([lat])),
                "lon": ("pos", np.array([lon])),
            }
        )
        assert bbox.is_inside(data) == expect


class TestLocation:
    """Test class for Location."""

    @pytest.mark.parametrize("lat,lon,max_distance,expect", (
        (10., 30., 1000., True),
        (10., 30.1, 1000., False),
        (10.1, 30., 1000., False),
        (10., 30.1, 20000., True),
    ))
    def test_is_inside(
        self,
        location: Location,
        lat: float,
        lon: float,
        max_distance: float,
        expect: bool,
    ):
        """Test is inside."""
        data = xr.Dataset(
            {
                "lat": ("pos", np.array([lat])),
                "lon": ("pos", np.array([lon])),
            }
        )
        assert location.is_inside(data, max_distance) == expect


class TestLandWaterMask:
    """Test class for LandWaterMask."""

    @pytest.mark.parametrize("mask_values,expect", (
        (
            np.array([
                LandWaterMask.DEEP_OCEAN.value,
                LandWaterMask.COASTLINES.value,
            ]),
            LandWaterMask.COASTLINES.value,
        ),
        (
            np.array([
                LandWaterMask.DEEP_OCEAN.value,
                LandWaterMask.LAND.value,
            ]),
            LandWaterMask.COASTLINES.value,
        ),
        (
            np.array([
                LandWaterMask.SHALLOW_OCEAN.value,
                LandWaterMask.DEEP_OCEAN.value,
                LandWaterMask.DEEP_OCEAN.value,
            ]),
            LandWaterMask.DEEP_OCEAN.value,
        ),

    ))
    def test_get_mask(
        self,
        mask_values: np.ndarray,
        expect: int,
    ):
        """Test get mask."""
        assert LandWaterMask.get_mask(mask_values) == expect

    def test_get_prefilled_array(self):
        """Test get prefilled array."""
        times = np.array([0, 1])
        prefilled_mask = LandWaterMask.get_prefilled_array(times)
        assert prefilled_mask.size == 2
