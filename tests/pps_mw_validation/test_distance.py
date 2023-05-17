import pytest  # type: ignore

from pps_mw_validation import distance


@pytest.mark.parametrize(
    "orig_lat,orig_lon,dest_lat,dest_lon,expect", (
        (0., 0., 1., 0., 111319.462),
        (0., 0., 0., 1., 111319.490),
        (45., 45., 45., 46., 78582.864),
        (45., 45., 46., 45., 111130.400),
        (90., 0., 89., 0., 110946.286),
    )
)
def test_get_distance(
    orig_lat: float,
    orig_lon: float,
    dest_lat: float,
    dest_lon: float,
    expect: float,
):
    dist = distance.get_distance(
        orig_lat, orig_lon, dest_lat, dest_lon
    )
    assert dist == pytest.approx(expect, abs=1e-3)
