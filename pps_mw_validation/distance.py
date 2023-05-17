from typing import Union

import numpy as np  # type: ignore


GEOID_SEMI_MAJOR_AXIS = 6378137.0
GEOID_SEMI_MINOR_AXIS = 6356752.3


def get_distance(
    orig_latitude: float,
    orig_longitude: float,
    dest_latitude: Union[float, np.ndarray],
    dest_longitude: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Get Haversine distance(s) in m between a given position
    and other position(s).

    :param orig_latitude: origin latitude [degrees]
    :param orig_longitude: origin longitude [degrees]
    :param dest_latitude: destination latitude(s) [degrees]
    :param dest_longitude: destination longitude(s) [degrees]

    Returns:
        distances between a given position on geoid and data points of
        the lookup dataset [m]
    """
    orig_lat_rad = np.radians(orig_latitude)
    orig_lon_rad = np.radians(orig_longitude)
    dest_lat_rad = np.radians(dest_latitude)
    dest_lon_rad = np.radians(dest_longitude)
    deltaLat = dest_lat_rad - orig_lat_rad
    deltaLon = dest_lon_rad - orig_lon_rad
    mean_lat_rad = (dest_lat_rad + orig_lat_rad) * 0.5
    a = (
        np.sin(0.5 * deltaLat) ** 2 + np.cos(orig_lat_rad)
        * np.cos(dest_lat_rad) * np.sin(0.5 * deltaLon) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius = np.sqrt(
        (
            (GEOID_SEMI_MAJOR_AXIS ** 2 * np.cos(mean_lat_rad)) ** 2
            + (GEOID_SEMI_MINOR_AXIS ** 2 * np.sin(mean_lat_rad)) ** 2
        ) / (
            (GEOID_SEMI_MAJOR_AXIS * np.cos(mean_lat_rad)) ** 2
            + (GEOID_SEMI_MINOR_AXIS * np.sin(mean_lat_rad)) ** 2
        )
    )
    return radius * c
