"""
Retrieve the solar direction for a given time and location.

Azimuth is with respect to 0 degrees (North) and increases clockwise.

"""

import pvlib
from datetime import datetime
import pytz


def get_solar_position(lat, lon, timestamp):
    # Convert timestamp to datetime if needed and set timezone
    timezone = pytz.timezone("UTC")  # Adjust as necessary
    timestamp = timezone.localize(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))

    # Get solar position data
    solar_position = pvlib.solarposition.get_solarposition(
        time=timestamp, latitude=lat, longitude=lon
    )

    # Extract azimuth and elevation
    azimuth = solar_position["azimuth"].values[0]
    elevation = solar_position["elevation"].values[0]

    return azimuth, elevation


# Example usage
latitude = 37.420973  # Example latitude
longitude = -122.158410  # Example longitude
timestamp = "2024-10-28 22:06:36"  # Example UTC timestamp

azimuth, elevation = get_solar_position(latitude, longitude, timestamp)
print(f"Solar Azimuth: {azimuth:.2f} degrees")
print(f"Solar Elevation: {elevation:.2f} degrees")
