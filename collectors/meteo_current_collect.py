"""
Current Meteorological Data Collection Module
============================================

This module collects current meteorological data for Western North America
using the Open-Meteo API. It creates a regular grid of points across the region
and fetches comprehensive weather data for each point.

Features:
1. Creates a grid of sampling points with ~100km spacing, adjusted for latitude
2. Fetches current meteorological conditions from Open-Meteo's forecast API
3. Implements efficient caching and retry mechanisms for API requests
4. Processes the data into a structured DataFrame
5. Saves the collected data to a Parquet file for efficient storage and access

Geographic Coverage:
- Bounding Box: (-130°W, 30°N) to (-100°W, 70°N)
- Western North America (western United States and most of Canada)

Dependencies:
- openmeteo_requests: Client library for Open-Meteo API
- requests_cache: HTTP request caching
- retry_requests: Automatic retry for failed requests
- pandas: Data manipulation and storage
"""

import math
import time

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Define geographic grid parameters (~100 km spacing)
WEST, SOUTH, EAST, NORTH = -130.0, 30.0, -100.0, 70.0
LAT_STEP = 100 / 111.0  # ~0.90° latitude step (111km per degree)

# Generate grid points with latitude-dependent longitude spacing
grid = []
lat = SOUTH
while lat <= NORTH + 1e-6:
    # Calculate longitude step for this latitude (accounts for converging meridians)
    lon_step = 100 / (111.0 * math.cos(math.radians(lat)))
    lon = WEST
    while lon <= EAST + 1e-6:
        grid.append((round(lat, 3), round(lon, 3)))
        lon += lon_step
    lat += LAT_STEP
print(f"{len(grid)} grid points")  # ~970 points covering the region

# Configure Open-Meteo client with caching and retry capability
cache = requests_cache.CachedSession(".cache", expire_after=3600)  # 1-hour cache
session = retry(cache, retries=5, backoff_factor=0.5)  # Exponential backoff
om = openmeteo_requests.Client(session=session)

# Open-Meteo API configuration
URL = "https://api.open-meteo.com/v1/forecast"

# Comprehensive list of meteorological variables to collect
HOURLY_LIST = [
    "temperature_2m",  # Air temperature at 2m above ground (°C)
    "apparent_temperature",  # Feels-like temperature (°C)
    "relative_humidity_2m",  # Relative humidity at 2m (%)
    "dew_point_2m",  # Dew point temperature (°C)
    "vapour_pressure_deficit",  # Vapor pressure deficit (kPa)
    "soil_temperature_0cm",  # Soil temperature at surface (°C)
    "soil_moisture_0_to_1cm",  # Volumetric soil moisture (m³/m³)
    "soil_moisture_1_to_3cm",  # Volumetric soil moisture (m³/m³)
    "soil_moisture_3_to_9cm",  # Volumetric soil moisture (m³/m³)
    "soil_moisture_9_to_27cm",  # Volumetric soil moisture (m³/m³)
    "soil_moisture_27_to_81cm",  # Volumetric soil moisture (m³/m³)
    "evapotranspiration",  # Actual evapotranspiration (mm)
    "et0_fao_evapotranspiration",  # Reference evapotranspiration (mm)
    "wind_speed_10m",  # Wind speed at 10m (km/h)
    "wind_direction_10m",  # Wind direction at 10m (°)
    "wind_gusts_10m",  # Wind gusts at 10m (km/h)
    "precipitation",  # Precipitation amount (mm)
    "precipitation_probability",  # Probability of precipitation (%)
    "shortwave_radiation",  # Solar radiation (W/m²)
    "cloud_cover",  # Total cloud cover (%)
    "cloud_cover_low",  # Low-level cloud cover (%)
    "cloud_cover_mid",  # Mid-level cloud cover (%)
    "cloud_cover_high",  # High-level cloud cover (%)
    "surface_pressure",  # Surface pressure (hPa)
    "visibility",  # Visibility (m)
    "is_day",  # Daylight (1) or night (0)
    "sunshine_duration",  # Sunshine duration (minutes)
    "weather_code",  # WMO weather code
]
HOURLY = ",".join(HOURLY_LIST)
CHUNK = 100  # Process in chunks to keep URL < 8 kB
RATE_S = 1.0  # Pause between API calls (seconds)


def fetch(chunk):
    """
    Fetch current weather data for a chunk of grid points.

    Makes a batch request to the Open-Meteo API for multiple locations
    at once, optimizing the number of API calls required.

    Args:
        chunk: List of (latitude, longitude) tuples

    Returns:
        List of API response objects, one per location
    """
    lats, lons = zip(*chunk)
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "hourly": HOURLY,
        "forecast_hours": 1,  # Just the current/next hour
        "timezone": "auto",
    }
    try:
        return om.weather_api(URL, params=params)
    except Exception as e:
        print(f"[ERR] {e} for chunk starting {chunk[0]}")
        return []


def main():
    """
    Main function to collect and process current meteorological data.

    This function:
    1. Collects weather data for all grid points in batches
    2. Extracts the relevant data from API responses
    3. Organizes the data into a structured DataFrame
    4. Saves the data to a Parquet file for further analysis

    The resulting dataset provides a comprehensive snapshot of current
    meteorological conditions across the entire region of interest.
    """
    records = []
    # Process grid points in chunks to avoid URL length limitations
    for i in range(0, len(grid), CHUNK):
        for r in fetch(grid[i : i + CHUNK]):
            hr = r.Hourly()

            # Extract timestamp for the data point
            raw_ts = hr.Time()
            timestamp = pd.to_datetime(raw_ts, unit="s", utc=True)

            # Extract the first (current) value of each weather variable
            vals = [
                hr.Variables(j).ValuesAsNumpy()[0] for j in range(hr.VariablesLength())
            ]
            if len(vals) != len(HOURLY_LIST):
                continue  # Skip if we didn't get all expected variables

            # Create a structured record with all variables
            records.append(
                {
                    "latitude": r.Latitude(),
                    "longitude": r.Longitude(),
                    "timestamp": timestamp,
                    **dict(zip(HOURLY_LIST, vals)),
                }
            )

        # Implement rate limiting between chunks
        if i + CHUNK < len(grid):
            time.sleep(RATE_S)

    # Create DataFrame from collected records
    weather_df = pd.DataFrame(records)
    print(weather_df.head())  # Display sample data

    # Save the collected data to a Parquet file
    weather_df.to_parquet("meteo_current.parquet", index=False)
    print(f"Saved {len(weather_df)} rows → meteo_current.parquet")


if __name__ == "__main__":
    main()
