"""
Weather Forecast Data Collection Module
======================================

This module collects 6-hour forecast meteorological data for Western North America
using the Open-Meteo API. It provides future weather conditions for wildfire risk
analysis and prediction.

Features:
1. Creates a grid of sampling points with ~100km spacing, adjusted for latitude
2. Retrieves 6-hour ahead weather forecasts for each grid point
3. Implements efficient caching and retry mechanisms for API requests
4. Processes the forecast data into a structured DataFrame
5. Validates that forecast timestamps are in the future
6. Saves the collected forecast data to a Parquet file for analysis

Geographic Coverage:
- Bounding Box: (-130°W, 30°N) to (-100°W, 70°N)
- Western North America (western United States and most of Canada)

The forecast data complements current meteorological conditions to provide
temporal context for wildfire risk assessment and prediction models.
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
CHUNK = 100  # Process in chunks to keep URL <8 kB
RATE_S = 1.0  # Pause between API calls (seconds)


def fetch(chunk):
    """
    Fetch weather forecast data for a chunk of grid points.

    Makes a batch request to the Open-Meteo API for multiple locations
    at once, retrieving a 6-hour forecast window to extract the relevant
    forecast point (6 hours ahead).

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
        "forecast_hours": 6,  # Request a 6-hour forecast window
        "timezone": "auto",
    }
    try:
        return om.weather_api(URL, params=params)
    except Exception as e:
        print(f"[ERR] {e} for chunk starting {chunk[0]}")
        return []


def main():
    """
    Main function to collect and process 6-hour forecast data.

    This function:
    1. Collects 6-hour forecast data for all grid points in batches
    2. Extracts the relevant forecast point (6 hours ahead) from each response
    3. Validates that the forecast timestamp is in the future
    4. Organizes the data into a structured DataFrame
    5. Normalizes data types for efficient storage
    6. Saves the data to a Parquet file for further analysis

    The resulting dataset provides a forward-looking view of meteorological
    conditions that can be used for predictive modeling of wildfire risk.
    """
    # Collect forecast data for all grid points
    records = []
    for i in range(0, len(grid), CHUNK):
        for r in fetch(grid[i : i + CHUNK]):
            hr = r.Hourly()

            # Extract metadata about the forecast time series
            start_ts = hr.Time()  # First slot UNIX timestamp
            end_ts = hr.TimeEnd()  # Exclusive end UNIX timestamp
            interval = hr.Interval()  # Seconds per slot (3600)

            # Verify we received enough forecast slots
            n_slots = int((end_ts - start_ts) // interval)
            assert n_slots >= 6, f"Expected ≥6 slots, got {n_slots}"

            # Extract the 6-hour ahead forecast (slot index 5, 0-based)
            idx = 5
            raw_ts = start_ts + interval * idx
            forecast_ts = pd.to_datetime(raw_ts, unit="s", utc=True)

            # Validate that the forecast time is actually in the future
            now = pd.Timestamp.utcnow()
            assert forecast_ts > now, (
                f"6-hour ahead forecast {forecast_ts} is not > current time ({now})"
            )

            # Extract all weather variables for the 6-hour ahead slot
            vals = [
                hr.Variables(j).ValuesAsNumpy()[idx]
                for j in range(hr.VariablesLength())
            ]
            if len(vals) != len(HOURLY_LIST):
                continue  # Skip if we didn't get all expected variables

            # Create a structured record with all forecast variables
            records.append(
                {
                    "latitude": r.Latitude(),
                    "longitude": r.Longitude(),
                    "timestamp": forecast_ts,
                    **dict(zip(HOURLY_LIST, vals)),
                }
            )

        # Implement rate limiting between chunks
        if i + CHUNK < len(grid):
            time.sleep(RATE_S)

    # Create DataFrame from collected records
    weather_df = pd.DataFrame(records)
    print(weather_df.head())  # Display sample data

    # Normalize data types for efficient storage
    weather_df = weather_df.convert_dtypes()
    weather_df.to_parquet("meteo_forecast.parquet", index=False)
    print(f"Saved {len(weather_df)} rows → meteo_forecast.parquet")


if __name__ == "__main__":
    main()
