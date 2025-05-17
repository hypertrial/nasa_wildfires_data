"""
FIRMS Fire Detection Data Collection Module
==========================================

This module fetches active fire detections from NASA's FIRMS (Fire Information for
Resource Management System) API for the Western North America region. It combines
data from multiple satellite sources into a single comprehensive dataset.

The module:
1. Makes parallel API requests to multiple satellite data sources
2. Processes the CSV responses containing fire detection data
3. Combines and deduplicates records from different satellites
4. Saves the unified dataset to a CSV file for subsequent analysis

Data Sources:
- VIIRS (Visible Infrared Imaging Radiometer Suite) - SNPP, NOAA-20, NOAA-21 satellites
- MODIS (Moderate Resolution Imaging Spectroradiometer)
- LANDSAT (Land Remote-Sensing Satellite System)

Geographic Coverage:
- Bounding Box: (-130°W, 30°N) to (-100°W, 70°N)
- Western North America (western United States and most of Canada)

Environment Variables:
- FIRMS_MAP_KEY: API key for NASA FIRMS (optional, default provided)
"""
import os
import io
import requests
import pandas as pd
import concurrent.futures as cf

# API configuration
API_KEY = os.getenv("FIRMS_MAP_KEY", "21d91674ac77fef5ac2dd945f0430548")
BBOX    = "-130.0,30.0,-100.0,70.0"      # Western North America bounding box
DAY     = "1"                            # Fetch detections from last 24 hours
DATE    = "2025-05-14"                   # Specific date for historical data
BASE    = "https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv"

# Satellite data sources to query
SOURCES = [
    "VIIRS_SNPP_NRT",     # VIIRS instrument on Suomi NPP satellite
    "VIIRS_NOAA20_NRT",   # VIIRS instrument on NOAA-20 satellite
    "VIIRS_NOAA21_NRT",   # VIIRS instrument on NOAA-21 satellite
    "LANDSAT_NRT",        # Landsat satellites (US/Canada only)
    "MODIS_NRT",          # MODIS instrument (lower resolution)
]

def fetch(src: str) -> pd.DataFrame:
    """
    Fetch fire detection data from a specific satellite source via FIRMS API.
    
    Makes an HTTP request to the FIRMS API for the specified satellite source
    and parses the CSV response into a DataFrame. Handles empty results and errors.
    
    Args:
        src: Satellite source identifier (e.g., "VIIRS_SNPP_NRT")
        
    Returns:
        pandas.DataFrame: Fire detection data from the source or empty DataFrame if no data
    """
    url = f"{BASE}/{API_KEY}/{src}/{BBOX}/{DAY}/{DATE}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        if r.text.count("\n") <= 1:           # header only → no detections
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"{src}: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to collect and process fire detection data.
    
    This function:
    1. Fetches data from multiple satellite sources in parallel
    2. Combines the data into a single DataFrame
    3. Deduplicates records based on coordinates, date, and time
    4. Standardizes the data structure with consistent columns
    5. Saves the results to a CSV file for further analysis
    
    Returns:
        pandas.DataFrame: Combined fire detection data or empty DataFrame if no fires
    """
    # Fetch data from all sources concurrently for better performance
    with cf.ThreadPoolExecutor() as pool:
        df = pd.concat(pool.map(fetch, SOURCES), ignore_index=True)

    if df.empty:
        print("No active fires detected.")
        return df

    # Remove duplicate detections of the same fire from different satellites
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    
    # Ensure consistent columns across all source datasets
    cols = ["latitude","longitude","bright_ti4","scan","track","acq_date","acq_time",
            "satellite","instrument","confidence","version","bright_ti5","frp","daynight"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]
    
    # Save processed data to CSV file
    df.to_csv("fires_combined.csv", index=False)
    print(f"{len(df)} unique detections → fires_combined.csv")
    
    # Display the first few rows for verification
    print(df.head())
    
    return df

if __name__ == "__main__":
    main()
