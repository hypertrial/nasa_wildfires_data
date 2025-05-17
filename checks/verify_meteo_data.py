"""
Meteorological Data Validation
=============================

This script validates the completeness and format of meteorological data by:
1. Checking for required fields (temperature, humidity, wind speed, etc.)
2. Validating data types and formats
3. Verifying temporal and spatial coverage
4. Ensuring no critical data fields are missing

The script helps ensure that both current and forecast meteorological data
are properly formatted and complete for accurate wildfire risk analysis.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define expected ranges for meteorological variables
VARIABLE_RANGES = {
    'temperature': (-50.0, 50.0),  # °C
    'humidity': (0.0, 100.0),      # %
    'wind_speed': (0.0, 100.0),    # km/h
    'precipitation': (0.0, 500.0), # mm
    'pressure': (900.0, 1100.0),   # hPa
}

# Define critical variables that must be present
CRITICAL_VARIABLES = ['temperature', 'humidity', 'wind_speed']

def verify_meteo_data(file_path, data_type='current', min_lat=30.0, max_lat=70.0, min_lon=-130.0, max_lon=-100.0):
    """
    Verifies the integrity and completeness of meteorological data.
    
    Parameters:
    -----------
    file_path : str
        Path to the meteorological data file (CSV or Parquet)
    data_type : str
        Type of meteorological data ('current' or 'forecast')
    min_lat, max_lat, min_lon, max_lon : float
        Boundaries of the region of interest
        
    Returns:
    --------
    bool
        True if data passes all integrity checks, False otherwise
    """
    print(f"Verifying {data_type} meteorological data: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return False
    
    # Load data based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            print(f"ERROR: Unsupported file format: {file_ext}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to load data: {str(e)}")
        return False
    
    # Check row count
    if len(df) == 0:
        print("ERROR: Meteorological data file is empty")
        return False
    
    print(f"Found {len(df)} meteorological records")
    
    # Check for required columns (coordinates and timestamp are always required)
    required_columns = ['latitude', 'longitude', 'timestamp']
    
    # Add data-type specific required columns
    if data_type == 'current':
        # For current weather, we need actual observations
        required_columns.extend(CRITICAL_VARIABLES)
    elif data_type == 'forecast':
        # For forecast weather, we need forecast variables and forecast_time
        required_columns.extend(CRITICAL_VARIABLES + ['forecast_time'])
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {', '.join(missing_columns)}")
        return False
    
    # Check for null values in critical columns
    spatial_temporal_columns = ['latitude', 'longitude', 'timestamp']
    null_counts = {col: df[col].isnull().sum() for col in spatial_temporal_columns}
    
    has_critical_nulls = any(count > 0 for count in null_counts.values())
    if has_critical_nulls:
        print("ERROR: Found null values in critical spatial/temporal columns:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  - {col}: {count} null values ({count/len(df)*100:.2f}%)")
        return False
    
    # Check for nulls in meteorological variables
    meteo_nulls = {}
    for var in CRITICAL_VARIABLES:
        if var in df.columns:
            meteo_nulls[var] = df[var].isnull().sum()
    
    if any(count > 0 for count in meteo_nulls.values()):
        print("WARNING: Found null values in meteorological variables:")
        for var, count in meteo_nulls.items():
            if count > 0:
                print(f"  - {var}: {count} null values ({count/len(df)*100:.2f}%)")
    
    # Verify variable ranges for meteorological variables
    issues_found = False
    for var, (min_val, max_val) in VARIABLE_RANGES.items():
        if var in df.columns:
            if df[var].min() < min_val or df[var].max() > max_val:
                print(f"WARNING: {var} values outside expected range [{min_val}, {max_val}]")
                print(f"  - Found range: [{df[var].min()}, {df[var].max()}]")
                issues_found = True
    
    # Check geographic coverage
    lat_range = (df['latitude'].min(), df['latitude'].max())
    lon_range = (df['longitude'].min(), df['longitude'].max())
    
    if lat_range[0] > min_lat or lat_range[1] < max_lat or lon_range[0] > min_lon or lon_range[1] < max_lon:
        print("WARNING: Meteorological data does not cover the entire region of interest")
        print(f"  - Expected region: [{min_lat}, {max_lat}] × [{min_lon}, {max_lon}]")
        print(f"  - Actual coverage: [{lat_range[0]}, {lat_range[1]}] × [{lon_range[0]}, {lon_range[1]}]")
        issues_found = True
    
    # Check temporal coverage
    try:
        timestamps = pd.to_datetime(df['timestamp'])
        time_range = (timestamps.min(), timestamps.max())
        time_span = time_range[1] - time_range[0]
        
        print(f"Temporal coverage: {time_range[0]} to {time_range[1]} ({time_span})")
        
        # For forecasts, check forecast lead times
        if data_type == 'forecast' and 'forecast_time' in df.columns:
            forecast_times = pd.to_datetime(df['forecast_time'])
            max_lead_time = forecast_times.max() - timestamps.min()
            print(f"Maximum forecast lead time: {max_lead_time}")
            
            if max_lead_time < timedelta(days=2):
                print("WARNING: Forecast lead time is less than 2 days")
                issues_found = True
    except Exception as e:
        print(f"ERROR: Failed to process timestamp information: {str(e)}")
        issues_found = True
    
    # Check for duplicate records
    if data_type == 'current':
        # For current data, check for duplicates at the same location and time
        duplicates = df.duplicated(subset=['latitude', 'longitude', 'timestamp']).sum()
    else:
        # For forecast data, check for duplicates with the same forecast time
        duplicates = df.duplicated(subset=['latitude', 'longitude', 'timestamp', 'forecast_time']).sum()
    
    if duplicates > 0:
        print(f"WARNING: Found {duplicates} duplicate records")
        issues_found = True
    
    # Calculate overall data completeness
    completeness = 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    print(f"Overall data completeness: {completeness:.2f}%")
    
    if completeness < 90:
        print("WARNING: Data completeness below 90%")
        issues_found = True
    
    # Check grid point consistency (if it's a gridded dataset)
    if len(df) > 100:  # Only for larger datasets that are likely gridded
        unique_lats = df['latitude'].nunique()
        unique_lons = df['longitude'].nunique()
        expected_points = unique_lats * unique_lons
        
        if data_type == 'forecast' and 'forecast_time' in df.columns:
            unique_forecast_times = df['forecast_time'].nunique()
            expected_points *= unique_forecast_times
        
        if expected_points != len(df):
            print(f"WARNING: Grid point count inconsistency")
            print(f"  - Expected {expected_points} grid points")
            print(f"  - Found {len(df)} data points")
            issues_found = True
    
    print(f"{data_type.capitalize()} meteorological data verification completed")
    return not issues_found

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_meteo_data.py <meteo_data_file> <data_type>")
        print("  <data_type> can be 'current' or 'forecast'")
        sys.exit(1)
    
    meteo_file = sys.argv[1]
    data_type = sys.argv[2]
    
    if data_type not in ['current', 'forecast']:
        print("ERROR: <data_type> must be 'current' or 'forecast'")
        sys.exit(1)
    
    success = verify_meteo_data(meteo_file, data_type)
    
    if not success:
        print(f"{data_type.capitalize()} meteorological data verification failed")
        sys.exit(1)
    
    print(f"{data_type.capitalize()} meteorological data verification passed")
    sys.exit(0) 