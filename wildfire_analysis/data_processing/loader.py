"""
Data Loading Module
==================

Functions for loading fire detection and weather datasets from disk.
"""

import os
import pandas as pd
from pathlib import Path


def get_data_path():
    """
    Get the path to the data directory.
    
    Returns:
        Path: Path object pointing to the data directory
    """
    # Get the absolute path to the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Get path to the data directory
    data_path = project_root / "data"
    
    return data_path


def load_fire_data(data_dir=None, filename="fires_combined.csv"):
    """
    Load the FIRMS wildfire detection data from CSV file.
    
    Args:
        data_dir (str or Path, optional): Directory containing the data files.
            If None, uses the default data directory.
        filename (str, optional): Name of the fire data CSV file.
    
    Returns:
        DataFrame: Pandas DataFrame containing fire detection data
    """
    if data_dir is None:
        data_dir = get_data_path()
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / filename
    
    print(f"Loading fire data from {file_path}")
    firms_df = pd.read_csv(file_path)
    
    return firms_df


def load_current_weather(data_dir=None, filename="meteo_current.parquet"):
    """
    Load the current meteorological data from Parquet file.
    
    Args:
        data_dir (str or Path, optional): Directory containing the data files.
            If None, uses the default data directory.
        filename (str, optional): Name of the current weather parquet file.
    
    Returns:
        DataFrame: Pandas DataFrame containing current weather conditions
    """
    if data_dir is None:
        data_dir = get_data_path()
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / filename
    
    print(f"Loading current weather data from {file_path}")
    current_weather_df = pd.read_parquet(file_path)
    
    return current_weather_df


def load_forecast_weather(data_dir=None, filename="meteo_forecast.parquet"):
    """
    Load the forecast meteorological data from Parquet file.
    
    Args:
        data_dir (str or Path, optional): Directory containing the data files.
            If None, uses the default data directory.
        filename (str, optional): Name of the forecast weather parquet file.
    
    Returns:
        DataFrame: Pandas DataFrame containing 6-hour weather forecasts
    """
    if data_dir is None:
        data_dir = get_data_path()
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / filename
    
    print(f"Loading forecast weather data from {file_path}")
    forecast_weather_df = pd.read_parquet(file_path)
    
    return forecast_weather_df


def load_all_data(data_dir=None):
    """
    Load all three datasets (fire, current weather, and forecast weather) at once.
    
    Args:
        data_dir (str or Path, optional): Directory containing the data files.
            If None, uses the default data directory.
    
    Returns:
        tuple: (firms_df, current_weather_df, forecast_weather_df)
            - firms_df: DataFrame containing fire detection data
            - current_weather_df: DataFrame containing current weather conditions
            - forecast_weather_df: DataFrame containing 6-hour weather forecasts
    """
    firms_df = load_fire_data(data_dir)
    current_weather_df = load_current_weather(data_dir)
    forecast_weather_df = load_forecast_weather(data_dir)
    
    return firms_df, current_weather_df, forecast_weather_df
