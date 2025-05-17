"""
Data Preprocessing Module
========================

Functions for cleaning and preprocessing fire and weather datasets.
"""

import pandas as pd
import numpy as np


def filter_to_bounding_box(df, min_lat=30, max_lat=70, min_lon=-130, max_lon=-100):
    """
    Filter a dataframe to only include points within the specified geographic bounding box.
    
    Args:
        df: DataFrame containing latitude and longitude columns
        min_lat: Minimum latitude (default: 30°N)
        max_lat: Maximum latitude (default: 70°N)
        min_lon: Minimum longitude (default: -130°W)
        max_lon: Maximum longitude (default: -100°W)
        
    Returns:
        DataFrame: Filtered to only include points within the bounding box
    """
    filtered_df = df[(df['latitude'] >= min_lat) & 
                      (df['latitude'] <= max_lat) & 
                      (df['longitude'] >= min_lon) & 
                      (df['longitude'] <= max_lon)].copy()
    
    return filtered_df


def handle_missing_values(df, numeric_cols=None, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame to process
        numeric_cols: List of numeric columns to fill missing values for.
            If None, uses all numeric columns.
        strategy: Strategy for filling missing values ('mean', 'median', or 'zero')
        
    Returns:
        DataFrame: Processed DataFrame with missing values handled
    """
    processed_df = df.copy()
    
    # If numeric_cols is not provided, use all numeric columns
    if numeric_cols is None:
        numeric_cols = processed_df.select_dtypes(
            include=['float32', 'float64', 'int64', 'Float32', 'Float64', 'Int64']
        ).columns
    
    # Handle missing values based on the specified strategy
    for col in numeric_cols:
        if col in processed_df.columns:
            if strategy == 'mean':
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            elif strategy == 'median':
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            elif strategy == 'zero':
                processed_df[col] = processed_df[col].fillna(0)
    
    return processed_df


def process_fire_data(firms_df, min_lat=30, max_lat=70, min_lon=-130, max_lon=-100):
    """
    Process the fire detection data.
    
    Args:
        firms_df: DataFrame containing fire detection data
        min_lat, max_lat, min_lon, max_lon: Bounding box parameters
        
    Returns:
        DataFrame: Processed fire data
    """
    # Filter to the bounding box
    processed_df = filter_to_bounding_box(firms_df, min_lat, max_lat, min_lon, max_lon)
    
    # Handle missing values in key fields
    numeric_cols = ['bright_ti4', 'frp']
    processed_df = handle_missing_values(processed_df, numeric_cols, strategy='median')
    
    return processed_df


def process_weather_data(weather_df, min_lat=30, max_lat=70, min_lon=-130, max_lon=-100):
    """
    Process the meteorological data.
    
    Args:
        weather_df: DataFrame containing weather data
        min_lat, max_lat, min_lon, max_lon: Bounding box parameters
        
    Returns:
        DataFrame: Processed weather data
    """
    # Filter to the bounding box
    processed_df = filter_to_bounding_box(weather_df, min_lat, max_lat, min_lon, max_lon)
    
    # Handle missing values for all numeric columns
    processed_df = handle_missing_values(processed_df, strategy='mean')
    
    return processed_df


def process_all_data(firms_df, current_weather_df, forecast_weather_df, 
                     min_lat=30, max_lat=70, min_lon=-130, max_lon=-100):
    """
    Process all datasets.
    
    Args:
        firms_df: DataFrame containing fire detection data
        current_weather_df: DataFrame containing current weather data
        forecast_weather_df: DataFrame containing forecast weather data
        min_lat, max_lat, min_lon, max_lon: Bounding box parameters
        
    Returns:
        tuple: (processed_firms_df, processed_current_weather_df, processed_forecast_weather_df)
    """
    processed_firms_df = process_fire_data(firms_df, min_lat, max_lat, min_lon, max_lon)
    processed_current_weather_df = process_weather_data(current_weather_df, min_lat, max_lat, min_lon, max_lon)
    processed_forecast_weather_df = process_weather_data(forecast_weather_df, min_lat, max_lat, min_lon, max_lon)
    
    return processed_firms_df, processed_current_weather_df, processed_forecast_weather_df
