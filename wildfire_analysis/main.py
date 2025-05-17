"""
Wildfire Risk Analysis - Main Module
===================================

Main entry point for the wildfire risk analysis pipeline.
Integrates the various components of the package to create a spatial graph representation
of wildfire risk by combining fire detection data with meteorological conditions.
"""

import time
import os
import argparse

from wildfire_analysis.data_processing.loader import load_all_data
from wildfire_analysis.data_processing.preprocessor import process_all_data
from wildfire_analysis.spatial.graph_builder import create_spatial_graph
from wildfire_analysis.utils.parallel import get_optimal_workers


def run_pipeline(data_dir=None, grid_size_km=10, radius_km=10, n_workers=None, 
                min_lat=30, max_lat=70, min_lon=-130, max_lon=-100):
    """
    Execute the complete data processing pipeline.
    
    This function orchestrates the entire workflow:
    1. Load the fire and weather datasets from files
    2. Apply preprocessing and cleaning operations
    3. Create the spatial graph representation
    4. Return the processed data and graph for further analysis
    
    Args:
        data_dir: Directory containing the data files.
            If None, uses the default data directory.
        grid_size_km: Size of each grid cell in kilometers
        radius_km: Search radius in kilometers for associating data with cells
        n_workers: Number of worker processes for parallel processing.
            If None, determines optimal number based on CPU cores.
        min_lat, max_lat, min_lon, max_lon: Bounding box coordinates
        
    Returns:
        tuple: (firms_df, current_weather_df, forecast_weather_df, G, node_features)
            - firms_df: Processed fire detection data
            - current_weather_df: Processed current weather data
            - forecast_weather_df: Processed forecast weather data
            - G: NetworkX graph of the spatial grid
            - node_features: Dictionary of features for each grid cell
    """
    # Determine optimal number of workers if not specified
    if n_workers is None:
        n_workers = get_optimal_workers()
    
    print(f"Starting wildfire risk analysis with {n_workers} worker processes")
    
    # Record total processing time
    total_start_time = time.time()
    
    # Load the datasets
    print("Loading datasets...")
    data_loading_time = time.time()
    firms_df, current_weather_df, forecast_weather_df = load_all_data(data_dir)
    print(f"Data loading completed in {time.time() - data_loading_time:.2f} seconds")
    
    # Process the data
    print("Preprocessing datasets...")
    preprocessing_time = time.time()
    firms_df, current_weather_df, forecast_weather_df = process_all_data(
        firms_df, current_weather_df, forecast_weather_df,
        min_lat, max_lat, min_lon, max_lon
    )
    print(f"Preprocessing completed in {time.time() - preprocessing_time:.2f} seconds")
    
    # Print some basic information about the datasets
    print(f"\nDataset sizes:")
    print(f"  Fire data: {len(firms_df)} records")
    print(f"  Current weather data: {len(current_weather_df)} records")
    print(f"  Forecast weather data: {len(forecast_weather_df)} records")
    
    # Create spatial graph with optimized parallel processing
    graph_time = time.time()
    G, node_features = create_spatial_graph(
        firms_df, 
        current_weather_df, 
        forecast_weather_df,
        grid_size_km=grid_size_km,
        radius_km=radius_km,
        n_workers=n_workers,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon
    )
    print(f"Graph creation completed in {time.time() - graph_time:.2f} seconds")
    
    # Print sample node features
    if len(node_features) > 0:
        sample_node = list(node_features.keys())[0]
        print(f"\nSample node features for grid cell {sample_node}:")
        # Show only first few and last few features to avoid overwhelming output
        all_keys = list(node_features[sample_node].keys())
        if len(all_keys) > 10:
            # Show first 5 and last 5 features
            for key in all_keys[:5]:
                print(f"  {key}: {node_features[sample_node][key]}")
            print(f"  ... {len(all_keys) - 10} more features ...")
            for key in all_keys[-5:]:
                print(f"  {key}: {node_features[sample_node][key]}")
        else:
            # Show all features if fewer than 10
            for key, value in node_features[sample_node].items():
                print(f"  {key}: {value}")
    
    print(f"\nTotal processing time: {time.time() - total_start_time:.2f} seconds")
    
    return firms_df, current_weather_df, forecast_weather_df, G, node_features


def main():
    """
    Command-line entry point for running the wildfire risk analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Wildfire Risk Analysis Pipeline')
    parser.add_argument('--data-dir', type=str, help='Directory containing data files')
    parser.add_argument('--grid-size', type=float, default=10.0, help='Grid cell size in kilometers')
    parser.add_argument('--radius', type=float, default=10.0, help='Search radius in kilometers')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    parser.add_argument('--min-lat', type=float, default=30.0, help='Minimum latitude')
    parser.add_argument('--max-lat', type=float, default=70.0, help='Maximum latitude')
    parser.add_argument('--min-lon', type=float, default=-130.0, help='Minimum longitude')
    parser.add_argument('--max-lon', type=float, default=-100.0, help='Maximum longitude')
    
    args = parser.parse_args()
    
    # Run the pipeline with command-line arguments
    firms_df, current_weather_df, forecast_weather_df, G, node_features = run_pipeline(
        data_dir=args.data_dir,
        grid_size_km=args.grid_size,
        radius_km=args.radius,
        n_workers=args.workers,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon
    )
    
    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    main()
