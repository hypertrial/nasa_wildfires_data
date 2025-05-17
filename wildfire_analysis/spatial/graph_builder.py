"""
Graph Builder Module
===================

Functions for creating a spatial graph representation of wildfire risk.
Constructs a graph where nodes represent grid cells and edges connect neighboring cells.
"""

import networkx as nx
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

from wildfire_analysis.spatial.distance import calculate_haversine_distances
from wildfire_analysis.spatial.grid import create_grid_cells, create_grid_edges


def process_node_features_fire(node_batch, node_coords_array, fire_coords, firms_df, radius_km):
    """
    Process fire features for a batch of nodes in parallel.
    
    Args:
        node_batch: Tuple of (start_idx, end_idx, node_ids) for the batch of nodes to process
        node_coords_array: Array of node coordinates
        fire_coords: Array of fire coordinates
        firms_df: DataFrame containing fire data
        radius_km: Search radius in kilometers
        
    Returns:
        dict: Dictionary mapping node IDs to fire features
    """
    start_idx, end_idx, node_ids_batch = node_batch
    batch_features = {}
    
    # Calculate distances for this batch of nodes to all fire points
    batch_coords = node_coords_array[start_idx:end_idx]
    distances = calculate_haversine_distances(batch_coords, fire_coords, radius_km=radius_km)
    
    # Process each node in the batch
    for i, node_id in enumerate(node_ids_batch):
        batch_features[node_id] = {
            'fire_count': 0,
            'avg_bright_ti4': 0,
            'avg_frp': 0,
            'max_frp': 0
        }
        
        # Find indices of fires within radius
        in_radius_indices = np.where(distances[i, :] <= radius_km)[0]
        
        if len(in_radius_indices) > 0:
            # Get subset of fire data within radius
            nearby_fires = firms_df.iloc[in_radius_indices]
            
            # Calculate fire-related features
            batch_features[node_id]['fire_count'] = len(nearby_fires)
            batch_features[node_id]['avg_bright_ti4'] = nearby_fires['bright_ti4'].mean()
            batch_features[node_id]['avg_frp'] = nearby_fires['frp'].mean()
            batch_features[node_id]['max_frp'] = nearby_fires['frp'].max() if not np.isnan(nearby_fires['frp']).all() else 0
    
    return batch_features


def process_node_features_weather(node_batch, node_coords_array, weather_coords, weather_df, radius_km, is_forecast=False):
    """
    Process weather features for a batch of nodes in parallel.
    
    Args:
        node_batch: Tuple of (start_idx, end_idx, node_ids) for the batch of nodes to process
        node_coords_array: Array of node coordinates
        weather_coords: Array of weather coordinates
        weather_df: DataFrame containing weather data
        radius_km: Search radius in kilometers
        is_forecast: Boolean indicating if this is forecast data
        
    Returns:
        dict: Dictionary mapping node IDs to weather features
    """
    start_idx, end_idx, node_ids_batch = node_batch
    batch_features = {}
    
    # Calculate distances for this batch of nodes to all weather points
    batch_coords = node_coords_array[start_idx:end_idx]
    distances = calculate_haversine_distances(batch_coords, weather_coords, radius_km=radius_km)
    
    # Get numeric columns for weather features
    numeric_cols = weather_df.select_dtypes(include=['float32', 'float64', 'int64', 'Float32', 'Float64', 'Int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
    
    # Initialize features dictionary for batch
    for node_id in node_ids_batch:
        batch_features[node_id] = {}
    
    # Process each node in the batch
    for i, node_id in enumerate(node_ids_batch):
        # Find indices of weather points within radius
        in_radius_indices = np.where(distances[i, :] <= radius_km)[0]
        
        if len(in_radius_indices) > 0:
            # Get subset of weather data within radius
            nearby_weather = weather_df.iloc[in_radius_indices]
            
            # If multiple weather points exist, prioritize the closest ones
            if len(nearby_weather) > 1:
                nearby_distances = distances[i, in_radius_indices]
                sorted_indices = np.argsort(nearby_distances)
                nearby_weather = nearby_weather.iloc[sorted_indices]
            
            # Add each weather variable as a feature
            prefix = 'forecast_' if is_forecast else 'current_'
            for col in numeric_cols:
                batch_features[node_id][f'{prefix}{col}'] = nearby_weather[col].mean()
    
    return batch_features


def process_fire_batch_wrapper(args):
    """Wrapper function for processing fire features batch to avoid lambda pickling issues."""
    return process_node_features_fire(*args)


def process_weather_batch_wrapper(args):
    """Wrapper function for processing weather features batch to avoid lambda pickling issues."""
    return process_node_features_weather(*args)


def create_spatial_graph(firms_df, current_weather_df, forecast_weather_df, 
                         grid_size_km=10, radius_km=10, n_workers=4,
                         min_lat=30, max_lat=70, min_lon=-130, max_lon=-100):
    """
    Create a spatial graph representation integrating fire and weather data.
    
    This function forms the core of the spatial analysis pipeline by:
    1. Creating a grid with cells of approximately equal area
    2. Constructing a graph where each node represents a grid cell
    3. Associating nearby fire detections and weather measurements with each cell
    4. Connecting neighboring cells with edges to represent spatial relationships
    
    Args:
        firms_df: DataFrame containing fire detection data
        current_weather_df: DataFrame containing current weather data
        forecast_weather_df: DataFrame containing forecast weather data
        grid_size_km: Size of each grid cell in kilometers (default: 10)
        radius_km: Search radius in kilometers for associating data with cells (default: 10)
        n_workers: Number of parallel workers for processing (default: 4)
        min_lat, max_lat, min_lon, max_lon: Bounding box coordinates
        
    Returns:
        tuple: (G, node_features)
            - G: NetworkX graph with nodes as grid cells and edges connecting neighbors
            - node_features: Dictionary mapping node IDs to feature vectors
    """
    start_time = time.time()
    print(f"Starting spatial graph creation with {n_workers} workers...")
    
    # Create grid cells
    print(f"Creating grid cells...")
    grid_construction_time = time.time()
    lat_bins, lon_bins_by_lat, cell_centers = create_grid_cells(
        min_lat, max_lat, min_lon, max_lon, grid_size_km
    )
    
    # Extract node coordinates and IDs for easier processing
    node_coords = [(lat, lon) for lat, lon, _ in cell_centers]
    node_ids = [grid_id for _, _, grid_id in cell_centers]
    total_cells = len(node_ids)
    
    # Initialize node feature dictionary
    node_features = {
        grid_id: {
            'fire_count': 0, 'avg_bright_ti4': 0, 'avg_frp': 0, 'max_frp': 0
        } for grid_id in node_ids
    }
    
    print(f"Created {total_cells} grid cells in {time.time() - grid_construction_time:.2f} seconds")
    
    # Convert node coordinates list to NumPy array for faster calculations
    node_coords_array = np.array(node_coords)
    
    # Process fire data and associate with grid cells using parallel processing
    if not firms_df.empty:
        print(f"Processing fire data with parallel workers...")
        fire_processing_time = time.time()
        
        # Extract coordinates of all fire points
        fire_coords = firms_df[['latitude', 'longitude']].values
        
        # Determine batch size for parallel processing
        n_nodes = len(node_ids)
        batch_size = max(1, n_nodes // n_workers)
        
        # Create batches of nodes for parallel processing
        node_batches = [
            (i, min(i + batch_size, n_nodes), node_ids[i:min(i + batch_size, n_nodes)])
            for i in range(0, n_nodes, batch_size)
        ]
        
        # Process node batches in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Create a list of arguments for each batch
            batch_args = [
                (batch, node_coords_array, fire_coords, firms_df, radius_km)
                for batch in node_batches
            ]
            
            # Process batches in parallel and collect results using the wrapper function
            results = list(executor.map(process_fire_batch_wrapper, batch_args))
            
            # Merge results into the main node_features dictionary
            for batch_result in results:
                for node_id, features in batch_result.items():
                    node_features[node_id].update(features)
        
        print(f"Fire data processing completed in {time.time() - fire_processing_time:.2f} seconds")
    
    # Process current weather data using parallel processing
    if not current_weather_df.empty:
        print(f"Processing current weather data with parallel workers...")
        current_weather_time = time.time()
        
        # Extract coordinates of all weather data points
        weather_coords = current_weather_df[['latitude', 'longitude']].values
        
        # Create batches of nodes for parallel processing (reuse the batches from fire processing)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Create a list of arguments for each batch
            batch_args = [
                (batch, node_coords_array, weather_coords, current_weather_df, radius_km, False)
                for batch in node_batches
            ]
            
            # Process batches in parallel and collect results using the wrapper function
            results = list(executor.map(process_weather_batch_wrapper, batch_args))
            
            # Merge results into the main node_features dictionary
            for batch_result in results:
                for node_id, features in batch_result.items():
                    node_features[node_id].update(features)
        
        print(f"Current weather processing completed in {time.time() - current_weather_time:.2f} seconds")
    
    # Process forecast weather data using parallel processing
    if not forecast_weather_df.empty:
        print(f"Processing forecast weather data with parallel workers...")
        forecast_weather_time = time.time()
        
        # Extract coordinates of all forecast points
        forecast_coords = forecast_weather_df[['latitude', 'longitude']].values
        
        # Create batches of nodes for parallel processing (reuse the batches from fire processing)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Create a list of arguments for each batch
            batch_args = [
                (batch, node_coords_array, forecast_coords, forecast_weather_df, radius_km, True)
                for batch in node_batches
            ]
            
            # Process batches in parallel and collect results using the wrapper function
            results = list(executor.map(process_weather_batch_wrapper, batch_args))
            
            # Merge results into the main node_features dictionary
            for batch_result in results:
                for node_id, features in batch_result.items():
                    node_features[node_id].update(features)
        
        print(f"Forecast weather processing completed in {time.time() - forecast_weather_time:.2f} seconds")
    
    # Create edges between neighboring grid cells
    print(f"Creating graph edges between neighboring cells...")
    edge_construction_time = time.time()
    
    # Create edges between neighboring cells
    edge_list = create_grid_edges(lat_bins, lon_bins_by_lat, cell_centers)
    
    # Initialize the graph structure
    G = nx.Graph()
    
    # Add nodes to the graph with position attribute for visualization
    for lat, lon, grid_id in cell_centers:
        G.add_node(grid_id, pos=(lon, lat))
    
    # Add all edges to the graph at once
    G.add_edges_from(edge_list)
    
    print(f"Created {len(edge_list)} edges in {time.time() - edge_construction_time:.2f} seconds")
    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return G, node_features
