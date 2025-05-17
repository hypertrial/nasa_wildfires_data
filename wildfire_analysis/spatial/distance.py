"""
Distance Calculation Module
==========================

Functions for calculating distances between geographic points.
Includes standard Haversine formula and optimized implementations.
"""

import numpy as np
import math
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial import cKDTree
import numba


def convert_to_radians(coords_array):
    """
    Convert latitude and longitude coordinates from degrees to radians.
    
    This function handles various input types and ensures consistent output format.
    
    Args:
        coords_array: NumPy array or similar of shape (n, 2) with latitude and longitude in degrees
        
    Returns:
        NumPy array of shape (n, 2) with coordinates converted to radians
    """
    # Convert to standard NumPy array with float64 dtype
    # This handles both pandas Float64 (nullable) and standard float data types
    coords_array = np.array(coords_array, dtype=np.float64)
    return np.radians(coords_array)


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point (decimal degrees)
        lat2, lon2: Latitude and longitude of the second point (decimal degrees)
        
    Returns:
        float: Great-circle distance in kilometers between the two points
    """
    # Earth's mean radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula calculation
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance


@numba.njit
def fast_haversine(lat1_rad, lon1_rad, lat2_rad, lon2_rad):
    """
    Fast Haversine distance calculation between two points using Numba.
    
    Args:
        lat1_rad, lon1_rad: Latitude and longitude of the first point (in radians)
        lat2_rad, lon2_rad: Latitude and longitude of the second point (in radians)
        
    Returns:
        float: Great-circle distance in kilometers between the two points
    """
    # Earth's mean radius in kilometers
    R = 6371.0
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula calculation
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    return distance


def calculate_haversine_distances(source_coords, target_coords, radius_km=None):
    """
    Calculate the Haversine distances between points efficiently.
    
    Uses scikit-learn's vectorized implementation for smaller datasets
    and a KDTree approximation for larger datasets to improve performance.
    
    Args:
        source_coords: NumPy array of shape (n, 2) with latitude and longitude in degrees
        target_coords: NumPy array of shape (m, 2) with latitude and longitude in degrees
        radius_km: Optional search radius in km, used for KDTree optimization
        
    Returns:
        NumPy array of shape (n, m) with distances in kilometers
    """
    # For large datasets, use a hybrid approach
    if radius_km is not None and len(source_coords) * len(target_coords) > 1000000:
        # Convert to radians for consistency
        source_radians = convert_to_radians(source_coords)
        target_radians = convert_to_radians(target_coords)
        
        # Create a KDTree for faster approximate neighborhood searches
        # This works in Euclidean space, so it's an approximation that works
        # reasonably well for smaller regions
        tree = cKDTree(target_radians)
        
        # For each source point, find nearby target points within a radius
        # We'll use a larger radius for the initial filter, then refine with exact calculation
        # Convert radius from km to radians: radius_km / Earth radius = radius_rad
        search_radius_rad = (radius_km * 1.2) / 6371.0  # Add 20% margin
        
        # Create empty distance matrix
        distances = np.full((len(source_coords), len(target_coords)), np.inf)
        
        # For each source point, find nearby target points and calculate exact distances
        for i, source_point in enumerate(source_radians):
            nearby_indices = tree.query_ball_point(source_point, search_radius_rad)
            if len(nearby_indices) > 0:
                nearby_targets = target_radians[nearby_indices]
                for j, idx in enumerate(nearby_indices):
                    distances[i, idx] = fast_haversine(
                        source_point[0], source_point[1], 
                        nearby_targets[j, 0], nearby_targets[j, 1]
                    )
        
        return distances
    else:
        # For smaller datasets, use scikit-learn's vectorized implementation
        source_radians = convert_to_radians(source_coords)
        target_radians = convert_to_radians(target_coords)
        distances = haversine_distances(source_radians, target_radians) * 6371.0
        return distances
