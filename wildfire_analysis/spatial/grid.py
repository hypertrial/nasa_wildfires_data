"""
Spatial Grid Module
==================

Functions for creating a grid covering a geographic area.
Handles the creation of approximately equal-area grid cells despite Earth's curvature.
"""

import numpy as np


def create_grid_cells(
    min_lat=30, max_lat=70, min_lon=-130, max_lon=-100, grid_size_km=10
):
    """
    Create a grid of cells covering the specified geographic area.

    The grid uses variable-width longitude bins to create approximately equal-area
    cells despite the convergence of longitude lines at higher latitudes.

    Args:
        min_lat, max_lat: Minimum and maximum latitude boundaries
        min_lon, max_lon: Minimum and maximum longitude boundaries
        grid_size_km: Target size of each grid cell in kilometers

    Returns:
        tuple: (lat_bins, lon_bins_by_lat, cell_centers)
            - lat_bins: Array of latitude bin edges
            - lon_bins_by_lat: List of arrays of longitude bin edges for each latitude band
            - cell_centers: List of (center_lat, center_lon, grid_id) tuples for all cells
    """
    # Constants for converting lat/lon to approximate distances
    # 1 degree of latitude is approximately 111 km (varies slightly with latitude)
    KM_PER_LAT_DEGREE = 111.0

    # Convert grid size from kilometers to degrees for latitude
    # This is constant across all latitudes
    lat_grid_size = grid_size_km / KM_PER_LAT_DEGREE

    # Create latitude bins with equal spacing
    lat_bins = np.arange(min_lat, max_lat + lat_grid_size, lat_grid_size)

    # Create latitude-dependent longitude spacing to ensure equal-area grid cells
    lon_bins_by_lat = []
    cell_centers = []

    # For each latitude band, calculate appropriate longitude spacing
    for lat_idx in range(len(lat_bins) - 1):
        # Calculate center latitude of this band
        center_lat = (lat_bins[lat_idx] + lat_bins[lat_idx + 1]) / 2

        # Calculate longitude step size specific to this latitude
        # Adjusts for the convergence of longitude lines at higher latitudes
        lon_grid_size = grid_size_km / (
            KM_PER_LAT_DEGREE * np.cos(np.radians(center_lat))
        )

        # Create longitude bins for this latitude band
        lon_band_bins = np.arange(min_lon, max_lon + lon_grid_size, lon_grid_size)
        lon_bins_by_lat.append(lon_band_bins)

        # Create cell centers for each cell in this latitude band
        for lon_idx in range(len(lon_band_bins) - 1):
            center_lon = (lon_band_bins[lon_idx] + lon_band_bins[lon_idx + 1]) / 2
            grid_id = f"{center_lat:.2f}_{center_lon:.2f}"
            cell_centers.append((center_lat, center_lon, grid_id))

    return lat_bins, lon_bins_by_lat, cell_centers


def calculate_grid_stats(lat_bins, lon_bins_by_lat):
    """
    Calculate statistics about the grid configuration.

    Args:
        lat_bins: Array of latitude bin edges
        lon_bins_by_lat: List of arrays of longitude bin edges for each latitude band

    Returns:
        dict: Statistics about the grid including counts and dimensions
    """
    stats = {
        "lat_bands": len(lat_bins) - 1,
        "min_lat": lat_bins[0],
        "max_lat": lat_bins[-1],
        "lon_bands_per_lat": [len(lon_bins) - 1 for lon_bins in lon_bins_by_lat],
        "min_lon": lon_bins_by_lat[0][0],
        "max_lon": lon_bins_by_lat[0][-1],
        "total_cells": sum(len(lon_bins) - 1 for lon_bins in lon_bins_by_lat),
    }

    return stats


def create_grid_edges(lat_bins, lon_bins_by_lat, cell_centers):
    """
    Create edges between neighboring grid cells.

    Args:
        lat_bins: Array of latitude bin edges
        lon_bins_by_lat: List of arrays of longitude bin edges for each latitude band
        cell_centers: List of (center_lat, center_lon, grid_id) tuples for all cells

    Returns:
        list: List of (grid_id1, grid_id2) tuples representing edges between cells
    """
    # Create dictionary mapping (lat_idx, lon_idx) to grid_id for fast lookup
    cell_lookup = {}
    lon_centers_by_lat = []

    # Populate the lookup dictionary and calculate longitude centers by latitude
    for lat_idx in range(len(lat_bins) - 1):
        lon_band_bins = lon_bins_by_lat[lat_idx]
        lon_centers = [
            (lon_band_bins[i] + lon_band_bins[i + 1]) / 2
            for i in range(len(lon_band_bins) - 1)
        ]
        lon_centers_by_lat.append(lon_centers)

        center_lat = (lat_bins[lat_idx] + lat_bins[lat_idx + 1]) / 2

        for lon_idx, center_lon in enumerate(lon_centers):
            grid_id = f"{center_lat:.2f}_{center_lon:.2f}"
            cell_lookup[(lat_idx, lon_idx)] = grid_id

    # Create edges between neighboring cells
    edge_list = []

    for lat_idx in range(len(lat_bins) - 1):
        for lon_idx in range(len(lon_centers_by_lat[lat_idx])):
            current_id = cell_lookup.get((lat_idx, lon_idx))
            if current_id is None:
                continue

            # Connect to right neighbor (east) in same latitude band
            if lon_idx < len(lon_centers_by_lat[lat_idx]) - 1:
                right_id = cell_lookup.get((lat_idx, lon_idx + 1))
                if right_id is not None:
                    edge_list.append((current_id, right_id))

            # Connect to bottom neighbor (south) if not in the last latitude band
            if lat_idx < len(lat_bins) - 2:
                # Find the closest longitude in the next latitude band
                if lat_idx + 1 < len(lon_centers_by_lat):
                    next_lon_centers = lon_centers_by_lat[lat_idx + 1]
                    current_lon = lon_centers_by_lat[lat_idx][lon_idx]

                    # Find the closest longitude center in the next band
                    closest_lon_idx = np.argmin(
                        np.abs(np.array(next_lon_centers) - current_lon)
                    )

                    # Connect to bottom neighbor
                    bottom_id = cell_lookup.get((lat_idx + 1, closest_lon_idx))
                    if bottom_id is not None:
                        edge_list.append((current_id, bottom_id))

                    # Connect to diagonal neighbor if available (southeast)
                    if closest_lon_idx < len(next_lon_centers) - 1:
                        diag_right_id = cell_lookup.get(
                            (lat_idx + 1, closest_lon_idx + 1)
                        )
                        if diag_right_id is not None:
                            edge_list.append((current_id, diag_right_id))

                    # Connect to diagonal neighbor if available (southwest)
                    if closest_lon_idx > 0:
                        diag_left_id = cell_lookup.get(
                            (lat_idx + 1, closest_lon_idx - 1)
                        )
                        if diag_left_id is not None:
                            edge_list.append((current_id, diag_left_id))

    return edge_list
