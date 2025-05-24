"""
Grid Cell Verification Utility
==============================

This script verifies the variable-spaced grid used in the wildfire analysis by:
1. Calculating actual cell dimensions at different latitudes
2. Checking the accuracy of the variable longitude spacing approach
3. Computing and displaying grid statistics across the entire spatial domain

The script helps ensure that our grid cells have approximately equal areas despite
the convergence of longitude lines at higher latitudes, which is critical for
accurate spatial analysis of wildfire risk.
"""

import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# Constants for geographic calculations
R = 6371.0  # Earth radius in kilometers
KM_PER_LAT_DEGREE = 111.0  # Approximate conversion at mid-latitudes
grid_size_km = 10  # Target grid cell size in kilometers
min_lat, max_lat = 30, 70  # North latitude boundaries
min_lon, max_lon = -130, -100  # West longitude boundaries (negative values)

# Calculate latitude step size (constant for all latitudes)
lat_grid_size = grid_size_km / KM_PER_LAT_DEGREE
lat_bins = np.arange(min_lat, max_lat + lat_grid_size, lat_grid_size)

print("Verifying variable-spaced grid with latitude-dependent longitude steps")
print(f"Target cell size: {grid_size_km} km x {grid_size_km} km")
print("-" * 60)

# Test cell dimensions at different latitudes to verify equal-area approach
latitudes = [30, 40, 50, 60, 70]
print("Cell dimensions at different latitudes:")

for lat in latitudes:
    # Calculate longitude step size specific to this latitude
    # Higher latitudes require larger longitude steps to maintain equal cell width in km
    lon_grid_size = grid_size_km / (KM_PER_LAT_DEGREE * np.cos(np.radians(lat)))

    # Calculate north-south distance (should be constant across all latitudes)
    p1 = np.array([[np.radians(lat), np.radians(-120)]])
    p2 = np.array([[np.radians(lat + lat_grid_size), np.radians(-120)]])
    ns_dist = haversine_distances(p1, p2)[0][0] * R

    # Calculate east-west distance using latitude-specific longitude step
    p3 = np.array([[np.radians(lat), np.radians(-120)]])
    p4 = np.array([[np.radians(lat), np.radians(-120 + lon_grid_size)]])
    ew_dist = haversine_distances(p3, p4)[0][0] * R

    # Calculate cell area and deviation from target
    cell_area = ns_dist * ew_dist

    print(f"At latitude {lat}°N:")
    print(f"  Longitude step size: {lon_grid_size:.6f}° (varies with latitude)")
    print(
        f"  Cell dimensions: {ns_dist:.2f} km (N-S) × {ew_dist:.2f} km (E-W) = {cell_area:.2f} km²"
    )
    print(
        f"  Deviation from target: N-S {abs(ns_dist - grid_size_km) / grid_size_km * 100:.1f}%, E-W {abs(ew_dist - grid_size_km) / grid_size_km * 100:.1f}%"
    )
    print()

# Calculate total cells in grid and analyze grid properties
total_lat_cells = len(lat_bins) - 1
total_lon_cells_by_lat = []

# Calculate how longitude cells vary across different latitude bands
for lat_idx in range(len(lat_bins) - 1):
    center_lat = (lat_bins[lat_idx] + lat_bins[lat_idx + 1]) / 2
    lon_grid_size = grid_size_km / (KM_PER_LAT_DEGREE * np.cos(np.radians(center_lat)))
    lon_band_bins = np.arange(min_lon, max_lon + lon_grid_size, lon_grid_size)
    total_lon_cells_by_lat.append(len(lon_band_bins) - 1)

total_cells = sum(total_lon_cells_by_lat)
avg_lon_cells = sum(total_lon_cells_by_lat) / len(total_lon_cells_by_lat)

# Output grid statistics
print("Grid Statistics:")
print(f"  Latitude bands: {total_lat_cells}")
print(
    f"  Longitude cells per latitude band: varies from {min(total_lon_cells_by_lat)} to {max(total_lon_cells_by_lat)}"
)
print(f"  Average longitude cells per latitude band: {avg_lon_cells:.1f}")
print(
    f"  Total grid cells: {total_cells} (compared to {total_lat_cells * 214} with fixed longitude spacing)"
)

# Calculate the geographic coverage and theoretical ideal grid size
print("\nGeographic Coverage:")
ns_distance = (max_lat - min_lat) * KM_PER_LAT_DEGREE
ew_distance_avg = (
    (max_lon - min_lon)
    * KM_PER_LAT_DEGREE
    * np.cos(np.radians((min_lat + max_lat) / 2))
)
area = ns_distance * ew_distance_avg
print(f"  North-South distance: {ns_distance:.1f} km")
print(f"  East-West distance at average latitude: {ew_distance_avg:.1f} km")
print(f"  Approximate area: {area:.1f} km²")
print(
    f"  Theoretical perfect grid cell count: {int(area / (grid_size_km * grid_size_km))}"
)
