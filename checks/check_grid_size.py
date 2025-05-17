"""
Grid Cell Size Evaluation Tool
=============================

This script evaluates the impact of using fixed vs. latitude-dependent longitude
grid spacing in geospatial analysis. It demonstrates why we need variable longitude
spacing to create true equal-area grid cells across different latitudes.

The script:
1. Calculates grid cell sizes in degrees based on target size in kilometers
2. Tests actual cell dimensions at different latitudes when using:
   - Global average longitude spacing (distortion increases with latitude)
   - Latitude-specific longitude spacing (maintains equal areas)
3. Computes grid statistics for the geographic domain
4. Verifies the correctness of grid dimensioning

This analysis ensures accurate spatial representation for wildfire risk assessment
by accounting for Earth's curvature and longitude convergence at high latitudes.
"""
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# Constants for geographic calculations
R = 6371.0  # Earth radius in kilometers
KM_PER_LAT_DEGREE = 111.0  # Approximate conversion factor
grid_size_km = 10  # Target grid cell size in kilometers

# Calculate grid sizes in degrees
lat_grid_size = grid_size_km / KM_PER_LAT_DEGREE
avg_lat = (30 + 70) / 2  # Average latitude of our study area (50°N)
lon_grid_size = grid_size_km / (KM_PER_LAT_DEGREE * np.cos(np.radians(avg_lat)))

print(f"Grid cell size configuration: {grid_size_km} km x {grid_size_km} km")
print(f"Converting to degrees:")
print(f"  Latitude step size: {lat_grid_size:.6f} degrees (constant at all latitudes)")
print(f"  Longitude step size: {lon_grid_size:.6f} degrees (at average latitude {avg_lat}°N)")

print("\nChecking actual grid cell sizes at different latitudes:")
print("-----------------------------------------------------------")

# Test grid cell dimensions at different latitudes from south to north
for lat in [30, 40, 50, 60, 70]:
    # Calculate east-west distance using the global longitude grid size
    # This shows distortion when using fixed longitude spacing
    p1 = np.array([[np.radians(lat), np.radians(-120)]])
    p2 = np.array([[np.radians(lat), np.radians(-120 + lon_grid_size)]])
    ew_dist = haversine_distances(p1, p2)[0][0] * R

    # Calculate north-south distance (consistent at all latitudes)
    p3 = np.array([[np.radians(lat), np.radians(-120)]])
    p4 = np.array([[np.radians(lat + lat_grid_size), np.radians(-120)]])
    ns_dist = haversine_distances(p3, p4)[0][0] * R
    
    # Calculate east-west distance using a latitude-specific longitude step
    # This demonstrates the correct approach for equal-area cells
    local_lon_grid_size = grid_size_km / (KM_PER_LAT_DEGREE * np.cos(np.radians(lat)))
    p5 = np.array([[np.radians(lat), np.radians(-120)]])
    p6 = np.array([[np.radians(lat), np.radians(-120 + local_lon_grid_size)]])
    local_ew_dist = haversine_distances(p5, p6)[0][0] * R
    
    print(f"At latitude {lat}°N:")
    print(f"  N-S distance: {ns_dist:.2f} km (target: {grid_size_km} km)")
    print(f"  E-W distance using global step: {ew_dist:.2f} km (target: {grid_size_km} km)")
    print(f"  E-W distance using local step ({local_lon_grid_size:.6f}°): {local_ew_dist:.2f} km")
    print(f"  E-W distortion: {abs(ew_dist - grid_size_km)/grid_size_km*100:.1f}%")
    print()

# Calculate total grid cells in our bounding box using global longitude step
min_lat, max_lat = 30, 70  # North latitude boundaries
min_lon, max_lon = -130, -100  # West longitude boundaries (negative values)

lat_bins = np.arange(min_lat, max_lat + lat_grid_size, lat_grid_size)
lon_bins = np.arange(min_lon, max_lon + lon_grid_size, lon_grid_size)

print("Grid dimensions:")
print(f"  Latitude bins: {len(lat_bins)} (covering {min_lat}°N to {max_lat}°N)")
print(f"  Longitude bins: {len(lon_bins)} (covering {min_lon}°W to {max_lon}°W)")
print(f"  Total grid cells: {(len(lat_bins)-1) * (len(lon_bins)-1)}")
print(f"  Expected grid dimensions: {len(lat_bins)-1} x {len(lon_bins)-1}")

# Calculate actual geographic coverage and verify grid sizing
ns_distance = (max_lat - min_lat) * KM_PER_LAT_DEGREE
ew_distance = (max_lon - min_lon) * KM_PER_LAT_DEGREE * np.cos(np.radians(avg_lat))

print(f"\nActual geographic area covered:")
print(f"  North-South distance: {ns_distance:.1f} km")
print(f"  East-West distance (at {avg_lat}°N): {ew_distance:.1f} km")
print(f"  Area: approximately {ns_distance * ew_distance:.1f} km²")

print(f"\nIs the grid correctly sized? ")
print(f"  Expected cells with perfect 10km x 10km grid: {int(ns_distance/grid_size_km)} x {int(ew_distance/grid_size_km)} = {int(ns_distance/grid_size_km) * int(ew_distance/grid_size_km)}")
print(f"  Actual cells in our grid: {len(lat_bins)-1} x {len(lon_bins)-1} = {(len(lat_bins)-1) * (len(lon_bins)-1)}")

if (len(lat_bins)-1) == int(ns_distance/grid_size_km) and (len(lon_bins)-1) == int(ew_distance/grid_size_km):
    print("  ✓ Grid size is correct!")
else:
    print("  ✗ Grid size does not match expected dimensions")
    print("  Note: Minor differences are expected due to rounding and edge effects") 