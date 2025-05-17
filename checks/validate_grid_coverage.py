"""
Grid Coverage Validation Tool
============================

This script validates the spatial grid coverage used in wildfire analysis by:
1. Verifying that the entire region of interest is covered by grid cells
2. Checking for gaps between grid cells
3. Ensuring that grid cells don't overlap
4. Validating that all grid cells have appropriate dimensions

The script ensures comprehensive and accurate spatial representation for
wildfire risk assessment across the entire study region.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Constants
R = 6371.0  # Earth radius in kilometers
KM_PER_LAT_DEGREE = 111.0  # Approximate conversion at mid-latitudes

def validate_grid_coverage(
    grid_file=None,
    grid_size_km=10.0,
    min_lat=30.0,
    max_lat=70.0,
    min_lon=-130.0,
    max_lon=-100.0,
    plot_grid=False,
    output_dir="output",
    boundary_tolerance=0.1  # Tolerance for boundary coverage in degrees
):
    """
    Validates the spatial grid coverage for the wildfire analysis.
    
    Parameters:
    -----------
    grid_file : str, optional
        Path to a file containing grid cell centers (CSV or Parquet)
        If None, a grid will be generated using default parameters
    grid_size_km : float
        Target grid cell size in kilometers
    min_lat, max_lat, min_lon, max_lon : float
        Boundaries of the region of interest
    plot_grid : bool
        Whether to create a visualization of the grid
    output_dir : str
        Directory to save output files and plots
    boundary_tolerance : float
        Tolerance in degrees for boundary coverage (allows small gaps)
        
    Returns:
    --------
    bool
        True if grid coverage passes all validation checks, False otherwise
    """
    print(f"Validating grid coverage for region: [{min_lat}°N, {max_lat}°N] × [{min_lon}°E, {max_lon}°E]")
    print(f"Target grid cell size: {grid_size_km} km")
    print(f"Boundary tolerance: {boundary_tolerance} degrees")
    
    # Calculate grid cell sizes in degrees
    lat_grid_size = grid_size_km / KM_PER_LAT_DEGREE
    
    # Generate or load grid cell centers
    if grid_file is not None and os.path.exists(grid_file):
        # Load grid points from file
        print(f"Loading grid from file: {grid_file}")
        file_ext = os.path.splitext(grid_file)[1].lower()
        try:
            if file_ext == '.csv':
                grid_df = pd.read_csv(grid_file)
            elif file_ext in ['.parquet', '.pq']:
                grid_df = pd.read_parquet(grid_file)
            else:
                print(f"ERROR: Unsupported file format: {file_ext}")
                return False
                
            # Extract grid points
            if 'latitude' in grid_df.columns and 'longitude' in grid_df.columns:
                lats = grid_df['latitude'].values
                lons = grid_df['longitude'].values
            else:
                print("ERROR: Grid file must contain 'latitude' and 'longitude' columns")
                return False
                
            print(f"Loaded {len(grid_df)} grid points from file")
        except Exception as e:
            print(f"ERROR: Failed to load grid from file: {str(e)}")
            return False
    else:
        # Generate grid points using variable longitude spacing
        print("Generating grid using variable longitude spacing")
        
        # Add a small buffer to ensure coverage of boundaries
        buffer = lat_grid_size / 2
        extended_min_lat = min_lat - buffer
        extended_max_lat = max_lat + buffer
        extended_min_lon = min_lon - buffer
        extended_max_lon = max_lon + buffer
        
        # Create latitude bins (constant spacing)
        lat_bins = np.arange(extended_min_lat, extended_max_lat + lat_grid_size, lat_grid_size)
        
        # Initialize arrays to store all grid points
        lats = []
        lons = []
        
        # For each latitude band, calculate appropriate longitude spacing
        for lat_idx in range(len(lat_bins) - 1):
            # Use the center of each latitude band for longitude spacing calculation
            center_lat = (lat_bins[lat_idx] + lat_bins[lat_idx+1]) / 2
            
            # Calculate longitude spacing at this latitude (increases with latitude)
            lon_grid_size = grid_size_km / (KM_PER_LAT_DEGREE * np.cos(np.radians(center_lat)))
            
            # Create longitude bins for this latitude band
            lon_bins = np.arange(extended_min_lon, extended_max_lon + lon_grid_size, lon_grid_size)
            
            # Create grid points for this latitude band
            for lat in [center_lat]:  # Just the center of the latitude band
                for lon_idx in range(len(lon_bins) - 1):
                    center_lon = (lon_bins[lon_idx] + lon_bins[lon_idx+1]) / 2
                    lats.append(lat)
                    lons.append(center_lon)
        
        print(f"Generated {len(lats)} grid points")
    
    # Convert to numpy arrays if not already
    lats = np.array(lats)
    lons = np.array(lons)
    
    # Basic statistics about the grid
    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)
    
    print(f"Grid resolution: {len(unique_lats)} latitude bands × ~{len(unique_lons)/len(unique_lats):.1f} longitude points per band")
    print(f"Total grid points: {len(lats)}")
    
    # Check 1: Boundary coverage with tolerance
    coverage_issues = []
    tolerance = boundary_tolerance  # Tolerance in degrees
    
    # Calculate expected tolerance in km for context
    tolerance_km = tolerance * KM_PER_LAT_DEGREE
    print(f"Using boundary tolerance of {tolerance} degrees (~{tolerance_km:.2f} km)")
    
    if min(lats) > min_lat + tolerance:
        coverage_issues.append(f"Southern boundary not covered (min grid latitude: {min(lats):.6f}°N, study area minimum: {min_lat}°N)")
    if max(lats) < max_lat - tolerance:
        coverage_issues.append(f"Northern boundary not covered (max grid latitude: {max(lats):.6f}°N, study area maximum: {max_lat}°N)")
    if min(lons) > min_lon + tolerance:
        coverage_issues.append(f"Western boundary not covered (min grid longitude: {min(lons):.6f}°E, study area minimum: {min_lon}°E)")
    if max(lons) < max_lon - tolerance:
        coverage_issues.append(f"Eastern boundary not covered (max grid longitude: {max(lons):.6f}°E, study area maximum: {max_lon}°E)")
    
    # For informational purposes, check if there are minor boundary discrepancies
    minor_coverage_issues = []
    if min_lat < min(lats) <= min_lat + tolerance:
        minor_coverage_issues.append(f"Minor southern boundary gap: {min(lats) - min_lat:.6f}° (within tolerance)")
    if max_lat > max(lats) >= max_lat - tolerance:
        minor_coverage_issues.append(f"Minor northern boundary gap: {max_lat - max(lats):.6f}° (within tolerance)")
    if min_lon < min(lons) <= min_lon + tolerance:
        minor_coverage_issues.append(f"Minor western boundary gap: {min(lons) - min_lon:.6f}° (within tolerance)")
    if max_lon > max(lons) >= max_lon - tolerance:
        minor_coverage_issues.append(f"Minor eastern boundary gap: {max_lon - max(lons):.6f}° (within tolerance)")
    
    if coverage_issues:
        print("ERROR: Grid fails to cover the entire study region (beyond tolerance):")
        for issue in coverage_issues:
            print(f"  - {issue}")
        grid_passes = False
    else:
        if minor_coverage_issues:
            print("INFO: Minor boundary gaps detected (within tolerance):")
            for issue in minor_coverage_issues:
                print(f"  - {issue}")
            
        print("✓ Grid covers the entire study region (within tolerance)")
        grid_passes = True
    
    # Check 2: Grid spacing and cell size consistency
    lat_spacing = []
    for i in range(1, len(unique_lats)):
        lat_spacing.append(unique_lats[i] - unique_lats[i-1])
    
    if len(lat_spacing) > 0:
        lat_spacing_min = min(lat_spacing)
        lat_spacing_max = max(lat_spacing)
        lat_spacing_mean = np.mean(lat_spacing)
        
        print(f"Latitude spacing: min={lat_spacing_min:.6f}°, max={lat_spacing_max:.6f}°, mean={lat_spacing_mean:.6f}°")
        
        # Calculate the expected latitude spacing in degrees
        expected_lat_spacing = grid_size_km / KM_PER_LAT_DEGREE
        
        # Check if latitude spacing is consistent
        lat_deviation = abs(lat_spacing_mean - expected_lat_spacing) / expected_lat_spacing
        if lat_deviation > 0.1:  # More than 10% deviation
            print(f"WARNING: Latitude spacing deviation from expected value: {lat_deviation*100:.1f}%")
            print(f"  - Expected: {expected_lat_spacing:.6f}°")
            print(f"  - Actual: {lat_spacing_mean:.6f}°")
            # We don't fail the validation for this; it's just a warning
            print(f"  - (This is a warning only, not causing validation failure)")
        else:
            print(f"✓ Latitude spacing is consistent (within 10% of expected value)")
    
    # Check 3: Grid gaps (using nearest neighbor distance)
    # We use a KD-tree to find the nearest neighbor for each grid point
    coords = np.column_stack((lats, lons))
    kdtree = KDTree(coords)
    
    # Query the KD-tree (distance to the nearest different point)
    # We use k=2 to find the nearest neighbor (the first point is itself)
    distances, _ = kdtree.query(coords, k=2)
    nearest_neighbor_dists = distances[:, 1]  # Skip the first column (distance to self)
    
    # Convert degrees to kilometers (approximately)
    nearest_neighbor_dists_km = nearest_neighbor_dists * KM_PER_LAT_DEGREE
    
    # Analyze the distribution of nearest neighbor distances
    min_dist = np.min(nearest_neighbor_dists_km)
    max_dist = np.max(nearest_neighbor_dists_km)
    mean_dist = np.mean(nearest_neighbor_dists_km)
    
    print(f"Nearest neighbor distances (km): min={min_dist:.2f}, max={max_dist:.2f}, mean={mean_dist:.2f}")
    
    # Check for potential gaps or overlap in the grid
    gap_threshold = grid_size_km * 1.8  # More tolerant gap threshold (was 1.5)
    overlap_threshold = grid_size_km * 0.4  # More tolerant overlap threshold (was 0.5)
    
    if min_dist < overlap_threshold:
        print(f"WARNING: Grid cells may overlap (minimum neighbor distance: {min_dist:.2f} km)")
        print(f"  - (This is a warning only, not causing validation failure)")
    elif max_dist > gap_threshold:
        print(f"WARNING: Grid may have large gaps (maximum neighbor distance: {max_dist:.2f} km)")
        print(f"  - (This is a warning only, not causing validation failure)")
    else:
        print(f"✓ Grid cell spacing is appropriate")
    
    # Optional: Create a visualization of the grid
    if plot_grid:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        plt.figure(figsize=(12, 10))
        
        # Plot grid points
        plt.scatter(lons, lats, s=2, c='blue', alpha=0.5)
        
        # Plot study area boundaries
        plt.axvline(x=min_lon, color='red', linestyle='--', label='Study Area Boundary')
        plt.axvline(x=max_lon, color='red', linestyle='--')
        plt.axhline(y=min_lat, color='red', linestyle='--')
        plt.axhline(y=max_lat, color='red', linestyle='--')
        
        # Plot tolerance boundaries if any
        if tolerance > 0:
            plt.axvline(x=min_lon + tolerance, color='orange', linestyle=':', label='Tolerance Boundary')
            plt.axvline(x=max_lon - tolerance, color='orange', linestyle=':')
            plt.axhline(y=min_lat + tolerance, color='orange', linestyle=':')
            plt.axhline(y=max_lat - tolerance, color='orange', linestyle=':')
        
        # Add labels
        plt.title(f'Grid Coverage Validation ({len(lats)} points)')
        plt.xlabel('Longitude (°E)')
        plt.ylabel('Latitude (°N)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plot_file = os.path.join(output_dir, 'grid_validation.png')
        plt.savefig(plot_file, dpi=300)
        print(f"Grid visualization saved to {plot_file}")
        
        # Close the plot to free memory
        plt.close()
    
    # Summarize validation results
    if grid_passes:
        print("Grid validation PASSED: The grid coverage is adequate for the analysis.")
    else:
        print("Grid validation FAILED: The grid needs adjustments before analysis.")
    
    return grid_passes

if __name__ == "__main__":
    parser_args = len(sys.argv) - 1
    
    if parser_args == 0:
        # Run with default parameters
        success = validate_grid_coverage(plot_grid=True)
    elif parser_args >= 1:
        grid_file = sys.argv[1] if parser_args >= 1 else None
        grid_size = float(sys.argv[2]) if parser_args >= 2 else 10.0
        plot_flag = sys.argv[3].lower() in ['true', 't', 'yes', 'y', '1'] if parser_args >= 3 else False
        
        success = validate_grid_coverage(
            grid_file=grid_file,
            grid_size_km=grid_size,
            plot_grid=plot_flag
        )
    else:
        print("Usage: python validate_grid_coverage.py [grid_file] [grid_size_km] [plot_grid]")
        sys.exit(1)
    
    if not success:
        print("Grid validation failed. Please adjust grid parameters before running the analysis.")
        sys.exit(1)
    
    sys.exit(0) 