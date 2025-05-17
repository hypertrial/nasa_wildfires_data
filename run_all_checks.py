#!/usr/bin/env python3
"""
Wildfire Data Validation Suite - Runner
======================================

This script provides a convenient way to run all or selected validation checks
for the wildfire analysis data.

Usage:
    python run_all_checks.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--checks CHECK1,CHECK2,...]

Available checks:
    firms - FIRMS data integrity checks
    meteo_current - Current meteorological data validation
    meteo_forecast - Forecast meteorological data validation
    grid - Grid coverage validation
    weather - Weather variable range and outlier detection
    all - Run all checks (default)
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure we can import from the checks directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from checks.verify_firms_data import verify_firms_data
from checks.verify_meteo_data import verify_meteo_data
from checks.validate_grid_coverage import validate_grid_coverage
from checks.validate_weather_variables import validate_weather_variables

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run validation checks for wildfire analysis data.')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save validation results')
    parser.add_argument('--checks', type=str, default='all', 
                      help='Comma-separated list of checks to run (firms,meteo_current,meteo_forecast,grid,weather,all)')
    parser.add_argument('--create-sample-data', action='store_true', help='Create sample data files if they do not exist')
    return parser.parse_args()

def create_sample_data(data_dir):
    """
    Create sample data files for testing validation checks.
    
    Parameters:
    -----------
    data_dir : str
        Directory where sample files will be created
    """
    import pandas as pd
    import numpy as np
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample FIRMS data
    firms_file = os.path.join(data_dir, "firms_fire_data.csv")
    if not os.path.exists(firms_file):
        print(f"Creating sample FIRMS data: {firms_file}")
        # Create a grid of locations
        lats = np.linspace(35, 65, 10)
        lons = np.linspace(-125, -105, 10)
        
        data = []
        for lat in lats:
            for lon in lons:
                data.append({
                    'latitude': lat,
                    'longitude': lon,
                    'acq_date': '2023-06-15',
                    'acq_time': 1200,
                    'confidence': 80,
                    'frp': 15.5,
                    'brightness': 320.0
                })
        
        firms_df = pd.DataFrame(data)
        firms_df.to_csv(firms_file, index=False)
    
    # Sample weather data
    weather_file = os.path.join(data_dir, "current_weather.csv")
    if not os.path.exists(weather_file):
        print(f"Creating sample weather data: {weather_file}")
        # Create a grid of locations
        lats = np.linspace(30, 70, 10)
        lons = np.linspace(-130, -100, 10)
        
        data = []
        for lat in lats:
            for lon in lons:
                data.append({
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': '2023-06-15 12:00:00',
                    'temperature': 25.0,
                    'humidity': 40.0,
                    'wind_speed': 15.0,
                    'pressure': 1013.0,
                    'precipitation': 0.0
                })
        
        weather_df = pd.DataFrame(data)
        weather_df.to_csv(weather_file, index=False)
    
    # Sample forecast data
    forecast_file = os.path.join(data_dir, "forecast_weather.csv")
    if not os.path.exists(forecast_file):
        print(f"Creating sample forecast data: {forecast_file}")
        # Create a grid of locations
        lats = np.linspace(30, 70, 10)
        lons = np.linspace(-130, -100, 10)
        
        data = []
        for lat in lats:
            for lon in lons:
                data.append({
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': '2023-06-15 12:00:00',
                    'forecast_time': '2023-06-17 12:00:00',
                    'temperature': 27.0,
                    'humidity': 35.0,
                    'wind_speed': 20.0,
                    'pressure': 1010.0,
                    'precipitation': 2.0
                })
        
        forecast_df = pd.DataFrame(data)
        forecast_df.to_csv(forecast_file, index=False)
    
    # Sample grid cells
    grid_file = os.path.join(data_dir, "grid_cells.csv")
    if not os.path.exists(grid_file):
        print(f"Creating sample grid data: {grid_file}")
        # Create a grid of cells
        lats = []
        lons = []
        
        # Latitude spacing (constant)
        lat_step = 0.09  # Approximately 10km
        lat_bins = np.arange(30, 70 + lat_step, lat_step)
        
        for lat_idx in range(len(lat_bins) - 1):
            center_lat = (lat_bins[lat_idx] + lat_bins[lat_idx+1]) / 2
            # Longitude spacing (varies with latitude)
            lon_step = 0.09 / np.cos(np.radians(center_lat))
            lon_bins = np.arange(-130, -100 + lon_step, lon_step)
            
            for lon_idx in range(len(lon_bins) - 1):
                center_lon = (lon_bins[lon_idx] + lon_bins[lon_idx+1]) / 2
                lats.append(center_lat)
                lons.append(center_lon)
        
        grid_df = pd.DataFrame({'latitude': lats, 'longitude': lons})
        grid_df.to_csv(grid_file, index=False)
    
    print(f"Sample data files created in {data_dir}")

def run_checks(data_dir, output_dir, checks='all', create_samples=False):
    """
    Run the specified validation checks.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing data files
    output_dir : str
        Path to the directory where validation results will be saved
    checks : str
        Comma-separated list of checks to run
    create_samples : bool
        Whether to create sample data files if they don't exist
        
    Returns:
    --------
    dict
        Dictionary with check names as keys and validation results as values
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data if requested
    if create_samples:
        create_sample_data(data_dir)
    
    # Setup logging to a file
    log_file = os.path.join(output_dir, f"validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_handle = open(log_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = log_handle
    
    print(f"Starting validation suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Checks to run: {checks}")
    print("=" * 80)
    
    # Parse the checks to run
    check_list = checks.lower().split(',')
    if 'all' in check_list:
        run_all = True
    else:
        run_all = False
    
    validation_results = {}
    all_validations_passed = True
    at_least_one_check_run = False
    
    # 1. Validate FIRMS data
    if run_all or 'firms' in check_list:
        print("\n\n" + "=" * 40)
        print("RUNNING FIRMS DATA VALIDATION")
        print("=" * 40)
        firms_file = os.path.join(data_dir, "firms_fire_data.csv")
        if os.path.exists(firms_file):
            try:
                firms_valid = verify_firms_data(firms_file)
                validation_results['firms_data'] = firms_valid
                if not firms_valid:
                    all_validations_passed = False
                at_least_one_check_run = True
            except Exception as e:
                print(f"ERROR: Failed to validate FIRMS data: {str(e)}")
                validation_results['firms_data'] = False
                all_validations_passed = False
                at_least_one_check_run = True
        else:
            print(f"WARNING: FIRMS data file not found at {firms_file}")
            validation_results['firms_data'] = None
    
    # 2. Validate current meteorological data
    if run_all or 'meteo_current' in check_list:
        print("\n\n" + "=" * 40)
        print("RUNNING CURRENT METEOROLOGICAL DATA VALIDATION")
        print("=" * 40)
        current_meteo_file = os.path.join(data_dir, "current_weather.csv")
        if os.path.exists(current_meteo_file):
            try:
                current_meteo_valid = verify_meteo_data(current_meteo_file, data_type='current')
                validation_results['current_meteo'] = current_meteo_valid
                if not current_meteo_valid:
                    all_validations_passed = False
                at_least_one_check_run = True
            except Exception as e:
                print(f"ERROR: Failed to validate current meteorological data: {str(e)}")
                validation_results['current_meteo'] = False
                all_validations_passed = False
                at_least_one_check_run = True
        else:
            print(f"WARNING: Current meteorological data file not found at {current_meteo_file}")
            validation_results['current_meteo'] = None
    
    # 3. Validate forecast meteorological data
    if run_all or 'meteo_forecast' in check_list:
        print("\n\n" + "=" * 40)
        print("RUNNING FORECAST METEOROLOGICAL DATA VALIDATION")
        print("=" * 40)
        forecast_meteo_file = os.path.join(data_dir, "forecast_weather.csv")
        if os.path.exists(forecast_meteo_file):
            try:
                forecast_meteo_valid = verify_meteo_data(forecast_meteo_file, data_type='forecast')
                validation_results['forecast_meteo'] = forecast_meteo_valid
                if not forecast_meteo_valid:
                    all_validations_passed = False
                at_least_one_check_run = True
            except Exception as e:
                print(f"ERROR: Failed to validate forecast meteorological data: {str(e)}")
                validation_results['forecast_meteo'] = False
                all_validations_passed = False
                at_least_one_check_run = True
        else:
            print(f"WARNING: Forecast meteorological data file not found at {forecast_meteo_file}")
            validation_results['forecast_meteo'] = None
    
    # 4. Validate grid coverage
    if run_all or 'grid' in check_list:
        print("\n\n" + "=" * 40)
        print("RUNNING GRID COVERAGE VALIDATION")
        print("=" * 40)
        grid_file = os.path.join(data_dir, "grid_cells.csv")
        try:
            # For grid validation, we can generate a grid if no file exists
            grid_valid = validate_grid_coverage(
                grid_file=grid_file if os.path.exists(grid_file) else None,
                plot_grid=True,
                output_dir=output_dir
            )
            validation_results['grid_coverage'] = grid_valid
            if not grid_valid:
                all_validations_passed = False
            at_least_one_check_run = True
        except Exception as e:
            print(f"ERROR: Failed to validate grid coverage: {str(e)}")
            validation_results['grid_coverage'] = False
            all_validations_passed = False
            at_least_one_check_run = True
    
    # 5. Validate weather variables and check for outliers
    if run_all or 'weather' in check_list:
        print("\n\n" + "=" * 40)
        print("RUNNING WEATHER VARIABLE VALIDATION AND OUTLIER DETECTION")
        print("=" * 40)
        
        # Check current weather variables if file exists
        current_meteo_file = os.path.join(data_dir, "current_weather.csv")
        if os.path.exists(current_meteo_file):
            try:
                weather_vars_valid = validate_weather_variables(
                    current_meteo_file,
                    data_type='current',
                    output_dir=output_dir,
                    generate_plots=True
                )
                validation_results['weather_variables'] = weather_vars_valid
                if not weather_vars_valid:
                    all_validations_passed = False
                at_least_one_check_run = True
            except Exception as e:
                print(f"ERROR: Failed to validate weather variables: {str(e)}")
                validation_results['weather_variables'] = False
                all_validations_passed = False
                at_least_one_check_run = True
        else:
            print(f"WARNING: Current weather data file not found at {current_meteo_file}")
            validation_results['weather_variables'] = None
    
    # Print validation summary
    print("\n\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    
    skipped_count = 0
    failed_count = 0
    passed_count = 0
    
    for check_name, result in validation_results.items():
        if result is None:
            status = "SKIPPED"
            skipped_count += 1
        elif result is True:
            status = "PASSED"
            passed_count += 1
        else:
            status = "FAILED"
            failed_count += 1
        print(f"{check_name.replace('_', ' ').title(): <30}: {status}")
    
    validation_status = "PASSED" if all_validations_passed or not at_least_one_check_run else "FAILED"
    
    print("\nChecks summary:")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(validation_results)}")
    
    print("\nOVERALL VALIDATION:", validation_status)
    if not at_least_one_check_run:
        print("WARNING: No checks were actually run. All specified checks were skipped.")
    
    print(f"Detailed validation log saved to: {log_file}")
    
    # Reset stdout
    sys.stdout = original_stdout
    log_handle.close()
    
    # Print final message to console
    print(f"Validation completed. Log saved to {log_file}")
    print("OVERALL VALIDATION:", validation_status)
    
    # Print summary of which checks were run
    print("\nChecks performed:")
    for check_name in validation_results.keys():
        if validation_results[check_name] is None:
            status = "SKIPPED"
        elif validation_results[check_name] is True:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  - {check_name.replace('_', ' ').title()}: {status}")
    
    if skipped_count == len(validation_results):
        print("\nWARNING: All checks were skipped. No data files found.")
        print("Use --create-sample-data to generate sample data files for testing.")
    
    return validation_results, all_validations_passed, at_least_one_check_run

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Run the checks
    _, success, any_run = run_checks(args.data_dir, args.output_dir, args.checks, args.create_sample_data)
    
    # If no checks were run, offer to create sample data
    if not any_run and not args.create_sample_data:
        print("\nNo checks were run because data files are missing.")
        response = input("Would you like to create sample data files for testing? (y/n): ")
        if response.lower() in ['y', 'yes']:
            create_sample_data(args.data_dir)
            print("\nRunning checks with newly created sample data...")
            _, success, any_run = run_checks(args.data_dir, args.output_dir, args.checks, False)
    
    if not any_run:
        print("\nNo validation checks were executed. Please check your data files.")
        return 1
    
    if not success:
        print("\nWARNING: Some validation checks failed. Review the validation log before proceeding.")
        return 1
    
    print("\nAll performed validation checks passed. The data is ready for analysis.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 