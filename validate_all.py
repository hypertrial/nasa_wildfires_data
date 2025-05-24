#!/usr/bin/env python3
"""
Wildfire Data Validation Suite
=============================

This script runs all data validation checks to ensure data quality and consistency
before running the wildfire risk analysis. It includes:
1. FIRMS data integrity checks
2. Meteorological data validation
3. Grid coverage verification
4. Weather variable range and outlier detection

Usage:
    python validate_all.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import sys
from datetime import datetime

# Import validation modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from checks.validate_grid_coverage import validate_grid_coverage
from checks.validate_weather_variables import validate_weather_variables
from checks.verify_firms_data import verify_firms_data
from checks.verify_meteo_data import verify_meteo_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all validation checks for wildfire analysis data."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save validation results",
    )
    return parser.parse_args()


def run_all_validations(data_dir, output_dir):
    """Run all validation checks and return overall validation status."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging to a file
    log_file = os.path.join(
        output_dir, f"validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    sys.stdout = open(log_file, "w")

    print(
        f"Starting validation suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    all_validations_passed = True
    validation_results = {}

    # 1. Validate FIRMS data
    print("\n\n" + "=" * 40)
    print("RUNNING FIRMS DATA VALIDATION")
    print("=" * 40)
    firms_file = os.path.join(data_dir, "firms_fire_data.csv")
    if os.path.exists(firms_file):
        firms_valid = verify_firms_data(firms_file)
        validation_results["firms_data"] = firms_valid
        if not firms_valid:
            all_validations_passed = False
    else:
        print(f"WARNING: FIRMS data file not found at {firms_file}")
        validation_results["firms_data"] = None

    # 2. Validate current meteorological data
    print("\n\n" + "=" * 40)
    print("RUNNING CURRENT METEOROLOGICAL DATA VALIDATION")
    print("=" * 40)
    current_meteo_file = os.path.join(data_dir, "current_weather.csv")
    if os.path.exists(current_meteo_file):
        current_meteo_valid = verify_meteo_data(current_meteo_file, data_type="current")
        validation_results["current_meteo"] = current_meteo_valid
        if not current_meteo_valid:
            all_validations_passed = False
    else:
        print(
            f"WARNING: Current meteorological data file not found at {current_meteo_file}"
        )
        validation_results["current_meteo"] = None

    # 3. Validate forecast meteorological data
    print("\n\n" + "=" * 40)
    print("RUNNING FORECAST METEOROLOGICAL DATA VALIDATION")
    print("=" * 40)
    forecast_meteo_file = os.path.join(data_dir, "forecast_weather.csv")
    if os.path.exists(forecast_meteo_file):
        forecast_meteo_valid = verify_meteo_data(
            forecast_meteo_file, data_type="forecast"
        )
        validation_results["forecast_meteo"] = forecast_meteo_valid
        if not forecast_meteo_valid:
            all_validations_passed = False
    else:
        print(
            f"WARNING: Forecast meteorological data file not found at {forecast_meteo_file}"
        )
        validation_results["forecast_meteo"] = None

    # 4. Validate grid coverage
    print("\n\n" + "=" * 40)
    print("RUNNING GRID COVERAGE VALIDATION")
    print("=" * 40)
    grid_file = os.path.join(data_dir, "grid_cells.csv")
    grid_valid = validate_grid_coverage(
        grid_file=grid_file if os.path.exists(grid_file) else None,
        plot_grid=True,
        output_dir=output_dir,
    )
    validation_results["grid_coverage"] = grid_valid
    if not grid_valid:
        all_validations_passed = False

    # 5. Validate weather variables and check for outliers
    print("\n\n" + "=" * 40)
    print("RUNNING WEATHER VARIABLE VALIDATION AND OUTLIER DETECTION")
    print("=" * 40)

    # Check current weather variables if file exists
    if os.path.exists(current_meteo_file):
        weather_vars_valid = validate_weather_variables(
            current_meteo_file,
            data_type="current",
            output_dir=output_dir,
            generate_plots=True,
        )
        validation_results["weather_variables"] = weather_vars_valid
        if not weather_vars_valid:
            all_validations_passed = False

    # Print validation summary
    print("\n\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)

    for check_name, result in validation_results.items():
        status = (
            "PASSED" if result is True else "FAILED" if result is False else "SKIPPED"
        )
        print(f"{check_name.replace('_', ' ').title(): <30}: {status}")

    print("\nOVERALL VALIDATION:", "PASSED" if all_validations_passed else "FAILED")
    print(f"Detailed validation log saved to: {log_file}")

    # Reset stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # Print final message to console
    print(f"Validation completed. Log saved to {log_file}")
    print("OVERALL VALIDATION:", "PASSED" if all_validations_passed else "FAILED")

    return all_validations_passed


def main():
    """Main entry point."""
    args = parse_arguments()
    success = run_all_validations(args.data_dir, args.output_dir)

    if not success:
        print(
            "\nWARNING: Some validation checks failed. Review the validation log before proceeding."
        )
        return 1

    print("\nAll validation checks passed. The data is ready for analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
