"""
Weather Variable Validation Tool
==============================

This script analyzes meteorological data to ensure quality and consistency by:
1. Checking that variables are within expected ranges
2. Detecting outliers using statistical methods
3. Identifying suspicious data patterns or anomalies
4. Validating temporal consistency

The script helps ensure that weather data used in wildfire analysis is
accurate and reliable for risk assessment.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define expected ranges for meteorological variables
# Format: (min_value, max_value, units)
VARIABLE_RANGES = {
    "temperature": (-50.0, 50.0, "°C"),
    "humidity": (0.0, 100.0, "%"),
    "wind_speed": (0.0, 100.0, "km/h"),
    "precipitation": (0.0, 500.0, "mm"),
    "pressure": (900.0, 1100.0, "hPa"),
    "cloud_cover": (0.0, 100.0, "%"),
    "wind_direction": (0.0, 360.0, "°"),
    "dewpoint": (-50.0, 50.0, "°C"),
}

# Z-score threshold for outlier detection (standard deviations from mean)
OUTLIER_THRESHOLD = 3.0


def safe_zscore(values):
    """
    Calculate z-scores safely, handling cases with no variation.

    Parameters:
    -----------
    values : array-like
        The values to calculate z-scores for

    Returns:
    --------
    array
        Z-scores for the input values, or zeros if std is near zero
    """
    # Check if standard deviation is very small (near-identical values)
    std = np.std(values)
    if std < 1e-6:
        print(
            "  NOTE: Very little variation in data (std < 1e-6), skipping outlier detection"
        )
        return np.zeros_like(values)

    # Calculate mean and std safely with numpy
    mean = np.mean(values)
    # Calculate z-scores manually to avoid scipy warning
    z_scores = np.abs((values - mean) / std)
    return z_scores


def validate_weather_variables(
    file_path, data_type="current", output_dir="output", generate_plots=True
):
    """
    Validates weather variables and identifies outliers in meteorological data.

    Parameters:
    -----------
    file_path : str
        Path to the meteorological data file (CSV or Parquet)
    data_type : str
        Type of meteorological data ('current' or 'forecast')
    output_dir : str
        Directory to save output files and plots
    generate_plots : bool
        Whether to generate visualization plots

    Returns:
    --------
    bool
        True if data passes all validation checks, False otherwise
    """
    print(f"Validating {data_type} weather variables in: {file_path}")

    # Suppress warnings about precision loss
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="Precision loss occurred"
    )

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return False

    # Load data based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path)
        else:
            print(f"ERROR: Unsupported file format: {file_ext}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to load data: {str(e)}")
        return False

    # Check row count
    if len(df) == 0:
        print("ERROR: Weather data file is empty")
        return False

    print(f"Found {len(df)} weather records")

    # Ensure all required columns are present
    required_columns = ["latitude", "longitude", "timestamp"]

    if data_type == "forecast" and "forecast_time" not in df.columns:
        required_columns.append("forecast_time")

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        print(f"ERROR: Missing required columns: {', '.join(missing_required)}")
        return False

    # Process timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "forecast_time" in df.columns:
        df["forecast_time"] = pd.to_datetime(df["forecast_time"])

    # Find which weather variables are available in the data
    weather_vars = [var for var in VARIABLE_RANGES.keys() if var in df.columns]

    if not weather_vars:
        print("ERROR: No recognized weather variables found in the data")
        return False

    print(f"Found {len(weather_vars)} weather variables: {', '.join(weather_vars)}")

    # Create output directory if generating plots
    if generate_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Prepare a summary results dictionary
    results = {
        "out_of_range": [],
        "outliers": {},
        "total_outliers": 0,
        "overall_valid": True,
    }

    # Analyze each weather variable
    for var in weather_vars:
        print(f"\nAnalyzing variable: {var}")

        # Skip if the column is entirely null
        if df[var].isnull().all():
            print(f"  WARNING: Variable '{var}' contains only NULL values")
            results["overall_valid"] = False
            continue

        # Get non-null values for analysis
        values = df[var].dropna().values

        if len(values) == 0:
            print(f"  WARNING: No valid values found for '{var}'")
            results["overall_valid"] = False
            continue

        # 1. Check variable range
        min_val, max_val, units = VARIABLE_RANGES[var]
        actual_min = np.min(values)
        actual_max = np.max(values)

        print(f"  Range: {actual_min:.2f} to {actual_max:.2f} {units}")
        print(f"  Expected range: {min_val:.2f} to {max_val:.2f} {units}")

        # Check if values are within the expected range
        out_of_range = np.logical_or(values < min_val, values > max_val)
        out_of_range_count = np.sum(out_of_range)
        out_of_range_percent = (out_of_range_count / len(values)) * 100

        if out_of_range_count > 0:
            print(
                f"  WARNING: {out_of_range_count} values ({out_of_range_percent:.2f}%) are outside the expected range"
            )
            results["out_of_range"].append(
                (var, out_of_range_count, out_of_range_percent)
            )
            results["overall_valid"] = False
        else:
            print("  ✓ All values are within expected range")

        # 2. Detect outliers using z-score method (with safe implementation)
        # Check if all values are identical or nearly identical
        value_range = actual_max - actual_min
        if value_range < 1e-6:
            print(
                f"  NOTE: All values are identical or nearly identical ({actual_min})"
            )
            outlier_count = 0
            outlier_percent = 0
        else:
            # Use our safe z-score function
            z_scores = safe_zscore(values)
            outliers = z_scores > OUTLIER_THRESHOLD
            outlier_count = np.sum(outliers)
            outlier_percent = (outlier_count / len(values)) * 100

        results["outliers"][var] = outlier_count
        results["total_outliers"] += outlier_count

        if outlier_count > 0:
            print(
                f"  Outliers: {outlier_count} values ({outlier_percent:.2f}%) are statistical outliers"
            )
            if outlier_percent > 5.0:
                print("  WARNING: High percentage of outliers detected")
                results["overall_valid"] = False
        else:
            print("  ✓ No outliers detected")

        # 3. Basic statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)

        print(
            f"  Statistics: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f} {units}"
        )

        # 4. Generate visualization plots if requested
        if generate_plots:
            # Skip detailed plots if all values are identical
            if value_range < 1e-6:
                print("  NOTE: Skipping plots because all values are identical")
                continue

            plt.figure(figsize=(12, 8))

            # Create a figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Histogram with normal distribution overlay
            sns.histplot(values, kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {var}")
            ax1.set_xlabel(f"{var} ({units})")
            ax1.set_ylabel("Frequency")

            # Add lines for expected range
            ax1.axvline(
                x=min_val,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Min expected: {min_val}",
            )
            ax1.axvline(
                x=max_val,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Max expected: {max_val}",
            )

            # Add a line for outlier thresholds if std is not near zero
            if std_val > 1e-6:
                lower_threshold = mean_val - OUTLIER_THRESHOLD * std_val
                upper_threshold = mean_val + OUTLIER_THRESHOLD * std_val
                ax1.axvline(
                    x=lower_threshold,
                    color="g",
                    linestyle=":",
                    alpha=0.7,
                    label="Outlier threshold (lower)",
                )
                ax1.axvline(
                    x=upper_threshold,
                    color="g",
                    linestyle=":",
                    alpha=0.7,
                    label="Outlier threshold (upper)",
                )
            ax1.legend()

            # Plot 2: Box plot to visualize outliers
            ax2.boxplot(values, vert=False, showfliers=True)
            ax2.set_title(f"Box Plot of {var} with Outliers")
            ax2.set_xlabel(f"{var} ({units})")
            ax2.grid(True, alpha=0.3)

            # Adjust layout and save the plot
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{data_type}_{var}_validation.png")
            plt.savefig(plot_file, dpi=300)
            plt.close()

            # For temporal variables, create time series plots
            if (
                data_type == "current" and len(df) < 10000
            ):  # Only for reasonably sized datasets
                # Group by timestamp and calculate statistics
                time_stats = df.groupby(df["timestamp"].dt.date)[var].agg(
                    ["mean", "min", "max"]
                )

                if (
                    len(time_stats) > 1
                ):  # Only create time series if we have multiple dates
                    plt.figure(figsize=(14, 6))
                    plt.plot(
                        time_stats.index, time_stats["mean"], marker="o", label="Mean"
                    )
                    plt.fill_between(
                        time_stats.index,
                        time_stats["min"],
                        time_stats["max"],
                        alpha=0.2,
                        label="Min-Max Range",
                    )
                    plt.axhline(
                        y=min_val,
                        color="r",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Min expected: {min_val}",
                    )
                    plt.axhline(
                        y=max_val,
                        color="r",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Max expected: {max_val}",
                    )

                    plt.title(f"Time Series of {var} (Daily Statistics)")
                    plt.xlabel("Date")
                    plt.ylabel(f"{var} ({units})")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save the time series plot
                    time_plot_file = os.path.join(
                        output_dir, f"{data_type}_{var}_time_series.png"
                    )
                    plt.savefig(time_plot_file, dpi=300)
                    plt.close()

    # Overall validation summary
    print("\nWeather Variable Validation Summary:")
    print(f"Total records analyzed: {len(df)}")
    print(f"Weather variables examined: {len(weather_vars)}")

    if results["out_of_range"]:
        print("\nVariables with values outside expected ranges:")
        for var, count, percent in results["out_of_range"]:
            print(f"  - {var}: {count} values ({percent:.2f}%) out of range")
    else:
        print("\n✓ All variables are within expected ranges")

    if results["total_outliers"] > 0:
        print("\nDetected outliers by variable:")
        for var, count in results["outliers"].items():
            if count > 0:
                percent = (count / len(df[var].dropna())) * 100
                print(f"  - {var}: {count} outliers ({percent:.2f}%)")
    else:
        print("\n✓ No significant outliers detected")

    # Check for temporal consistency if applicable
    if data_type == "current" and len(df) > 100:
        # Check for suspicious jumps in values over time
        # This is a simplified check - in a real application, more sophisticated
        # time series analysis would be used
        try:
            for var in weather_vars:
                # Group by timestamp and calculate mean
                time_series = df.groupby(df["timestamp"].dt.date)[var].mean()

                # Only perform this check if we have multiple dates
                if len(time_series) > 2:
                    # Calculate day-to-day changes
                    changes = time_series.diff().dropna()

                    # Only check for jumps if we have some variation
                    std_change = np.std(changes)
                    if std_change > 1e-6:
                        large_jumps = np.abs(changes) > (
                            4 * std_change
                        )  # More than 4 standard deviations

                        if np.sum(large_jumps) > 0:
                            print(
                                f"\nWARNING: Detected {np.sum(large_jumps)} suspicious jumps in {var} time series"
                            )
                            results["overall_valid"] = False
        except Exception as e:
            print(f"INFO: Could not perform temporal consistency check: {str(e)}")

    # Final verdict
    if results["overall_valid"]:
        print(
            "\nWeather variable validation PASSED: Data appears to be valid for analysis."
        )
        return True
    else:
        print(
            "\nWeather variable validation WARNING: Issues detected in the weather data."
        )
        print(
            "The data may still be usable but should be reviewed carefully before analysis."
        )
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python validate_weather_variables.py <weather_data_file> <data_type> [output_dir] [generate_plots]"
        )
        print("  <data_type> can be 'current' or 'forecast'")
        sys.exit(1)

    weather_file = sys.argv[1]
    data_type = sys.argv[2]

    if data_type not in ["current", "forecast"]:
        print("ERROR: <data_type> must be 'current' or 'forecast'")
        sys.exit(1)

    output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    generate_plots = (
        True
        if len(sys.argv) <= 4
        else sys.argv[4].lower() in ["true", "t", "yes", "y", "1"]
    )

    success = validate_weather_variables(
        weather_file,
        data_type=data_type,
        output_dir=output_dir,
        generate_plots=generate_plots,
    )

    if not success:
        print(
            "Weather variable validation detected issues. Review the data before proceeding."
        )
        sys.exit(1)

    sys.exit(0)
