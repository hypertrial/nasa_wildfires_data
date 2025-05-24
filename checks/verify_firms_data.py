"""
FIRMS Data Integrity Verification
================================

This script verifies the integrity and completeness of collected FIRMS
(Fire Information for Resource Management System) data by:
1. Checking for the presence of all required fields
2. Validating data types and ranges for each field
3. Detecting and reporting missing or anomalous values
4. Verifying temporal and spatial coverage

The script helps ensure that wildfire data used in the analysis is
complete, consistent, and reliable.
"""

import os
import sys

import pandas as pd


def verify_firms_data(
    file_path, min_lat=30.0, max_lat=70.0, min_lon=-130.0, max_lon=-100.0
):
    """
    Verifies the integrity and completeness of FIRMS data.

    Parameters:
    -----------
    file_path : str
        Path to the FIRMS data file (CSV or Parquet)
    min_lat, max_lat, min_lon, max_lon : float
        Boundaries of the region of interest

    Returns:
    --------
    bool
        True if data passes all integrity checks, False otherwise
    """
    print(f"Verifying FIRMS data integrity for file: {file_path}")

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
        print("ERROR: FIRMS data file is empty")
        return False

    print(f"Found {len(df)} fire records")

    # Check required columns
    required_columns = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "confidence",
        "frp",
        "brightness",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {', '.join(missing_columns)}")
        return False

    # Check for null values in critical columns
    critical_columns = ["latitude", "longitude", "acq_date"]
    null_counts = {col: df[col].isnull().sum() for col in critical_columns}

    has_critical_nulls = any(count > 0 for count in null_counts.values())
    if has_critical_nulls:
        print("ERROR: Found null values in critical columns:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  - {col}: {count} null values ({count / len(df) * 100:.2f}%)")
        return False

    # Check value ranges
    if df["latitude"].min() < min_lat or df["latitude"].max() > max_lat:
        print(f"WARNING: Latitude values outside expected range [{min_lat}, {max_lat}]")
        print(f"  - Found range: [{df['latitude'].min()}, {df['latitude'].max()}]")

    if df["longitude"].min() < min_lon or df["longitude"].max() > max_lon:
        print(
            f"WARNING: Longitude values outside expected range [{min_lon}, {max_lon}]"
        )
        print(f"  - Found range: [{df['longitude'].min()}, {df['longitude'].max()}]")

    # Check confidence values
    if "confidence" in df.columns:
        if df["confidence"].min() < 0 or df["confidence"].max() > 100:
            print("WARNING: Confidence values outside expected range [0, 100]")
            print(
                f"  - Found range: [{df['confidence'].min()}, {df['confidence'].max()}]"
            )

    # Check temporal coverage
    if "acq_date" in df.columns:
        try:
            dates = pd.to_datetime(df["acq_date"])
            date_range = (dates.min(), dates.max())
            days_covered = (date_range[1] - date_range[0]).days + 1

            print(
                f"Temporal coverage: {date_range[0].date()} to {date_range[1].date()} ({days_covered} days)"
            )

            # Check for date gaps
            unique_dates = pd.Series(dates.dt.date.unique()).sort_values()
            expected_dates = pd.date_range(unique_dates.min(), unique_dates.max()).date
            missing_dates = set(expected_dates) - set(unique_dates)

            if missing_dates:
                print(
                    f"WARNING: Found {len(missing_dates)} days with missing fire data:"
                )
                for date in sorted(list(missing_dates))[
                    :5
                ]:  # Show first 5 missing dates
                    print(f"  - {date}")
                if len(missing_dates) > 5:
                    print(f"  ... and {len(missing_dates) - 5} more dates")
        except Exception as e:
            print(f"ERROR: Failed to process date information: {str(e)}")

    # Calculate data completeness percentage
    completeness = 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    print(f"Overall data completeness: {completeness:.2f}%")

    if completeness < 90:
        print("WARNING: Data completeness below 90%")

    # Check for duplicates
    duplicates = df.duplicated(
        subset=["latitude", "longitude", "acq_date", "acq_time"]
    ).sum()
    if duplicates > 0:
        print(f"WARNING: Found {duplicates} duplicate records")

    print("FIRMS data verification completed")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_firms_data.py <firms_data_file>")
        sys.exit(1)

    firms_file = sys.argv[1]
    success = verify_firms_data(firms_file)

    if not success:
        print("FIRMS data verification failed")
        sys.exit(1)

    print("FIRMS data verification passed")
    sys.exit(0)
