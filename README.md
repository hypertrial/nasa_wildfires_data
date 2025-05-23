# NASA FIRMS Wildfire Analysis System

A comprehensive spatial analysis system for monitoring, analyzing, and predicting wildfire risk through the integration of NASA FIRMS (Fire Information for Resource Management System) satellite fire detection data and meteorological information.

## Overview

This project provides a modular, maintainable Python package for processing and analyzing NASA FIRMS wildfire data. The system creates a spatial graph representation of wildfire risk by integrating data from multiple sources:

1. **NASA FIRMS** satellite fire detections (latitude, longitude, acquisition date/time, confidence, FRP, brightness)
2. Current meteorological conditions (temperature, humidity, wind speed/direction, etc.)
3. Forecast meteorological conditions (6-hour ahead predictions)

The system covers western North America (approximately (-130°W, 30°N) to (-100°W, 70°N)), including most of Canada and the western United States.

## NASA FIRMS Data Integration

The system processes NASA FIRMS data which includes:

- **Fire locations**: Precise latitude/longitude coordinates of detected fire hotspots
- **Fire intensity**: Fire Radiative Power (FRP) and brightness temperature measurements
- **Temporal information**: Acquisition date and time of satellite observations
- **Confidence values**: Detection confidence scores for each fire point

This data is combined with meteorological information to create a comprehensive spatial representation of wildfire risk.

## Directory Structure

```
wildfire_analysis/               # Main package directory
├── __init__.py                  # Package initialization
├── data_processing/             # Data processing modules
│   ├── __init__.py
│   ├── loader.py                # FIRMS and weather data loading functions
│   └── preprocessor.py          # Data preprocessing functions
├── spatial/                     # Spatial analysis modules
│   ├── __init__.py
│   ├── distance.py              # Distance calculation functions
│   ├── grid.py                  # Spatial grid creation functions
│   └── graph_builder.py         # Graph construction functions
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── parallel.py              # Parallel processing utilities
├── visualization/               # Visualization modules (placeholder)
│   └── __init__.py
└── main.py                      # Main entry point for the pipeline

data/                           # Data directory
├── fires_combined.csv           # FIRMS fire detection data
├── meteo_current.parquet        # Current meteorological conditions
└── meteo_forecast.parquet       # Forecast meteorological conditions

checks/                         # Data validation and quality checks
├── verify_firms_data.py         # FIRMS data integrity validation
├── verify_meteo_data.py         # Meteorological data format validation
├── validate_grid_coverage.py    # Grid coverage and consistency checks
├── validate_weather_variables.py # Weather variable range and outlier detection
├── verify_grid.py               # Grid cell verification
└── check_grid_size.py           # Grid dimensioning validation

run_analysis.py                 # Command-line script to run the analysis
validate_all.py                 # Script to run all validation checks
setup.py                        # Package installation script
```

## Key Features

1. **NASA FIRMS Data Processing**: Efficient processing of satellite fire detection data
2. **Modular Design**: Organized into logical components with clear responsibilities
3. **Spatial Grid System**: Creates equal-area grid cells (10km × 10km) despite Earth's curvature
4. **Graph-Based Representation**: Nodes represent grid cells, edges connect neighboring cells
5. **Feature Extraction**: Computes fire density, intensity, and meteorological features per cell
6. **Data Validation**: Comprehensive checks to ensure FIRMS data quality and consistency
7. **Parallel Processing**: Utilizes multiple CPU cores for efficient data processing

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/wildfire_analysis.git
cd wildfire_analysis

# Install package
pip install -e .
```

## Usage

### Command-line Interface

```bash
# Default configuration
python run_analysis.py --data-dir data

# With custom parameters
python run_analysis.py --data-dir data --grid-size 20 --radius 15 --workers 4
```

### Python API

```python
from wildfire_analysis.main import run_pipeline

# Run analysis
firms_df, current_weather_df, forecast_weather_df, G, node_features = run_pipeline(
    data_dir='data',
    grid_size_km=10,
    radius_km=10,
    n_workers=4
)

# Access graph and features
print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"Each node has {len(list(node_features.values())[0])} features")
```

## NASA FIRMS Data Validation

Before running the analysis, you can validate the FIRMS data to ensure quality and consistency:

```bash
# Run all validation checks
python validate_all.py --data-dir data --output-dir output

# Run specific FIRMS data validation
python checks/verify_firms_data.py data/firms_fire_data.csv
```

The system includes several data validation checks specific to FIRMS data:

1. **FIRMS Data Integrity**:

   - Validates required columns (latitude, longitude, acq_date, acq_time, confidence, frp, brightness)
   - Checks for missing values in critical fields
   - Verifies temporal and spatial coverage
   - Detects duplicate fire records

2. **Spatial Coverage Validation**:
   - Ensures FIRMS data covers the study region
   - Validates consistent spatial distribution
   - Creates visualization of fire detection coverage

## Performance

The system efficiently processes NASA FIRMS datasets:

- Processes thousands of fire detection records in seconds
- Creates spatial graphs with thousands of nodes and edges
- Leverages parallel processing to utilize multiple CPU cores
- Uses optimized distance calculations for geographic data

## Dependencies

All required Python packages are listed in `requirements.txt` and `setup.py`:

- numpy>=1.20.0
- pandas>=1.3.0
- networkx>=2.6.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- pyarrow>=8.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- statsmodels>=0.13.0
- numba>=0.56.0

## License

MIT
