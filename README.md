# Wildfire Risk Analysis System (Refactored)

A comprehensive spatial analysis system for monitoring, analyzing, and predicting wildfire risk through the integration of satellite fire detection data and meteorological information.

## Overview

This project has been refactored from a single monolithic script (`cleaner.py`) into a modular, maintainable Python package structure. The system creates a spatial graph representation of wildfire risk by integrating data from multiple sources:

1. NASA FIRMS (Fire Information for Resource Management System) satellite fire detections
2. Current meteorological conditions
3. Forecast meteorological conditions (6-hour ahead predictions)

The system covers western North America (approximately (-130°W, 30°N) to (-100°W, 70°N)), including most of Canada and the western United States.

## Directory Structure

```
wildfire_analysis/               # Main package directory
├── __init__.py                  # Package initialization
├── data_processing/             # Data processing modules
│   ├── __init__.py
│   ├── loader.py                # Data loading functions
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
README.md                       # Original project README
README_refactored.md            # This file
```

## Key Improvements from Refactoring

1. **Modular Design**: Organized into logical components with clear responsibilities
2. **Improved Maintainability**: Smaller, focused modules instead of one large script
3. **Better Documentation**: Comprehensive docstrings and module-level documentation
4. **Reusable Components**: Core functionality organized into importable modules
5. **Enhanced Testability**: Modular structure makes unit testing easier
6. **Flexible Configuration**: Command-line options and programmatic API
7. **Installable Package**: Can be installed with pip for use in other projects
8. **Data Validation**: Comprehensive checks to ensure data quality and consistency

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

### Data Validation

Before running the analysis, you can validate all input data to ensure quality and consistency:

```bash
# Run all validation checks
python validate_all.py --data-dir data --output-dir output

# Run specific validation checks
python checks/verify_firms_data.py data/firms_fire_data.csv
python checks/verify_meteo_data.py data/current_weather.csv current
python checks/validate_grid_coverage.py data/grid_cells.csv 10 true
python checks/validate_weather_variables.py data/current_weather.csv current
```

## Core Features

- **Multi-source Data Integration**: Combines satellite fire detections with weather variables
- **Adaptive Spatial Grid**: Creates equally-sized cells (10km × 10km) despite Earth's curvature
- **Optimized Distance Calculations**: Uses vectorized haversine distance calculations for efficiency
- **Comprehensive Feature Extraction**: Generates rich feature vectors for each geographic cell
- **Graph-based Representation**: Enables spatial analysis and network-based modeling approaches
- **Parallel Processing**: Utilizes multiple CPU cores for faster data processing
- **Data Quality Validation**: Ensures input data meets quality standards before analysis

## Validation Checks

The system includes several data validation checks to ensure data quality:

1. **FIRMS Data Integrity**:

   - Validates required columns and data types
   - Checks for missing values in critical fields
   - Verifies temporal and spatial coverage
   - Detects duplicate records

2. **Meteorological Data Validation**:

   - Confirms completeness of required variables
   - Validates data formats and structure
   - Checks for missing time periods
   - Verifies spatial coverage matches study area

3. **Grid Coverage Validation**:

   - Ensures complete coverage of the study region
   - Checks for gaps or overlaps between grid cells
   - Validates consistent cell spacing
   - Creates visualization of grid coverage

4. **Weather Variable Checks**:
   - Verifies values are within expected ranges
   - Detects statistical outliers
   - Identifies suspicious temporal patterns
   - Generates diagnostic plots for visual inspection

## Performance

The refactored code maintains the high performance of the original implementation:

- Efficiently processes datasets with thousands of records
- Creates spatial graphs with thousands of nodes and edges in seconds
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
