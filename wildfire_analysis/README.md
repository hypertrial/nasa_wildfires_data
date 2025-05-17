# Wildfire Risk Analysis Package

A refactored implementation of the wildfire risk analysis pipeline, organized as a modular Python package.

## Overview

This package creates a spatial graph representation of wildfire risk by integrating fire detection data with meteorological conditions. The resulting graph enables advanced spatial analysis, pattern recognition, and predictive modeling of wildfire behavior and risk factors.

## Package Structure

The codebase is organized into the following modules:

```
wildfire_analysis/
├── __init__.py                      # Package initialization
├── data_processing/
│   ├── __init__.py
│   ├── loader.py                    # Data loading functions
│   └── preprocessor.py              # Data preprocessing functions
├── spatial/
│   ├── __init__.py
│   ├── distance.py                  # Distance calculation functions
│   ├── grid.py                      # Spatial grid creation functions
│   └── graph_builder.py             # Graph construction functions
├── utils/
│   ├── __init__.py
│   └── parallel.py                  # Parallel processing utilities
├── visualization/
│   └── __init__.py                  # Visualization utilities (placeholder)
└── main.py                          # Main entry point for the pipeline
```

## Input Data

The package requires the following input data files:

1. `fires_combined.csv` - FIRMS wildfire detection data

   - Contains satellite-detected hotspots with coordinates, intensity, and metadata

2. `meteo_current.parquet` - Current meteorological data

   - Contains current weather variables at ~100km grid points

3. `meteo_forecast.parquet` - Forecast meteorological data
   - Contains 6-hour ahead weather forecasts at ~100km grid points

## Geographic Coverage

- Bounding Box: (-130°W, 30°N) to (-100°W, 70°N)
- Region: Western North America (western United States and most of Canada)

## Usage

```python
from wildfire_analysis.main import run_pipeline

# Run the pipeline with default parameters
firms_df, current_weather_df, forecast_weather_df, G, node_features = run_pipeline()

# Or customize the parameters
firms_df, current_weather_df, forecast_weather_df, G, node_features = run_pipeline(
    data_dir='path/to/data',
    grid_size_km=20,
    radius_km=15,
    n_workers=4,
    min_lat=35,
    max_lat=65,
    min_lon=-125,
    max_lon=-105
)
```

## Command-line Usage

The package can also be run from the command line using the provided `run_analysis.py` script:

```bash
python run_analysis.py --data-dir data --grid-size 10 --radius 10 --workers 4
```

Available command-line arguments:

- `--data-dir`: Directory containing the data files
- `--grid-size`: Grid cell size in kilometers (default: 10.0)
- `--radius`: Search radius in kilometers (default: 10.0)
- `--workers`: Number of worker processes (default: auto-detect)
- `--min-lat`, `--max-lat`, `--min-lon`, `--max-lon`: Bounding box coordinates

## Core Features

- **Multi-source Data Integration**: Combines satellite fire detections with weather variables
- **Adaptive Spatial Grid**: Creates equally-sized cells despite Earth's curvature
- **Optimized Distance Calculations**: Uses vectorized haversine distance calculations for efficiency
- **Comprehensive Feature Extraction**: Generates rich feature vectors for each geographic cell
- **Graph-based Representation**: Enables spatial analysis and network-based modeling approaches
- **Parallel Processing**: Utilizes multiple CPU cores for faster data processing

## Dependencies

See requirements.txt for the full list of dependencies.

- numpy
- pandas
- networkx
- scikit-learn
- scipy
- numba
- pyarrow
