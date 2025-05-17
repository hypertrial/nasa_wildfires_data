#!/usr/bin/env python3
"""
Wildfire Risk Analysis - Main Entry Point
=======================================

Script to run the wildfire risk analysis pipeline using the refactored package.
"""

import argparse
import sys
import os
import pickle
import networkx as nx
from wildfire_analysis.main import run_pipeline


def main():
    """
    Main entry point for running the wildfire risk analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Wildfire Risk Analysis Pipeline')
    parser.add_argument('--data-dir', type=str, help='Directory containing data files')
    parser.add_argument('--grid-size', type=float, default=10.0, help='Grid cell size in kilometers')
    parser.add_argument('--radius', type=float, default=10.0, help='Search radius in kilometers')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    parser.add_argument('--min-lat', type=float, default=30.0, help='Minimum latitude')
    parser.add_argument('--max-lat', type=float, default=70.0, help='Maximum latitude')
    parser.add_argument('--min-lon', type=float, default=-130.0, help='Minimum longitude')
    parser.add_argument('--max-lon', type=float, default=-100.0, help='Maximum longitude')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--save-format', type=str, default='pickle', 
                        choices=['graphml', 'gexf', 'pickle', 'adjlist'], 
                        help='Format to save the graph (pickle recommended for data analysis)')
    
    args = parser.parse_args()
    
    # Run the pipeline with command-line arguments
    print("Starting Wildfire Risk Analysis")
    firms_df, current_weather_df, forecast_weather_df, G, node_features = run_pipeline(
        data_dir=args.data_dir,
        grid_size_km=args.grid_size,
        radius_km=args.radius,
        n_workers=args.workers,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon
    )
    
    print("Analysis pipeline completed successfully.")
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Each node has {len(list(node_features.values())[0])} features")
    
    # Save the graph if output directory is provided
    if args.output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Add node features to the graph
        for node_id, features in node_features.items():
            if node_id in G.nodes:
                for key, value in features.items():
                    # Convert numpy arrays or other non-serializable types if needed
                    if hasattr(value, 'tolist'):
                        value = value.tolist()
                    G.nodes[node_id][key] = value
        
        # Save the graph in the specified format
        if args.save_format == 'pickle':
            output_file = os.path.join(args.output_dir, 'wildfire_graph.pickle')
            print(f"Saving graph to {output_file}")
            # Use standard pickle instead of nx.write_gpickle
            with open(output_file, 'wb') as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            output_file = os.path.join(args.output_dir, f'wildfire_graph.{args.save_format}')
            print(f"Saving graph to {output_file}")
            
            if args.save_format == 'graphml':
                nx.write_graphml(G, output_file)
            elif args.save_format == 'gexf':
                nx.write_gexf(G, output_file)
            elif args.save_format == 'adjlist':
                nx.write_adjlist(G, output_file)
            
        print(f"Graph saved successfully to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 