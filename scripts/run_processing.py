#!/usr/bin/env python
"""
Main script to run the complete meteor data processing pipeline.
"""
import os
import sys
import argparse
import logging
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, BATCH_SIZE
from src.data_loader import find_csv_files, validate_and_load_data, get_column_metadata
from src.data_processor import process_and_save_data, merge_datasets
from src.data_analyzer import generate_basic_statistics, analyze_meteor_streams, identify_outliers
from src.data_visualizer import create_visualization_report
from src.utils import setup_logging, optimize_dataframe, save_to_parquet, save_metadata, time_execution


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process meteor data from CSV files.')
    
    parser.add_argument('--input-dir', type=str, default=RAW_DATA_DIR,
                        help='Directory containing input CSV files')
    
    parser.add_argument('--output-dir', type=str, default=PROCESSED_DATA_DIR,
                        help='Directory for output files')
    
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (default: log to console only)')
    
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for processing large files')
    
    parser.add_argument('--skip-cleaning', action='store_true',
                        help='Skip data cleaning step')
    
    parser.add_argument('--skip-derived-columns', action='store_true',
                        help='Skip calculation of derived columns')
    
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization generation')
    
    parser.add_argument('--optimize-memory', action='store_true',
                        help='Optimize DataFrame memory usage')
    
    parser.add_argument('--stream-filter', type=str, default=None,
                        help='Filter data by specific meteor stream')
    
    parser.add_argument('--save-parquet', action='store_true',
                        help='Save processed data in Parquet format')
    
    return parser.parse_args()


@time_execution
def main():
    """Main function to run the processing pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Find CSV files
    logger.info("Step 1: Finding CSV files")
    unified_files, all_files = find_csv_files()
    
    if not unified_files and not all_files:
        logger.error("No CSV files found. Exiting.")
        return 1
    
    # Step 2: Load and validate data
    logger.info("Step 2: Loading and validating data")
    unified_data = validate_and_load_data(unified_files, args.batch_size)
    all_data = validate_and_load_data(all_files, args.batch_size)
    
    if not unified_data and not all_data:
        logger.error("No valid data loaded. Exiting.")
        return 1
    
    # Step 3: Process data
    logger.info("Step 3: Processing data")
    processing_kwargs = {
        'cleaning': not args.skip_cleaning,
        'calculate_derived': not args.skip_derived_columns,
    }
    
    # Add filter if specified
    if args.stream_filter:
        processing_kwargs['filters'] = {'_stream': args.stream_filter}
    
    # Process unified data
    if unified_data:
        unified_processed = process_and_save_data(
            unified_data,
            output_prefix=f"unified_{timestamp}",
            **processing_kwargs
        )
        logger.info(f"Processed {len(unified_processed)} unified files")
    
    # Process all data
    if all_data:
        all_processed = process_and_save_data(
            all_data,
            output_prefix=f"all_{timestamp}",
            **processing_kwargs
        )
        logger.info(f"Processed {len(all_processed)} all files")
    
    # Step 4: Optimize memory usage if requested
    if args.optimize_memory and unified_data:
        logger.info("Step 4: Optimizing memory usage")
        
        # Get the first DataFrame (or first chunk if using chunking)
        first_df_name = list(unified_data.keys())[0]
        first_df = unified_data[first_df_name]
        
        if hasattr(first_df, 'get_chunk'):
            # For chunked data
            chunk = next(first_df)
            optimized_chunk, optimization_stats = optimize_dataframe(chunk)
        else:
            # For complete DataFrame
            optimized_df, optimization_stats = optimize_dataframe(first_df)
        
        logger.info(f"Memory optimization results: {optimization_stats['memory_reduction_percent']}% reduction")
    
    # Step 5: Save in Parquet format if requested
    if args.save_parquet and unified_processed:
        logger.info("Step 5: Saving data in Parquet format")
        
        for original_file, csv_path in unified_processed.items():
            parquet_path = csv_path.replace('.csv', '.parquet')
            df = pd.read_csv(csv_path)
            save_to_parquet(df, parquet_path)
    
    # Step 6: Generate statistics and analysis
    logger.info("Step 6: Generating statistics and analysis")
    
    # Load the first processed file for analysis
    if unified_processed:
        analysis_file_path = list(unified_processed.values())[0]
        analysis_df = pd.read_csv(analysis_file_path)
        
        # Generate basic statistics
        statistics = generate_basic_statistics(analysis_df)
        stats_path = os.path.join(args.output_dir, f"statistics_{timestamp}.json")
        save_metadata(statistics, stats_path)
        
        # Analyze meteor streams
        if '_stream' in analysis_df.columns:
            stream_analysis = analyze_meteor_streams(analysis_df)
            stream_path = os.path.join(args.output_dir, f"stream_analysis_{timestamp}.csv")
            stream_analysis.to_csv(stream_path, index=False)
            logger.info(f"Saved stream analysis to {stream_path}")
        
        # Identify outliers
        outlier_columns = ['_amag', '_vg', '_a', '_e', '_incl'] if '_a' in analysis_df.columns else ['_amag', '_vg']
        outliers_df = identify_outliers(analysis_df, outlier_columns)
        outliers_path = os.path.join(args.output_dir, f"outliers_{timestamp}.csv")
        outliers_df.to_csv(outliers_path, index=False)
        logger.info(f"Saved outlier analysis to {outliers_path}")
    
    # Step 7: Generate visualizations if not skipped
    if not args.skip_visualization and unified_processed:
        logger.info("Step 7: Generating visualizations")
        
        # Use the same DataFrame as for analysis
        visualization_dir = create_visualization_report(
            analysis_df,
            output_dir=os.path.join(args.output_dir, f"viz_{timestamp}")
        )
        logger.info(f"Generated visualization report in {visualization_dir}")
    
    logger.info("Processing completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
