"""
Core data processing functions for meteor data.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import logging
from datetime import datetime
from src.config import PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values, converting data types, etc.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with missing values in critical columns (you can customize this)
    critical_columns = ['_amag', '_ra_o', '_dc_o', '_vg', '_stream']
    cleaned_df = cleaned_df.dropna(subset=critical_columns)
    
    # Convert date-time columns
    if '_localtime' in cleaned_df.columns:
        try:
            cleaned_df['_datetime'] = pd.to_datetime(cleaned_df['_localtime'])
        except:
            logger.warning("Could not convert _localtime to datetime")
    
    # Fill missing values where appropriate (e.g., use mean for numeric columns)
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Skip columns where NaN might be meaningful
        if col not in ['_stream']:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Convert string columns to proper case or format
    if '_stream' in cleaned_df.columns:
        cleaned_df['_stream'] = cleaned_df['_stream'].str.strip().str.upper()
    
    return cleaned_df


def filter_data(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Filter data based on specified criteria.
    
    Args:
        df: Input DataFrame
        filters: Dictionary mapping column names to filter functions or values
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for column, filter_criteria in filters.items():
        if column not in filtered_df.columns:
            logger.warning(f"Column {column} not found in DataFrame, skipping filter")
            continue
        
        if callable(filter_criteria):
            # Apply custom filter function
            filtered_df = filtered_df[filter_criteria(filtered_df[column])]
        else:
            # Apply simple equality filter
            filtered_df = filtered_df[filtered_df[column] == filter_criteria]
    
    logger.info(f"Filtered data from {len(df)} to {len(filtered_df)} rows")
    return filtered_df


def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived columns based on existing data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional derived columns
    """
    result_df = df.copy()
    
    # Example: Calculate apparent velocity from components (if applicable)
    if '_vo' in result_df.columns and '_vs' in result_df.columns:
        try:
            result_df['_apparent_velocity_diff'] = result_df['_vo'] - result_df['_vs']
        except:
            logger.warning("Could not calculate apparent velocity difference")
    
    # Calculate mean values per meteor stream (if applicable)
    if '_stream' in result_df.columns:
        try:
            # Group by stream and calculate statistics
            stream_stats = result_df.groupby('_stream').agg({
                '_vg': 'mean',
                '_a': 'mean',
                '_q': 'mean',
                '_e': 'mean',
                '_incl': 'mean'
            }).rename(columns={
                '_vg': '_stream_mean_vg',
                '_a': '_stream_mean_a',
                '_q': '_stream_mean_q',
                '_e': '_stream_mean_e',
                '_incl': '_stream_mean_incl'
            })
            
            # Merge back to the main dataframe
            result_df = pd.merge(
                result_df, 
                stream_stats, 
                left_on='_stream', 
                right_index=True, 
                how='left'
            )
        except:
            logger.warning("Could not calculate stream statistics")
    
    return result_df


def process_data(df: pd.DataFrame, 
                cleaning: bool = True, 
                filters: Optional[Dict] = None,
                calculate_derived: bool = True) -> pd.DataFrame:
    """
    Process data by cleaning, filtering, and calculating derived columns.
    
    Args:
        df: Input DataFrame
        cleaning: Whether to apply cleaning
        filters: Filters to apply (dict mapping columns to filter values/functions)
        calculate_derived: Whether to calculate derived columns
        
    Returns:
        Processed DataFrame
    """
    result_df = df.copy()
    
    # Apply processing steps based on flags
    if cleaning:
        result_df = clean_data(result_df)
    
    if filters:
        result_df = filter_data(result_df, filters)
    
    if calculate_derived:
        result_df = calculate_derived_columns(result_df)
    
    return result_df


def process_and_save_data(data_dict: Dict[str, pd.DataFrame], 
                         output_prefix: str = "processed",
                         **processing_kwargs) -> Dict[str, str]:
    """
    Process multiple DataFrames and save results.
    
    Args:
        data_dict: Dictionary mapping file names to DataFrames
        output_prefix: Prefix for output files
        processing_kwargs: Keyword arguments for process_data()
        
    Returns:
        Dictionary mapping original file names to output file paths
    """
    output_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file_name, df in data_dict.items():
        # For chunked data (generators)
        if isinstance(df, pd.io.parsers.TextFileReader):
            # Process each chunk and append to a list
            processed_chunks = []
            for chunk in df:
                processed_chunk = process_data(chunk, **processing_kwargs)
                processed_chunks.append(processed_chunk)
            
            # Concatenate chunks
            processed_df = pd.concat(processed_chunks, ignore_index=True)
        else:
            # Process the entire DataFrame
            processed_df = process_data(df, **processing_kwargs)
        
        # Generate output file name
        output_file = f"{output_prefix}_{file_name.replace('.csv', '')}_{timestamp}.csv"
        output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
        
        # Save processed data
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        output_files[file_name] = output_path
    
    return output_files


def merge_datasets(unified_data: pd.DataFrame, all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge unified and all datasets based on common keys.
    
    Args:
        unified_data: DataFrame containing unified data
        all_data: DataFrame containing all data
        
    Returns:
        Merged DataFrame
    """
    # Determine appropriate merge keys based on the data
    # This will depend on the exact structure of your data
    merge_keys = ['_ID1', '_ID2', '_mjd']  # Example merge keys
    
    # Check if all merge keys exist in both dataframes
    for key in merge_keys:
        if key not in unified_data.columns or key not in all_data.columns:
            logger.warning(f"Merge key {key} not found in both dataframes")
            return None
    
    try:
        # Merge the dataframes
        merged_df = pd.merge(
            unified_data,
            all_data,
            on=merge_keys,
            how='outer',
            suffixes=('_unified', '_all')
        )
        
        logger.info(f"Merged dataframes, resulting in {len(merged_df)} rows")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging dataframes: {str(e)}")
        return None
