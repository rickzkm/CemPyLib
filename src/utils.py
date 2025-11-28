"""
Utility functions for the meteor data processing project.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, log to console only)
        level: Logging level
        
    Returns:
        Configured logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_memory_usage(df: pd.DataFrame) -> Dict:
    """
    Calculate memory usage of a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    # Calculate memory usage by dtype
    dtype_memory = df.dtypes.value_counts().to_dict()
    dtype_memory = {str(k): v for k, v in dtype_memory.items()}
    
    # Calculate memory usage by column
    column_memory = {col: memory_usage[i] for i, col in enumerate(df.columns)}
    column_memory = {k: int(v) for k, v in sorted(column_memory.items(), key=lambda item: item[1], reverse=True)}
    
    result = {
        'total_memory_bytes': int(total_memory),
        'total_memory_mb': round(total_memory / (1024 * 1024), 2),
        'row_count': len(df),
        'column_count': len(df.columns),
        'dtypes': dtype_memory,
        'column_memory': column_memory
    }
    
    return result


def optimize_dataframe(df: pd.DataFrame, category_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimize DataFrame memory usage by converting data types.
    
    Args:
        df: Input DataFrame
        category_threshold: Threshold for converting string columns to category
            (ratio of unique values to total values)
        
    Returns:
        Tuple of (optimized DataFrame, optimization statistics)
    """
    result_df = df.copy()
    
    original_memory = get_memory_usage(df)
    optimization_stats = {
        'original_memory_mb': original_memory['total_memory_mb'],
        'conversions': {}
    }
    
    # Process each column
    for col in df.columns:
        col_type = df[col].dtype
        
        # For integer columns
        if pd.api.types.is_integer_dtype(col_type):
            # Check if column can be downcasted
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max <= 255:
                    result_df[col] = df[col].astype(np.uint8)
                    optimization_stats['conversions'][col] = f'{col_type} -> np.uint8'
                elif col_max <= 65535:
                    result_df[col] = df[col].astype(np.uint16)
                    optimization_stats['conversions'][col] = f'{col_type} -> np.uint16'
                elif col_max <= 4294967295:
                    result_df[col] = df[col].astype(np.uint32)
                    optimization_stats['conversions'][col] = f'{col_type} -> np.uint32'
            else:  # Signed integers
                if col_min >= -128 and col_max <= 127:
                    result_df[col] = df[col].astype(np.int8)
                    optimization_stats['conversions'][col] = f'{col_type} -> np.int8'
                elif col_min >= -32768 and col_max <= 32767:
                    result_df[col] = df[col].astype(np.int16)
                    optimization_stats['conversions'][col] = f'{col_type} -> np.int16'
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    result_df[col] = df[col].astype(np.int32)
                    optimization_stats['conversions'][col] = f'{col_type} -> np.int32'
        
        # For float columns
        elif pd.api.types.is_float_dtype(col_type):
            # Check if column can be downcasted to float32
            col_min = df[col].min()
            col_max = df[col].max()
            
            if np.isnan(col_min) or np.isnan(col_max):
                continue
                
            if (col_min >= np.finfo(np.float32).min and 
                col_max <= np.finfo(np.float32).max):
                result_df[col] = df[col].astype(np.float32)
                optimization_stats['conversions'][col] = f'{col_type} -> np.float32'
        
        # For string columns
        elif pd.api.types.is_string_dtype(col_type) or pd.api.types.is_object_dtype(col_type):
            # Check if column has a manageable number of unique values
            unique_values = df[col].nunique()
            unique_ratio = unique_values / len(df)
            
            # Convert to category if ratio of unique values is below threshold
            if unique_ratio < category_threshold:
                result_df[col] = df[col].astype('category')
                optimization_stats['conversions'][col] = f'{col_type} -> category'
    
    # Calculate final memory usage
    optimized_memory = get_memory_usage(result_df)
    optimization_stats['optimized_memory_mb'] = optimized_memory['total_memory_mb']
    optimization_stats['memory_reduction_percent'] = round(
        (original_memory['total_memory_mb'] - optimized_memory['total_memory_mb']) /
        original_memory['total_memory_mb'] * 100, 2
    )

    return result_df, optimization_stats


def save_to_parquet(df: pd.DataFrame, output_path: str) -> str:
    """
    Save DataFrame to Parquet format for efficient storage and future loading.
    
    Args:
        df: Input DataFrame
        output_path: Path to save the Parquet file
        
    Returns:
        Path to the saved file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to Parquet format
        df.to_parquet(output_path, index=False, compression='snappy')
        
        logger.info(f"Saved DataFrame to Parquet: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving DataFrame to Parquet: {str(e)}")
        return None


def load_from_parquet(input_path: str) -> pd.DataFrame:
    """
    Load DataFrame from Parquet format.
    
    Args:
        input_path: Path to the Parquet file
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded DataFrame from Parquet: {input_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from Parquet: {str(e)}")
        return None


def save_metadata(metadata: Dict, output_path: str) -> str:
    """
    Save metadata dictionary to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to Python types
        metadata_json = json.dumps(metadata, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else (int(x) if isinstance(x, np.integer) else str(x)), indent=2)
        
        with open(output_path, 'w') as f:
            f.write(metadata_json)
        
        logger.info(f"Saved metadata to JSON: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving metadata to JSON: {str(e)}")
        return None


def time_execution(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper