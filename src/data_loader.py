"""
Functions to load and validate meteor data from CSV files.
"""
import os
import glob
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from src.config import RAW_DATA_DIR, UNIFIED_PATTERN, ALL_PATTERN, EXPECTED_COLUMNS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_csv_files() -> Tuple[List[str], List[str]]:
    """
    Find all unified and all CSV files in the raw data directory.
    
    Returns:
        Tuple containing lists of paths to unified and all CSV files
    """
    unified_files = glob.glob(os.path.join(RAW_DATA_DIR, UNIFIED_PATTERN))
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, ALL_PATTERN))
    
    logger.info(f"Found {len(unified_files)} unified files and {len(all_files)} all files")
    
    return unified_files, all_files


def validate_csv_structure(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the structure of a CSV file, checking for header issues and expected columns.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Read just the header to check column count
        header = pd.read_csv(file_path, nrows=0)
        
        # Check number of columns
        if len(header.columns) != EXPECTED_COLUMNS:
            return False, f"Expected {EXPECTED_COLUMNS} columns, found {len(header.columns)}"
        
        # Check for duplicate headers
        if len(header.columns) != len(set(header.columns)):
            duplicates = [col for col in header.columns if list(header.columns).count(col) > 1]
            return False, f"Duplicate column names found: {duplicates}"
        
        # Check for unnamed columns (which could indicate header issues)
        unnamed_cols = [col for col in header.columns if "Unnamed" in str(col)]
        if unnamed_cols:
            return False, f"Unnamed columns found: {unnamed_cols}"
        
        # Read a small sample to check for data consistency
        sample = pd.read_csv(file_path, nrows=5)
        
        # Check for all numeric columns that should be numeric
        # (You can add more specific validation based on the column metadata)
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"


def load_csv_data(file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from a CSV file, with optional chunking for large files.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Size of chunks to load (if None, load entire file)
        
    Returns:
        DataFrame containing the data
    """
    logger.info(f"Loading data from {file_path}")
    
    if chunk_size:
        # Return a generator of chunks for memory-efficient processing
        return pd.read_csv(file_path, chunksize=chunk_size)
    else:
        # Load the entire file
        return pd.read_csv(file_path)


def validate_and_load_data(file_paths: List[str], chunk_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Validate and load data from multiple CSV files.
    
    Args:
        file_paths: List of paths to CSV files
        chunk_size: Size of chunks to load (if None, load entire files)
        
    Returns:
        Dictionary mapping file names to DataFrames
    """
    data_dict = {}
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # Validate file structure
        is_valid, error_message = validate_csv_structure(file_path)
        
        if not is_valid:
            logger.error(f"Validation failed for {file_name}: {error_message}")
            continue
        
        # Load data
        try:
            data = load_csv_data(file_path, chunk_size)
            data_dict[file_name] = data
            logger.info(f"Successfully loaded data from {file_name}")
        except Exception as e:
            logger.error(f"Error loading data from {file_name}: {str(e)}")
    
    return data_dict


def get_column_metadata(df: pd.DataFrame) -> Dict:
    """
    Get metadata about the columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing column metadata
    """
    metadata = {
        "total_columns": len(df.columns),
        "column_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "string_columns": df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return metadata
