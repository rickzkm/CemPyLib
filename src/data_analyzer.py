"""
Analysis functions for meteor data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import logging
from scipy import stats
from sklearn.cluster import DBSCAN
from src.config import PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_basic_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate basic statistics for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of statistics
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Calculate statistics
    stats_dict = {
        'column_count': len(df.columns),
        'row_count': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'descriptive_stats': df[numeric_cols].describe().to_dict()
    }
    
    # Calculate distribution metrics for key columns
    key_columns = ['_amag', '_vg', '_a', '_q', '_e', '_incl']
    distribution_stats = {}
    
    for col in key_columns:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                distribution_stats[col] = {
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'normality_test': stats.normaltest(data)[1] if len(data) >= 8 else None
                }
    
    stats_dict['distribution_stats'] = distribution_stats
    
    return stats_dict


def analyze_meteor_streams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze meteor streams in the data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with stream analysis results
    """
    if '_stream' not in df.columns:
        logger.warning("No '_stream' column found in the data")
        return pd.DataFrame()
    
    # Count meteors per stream
    stream_counts = df['_stream'].value_counts().reset_index()
    stream_counts.columns = ['stream', 'count']
    
    # Calculate statistics per stream
    stream_stats = df.groupby('_stream').agg({
        '_amag': ['mean', 'std', 'min', 'max'],
        '_vg': ['mean', 'std', 'min', 'max'],
        '_a': ['mean', 'std'],
        '_q': ['mean', 'std'],
        '_e': ['mean', 'std'],
        '_incl': ['mean', 'std']
    }).reset_index()
    
    # Flatten the multi-index columns
    stream_stats.columns = ['_'.join(col).strip('_') for col in stream_stats.columns.values]

    # Rename the first column back to 'stream' for merging
    stream_stats.rename(columns={'stream': '_stream'}, inplace=True)

    # Merge with counts
    result = pd.merge(stream_counts, stream_stats, left_on='stream', right_on='_stream')

    return result


def identify_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Identify outliers in the data.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outlier flags
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        # Skip non-numeric columns
        if not np.issubdtype(result_df[col].dtype, np.number):
            logger.warning(f"Column {col} is not numeric, skipping outlier detection")
            continue
        
        if method == 'iqr':
            # IQR method
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            result_df[f'{col}_outlier'] = (
                (result_df[col] < lower_bound) | 
                (result_df[col] > upper_bound)
            )
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(result_df[col], nan_policy='omit'))
            result_df[f'{col}_outlier'] = z_scores > threshold
        
        else:
            logger.warning(f"Unknown method {method}, skipping outlier detection")
    
    return result_df


def cluster_meteors(df: pd.DataFrame, features: List[str], eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    """
    Cluster meteors based on their orbital parameters.
    
    Args:
        df: Input DataFrame
        features: Features to use for clustering
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        DataFrame with cluster assignments
    """
    # Check if all features exist in the dataframe
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in DataFrame")
            return df
    
    # Extract features and scale them
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Normalize features
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    # Add cluster labels to the original dataframe
    result_df = df.copy()
    result_df['cluster'] = cluster_labels
    
    # Count observations in each cluster
    cluster_counts = pd.DataFrame(result_df['cluster'].value_counts()).reset_index()
    cluster_counts.columns = ['cluster', 'count']
    
    logger.info(f"Found {len(cluster_counts)} clusters (including noise cluster -1)")
    logger.info(f"Noise points: {len(result_df[result_df['cluster'] == -1])}")
    
    return result_df


def save_analysis_results(results: Dict, filename: str) -> str:
    """
    Save analysis results to a file.
    
    Args:
        results: Analysis results dictionary
        filename: Output filename
        
    Returns:
        Path to the saved file
    """
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    
    # Convert results to a format suitable for saving
    if isinstance(results, pd.DataFrame):
        results.to_csv(output_path, index=False)
    else:
        # Convert dictionary to DataFrame for saving
        result_rows = []
        for key, value in _flatten_dict(results).items():
            result_rows.append({'metric': key, 'value': value})
        
        pd.DataFrame(result_rows).to_csv(output_path, index=False)
    
    logger.info(f"Saved analysis results to {output_path}")
    return output_path


def _flatten_dict(d: Dict, parent_key: str = '') -> Dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    
    return dict(items)
