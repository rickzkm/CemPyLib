"""
Configuration settings for the meteor data processing project.
"""
import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File patterns
UNIFIED_PATTERN = "*unified.csv"
ALL_PATTERN = "*all.csv"

# Column configurations
EXPECTED_COLUMNS = 106  # Based on your CSV metadata

# Processing settings
BATCH_SIZE = 10000  # Process data in batches for memory efficiency
