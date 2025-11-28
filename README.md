# CemPyLib - Meteor Data Processing Library

A comprehensive Python project for processing, analyzing, and visualizing single and unified meteor observing data from CSV files.

## Project Overview

CemPyLib provides a structured framework for working with meteor observation data. It includes:

- Data loading and validation for unified and all meteor datasets
- Data cleaning and processing with batch support for large files
- Statistical analysis and meteor stream identification
- Visualization of meteor characteristics and orbital elements
- Memory optimization for efficient processing
- Multiple output formats (CSV, Parquet, JSON)

## Project Structure

```
CemPyLib/
│
├── data/                   # Data directory
│   ├── raw/                # Place your CSV files here
│   │   ├── *_unified.csv   # Unified meteor observations
│   │   └── *_all.csv       # All meteor observations
│   └── processed/          # Output will go here
│
├── src/                    # Source code
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Functions to load and validate CSV data
│   ├── data_processor.py   # Core data processing functions
│   ├── data_analyzer.py    # Analysis functions
│   ├── data_visualizer.py  # Visualization functions
│   └── utils.py            # Utility functions
│
├── scripts/
│   └── run_processing.py   # Main script to run the complete pipeline
│
├── tests/                  # Unit tests (for future development)
│
├── requirements.txt        # Python dependencies
└── readme.md               # This file
```

## Installation

1. Clone or download the repository:
   ```bash
   git clone https://github.com/yourusername/CemPyLib.git
   cd CemPyLib
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If using Anaconda/conda, you can also install with:
   ```bash
   conda install pandas numpy scipy matplotlib seaborn scikit-learn pyarrow tqdm joblib
   ```

3. Verify installation:
   ```bash
   python scripts/run_processing.py --help
   ```

## Usage

### Basic Usage

1. Place your meteor CSV files in the `data/raw/` directory.

2. Run the processing pipeline:
   ```
   python scripts/run_processing.py
   ```

3. View the results in the `data/processed/` directory.

### Command Line Options

The main script supports various command line options:

```
python scripts/run_processing.py --help
```

Common options:

- `--input-dir PATH`: Custom input directory
- `--output-dir PATH`: Custom output directory
- `--batch-size N`: Process data in batches of size N
- `--skip-cleaning`: Skip data cleaning step
- `--skip-visualization`: Skip visualization generation
- `--optimize-memory`: Optimize DataFrame memory usage
- `--stream-filter STREAM`: Filter data by specific meteor stream
- `--save-parquet`: Save processed data in Parquet format

## Output Files

The processing pipeline generates several output files in `data/processed/`:

1. **Processed CSV files**: Cleaned and processed meteor data
   - `unified_YYYYMMDD_HHMMSS_*.csv` - Processed unified observations
   - `all_YYYYMMDD_HHMMSS_*.csv` - Processed all observations

2. **Statistics JSON**: Statistical summary of the data
   - `statistics_YYYYMMDD_HHMMSS.json` - Descriptive statistics, distribution metrics

3. **Stream Analysis**: Analysis of meteor streams
   - `stream_analysis_YYYYMMDD_HHMMSS.csv` - Stream counts and statistics

4. **Outlier Detection**: Identified outliers in the data
   - `outliers_YYYYMMDD_HHMMSS.csv` - Flagged outliers for key parameters

5. **Visualizations** (if enabled):
   - `meteor_distribution.png` - Sky distribution (RA/Dec)
   - `velocity_distribution.png` - Velocity histograms and box plots
   - `orbital_elements.png` - Orbital parameter analysis
   - `stream_analysis.png` - Meteor stream comparisons
   - `report.html` - Interactive HTML report with all visualizations

## Data Format

The project expects CSV files with 106 columns of meteor data including:

**Key Columns:**
- `_Version`, `_#` - Version and identifier
- `_localtime`, `_mjd`, `_sol` - Time information
- `_ID1`, `_ID2` - Observer identifiers
- `_amag` - Apparent magnitude
- `_ra_o`, `_dc_o` - Right ascension and declination (observed)
- `_ra_t`, `_dc_t` - Right ascension and declination (theoretical)
- `_vg`, `_vo`, `_vi`, `_vs` - Velocity measurements
- `_a`, `_q`, `_e`, `_incl`, `_peri`, `_node` - Orbital elements
- `_stream` - Meteor stream identifier

**File Naming Convention:**
- Unified observations: `*_unified.csv` (e.g., `U2_20240105_unified.csv`)
- All observations: `*_all.csv` (e.g., `U2_20240105_all.csv`)

## Troubleshooting

**ImportError: No module named 'pandas'**
- Solution: Install dependencies with `pip install -r requirements.txt`

**FileNotFoundError: No CSV files found**
- Solution: Ensure CSV files are placed in `data/raw/` directory
- Check that files follow naming convention (`*_unified.csv` or `*_all.csv`)

**MemoryError: Large datasets**
- Solution: Use `--batch-size` option to process in smaller chunks
- Example: `python scripts/run_processing.py --batch-size 5000`

**Visualization issues**
- Solution: Use `--skip-visualization` to skip plots if matplotlib has issues
- Example: `python scripts/run_processing.py --skip-visualization`

## Project Improvements (v2024.11)

This version includes several fixes and improvements:
- Fixed file naming inconsistencies (hyphen vs underscore)
- Added missing pandas import in main script
- Fixed missing return statement in `optimize_dataframe()` function
- Completed incomplete `plot_velocity_distribution()` function
- Fixed column name issue in `analyze_meteor_streams()` function
- Updated README with accurate instructions and troubleshooting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
