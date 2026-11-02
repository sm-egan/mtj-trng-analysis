# MTJ-TRNG Analysis Setup Guide

This repository contains tools for analyzing MTJ-TRNG experimental data from Google Drive.

## Overview

The system imports and processes experimental voltage measurements from Magnetic Tunnel Junction (MTJ) devices used for True Random Number Generation (TRNG). Data is stored in Apache Parquet file format for high compression and fast retrieval of numerical data.

## Prerequisites

- Python 3.8 or higher
- Access to the Google Drive folder containing experimental data
- Git (for cloning the repository)

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone [repository-url]
cd mtj-trng-analysis

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Google Drive Setup

1. Request access to the experiment data folder from the project admin if you don't already have it
2. Set up Google Drive API credentials (two options):

   **Option A: Use existing credentials**
   - Ask the project admin for the service_account.json file
   - Place it in the config directory

   **Option B: Create your own credentials**
   - Go to the Google Cloud Console
   - Create a new project or select an existing one
   - Enable the Google Drive API for your project
   - Create a Service Account:
     - Go to "IAM & Admin" > "Service Accounts"
     - Click "Create Service Account"
     - Give it a name and description
     - Grant it Basic > Editor role
     - Click "Create Keys" and select JSON format
     - Save the downloaded JSON file as service_account.json in the config directory

3. Share your Drive folder with the service account email (found in the JSON file)

4. Create your configuration file:
   ```bash
   # Copy the template
   cp config/config.template.json config/config.json
   ```

   Edit config.json to include:
   ```json
   {
     "drive": {
       "credentials_path": "config/service_account.json",
       "default_folder_id": "your_folder_id_here"
     },
     "database": {
       "path": "database/mtj_switching_data_parquet.db"
     }
   }
   ```

### 3. Running Data Import (Recommended Parquet Method)

The Parquet storage method is strongly recommended as it offers:

- Significantly reduced storage requirements
- Much faster data access for analysis
- Better organization of large datasets

```bash
# Make sure your virtual environment is activated
python scripts/import_drive_data_parquet.py
```

Optional parameters:
```bash
# Specify a different folder ID
python scripts/import_drive_data_parquet.py --folder_id "your_alternative_folder_id"

# Import a specific number of files (useful for testing)
python scripts/import_drive_data_parquet.py --n_files 10

# Import randomly selected files instead of first N files
python scripts/import_drive_data_parquet.py --n_files 10 --shuffle_files True

# Enable debug logging
python scripts/import_drive_data_parquet.py --logging_level DEBUG
```

### Alternative: Legacy SQLite Import Method

The original SQLite storage method has been preserved for backward compatibility or when memory usage is a concern.

```bash
# Make sure your virtual environment is activated
python scripts/import_drive_data.py
```

This method stores all data directly in a SQLite database rather than using Parquet files. It may be useful if:

- You're running on a system with very limited disk space
- You need compatibility with older analysis tools
- You're dealing with a very small dataset where the performance benefits are minimal

### 4. Opening Jupyter Notebook for Data Analysis

The repository ships a single template notebook, `notebooks/TEMPLATE_device_analysis.ipynb`, with the standard imports, plot style config, and data-loading cells for a per-device analysis. Copy it rather than editing it directly:

```bash
# Register the kernel for Jupyter
python -m ipykernel install --user --name=venv

# Start a new analysis from the template
cp notebooks/TEMPLATE_device_analysis.ipynb notebooks/Device_011_08.ipynb
jupyter notebook notebooks/Device_011_08.ipynb
```

Everything in `notebooks/` except the template is gitignored, so your working notebooks stay local. If you improve the shared setup cells, make the change in the template and commit it.

## Analyzing Data

The ParquetExperimentalDataManager class provides an API for loading and analyzing experimental data:

```python
from src.parquet_data_manager import ParquetExperimentalDataManager

# Initialize the data manager
data_manager = ParquetExperimentalDataManager()

# Get measurements for specific run IDs
run_ids = data_manager.get_runs_by_criteria({'device_id': '023_07'})
measurements = data_manager.get_measurements(run_ids[:10])

# Stack measurements for analysis
stacked_measurements = data_manager.stack_measurements(measurements)

# Access the analysis tools
from src.analysis.processing import find_p_ap_states, calculate_switching_probability

# Find P and AP states
threshold, peaks, tmr = find_p_ap_states(stacked_measurements)

# Calculate switching probabilities for each run
switching_df = calculate_switching_probability(stacked_measurements, group_by='run_id')
```

## Troubleshooting

Common issues:

- If you see credential errors, check that service_account.json is in the correct location
- If the script can't find files, verify your folder ID is correct
- For environment issues, ensure you've activated the virtual environment before running scripts

## Questions?

Contact the project admin for assistance or access requests.
