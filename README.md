# MTJ-TRNG Analysis

Analysis code for MTJ-based true random number generators (TRNGs). This repository implements the analysis described in

> Q. Jia, S. Egan, Y. Lv and J.-P. Wang, "Widening the High-Entropy Operating Window of
> Perpendicular Magnetic Tunnel Junction TRNGs via Multi-Bit XOR Whitening,"
> *IEEE Transactions on Electron Devices*, 2026.
> [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11627766)

It processes voltage measurements from Magnetic Tunnel Junction (MTJ) devices into bitstreams, applies multi-bit XOR whitening, and runs NIST SP 800-90B entropy assessments on the results. Measurement data is stored in Apache Parquet files with run metadata in SQLite.

## Prerequisites

- Python 3.8 or higher
- The NIST [SP800-90B_EntropyAssessment](https://github.com/usnistgov/SP800-90B_EntropyAssessment) tools (`ea_iid`, `ea_non_iid`), built locally

## Setup Instructions

### 1. Environment Setup

```bash
git clone https://github.com/sm-egan/mtj-trng-analysis.git
cd mtj-trng-analysis

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configuration

```bash
cp config/config.template.json config/config.json
```

Set `ea_binaries.path` to the directory containing the built `ea_iid` and `ea_non_iid` binaries.

### 3. Getting Data

The repository includes code to import raw measurement files from Google Drive (`scripts/import_drive_data_parquet.py`, configured via the `drive` section of `config.json`). You don't need it to use the analysis: you can start from your own data instead, either

- Parquet measurement files plus run metadata, loaded through `ParquetExperimentalDataManager` (`src/parquet_data_manager.py`), or
- 1-bit `.bin` bitstreams, passed straight to the SP 800-90B entropy evaluation in `src/analysis/entropy_evaluation_pipeline.py`.

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

- `ea_iid not found`: check `ea_binaries.path` in `config/config.json` points at the built SP 800-90B tools
- For environment issues, ensure you've activated the virtual environment before running scripts
