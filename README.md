# MTJ-TRNG Analysis Setup Guide

This repository contains tools for analyzing MTJ-TRNG experimental data from Google Drive.

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

#Register the kernel for Jupyter:
python -m ipykernel install --user --name=venv

#Start Jupyter notebook:
jupyter notebooks/test_histogram_switching_prob.ipynb
```

### 2. Google Drive Setup

1. Request access to the experiment data folder from the project admin if you don't already have it

2. Set up Google Drive API credentials (two options):

   Option A: Use existing credentials
   - Ask the project admin for the `service_account.json` file
   - Place it in the `config` directory

   Option B: Create your own credentials
   - Go to the [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select an existing one
   - Enable the Google Drive API for your project
   - Create a Service Account:
     1. Go to "IAM & Admin" > "Service Accounts"
     2. Click "Create Service Account"
     3. Give it a name and description
     4. Grant it Basic > Editor role
     5. Click "Create Keys" and select JSON format
     6. Save the downloaded JSON file as `service_account.json` in the `config` directory
   - Share your Drive folder with the service account email (found in the JSON file)

3. Create your configuration file:
   ```bash
   # Copy the template
   cp config/config.template.json config/config.json
   ```
   
   Edit `config.json` to include:
   ```json
   {
     "drive": {
       "credentials_path": "config/service_account.json",
       "default_folder_id": "your_folder_id_here"
     },
     "database": {
       "path": "database/mtj_switching_data.db"
     }
   }
   ```

### 3. Running Data Import

```bash
# Make sure your virtual environment is activated
python scripts/import_drive_data.py
```

Optional: Specify a different folder ID:
```bash
python scripts/import_drive_data.py --folder_id "your_alternative_folder_id"
```

## Troubleshooting

Common issues:
- If you see credential errors, check that `service_account.json` is in the correct location
- If the script can't find files, verify your folder ID is correct
- For environment issues, ensure you've activated the virtual environment before running scripts

## Questions?

Contact Shannon for assistance or access requests.
