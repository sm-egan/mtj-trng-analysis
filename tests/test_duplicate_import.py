# test_duplicate_import.py
from src.data_manager import ExperimentalDataManager
from pathlib import Path

def test_duplicate_import():
    # Initialize data manager
    dm = ExperimentalDataManager()
    
    # Try to import the sample file again
    sample_file = Path('sample_data/Hz_35__Row  270_RAmp-7.00E-1_RWid1.000000E-7_SAmp4.73E-1_SWid8.50E-9_Field0.00E+0_Source0.00E+0_Repeats1.00E+4.txt')
    
    print("\nAttempting to import potentially duplicate file...")
    run_id = dm.import_data_file(sample_file)
    
    if run_id is None:
        print("File was skipped (invalid format)")
    else:
        # Check if it's an existing run_id
        existing_runs = dm.get_runs()
        if run_id in existing_runs['run_id'].values:
            print(f"Duplicate detected! Returned existing run_id: {run_id}")
        else:
            print(f"New run created with run_id: {run_id}")
    
    # Let's also see what's in the database
    print("\nCurrent runs in database:")
    print(dm.get_runs()[['run_id', 'filename', 'hz', 'row_number']])

if __name__ == "__main__":
    test_duplicate_import()