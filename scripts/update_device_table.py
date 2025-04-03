# update_device_table.py
from src.parquet_data_manager import ParquetExperimentalDataManager
import polars as pl

def update_device_table():
    # Initialize data manager
    data_manager = ParquetExperimentalDataManager()
    
    # Get all unique device IDs
    device_ids = data_manager.get_runs_by_criteria({"device_id": ["%"]})
    device_ids = list(set([run.split("_")[0].replace("D", "") for run in device_ids]))
    
    print(f"Found {len(device_ids)} devices to analyze")
    
    # Import the find_p_ap_states function
    from src.analysis.processing import find_p_ap_states
    
    # Process each device
    for device_id in device_ids:
        # Check if device already in table
        device_params = data_manager.get_device_parameters(device_id)
        if device_params:
            print(f"Device {device_id} already analyzed, skipping")
            continue
            
        print(f"Analyzing device {device_id}...")
        
        # Get runs for this device
        run_ids = data_manager.get_runs_by_criteria({'device_id': device_id})
        if not run_ids:
            print(f"No runs found for device {device_id}")
            continue
            
        # Get measurements for first 10 runs directly
        measurements = data_manager.get_measurements(run_ids[:10])
        
        if measurements.is_empty():
            print(f"No measurement data found for device {device_id}")
            continue
            
        # Stack measurements
        stacked_data = data_manager.stack_measurements(measurements)
        
        # Calculate threshold and peaks
        try:
            threshold, peaks, tmr = find_p_ap_states(stacked_data)
            
            # Save to device table
            device_params = {
                'device_id': device_id,
                'p_state_voltage': float(peaks[0]),
                'ap_state_voltage': float(peaks[1]),
                'threshold': float(threshold),
                'tmr': float(tmr)
            }
            
            data_manager.update_device_parameters(device_params)
            print(f"Successfully analyzed device {device_id}: threshold={threshold:.6f}, TMR={tmr:.6f}")
            
        except Exception as e:
            print(f"Error analyzing device {device_id}: {str(e)}")
            
    print("Device table update complete")

if __name__ == "__main__":
    update_device_table()