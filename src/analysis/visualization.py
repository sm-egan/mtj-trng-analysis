# src/analysis/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_peaks_polars(df, threshold=None, peaks=None, sample_size=100000, bins=100):
    """
    Plot a histogram of data with threshold and peaks marked.
    
    Args:
        df: Polars DataFrame with 'device_voltage' column
        threshold: Optional voltage threshold (will be calculated if None)
        peaks: Optional peaks array (will be calculated if None)
        sample_size: Number of samples to use if data is large
        bins: Number of histogram bins
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    from src.analysis.processing import find_p_ap_states
    
    # Sample data if it's very large
    if len(df) > sample_size:
        sample_df = df.sample(sample_size)
    else:
        sample_df = df
    
    # Get voltage data as numpy array for plotting
    voltage_data = sample_df["device_voltage"].to_numpy()
    
    # Calculate threshold and peaks if not provided
    if threshold is None or peaks is None:
        threshold, peaks, _ = find_p_ap_states(df, sample_size)
    
    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    plt.hist(voltage_data, bins=bins, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}V')
    plt.axvline(peaks[0], color='g', linestyle='-', label=f'Peak 1: {peaks[0]:.6f}V')
    plt.axvline(peaks[1], color='g', linestyle='-', label=f'Peak 2: {peaks[1]:.6f}V')
    plt.xlabel('Voltage')
    plt.ylabel('Frequency')
    plt.title('MTJ Device Voltage Distribution')
    plt.legend()
    
    return fig