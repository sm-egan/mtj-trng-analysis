# src/analysis/processing.py
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

def find_p_ap_states(df, sample_size=100000, init_method='maxmin'):
    """
    Robustly analyze MTJ device data to find two voltage peaks using sampling
    and better initialization.
    
    Args:
        df: Polars DataFrame with voltage data in 'device_voltage' column
        sample_size: Number of points to use for initial estimation
    
    Returns:
        tuple: (threshold, peaks, tmr)
            - threshold: Voltage threshold for bit classification
            - peaks: The two identified peak values (sorted)
            - tmr: Tunneling magnetoresistance ratio
    """
    # Take a random sample for initial estimation
    sample_df = df.sample(sample_size, seed=42)
    voltages = sample_df["device_voltage"].to_numpy().reshape(-1, 1)
    
    # Look at histogram to estimate initial centroids
    hist, bin_edges = np.histogram(voltages, bins=100)
    
    print(f"Setting initial k-means centers by method {init_method}")
    if init_method == 'max_freq':
        # Set initial centers based on the two highest peaks in the histogram
        peak_indices = np.argsort(hist)[-2:]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        initial_centers = bin_centers[peak_indices].reshape(-1, 1)
    elif init_method == 'maxmin':
        # Set initial centers to max and min of the data
        initial_centers = np.array([[voltages.min()], [voltages.max()]])
    elif init_method == 'random':
        # Set initial centers to random values in the range of the data
        initial_centers = np.random.rand(2, 1)*(voltages.max() - voltages.min()) + voltages.min()
    elif init_method == 'range_quarters':
        # Set initial centers to 25% and 75% of the range of the data
        initial_centers = np.array([[voltages.min() + 0.25*(voltages.max() - voltages.min())], 
                                     [voltages.min() + 0.75*(voltages.max() - voltages.min())]])
    
    # Use regular KMeans with these better initial centers
    kmeans = KMeans(
        n_clusters=2, 
        init=initial_centers,
        n_init=1,  # Use our initialization, not random
        random_state=42
    )
    
    kmeans.fit(voltages)
    
    # Get the two cluster centers (voltage peaks)
    peaks = kmeans.cluster_centers_.flatten()
    peaks.sort()
    
    # Calculate threshold as midpoint between peaks
    threshold = (peaks[0] + peaks[1]) / 2
    
    # Calculate TMR
    tmr = (peaks[1] - peaks[0]) / peaks[0]
    
    return threshold, peaks, tmr

# def calculate_switching_probability(measurements, threshold):
#     """
#     Calculate the switching probability of MTJ device.
    
#     Args:
#         stacked_measurements: Polars DataFrame with 'device_voltage' column 
#         threshold: Voltage threshold for bit classification
        
#     Returns:
#         float: Switching probability (0.0 to 1.0)
#     """
#     sw_prob = 1 - (measurements['device_voltage'] > threshold).sum() / len(measurements)
#     return float(sw_prob)

def calculate_switching_probability(measurements, threshold=None, group_by=None):
    """
    Calculate the switching probability of MTJ device.
    
    Args:
        measurements: Polars DataFrame with 'device_voltage' column
        threshold: Voltage threshold for bit classification (calculated if None)
        group_by: Optional column name to group by (e.g., 'run_id', 'field')
        
    Returns:
        If group_by is None:
            float: Switching probability (0.0 to 1.0)
        If group_by is specified:
            polars.DataFrame: Grouped switching probabilities with columns:
                - group_by column
                - 'switching_probability'
                - 'count' (number of measurements in group)
    """
    import polars as pl
    
    # Calculate threshold if not provided
    if threshold is None:
        threshold, _, _ = find_p_ap_states(measurements)
    
    if group_by is None:
        # Simple case - calculate overall probability
        sw_prob = 1 - (measurements['device_voltage'] > threshold).sum() / len(measurements)
        return float(sw_prob)
    else:
        # Group by case
        if not isinstance(group_by, str):
            raise ValueError("group_by must be a string column name")
            
        if group_by not in measurements.columns:
            raise ValueError(f"Column '{group_by}' not found in measurements DataFrame")
        
        # Define aggregation expression for switching probability
        result = (
            measurements
            .group_by(group_by)
            .agg([
                (1 - pl.col('device_voltage').gt(threshold).sum() / pl.count()).alias('switching_probability'),
                pl.count().alias('count')
            ])
            .sort(group_by)
        )
        
        return result