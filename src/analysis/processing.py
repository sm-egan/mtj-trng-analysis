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
    TO DO: Add check for sign of s_amplitude to determin if sw_prob = 1 - avg or sw_prob = avg (avg of binary values)

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

def xorn_whitening(binary_data, n=2, method="nonoverlapping", rolling_lag=1, verbose=False):
    L = len(binary_data)

    if method == "nonoverlapping":
        nout = int(L / n)

        if not (L / n).is_integer():
            binary_data = binary_data[0:nout * n]
        
        # Implement n bit XOR whitening
        xorn = np.sum(binary_data.reshape(nout, n),  axis=1) % 2

    elif method == "rolling":
        nout = L - n + 1
        
        rolling_data = binary_data.copy()
        for i in range(1, n):
            rolling_data += np.roll(binary_data, -i)
        xorn = rolling_data[:nout] % 2 
        if rolling_lag < n:
            #To implement rolling lag, just take a subset of the bitstream computed with the simple roll
            ## Why this works: We still want to use sequential bits as XOR inputs, and the loop from 1 to n with np.roll
            ## is a nice simple way to implement this. Then we control how much the windows overlap with the lag parameter
            ## if n = rolling_lag, this is equivalent to non-overlapping
            if rolling_lag > 1:
                xorn = xorn[::rolling_lag]
        else:
            print('WARNING: rolling_lag >= n, either use nonoverlapping method or decrease rolling_lag. Returning None')
            return None

    if verbose:
        print(f'Performing XOR{n} whitening')
        print('Input:')
        print(binary_data[0:10 * n])
        print('\n\nOutput:')

        print('---------------------')
        if method == "nonoverlapping":
            print(xorn[0:10])
        elif method == "rolling":
            print(xorn[0:10 * n])

    return xorn.astype(np.uint8)