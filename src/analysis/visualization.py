# src/analysis/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

__all__ = [
    'visualize_peaks_polars',
    'plot_binomial_histogram'
]

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

def plot_binomial_histogram(values, ntrials, bins=None, normalize=False, binomial_pdf=False, align='left'):
    if normalize and np.all(values >= 1):
        values = values / len(values)
    if np.all(values < 1):
        normalize = True #Set correct labels for switching probability
    mean = np.mean(values)
    std = np.std(values)

    if not bins and not normalize:
        bins = np.arange(np.min(values), np.max(values)+1)
    
    n, _, _, = plt.hist(values, bins=bins, alpha=0.7, color='blue', align='left')
    
    if normalize:
        plt.title(f'Histogram of switching probability over {str(ntrials)} trials')
        plt.xlabel('Switching probability')
    else:
        plt.title(f'Histogram of successful switching attempts over {str(ntrials)} trials')
        plt.xlabel('Successes')
    plt.ylabel('Frequency')
    if normalize:
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean = {mean:.3f}%')
        plt.axvspan(mean - std, mean + std, color='r', alpha=0.2, label=f'Std = {std:.3f}%')
    else:
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean = {mean:.3f}')
        plt.axvspan(mean - std, mean + std, color='r', alpha=0.2, label=f'Std = {std:.3f}')

    if binomial_pdf:
        prefactor = comb(ntrials, bins)
        pdf = prefactor * (np.power(mean / ntrials, bins) * np.power(1 - mean / ntrials, ntrials - bins))
        plt.plot(bins, pdf * np.sum(n), 'k-.', label='Binomial distribution')

    plt.legend()
    plt.show()