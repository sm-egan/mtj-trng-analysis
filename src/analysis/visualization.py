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

def plot_binomial_histogram(values, ntrials, bins=None, plot_probabilities=False, binomial_pdf=True, align='left', title=None):
    if plot_probabilities and np.all(values >= 1):
        values = values / len(values)
    if np.all(values < 1):
        plot_probabilities = True #Set correct labels for switching probability

    mean = np.mean(values)
    std = np.std(values)

    if not bins and not plot_probabilities:
        bins = np.arange(np.min(values), np.max(values)+1)
    
    n, _, _, = plt.hist(values, bins=bins, alpha=0.7, color='blue', align='left')
    
    if plot_probabilities:
        if title is None:
            plt.title(f'Histogram of switching probability over {str(ntrials)} trials')
        plt.xlabel('Switching probability')
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean = {mean:.3f}%')
        plt.axvspan(mean - std, mean + std, color='r', alpha=0.2, label=f'Std = {std:.3f}%')
    else:
        if title is None:
            plt.title(f'Histogram of successful switching attempts over {str(ntrials)} trials')
        plt.xlabel('Successes')
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean = {mean:.3f}')
        plt.axvspan(mean - std, mean + std, color='r', alpha=0.2, label=f'Std = {std:.3f}')
    
    if title is not None:
        plt.title(title)

    plt.ylabel('Frequency')

    if binomial_pdf:
        if plot_probabilities:
            bins = bins / ntrials
        prefactor = comb(ntrials, bins)
        pdf = prefactor * (np.power(mean / ntrials, bins) * np.power(1 - mean / ntrials, ntrials - bins))
        plt.plot(bins, pdf * np.sum(n), 'k-.', label='Binomial distribution')

    plt.legend()
    plt.show()

def plot_switching_scatter(data, ntrials, markersize=None, title=None):
    '''

    '''
    if data.dtype == np.uint8:
        # If data is just an array of binary values (0's and 1's), convert to switching probability
        nsubsets = int(len(data) / ntrials)
        data = np.sum(data[0:len(data) * ntrials].reshape((nsubsets, ntrials)), axis = 1) / ntrials

    sp_mean = np.mean(data)
    sp_std = np.std(data)
    sp_median = np.median(data)

    if sp_mean < 1:
        p = sp_mean
    else:
        p = sp_mean / ntrials
    expected_std = np.sqrt(sp_mean * (1 - sp_mean) / ntrials)

    x = np.arange(len(data)) * ntrials
    if markersize is None:
        markersize = -3e-4 * len(data) + 5 # set markersize based on the number of data points
    

    print(f"Mean switching probability: {sp_mean:.5f} +/- {sp_std:.5f}")
    print(f"Median switching probability: {sp_median:.5f}")
    print(f"Expected standard deviation: {expected_std:.5f}")

    fig = plt.figure()
    plt.plot(x, data, 'o', markersize=markersize if markersize >= 1 else 1)
    plt.xlabel('Trial Number')
    if np.all(data < 1):
        plt.hlines(sp_mean, 0, np.max(x), colors='r', linestyles='dashed', label=f"Mean = {sp_mean*100:.2f}%")
        plt.axhspan(sp_mean - sp_std, sp_mean + sp_std, color='r', alpha=0.1, label=f"Std = {sp_std*100:.2f}%")
        plt.ylabel('Switching Probability')
    else:
        plt.hlines(sp_mean, 0, np.max(x), colors='r', linestyles='dashed', label=f"Mean = {sp_mean:.1f}")
        plt.axhspan(sp_mean - sp_std, sp_mean + sp_std, color='r', alpha=0.1, label=f"Std = {sp_std:.1f}")
        plt.ylabel('Number of successes')
    if title is None:
        if np.all(data < 1):
            plt.title(f'Switching probability over {str(int(ntrials))} trials')
        else:
            plt.title(f"Number of successes over {str(int(ntrials))} trials")
    else:
        plt.title(title)

    plt.legend()
    fig.show()

    print(f"Markersize = {markersize}")