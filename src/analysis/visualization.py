# src/analysis/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

__all__ = [
    'visualize_peaks_polars',
    'plot_binomial_histogram',
    'plot_switching_scatter',
    'plot_comparison_subplots'
]

def plot_binomial_histogram(values, ntrials, bins=None, plot_probabilities=False, 
                          binomial_pdf=True, binomial_p=None, density=True, align='left', title=None, ax=None, 
                          fontsize=14, hide_ylabel=False, show_plot=True, show_legend=False, show_std=True,
                          color='b', alpha=0.8, label=None):
    """
    Plot histogram of binomial data with optional binomial PDF overlay.
    
    Args:
        ax: matplotlib Axes object. If None, creates new figure.
        fontsize: Font size for labels and legends
        hide_ylabel: If True, don't show y-axis label (useful for subplots)
        Other args same as before...
    
    Returns:
        tuple: (fig, ax) - Figure and Axes objects
    """
    # Create figure/axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        meanstd_color = 'r' #set color differently if there is only one histogram on the axis
    else:
        fig = ax.get_figure()
        meanstd_color = color
        show_plot = False
    
    if plot_probabilities and np.all(values >= 1):
        values = values / len(values)
    if np.all(values < 1):
        plot_probabilities = True
    
    mean = np.mean(values)
    std = np.std(values)
    
    if not bins and not plot_probabilities:
        bins = np.arange(np.min(values), np.max(values)+1)
    
    if color:
        n, _, _ = ax.hist(values, bins=bins, alpha=alpha, color=color, 
                        align='left', density=density)
    else:
        n, _, _ = ax.hist(values, bins=bins, alpha=alpha, 
                        align='left', density=density)
    
    # Set labels and title
    if plot_probabilities:
        if title is None:
            ax.set_title(f'Histogram of switching probability over {str(ntrials)} trials', 
                        fontsize=fontsize)
        ax.set_xlabel('Switching probability', fontsize=fontsize)
        ax.axvline(mean, color=meanstd_color, linestyle='dashed', linewidth=1, 
                   label=f'Mean = {mean:.3f}')
        if show_std:
            ax.axvspan(mean - std, mean + std, color=meanstd_color, alpha=0.2, 
                    label=f'Std = {std:.3f}')
    else:
        if title is None:
            ax.set_title(f'Histogram of successful switching attempts over {str(ntrials)} trials',
                        fontsize=fontsize)
        ax.set_xlabel('Successes', fontsize=fontsize)
        ax.axvline(mean, color=meanstd_color, linestyle='dashed', linewidth=3, 
                   label=f'Mean = {mean:.3f}')
        if show_std:
            ax.axvspan(mean - std, mean + std, color=meanstd_color, alpha=0.2, 
                    label=f'Std = {std:.3f}')
    
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    
    if not hide_ylabel:
        ax.set_ylabel('Frequency', fontsize=fontsize)
    
    # Add binomial PDF if requested
    if binomial_pdf:
        if plot_probabilities:
            bins_for_pdf = bins / ntrials if hasattr(bins, '__len__') else bins / ntrials
        else:
            bins_for_pdf = bins
            
        if hasattr(bins_for_pdf, '__len__'):
            prefactor = comb(ntrials, bins_for_pdf)
            if binomial_p:
                p = binomial_p
            else:
                p = mean / ntrials if not plot_probabilities else mean
            pdf = prefactor * (np.power(p, bins_for_pdf) * np.power(1 - p, ntrials - bins_for_pdf))
            ax.plot(bins_for_pdf if plot_probabilities else bins, pdf * np.sum(n), 
                    'k-.', label=f'Binomial pdf: n={ntrials}, p={p:.3f}', linewidth=5)
    
    # Legend on plot
    ax.legend(fontsize=fontsize-4, loc='upper left')
    # Legend above plot
    #ax.legend(fontsize=fontsize-4, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=1)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_switching_scatter(data, ntrials, markersize=None, title=None, ax=None,
                         fontsize=14, hide_ylabel=False, verbose=True):
    """
    Plot switching probability scatter plot over trials.
    
    Args:
        ax: matplotlib Axes object. If None, creates new figure.
        fontsize: Font size for labels and legends
        hide_ylabel: If True, don't show y-axis label (useful for subplots)
        verbose: If True, print statistics to console
        Other args same as before...
    
    Returns:
        tuple: (fig, ax) - Figure and Axes objects
    """
    # Create figure/axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True
    else:
        fig = ax.get_figure()
        show_plot = False
    
    # Process data
    if data.dtype == np.uint8:
        nsubsets = int(len(data) / ntrials)
        data = np.sum(data[0:len(data) * ntrials].reshape((nsubsets, ntrials)), axis=1) / ntrials
    
    sp_mean = np.mean(data)
    sp_std = np.std(data)
    sp_median = np.median(data)
    
    if sp_mean < 1:
        p = sp_mean
    else:
        p = sp_mean / ntrials
    expected_std = np.sqrt(sp_mean * (1 - sp_mean) / ntrials)
    
    x = np.arange(len(data)) * ntrials / 1_000_000
    
    # Smart markersize calculation based on data density
    if markersize is None:
        # Base size on number of points, but keep reasonable bounds
        base_size = max(0.5, min(8, -2e-4 * len(data) + 4))
        markersize = base_size
    
    # Print statistics (optional)
    if verbose:
        print(f"Mean switching probability: {sp_mean:.5f} +/- {sp_std:.5f}")
        print(f"Median switching probability: {sp_median:.5f}")
        print(f"Expected standard deviation: {expected_std:.5f}")
        print(f"Markersize = {markersize}")
    
    # Create plot
    ax.plot(x, data, 'bo', markersize=markersize)
    ax.set_xlabel('Trial Number (millions)', fontsize=fontsize)
    
    if np.all(data < 1):
        ax.hlines(sp_mean, 0, np.max(x), colors='r', linestyles='dashed', 
                  label=f"Mean = {sp_mean*100:.2f}%")
        ax.axhspan(sp_mean - sp_std, sp_mean + sp_std, color='r', alpha=0.1, 
                   label=f"Std = {sp_std*100:.2f}%")
        if not hide_ylabel:
            ax.set_ylabel('Switching Probability', fontsize=fontsize)
    else:
        ax.hlines(sp_mean, 0, np.max(x), colors='r', linestyles='dashed', 
                  label=f"Mean = {sp_mean:.1f}", linewidth=5)
        ax.axhspan(sp_mean - sp_std, sp_mean + sp_std, color='r', alpha=0.1, 
                   label=f"Std = {sp_std:.1f}")
        if not hide_ylabel:
            ax.set_ylabel('Successes', fontsize=fontsize)
    
    if title is None:
        if np.all(data < 1):
            ax.set_title(f'Switching probability over {str(int(ntrials))} trials',
                        fontsize=fontsize)
        else:
            ax.set_title(f"Number of successes over {str(int(ntrials))} trials",
                        fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize)
    
    ax.legend(fontsize=fontsize-2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_comparison_subplots(data1, data2, ntrials, plot_type='histogram', 
                           titles=None, figsize=(12, 10), sharey=True, 
                           fontsize=14, **kwargs):
    """
    Convenience function to plot two datasets in vertically aligned subplots.
    
    Args:
        data1, data2: Data arrays for top and bottom plots
        ntrials: Number of trials for both datasets
        plot_type: 'histogram' or 'scatter'
        titles: List of titles for subplots [top_title, bottom_title]
        figsize: Figure size tuple
        sharey: Bool, whether to share y-axis between subplots (default True)
        fontsize: Font size for labels and legends
        **kwargs: Additional arguments passed to plotting functions
    
    Returns:
        tuple: (fig, axes) where axes is [ax1, ax2] for top and bottom
    """
    
    if plot_type == 'histogram':
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        # Left plot - keep y-label
        plot_binomial_histogram(data1, ntrials, ax=axes[0], 
                              title=titles[0] if titles else None, 
                              fontsize=fontsize, hide_ylabel=False, **kwargs)
        # Right plot - hide y-label since it's shared
        plot_binomial_histogram(data2, ntrials, ax=axes[1], 
                              title=titles[1] if titles else None, 
                              fontsize=fontsize, hide_ylabel=True, **kwargs)
        
    elif plot_type == 'scatter':
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        # Left plot - keep y-label
        plot_switching_scatter(data1, ntrials, ax=axes[0], 
                             title=titles[0] if titles else None, 
                             fontsize=fontsize, hide_ylabel=False, **kwargs)
        # Right plot - hide y-label since it's shared
        plot_switching_scatter(data2, ntrials, ax=axes[1], 
                             title=titles[1] if titles else None, 
                             fontsize=fontsize, hide_ylabel=True, **kwargs)
    
    plt.tight_layout()
    return fig, axes


# Keep original function for backward compatibility
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
    plt.legend()
    
    return fig