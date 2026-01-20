# src/analysis/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

__all__ = [
    'visualize_peaks_polars',
    'plot_binomial_histogram',
    'plot_switching_scatter',
    'plot_comparison_subplots',
    'plot_entropy_contour',
    'plot_entropy_lines',
    'plot_emphasis_circles'
]

def plot_binomial_histogram(values, ntrials, bins=None, plot_probabilities=False, 
                          binomial_pdf=True, binomial_p=None, density=True, align='left', title=None, ax=None, 
                          fontsize=16, hide_ylabel=False, show_plot=True, show_std=True,
                          color=None, alpha=0.8, label=None, linewidth=None, legend_ncols=1):
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
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = fontsize

    # Create figure/axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
        show_plot = False
    
    mean = np.mean(values)
    std = np.std(values)

    if not bins:
        bins = np.arange(np.min(values), np.max(values)+1)

    if np.all(values < 1):
        plot_probabilities = True
    else:
        # If values are all greater than 1 and you wish to plot probabilities, divide by ntrials
        if plot_probabilities:
            values = values / ntrials
            bins = bins / ntrials

    if not color:
        hist_color = 'blue'
        meanstd_color = 'red'
    else:
        hist_color = color
        meanstd_color = color
    
    n, _, _ = ax.hist(values, bins=bins, alpha=alpha, color=hist_color, 
                    align='left', density=density, label=label)
    
    # Set labels and title
    if plot_probabilities:
        if title is None:
            ax.set_title(f'Histogram of switching probability over {str(ntrials)} trials', 
                        fontsize=fontsize, fontweight='bold'
                        )
        ax.set_xlabel('Switching probability', fontsize=fontsize)
        ax.axvline(mean, color=meanstd_color, linestyle='dashed', linewidth=linewidth if linewidth else 1, 
                   label=f'Mean = {mean:.3f}')
        if show_std:
            ax.axvspan(mean - std, mean + std, color=meanstd_color, alpha=0.2, 
                    label=r'$\sigma$=' + f'{std:.1f}')
    else:
        if title is None:
            ax.set_title(f'Histogram of successful switching attempts over {str(ntrials)} trials',
                        fontsize=fontsize, fontweight='bold')
        ax.set_xlabel('Successes', fontsize=fontsize, fontweight='bold')
        ax.axvline(mean, color=meanstd_color, linestyle='dashed', linewidth=linewidth*3 if linewidth else 3, 
                   label=f'Mean = {int(mean)}')
        if show_std:
            ax.axvspan(mean - std, mean + std, color=meanstd_color, alpha=0.2, 
                    label=r'$\sigma$=' + f'{std:.1f}')
    
    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontweight='bold')
    
    if not hide_ylabel:
        ax.set_ylabel('Frequency', fontsize=fontsize, fontweight='bold')
    
    # Add binomial PDF if requested
    if binomial_pdf:
        from scipy.stats import binom
        if binomial_p:
            p = binomial_p
        else:
            # If binomial_p is not given as kwarg, set it to the mean
            p = mean / ntrials if not plot_probabilities else mean

        if p == 0.5:
                binomial_label = binomial_label = f'$B({ntrials}, {p:.1f})$'
        else:
            binomial_label = f'$B({ntrials}, {p:.4f})$'
        
        #Set bins for the binomial pdf to the same range as hist bins
        bins_for_pdf = bins if not plot_probabilities else (bins * ntrials).astype(int)
            
        if hasattr(bins_for_pdf, '__len__'):
            #prefactor = comb(ntrials, bins_for_pdf)
            pdf = binom.pmf(bins_for_pdf, ntrials, p)
            #prefactor * (np.power(p, bins_for_pdf) * np.power(1 - p, ntrials - bins_for_pdf))
            ax.plot(bins, pdf, 'k-.', label=binomial_label, linewidth=linewidth*5 if linewidth else 5)
    
    # Legend on plot
    ax.legend(fontsize=fontsize-2, loc='upper left', ncol=legend_ncols)
    # Legend above plot
    #ax.legend(fontsize=fontsize-4, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=1)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_switching_scatter(data, ntrials, markersize=None, title=None, ax=None,
                         fontsize=14, hide_ylabel=False, verbose=True, linewidth=None, legend_loc='lower left',
                         edgecolor=None):
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
    
    ax.scatter(x, data, s=markersize, color='b', marker='o', edgecolors=edgecolor, linewidth=0.5)
    ax.set_xlabel(r'Sample Number ($10^6$)', fontsize=fontsize, fontweight='bold')
    
    if np.all(data < 1):
        ax.hlines(sp_mean, 0, np.max(x), colors='r', linestyles='dashed', 
                  label=f"Mean = {sp_mean:.4f}", linewidth=linewidth if linewidth else 5)
        ax.axhspan(sp_mean - sp_std, sp_mean + sp_std, color='r', alpha=0.1, 
                   label=r'$\sigma$=' + f"{sp_std:.4f}")
        if not hide_ylabel:
            ax.set_ylabel('Switching Probability', fontsize=fontsize, fontweight='bold')
    else:
        ax.hlines(sp_mean, 0, np.max(x), colors='r', linestyles='dashed', 
                  label=f"Mean = {int(sp_mean)}", linewidth=linewidth if linewidth else 5)
        ax.axhspan(sp_mean - sp_std, sp_mean + sp_std, color='r', alpha=0.1, 
                   label=r'$\sigma$=' + f'{sp_std:.1f}')
        if not hide_ylabel:
            ax.set_ylabel('Successes', fontsize=fontsize, fontweight='bold')
    
    if title is None:
        if np.all(data < 1):
            ax.set_title(f'Switching probability over {str(int(ntrials))} trials',
                        fontsize=fontsize, fontweight='bold')
        else:
            ax.set_title(f"Number of successes over {str(int(ntrials))} trials",
                        fontsize=fontsize, fontweight='bold')
    else:
        ax.set_title(title, fontsize=fontsize, fontweight='bold')
    
    ax.legend(fontsize=fontsize-2, loc=legend_loc)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
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
# def visualize_peaks_polars(df, threshold=None, peaks=None, sample_size=100000, bins=100):
#     """
#     Plot a histogram of data with threshold and peaks marked.
    
#     Args:
#         df: Polars DataFrame with 'device_voltage' column
#         threshold: Optional voltage threshold (will be calculated if None)
#         peaks: Optional peaks array (will be calculated if None)
#         sample_size: Number of samples to use if data is large
#         bins: Number of histogram bins
        
#     Returns:
#         matplotlib.figure.Figure: The created figure
#     """
#     import matplotlib.pyplot as plt
#     from src.analysis.processing import find_p_ap_states
    
#     # Sample data if it's very large
#     if len(df) > sample_size:
#         sample_df = df.sample(sample_size)
#     else:
#         sample_df = df
    
#     # Get voltage data as numpy array for plotting
#     voltage_data = sample_df["device_voltage"].to_numpy()
    
#     # Calculate threshold and peaks if not provided
#     if threshold is None or peaks is None:
#         threshold, peaks, _ = find_p_ap_states(df, sample_size)
    
#     # Create the plot
#     fig = plt.figure(figsize=(10, 6))
#     plt.hist(voltage_data, bins=bins, alpha=0.7, density=True)
#     plt.axvline(threshold, color='r', linestyle='-', linewidth=3, label=f'Threshold: {threshold:.4f}V')
#     plt.axvline(peaks[0], color='g', linestyle='--', linewidth=3, label=f'P: {peaks[0]:.4f}V')
#     plt.axvline(peaks[1], color='g', linestyle='-.', linewidth=3, label=f'AP: {peaks[1]:.4f}V')
#     plt.xlabel('Voltage')
#     plt.ylabel('Frequency')
#     plt.legend(bbox_to_anchor=(0.5, 1.12), loc='upper center', ncol=3)
    
#     return fig

def visualize_peaks_polars(df, threshold=None, peaks=None, sample_size=100000, bins=100, **kwargs):
    """
    Plot a histogram of data with threshold and peaks marked.
    
    Args:
        df: Polars DataFrame with 'device_voltage' column
        threshold: Optional voltage threshold (will be calculated if None)
        peaks: Optional peaks array (will be calculated if None)
        sample_size: Number of samples to use if data is large
        bins: Number of histogram bins
        **kwargs: Additional plotting parameters:
            - figsize: Figure size (default: (10, 6))
            - fontsize: Font size for labels (default: matplotlib default)
            - linewidth: Line width for threshold/peak lines (default: 3)
            - hist_alpha: Histogram transparency (default: 0.7)
            - Other kwargs passed to plt.hist()
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    from src.analysis.processing import find_p_ap_states
    
    # Extract specific kwargs with defaults
    figsize = kwargs.pop('figsize', (10, 6))
    fontsize = kwargs.pop('fontsize', None)
    linewidth = kwargs.pop('linewidth', 3)
    hist_alpha = kwargs.pop('hist_alpha', 0.7)
    density = kwargs.pop('density', True)
    legend_ncol = kwargs.pop('legend_ncol', 2)
    legendy = kwargs.pop('legendy', 1.3)
    
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
    fig = plt.figure(figsize=figsize)
    
    # Pass remaining kwargs to histogram
    plt.hist(voltage_data*1e3, bins=bins, alpha=hist_alpha, density=density, **kwargs)

    plt.axvline(peaks[0]*1e3, color='k', linestyle='--', linewidth=linewidth, label=f'P: {peaks[0]*1e3:.1f} mV')
    plt.axvline(peaks[1]*1e3, color='k', linestyle=':', linewidth=linewidth, label=f'AP: {peaks[1]*1e3:.1f} mV')  
    plt.axvline(threshold*1e3, color='r', linestyle='-', linewidth=linewidth, label=f'Threshold: {threshold*1e3:.1f} mV')
    
    plt.xlabel('Voltage (mV)', fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(bbox_to_anchor=(0.5, legendy), loc='upper center', ncol=legend_ncol, fontsize=fontsize)

    return fig


def plot_entropy_contour(X, Y, Z, cmap='plasma', vmin=None, vmax=None, levels=None,
                         ax=None, figsize=(8, 6), add_colorbar=True,
                         contour_lines=False, contour_kwargs=None, **contourf_kwargs):
    """
    Create a filled contour plot from 2D meshgrid data.

    This function is designed to work like matplotlib axis methods, creating only
    the contour plot and colorbar while leaving all other customization (labels,
    ticks, titles, etc.) to be added afterward by the user.

    Args:
        X: 2D array of x-coordinates (from np.meshgrid)
        Y: 2D array of y-coordinates (from np.meshgrid)
        Z: 2D array of z-values to plot
        cmap: Colormap name (default: 'plasma')
        vmin: Minimum value for colormap (default: None, uses data min)
        vmax: Maximum value for colormap (default: None, uses data max)
        levels: Number of contour levels or array of levels (default: None, matplotlib chooses)
                If int, creates that many levels between vmin and vmax
                If array, uses those specific level values
        ax: matplotlib Axes object. If None, creates new figure (default: None)
        figsize: Figure size tuple, only used if ax is None (default: (8, 6))
        add_colorbar: If True, adds a colorbar (default: True)
        contour_lines: If True, overlay contour lines on filled contours (default: False)
        contour_kwargs: Dict of kwargs to pass to ax.contour() for lines (default: None)
        **contourf_kwargs: Additional keyword arguments passed to ax.contourf()

    Returns:
        tuple: (fig, ax, contourf_object, colorbar_object)
            - fig: matplotlib Figure object
            - ax: matplotlib Axes object
            - contourf_object: The ContourSet from contourf()
            - colorbar_object: The Colorbar object (or None if add_colorbar=False)

    Example:
        >>> # Prepare your data
        >>> x = data.select("s_width").unique(maintain_order=True).to_numpy().T[0]
        >>> y = np.array([0, 2, 4, 8])
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = data.pivot("s_width", index="processing", values="h_final").to_numpy()[:, 1:].astype(float)
        >>>
        >>> # Create the contour plot
        >>> fig, ax, cf, cbar = plot_entropy_contour(
        ...     X*1e9, Y, Z,
        ...     vmin=0, vmax=1.0,
        ...     levels=np.linspace(0, 1.0, 51)
        ... )
        >>>
        >>> # Customize afterward
        >>> cbar.set_label('Min-entropy / bit', rotation=270, labelpad=20)
        >>> cbar.set_ticks(np.arange(0, 1.1, 0.1))
        >>> ax.set_xlabel("Perturb pulse width (ns)")
        >>> ax.set_ylabel("Number of XOR bits")
        >>> ax.set_yticks([1, 2, 4, 8])
        >>> plt.show()
    """
    # Create figure/axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Process levels parameter
    if levels is not None and isinstance(levels, int):
        # If levels is an integer, create that many levels between vmin and vmax
        if vmin is None:
            vmin = np.nanmin(Z)
        if vmax is None:
            vmax = np.nanmax(Z)
        levels = np.linspace(vmin, vmax, levels)

    # Create filled contour plot
    cf = ax.contourf(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, **contourf_kwargs)

    # Add contour lines if requested
    if contour_lines:
        if contour_kwargs is None:
            contour_kwargs = {'colors': 'k', 'linestyles': '--', 'linewidths': 0.5}
        ax.contour(X, Y, Z, levels=levels if levels is not None else cf.levels, **contour_kwargs)

    # Create colorbar if requested
    cbar = None
    if add_colorbar:
        cbar = fig.colorbar(cf, ax=ax)

    return fig, ax, cf, cbar