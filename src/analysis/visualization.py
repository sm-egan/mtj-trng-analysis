# src/analysis/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from cycler import cycler

__all__ = [
    'visualize_peaks_polars',
    'plot_binomial_histogram',
    'plot_switching_scatter',
    'plot_comparison_subplots',
    'plot_entropy_contour',
    'plot_entropy_lines',
    'plot_emphasis_circles',
    'plot_Sn_variance',
    'plot_entropy_vs_sp',
    'figsize_from_width_cm',
    'ENTROPY_CYCLER',
]

ENTROPY_CYCLER = (
    cycler(color=['blue', 'forestgreen', 'gold', 'hotpink', 'darkviolet',
                  'orange', 'crimson', 'teal', 'sienna']) +
    cycler(marker=['o', '^', 's', 'P', 'v', 'X', 'D', 'h', '*'])
)


def figsize_from_width_cm(width_cm, aspect=1):
    cm_to_inch = 1 / 2.54
    return (width_cm * cm_to_inch, aspect * width_cm * cm_to_inch)

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

def plot_Sn_variance(
    binary_data,
    xor_levels=(2, 4, 8, 16),
    Ns=None,
    p_hat=None,
    confidence_level=0.95,
    ax=None,
    figsize=None,
    fontsize=12.5,
    ylim=None,
    markersize=3,
    band_alpha=0.12,
    legend_ncol=3,
    legend_kwargs=None,
    normalize='ratio',
    point_alpha=1.0,
    yscale='linear',
):
    """
    Plot sample variance normalized by expected binomial variance across trial sizes,
    for raw binary data and XOR-whitened versions, with confidence bands.

    A value of 1.0 means the data behaves like ideal Bernoulli trials at that
    trial size. Deviations outside the confidence band indicate statistically
    significant correlations or bias.

    The null hypothesis throughout is S_n ~ B(n, 1/2). XOR whitening enforces
    this: XORing k independent bits produces a 0 or 1 with equal probability
    as long as the underlying bits are uncorrelated, regardless of any
    individual bit bias. The raw sequence is included for comparison but its
    null variance uses p_hat rather than 1/2.

    When normalize='ratio' (default), the y-axis shows the raw variance ratio
    Var_hat / Var_true with per-curve chi-squared CI bands. Each processing
    level has its own band width because XOR whitening reduces the sequence
    length and therefore the number of blocks k.

    When normalize='zscore', the y-axis is z-scored as
    (R - 1) / sqrt(2 / (k-1)), which is approximately N(0, 1) under the null
    for all processing levels. This makes the confidence band a single shared
    band at ±z_crit (normal approximation), allowing fair visual comparison
    across XOR levels with different sample counts.

    Parameters
    ----------
    binary_data : np.ndarray (uint8)
        Raw binary sequence (0s and 1s).
    xor_levels : tuple of int, default (2, 4, 8, 16)
        XOR whitening levels to compute and plot. Each value n means
        "XOR together n consecutive bits."
    Ns : np.ndarray, optional
        Array of trial sizes for variance computation.
        Default: np.arange(100, 10001, 100).
    p_hat : float, optional
        Estimated probability of bit=1. If None, computed as np.mean(binary_data).
    confidence_level : float, default 0.95
        Confidence level for the bands (0.95 = 95% CI).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (only used if ax is None).
    fontsize : float, default 12.5
        Font size for labels and legend.
    ylim : tuple, optional
        Y-axis limits. If None (default), auto-scaled to the data.
    markersize : float, default 3
        Marker size for data points.
    band_alpha : float, default 0.12
        Transparency for confidence band shading. Kept low since multiple
        bands overlap.
    legend_ncol : int, default 3
        Number of columns in the legend.
    legend_kwargs : dict, optional
        Override any other default legend keyword arguments.
    normalize : {'ratio', 'zscore', 'variance'}, default 'ratio'
        'ratio'    — plot raw variance ratio R = Var_hat / Var_true with
                     per-curve chi-squared CI bands (original behavior).
        'zscore'   — plot (R-1) / sqrt(2/(k-1)), i.e. standard deviations
                     from ideal Bernoulli. The confidence band becomes a
                     single shared band at ±z_crit for all processing levels,
                     enabling fair comparison across XOR degrees.
        'variance' — plot raw sample variance Var(Sn) directly, with
                     per-curve chi-squared CI bands scaled to variance units.
                     Reference curve is the expected binomial variance
                     n·p̂·(1−p̂), which grows linearly with n.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    stats_dict : dict
        Computed statistics for each processing level. Structure::

            {
                'raw': {
                    'Ns': array, 'means': array, 'vars': array,
                    'medians': array, 'normalized_vars': array,
                    'z_scores': array,   # (R-1) / sqrt(2/(k-1))
                    'n_samples': int, 'n_blocks': array,
                    'ci_lower': array, 'ci_upper': array,
                },
                'xor02': { ... same structure ... },
                ...
                'reference': {
                    'Ns': array, 'p_hat': float, 'binomial_var': array,
                },
                'z_crit': float,   # normal-approx CI threshold (e.g. 1.96)
            }

    Example
    -------
    >>> fig, ax, stats = plot_Sn_variance(binary_data, xor_levels=[2, 4, 8])
    >>> fig.savefig(figpath / reg.figname(FID, "norm_variance"))
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import chi2, norm as scipy_norm
    from src.analysis.processing import xorn_whitening
    from src.analysis.stats import (
        get_bernoulli_trials_statistics,
        binomial_var,
    )

    if Ns is None:
        Ns = np.arange(100, 10001, 100)

    if p_hat is None:
        p_hat = float(np.mean(binary_data))

    # Chi-squared quantiles for confidence bands
    # The normalized ratio (sample_var / true_var) follows (1/(k-1)) * chi^2_{k-1}
    # So CI for the ratio is: [chi2.ppf(alpha/2, k-1)/(k-1), chi2.ppf(1-alpha/2, k-1)/(k-1)]
    alpha = 1 - confidence_level

    def compute_ci_bounds(n_blocks_array):
        """Compute CI bounds on the normalized variance ratio for each trial size."""
        dof = n_blocks_array - 1  # degrees of freedom = k - 1
        ci_lower = chi2.ppf(alpha / 2, dof) / dof
        ci_upper = chi2.ppf(1 - alpha / 2, dof) / dof
        return ci_lower, ci_upper

    def compute_stats_for_array(data_array, label):
        """Compute variance statistics and confidence bands for one binary array."""
        n_samples = len(data_array)
        Ns_out, means, vars_, medians, _ = get_bernoulli_trials_statistics(data_array, Ns=Ns)
        bvar = binomial_var(Ns_out, p=p_hat)
        n_blocks = n_samples // Ns_out
        ci_lower, ci_upper = compute_ci_bounds(n_blocks)

        normalized_vars = vars_ / bvar
        z_scores = (normalized_vars - 1) / np.sqrt(2 / (n_blocks - 1))

        return {
            'Ns': Ns_out,
            'means': means,
            'vars': vars_,
            'medians': medians,
            'normalized_vars': normalized_vars,
            'z_scores': z_scores,
            'n_samples': n_samples,
            'n_blocks': n_blocks,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    # --- Compute statistics for raw data and each XOR level ---

    stats_dict = {}

    # Raw data
    stats_dict['raw'] = compute_stats_for_array(binary_data, 'raw')

    # XOR-whitened versions
    for n in xor_levels:
        key = f"xor{n:02d}"
        whitened = xorn_whitening(binary_data, n=n)
        stats_dict[key] = compute_stats_for_array(whitened, key)

    # Reference line data
    stats_dict['reference'] = {
        'Ns': Ns,
        'p_hat': p_hat,
        'binomial_var': binomial_var(Ns, p=p_hat),
    }
    stats_dict['z_crit'] = float(scipy_norm.ppf(1 - alpha / 2))

    # --- Plot ---

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Colors and markers for all curves (raw + XOR levels)
    all_colors = [p['color'] for p in ENTROPY_CYCLER]
    all_markers = [p['marker'] for p in ENTROPY_CYCLER]
    all_keys = ['raw'] + [f"xor{n:02d}" for n in xor_levels]
    all_labels = ['Raw'] + [f"XOR{n:02d}".upper() for n in xor_levels]

    ci_pct = int(confidence_level * 100)

    if normalize == 'zscore':
        z_crit = stats_dict['z_crit']

        for i, (key, label) in enumerate(zip(all_keys, all_labels)):
            s = stats_dict[key]
            color = all_colors[i % len(all_colors)]
            marker = all_markers[i % len(all_markers)]

            ax.plot(
                s['Ns'], s['z_scores'],
                color=color, alpha=point_alpha, marker=marker, linestyle='',
                label=label, markersize=markersize, zorder=2,
            )

        # Single shared CI band — same for all processing levels
        ax.axhspan(
            -z_crit, z_crit,
            color='gray', alpha=band_alpha * 2,
            label=f'{ci_pct}% CI (normal approx.)', zorder=0,
        )

        # Reference line at z=0
        ax.axhline(0.0, color='k', linewidth=2, label='Expected (z=0)')

        ax.set_ylabel(
            r'$z$-score of $S_n$  $(H_0\!: B(n,\frac{1}{2}))$',
            fontsize=fontsize,
        )
        if ylim is not None:
            ax.set_ylim(ylim)

    elif normalize == 'ratio':
        for i, (key, label) in enumerate(zip(all_keys, all_labels)):
            s = stats_dict[key]
            color = all_colors[i % len(all_colors)]
            marker = all_markers[i % len(all_markers)]

            # Data points
            ax.plot(
                s['Ns'], s['normalized_vars'],
                color=color, alpha=point_alpha, marker=marker, linestyle='',
                label=label, markersize=markersize, zorder=2,
            )

            # Per-curve confidence band
            ax.fill_between(
                s['Ns'], s['ci_lower'], s['ci_upper'],
                color=color, alpha=band_alpha, zorder=0,
            )

        # Reference line at 1.0
        ax.axhline(
            1.0, color='k', linewidth=2,
            label='Expected \nbinomial', zorder=1,
        )

        ax.set_ylabel(
            r'$\frac{\mathrm{Var}(S_n)}{\mathrm{Var}(B(n, 0.5))}$',
            fontsize=fontsize,
        )
        if ylim is not None:
            ax.set_ylim(ylim)

    elif normalize == 'variance':
        for i, (key, label) in enumerate(zip(all_keys, all_labels)):
            s = stats_dict[key]
            color = all_colors[i % len(all_colors)]
            marker = all_markers[i % len(all_markers)]

            # Scale CI bounds from ratio space back to variance units
            bvar = s['vars'] / s['normalized_vars']

            ax.plot(
                s['Ns'], s['vars'],
                color=color, alpha=point_alpha, marker=marker, linestyle='',
                label=label, markersize=markersize, zorder=2,
            )

            ax.fill_between(
                s['Ns'], s['ci_lower'] * bvar, s['ci_upper'] * bvar,
                color=color, alpha=band_alpha, zorder=0,
            )

        # Reference curve: expected binomial variance n·p̂·(1-p̂)
        ref = stats_dict['reference']
        ax.plot(
            ref['Ns'], ref['binomial_var'],
            color='k', linewidth=2,
            label=r'$np(1-p)$',
        )

        ax.set_ylabel(r'$\mathrm{Var}(S_n)$', fontsize=fontsize)
        if ylim is not None:
            ax.set_ylim(ylim)

    # Formatting common to both modes
    ax.set_yscale(yscale)
    ax.set_xlabel('Number of trials (n)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    default_legend = {
        'fontsize': fontsize - 2,
        'bbox_to_anchor': (0.4, 1.05),
        'loc': 'lower center',
        'ncol': legend_ncol,
        'title_fontsize': fontsize - 3,
    }
    if legend_kwargs:
        default_legend.update(legend_kwargs)
    ax.legend(**default_legend)

    plt.tight_layout()

    return fig, ax, stats_dict


def plot_entropy_vs_sp(
    df,
    entropy_per_bit,
    entropy_col="h_original_non_iid",
    exclude_processing=None,
    iid_label_trigger="xor02",
    width_cm=16,
    aspect=0.42,
    fontsize=13,
    reference_legend_title="This work",
):
    """
    Two-panel figure: min-entropy vs. raw switching probability (left) with a
    NIST entropy histogram for comparison (right).

    The right panel draws two horizontal reference lines derived from the data:
    - Solid: max entropy achieved by the raw (unprocessed) data
    - Dashed: max entropy achieved across all XOR post-processed data

    Args:
        df: Polars DataFrame containing columns `processing`, `switching_probability`,
            `iid_pass`, `restart_pass`, and the column named by `entropy_col`.
        entropy_per_bit: Polars Series of entropy values for the NIST histogram.
        entropy_col: Column in `df` to use as the y-axis (default: "h_original_non_iid").
        exclude_processing: List of substrings — skip any processing group whose name
            contains one of these. Default: ["xor24"].
        iid_label_trigger: Processing name substring that triggers the "IID track" legend
            label (default: "xor02").
        width_cm: Figure width in centimetres (default: 16).
        aspect: Height-to-width ratio (default: 0.42).
        fontsize: Base font size applied to all text elements (default: 13).
        reference_legend_title: Title for the right-panel legend (default: "This work").

    Returns:
        tuple: (fig, axl, axh, plot_data)
            - fig: matplotlib Figure
            - axl: left Axes (entropy vs. switching probability)
            - axh: right Axes (NIST histogram)
            - plot_data: dict mapping processing name -> {"sps": np.ndarray, "entropies": np.ndarray}
    """
    import polars as pl

    if exclude_processing is None:
        exclude_processing = ["xor24"]

    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize - 3.5,
        'legend.title_fontsize': fontsize - 3.5,
    })

    figsize = figsize_from_width_cm(width_cm, aspect)
    fig, (axl, axh) = plt.subplots(
        ncols=2, width_ratios=[3, 1], figsize=figsize,
        sharey=True, gridspec_kw={'wspace': 0.1}
    )

    axl.set_prop_cycle(ENTROPY_CYCLER)

    plot_data = {}

    # Left panel: entropy vs. switching probability per processing group
    for name, data in df.sort(["processing", "switching_probability"]).group_by(["processing"]):
        proc = name[0]
        if any(ex in proc for ex in exclude_processing):
            continue
        sps = data.select(["switching_probability"]).to_numpy()
        entropies = data.select([entropy_col]).to_numpy()
        label = proc.capitalize() if proc == "raw" else proc.split('_')[0].upper()
        axl.plot(sps, entropies, markersize=8, label=label)
        plot_data[proc] = {"sps": sps, "entropies": entropies}

    # Left panel: IID track overlay
    for name, data in df.sort(["processing", "switching_probability"]).group_by(["processing"]):
        proc = name[0]
        iid_mask = (pl.col("iid_pass")) & (pl.col("restart_pass"))
        iid_entropies = data.filter(iid_mask).select([entropy_col]).to_numpy()
        iid_sps = data.filter(iid_mask).select(["switching_probability"]).to_numpy()
        axl.plot(
            iid_sps, iid_entropies,
            markersize=15, marker='o', markerfacecolor='none',
            markeredgewidth=1, linestyle='', color='grey',
            label='IID track' if iid_label_trigger in proc else '',
        )

    axl.set_xlabel('Raw switching probability')
    axl.set_ylabel('Min-entropy / bit')
    axl.set_xlim(0.0, 1)
    axl.set_ylim(-0.05, 1.01)
    axl.set_xticks(np.linspace(0, 1, 11, endpoint=True))
    axl.set_yticks(np.linspace(0, 1, 11, endpoint=True), minor=True)
    axl.legend(bbox_to_anchor=(0.6, 1.02), loc='lower center', ncol=4, fontsize=fontsize - 1)
    axl.grid(axis='both', which='both')

    # Right panel: NIST entropy histogram
    bins = np.linspace(0, 1, 51, endpoint=True)
    axh.hist(entropy_per_bit.to_list(), bins=bins, color='grey', orientation='horizontal')
    axh.set_ylim(-0.05, 1.1)
    axh.set_xlabel('Count')

    # Derive reference lines from data
    raw_max = df.filter(pl.col("processing") == "raw")[entropy_col].max()
    xor_max = (
        df.filter(pl.col("processing") != "raw")[entropy_col].max()
    )
    axh.axhline(raw_max, c='r', linewidth=2, label="Raw")
    axh.axhline(xor_max, color='r', linestyle='--', linewidth=2, label="XOR max", alpha=0.7)

    axh.set_yticks(np.linspace(0, 1, 11))
    axh.set_xticks(np.arange(0, 21, 5))
    axh.grid(axis='y')
    axh.legend(loc='lower right', title=reference_legend_title, bbox_to_anchor=(1.02, -0.025))

    plt.subplots_adjust(wspace=0.01)
    plt.tight_layout()

    return fig, axl, axh, plot_data