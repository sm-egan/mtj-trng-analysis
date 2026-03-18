import numpy as np
import polars as pl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import t as t_dist

__all__ = [
    "get_bernoulli_trials_statistics",
    "binomial_var",
    "binomial_var_of_sample_var",
    "chisquare_binomial_sample_var",
    "binomial_sample_var_se",
    "compute_correlation_coefficients",
    "compute_binary_statistics",
    "load_binary_sequence",
    "acf_df_from_entropy_results",
    "acf_df_from_binary",
    "BetaBinomialFit",
    "fit_beta_binomial",
    "fit_beta_binomial_from_stats",
    "fit_beta_binomial_all_xor",
]

def get_bernoulli_trials_statistics(binary_data, Ns=None):
    #Ns = []
    means = []
    vars = []
    medians = []

    sum_over_N = {}

    data = binary_data

    if Ns is None:
        order = np.floor(np.log10(len(data) / 100))
        maxN = np.power(10, order).astype(int) #Set the max N to two orders of magnitude less than the length of the data

        if maxN < 100:
            maxN = 100

        # Generate Ns as powers of 10, up to the maximum N
        # This will create Ns like 100, 200, ..., 900, 1000, 2000, ..., 9000, 10000, ..., maxN
        Ns = np.transpose(np.arange(1,10).reshape((9,1))*np.power(10, np.arange(2, order + 1))).flatten().astype(int)
        Ns = Ns[Ns <= maxN]

    print(f'Ns array: {Ns}')

    for N in Ns:
        # Calculate number of nonoverlapping subsets of length N in 1 million bits
        nsubsets = len(data) // N

        if nsubsets * N < len(data):
            print(f"Warning: {len(data)} bits is not a multiple of {N}, truncating data to {nsubsets * N} bits.")
            data_N = data[:nsubsets * N]
        else:
            data_N = data

        sum_array = np.sum(data_N.reshape((int(nsubsets), N)), axis=1) # Sums over N consecutive bits
        sum_over_N[str(N)] = sum_array # Add to dictionary of arrays

        # Append basic stats to list
        means.append(np.mean(sum_array))
        vars.append(np.var(sum_array, ddof=1)) # Sample variance
        medians.append(np.median(sum_array))

    Ns = np.array(Ns)
    means = np.array(means)
    vars = np.array(vars)
    medians = np.array(medians)

    return Ns, means, vars, medians, sum_over_N

def binomial_var(N_array, p=0.5):
    return p*(1-p)*N_array

def binomial_var_of_sample_var(N_array, nsubsets, p=0.5):
    return 2*np.power(binomial_var(N_array, p),2) / (nsubsets - 1)

def chisquare_binomial_sample_var(N_array, vars, nsamples, p=0.5):
    var_of_sample_var = binomial_var_of_sample_var(N_array, 1e6 / N_array, p)

    return np.sum(np.power(vars - binomial_var(N_array), 2) / var_of_sample_var)

def binomial_sample_var_se(N_array, n_blocks, p=0.5):
    """
    Standard error of the unbiased sample variance of S_n ~ Binomial(n, p),
    computed from L = n_blocks independent draws.
 
    Uses the exact binomial central moments:
        mu2 = n·p·q
        mu4 = 3·n²·p²·q² + n·p·q·(1 - 6·p·q)
 
    and the general result:
        Var(S²) = (1/L) · [mu4 - ((L-3)/(L-1)) · mu2²]
 
    Parameters
    ----------
    N_array : array-like
        Block sizes (n in S_n ~ B(n,p)).
    n_blocks : array-like
        Number of independent blocks L at each block size.
    p : float
        Bit probability.
 
    Returns
    -------
    se : np.ndarray
        Standard error sqrt(Var(S²)) at each block size.
    """
    N_array = np.asarray(N_array, dtype=float)
    n_blocks = np.asarray(n_blocks, dtype=float)
    q = 1.0 - p
    npq = N_array * p * q
 
    mu2 = npq
    mu4 = 3.0 * N_array**2 * p**2 * q**2 + npq * (1.0 - 6.0 * p * q)
    var_S2 = (1.0 / n_blocks) * (mu4 - ((n_blocks - 3.0) / (n_blocks - 1.0)) * mu2**2)
 
    var_S2 = np.maximum(var_S2, 0.0)  # clamp numerical noise
    return np.sqrt(var_S2)

def compute_correlation_coefficients(
    data: np.ndarray,
    lags: Optional[List[int]] = None,
    max_len: int = 2**20
) -> Dict[int, float]:
    """
    Compute autocorrelation coefficients for binary data at specified lags.
    
    Uses the formula:
        correlation(lag) = Σ((x_i - x̄)(x_{i+lag} - ȳ)) / sqrt(Σ(x_i - x̄)² × Σ(x_{i+lag} - ȳ)²)
    
    where x̄ and ȳ are the means of the respective subsequences.
    
    Parameters
    ----------
    data : np.ndarray
        Binary data array (0s and 1s), will be converted to float64 internally.
    lags : list of int, optional
        Lag values to compute. Default: powers of 2 from 1 to 2^20.
    max_len : int, optional
        Maximum number of terms (N-k) in the autocorrelation sum.
        Default: 2^20 (1,048,576). Set to None to use all available data.
        
    Returns
    -------
    dict
        Dictionary mapping lag values to correlation coefficients.
        
    Examples
    --------
    >>> data = np.random.randint(0, 2, size=1_000_000, dtype=np.uint8)
    >>> corr = compute_correlation_coefficients(data, lags=[1, 2, 4, 8])
    >>> print(corr[1])  # Lag-1 autocorrelation
    """
    if lags is None:
        lags = (2 ** np.arange(0, 21)).tolist()
    
    # Convert to float for calculations
    data = data.astype(np.float64)
    correlations = {}
    
    for lag in lags:
        if lag >= len(data):
            # Skip lags that exceed data length
            continue
        
        # Determine the subsequences based on max_len
        if max_len is not None and (len(data) - lag) >= max_len:
            x = data[:max_len]
            y = data[lag:max_len + lag]
        else:
            x = data[:-lag] if lag > 0 else data
            y = data[lag:] if lag > 0 else data
        
        # Compute means of each subsequence
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Center the sequences
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Compute correlation
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        
        if denominator > 0:
            correlations[lag] = float(numerator / denominator)
        else:
            correlations[lag] = 0.0
    
    return correlations

def compute_binary_statistics(
    data: np.ndarray,
    correlation_lags: Optional[List[int]] = None,
    corr_sequence_max_len: int = 2**20
) -> Dict:
    """
    Compute summary statistics for binary data.
    
    Parameters
    ----------
    data : np.ndarray
        Binary data array (0s and 1s as uint8).
    correlation_lags : list of int, optional
        Lag values for autocorrelation. Default: powers of 2 from 1 to 2^20.
    max_correlation_len : int, optional
        Maximum number of terms in autocorrelation sum. Default: 2^20.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_samples': Number of samples
        - 'mean': Mean value (probability of 1)
        - 'variance': Variance of the data
        - 'correlations': Dict of {lag: correlation_coefficient}
    """
    if len(data) == 0:
        return {
            'n_samples': 0,
            'mean': None,
            'variance': None,
            'correlations': {}
        }
    
    if correlation_lags is None:
        correlation_lags = (2 ** np.arange(0, 21)).tolist()
    
    return {
        'n_samples': len(data),
        'mean': float(np.mean(data)),
        'variance': float(np.var(data)),
        'correlations': compute_correlation_coefficients(
            data,
            lags=correlation_lags,
            max_len=corr_sequence_max_len
        )
    }


def load_binary_sequence(
    device_id: str,
    process_name: str,
    raw_hash: str,
    bin_dir,
) -> np.ndarray:
    """
    Load a binary sequence file as a uint8 numpy array.

    Parameters
    ----------
    device_id : str
        Device ID, e.g. '023_07'.
    process_name : str
        Process name, e.g. 'raw', 'xor02'.
    raw_hash : str
        SHA-256 hash; the stem of the .bin file.
    bin_dir : path-like
        Root binary file directory (e.g. project_root / 'bin').

    Returns
    -------
    np.ndarray
        Binary data as uint8 array (values 0 and 1).
    """
    bin_file = Path(bin_dir) / device_id / process_name / f"{raw_hash}.bin"
    with open(bin_file, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def acf_df_from_entropy_results(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract precomputed autocorrelation coefficients from an entropy_results
    CSV DataFrame into a long-form (tidy) Polars DataFrame.

    Detects all ``corr_lag_*`` columns by name, parses the integer lag from
    the column suffix, and unpivots to one row per (run, lag).

    Parameters
    ----------
    df : pl.DataFrame
        One or more rows loaded from an ``entropy_results_yyyymmdd.csv`` file
        via ``pl.read_csv``. Must contain ``device``, ``processing``,
        ``raw_hash``, and at least one ``corr_lag_*`` column.

    Returns
    -------
    pl.DataFrame
        Columns: ``device``, ``processing``, ``raw_hash``,
        ``lag`` (Int64), ``acf`` (Float64). Sorted by
        [``raw_hash``, ``processing``, ``lag``].

    Notes
    -----
    The ACF values are precomputed — ``corr_sequence_max_len`` is not a
    parameter here. The confidence level of the stored values depends on
    the ``corr_sequence_max_len`` used when the CSV was generated (see
    ``scripts/extract_entropy_results.py``).
    """
    id_cols = ["device", "processing", "raw_hash"]
    lag_cols = sorted(
        [c for c in df.columns if c.startswith("corr_lag_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    return (
        df.select(id_cols + lag_cols)
        .unpivot(on=lag_cols, index=id_cols, variable_name="_col", value_name="acf")
        .with_columns(
            pl.col("_col").str.replace("corr_lag_", "").cast(pl.Int64).alias("lag")
        )
        .drop("_col")
        .select(id_cols + ["lag", "acf"])
        .sort(["raw_hash", "processing", "lag"])
    )


def acf_df_from_binary(
    raw_hash: str,
    device_id: str,
    process_name: str,
    bin_dir,
    lags: Optional[List[int]] = None,
    corr_sequence_max_len: int = 2**20,
) -> pl.DataFrame:
    """
    Load a binary sequence file and compute autocorrelation coefficients,
    returning a long-form (tidy) Polars DataFrame with run metadata.

    Parameters
    ----------
    raw_hash : str
        SHA-256 hash; the stem of the .bin file.
    device_id : str
        Device ID, e.g. '023_07'.
    process_name : str
        Process name, e.g. 'raw', 'xor02'.
    bin_dir : path-like
        Root binary file directory (e.g. project_root / 'bin').
    lags : list of int, optional
        Lag values to compute. Default: powers of 2 from 2^0 to 2^20.
    corr_sequence_max_len : int, optional
        Maximum number of terms (N-k) in each autocorrelation sum.
        Controls the 95 %% confidence level: CL ≈ 2/sqrt(corr_sequence_max_len).
        Default: 2^20 (1,048,576).

    Returns
    -------
    pl.DataFrame
        Columns: ``device``, ``processing``, ``raw_hash``,
        ``lag`` (Int64), ``acf`` (Float64). Sorted by ``lag``.
        Schema matches ``acf_df_from_entropy_results`` for use with
        ``pl.concat``.
    """
    data = load_binary_sequence(device_id, process_name, raw_hash, bin_dir)
    stats = compute_binary_statistics(
        data,
        correlation_lags=lags,
        corr_sequence_max_len=corr_sequence_max_len,
    )
    corr = stats["correlations"]  # {lag: float}
    return pl.DataFrame({
        "device": device_id,
        "processing": process_name,
        "raw_hash": raw_hash,
        "lag": list(corr.keys()),
        "acf": list(corr.values()),
    }).with_columns(pl.col("lag").cast(pl.Int64)).sort("lag")


# ---------------------------------------------------------------------------
# Beta-Binomial fitting
# ---------------------------------------------------------------------------

@dataclass
class BetaBinomialFit:
    """
    Results of a Beta-Binomial fit to block-sum variance data.

    Model
    -----
    Var(S_n) = n · p · (1-p) · (1 + (n-1) · ρ)

    Fit
    ---
    With p fixed to the empirical bit mean, q = p·(1-p) is known and the
    model reduces to a 1-parameter WLS fit through the origin::

        [Var(S_n)/n  −  q]  =  ρ · q · (n − 1)

    p is not estimated from the variance because q = p(1-p) is flat near its
    maximum at p = 0.5, so dp/dq diverges and any small error in q produces
    a large, unreliable error in p.

    Uncertainty
    -----------
    Weights w_i = (k_i − 1) / (2 · (Var(S_n_i)/n_i)²) from variance
    propagation: Var(Var(S_n)/n) = 2·(Var(S_n)/n)² / (k−1).

    Two SE estimates are stored:

    * ``rho_se``           – independence lower bound (σ² = 1).
    * ``rho_se_corrected`` – inflated by 1/√wmse to account for correlations
      between R(n) values at different n (same underlying bitstream).

    The 95 % CI uses the corrected SE.  wmse < 1 confirms correlated
    observations; the corrected SE is a conservative practical upper bound.

    Attributes
    ----------
    rho, p, q : float
        Fitted ICC, fixed mean probability, and Bernoulli variance q = p(1-p).
    rho_se : float
        SE under the independence assumption (lower bound).
    rho_se_corrected : float
        SE corrected for inter-observation correlations (upper bound).
    rho_lo, rho_hi : float
        95 % CI using ``rho_se_corrected``.
    wmse : float
        Weighted MSE of residuals.  Expected ≈ 1 for independent observations;
        wmse < 1 indicates correlated R(n) values.
    dof : int
        Degrees of freedom (n_points − 1).
    Ns, vars_Sn, n_blocks : np.ndarray
        Block sizes, sample variances, and block counts used in the fit
        (after applying the n_min mask).
    n_min : int
        Minimum block size included in the fit.
    """
    rho:               float
    p:                 float
    q:                 float
    rho_se:            float
    rho_se_corrected:  float
    rho_lo:            float
    rho_hi:            float
    wmse:              float
    dof:               int
    Ns:                np.ndarray
    vars_Sn:           np.ndarray
    n_blocks:          np.ndarray
    n_min:             int = 0

    @property
    def alpha_bb(self) -> float:
        """Beta-binomial α = p·(1/ρ − 1).  NaN if ρ ≤ 0."""
        return self.p * (1.0 / self.rho - 1.0) if self.rho > 0 else float("nan")

    @property
    def beta_bb(self) -> float:
        """Beta-binomial β = (1-p)·(1/ρ − 1).  NaN if ρ ≤ 0."""
        return (1.0 - self.p) * (1.0 / self.rho - 1.0) if self.rho > 0 else float("nan")

    @property
    def var_ratio(self) -> np.ndarray:
        """Observed Var(S_n)/(n·q) — the variance ratio relative to binomial(n, p)."""
        return self.vars_Sn / (self.Ns * self.q)

    @property
    def fitted_var_per_n(self) -> np.ndarray:
        """Fitted Var(S_n)/n = q·(1 + (n−1)·ρ) at each fitted block size."""
        return self.q * (1.0 + (self.Ns - 1.0) * self.rho)

    def __repr__(self) -> str:
        return (
            f"BetaBinomialFit(ρ={self.rho:.4e} ± {self.rho_se_corrected:.1e} [corrected], "
            f"p={self.p:.5f}, wmse={self.wmse:.3f}, n_min={self.n_min})"
        )


def fit_beta_binomial(
    Ns: np.ndarray,
    vars_Sn: np.ndarray,
    n_blocks: np.ndarray,
    p: float,
    n_min: int = 0,
    confidence: float = 0.95,
) -> BetaBinomialFit:
    """
    Fit a Beta-Binomial model to block-sum variance data.

    Parameters
    ----------
    Ns : array-like
        Block sizes used to compute ``vars_Sn``.
    vars_Sn : array-like
        Sample variances Var(S_n) at each block size, as returned by
        ``get_bernoulli_trials_statistics``.
    n_blocks : array-like
        Number of non-overlapping blocks at each block size (k = L / n).
    p : float
        Empirical mean bit probability: ``float(np.mean(binary_data))``.
        Do not substitute 0.5 unless the data is known to be perfectly balanced.
    n_min : int, optional
        Exclude block sizes below this value.  Useful when short-range
        negative correlations cause Var(S_n)/n to dip below q at small n,
        violating the linear beta-binomial assumption.  Default: 0 (use all).
    confidence : float, optional
        Confidence level for the returned CI.  Default: 0.95.

    Returns
    -------
    BetaBinomialFit
    """
    Ns       = np.asarray(Ns,       dtype=float)
    vars_Sn  = np.asarray(vars_Sn,  dtype=float)
    n_blocks = np.asarray(n_blocks, dtype=float)

    mask = Ns >= n_min
    Ns_f, y_f, k_f = Ns[mask], vars_Sn[mask] / Ns[mask], n_blocks[mask]

    q = p * (1.0 - p)

    # 1-parameter WLS through origin:  (y − q) = ρ · q·(n−1)
    x  = q * (Ns_f - 1.0)
    dy = y_f - q
    w  = (k_f - 1.0) / (2.0 * y_f ** 2)

    rho_fit   = np.sum(w * x * dy) / np.sum(w * x ** 2)
    residuals = dy - rho_fit * x
    dof       = max(len(Ns_f) - 1, 1)
    wmse      = np.sum(w * residuals ** 2) / dof

    rho_se      = 1.0 / np.sqrt(np.sum(w * x ** 2))
    rho_se_corr = rho_se / np.sqrt(max(wmse, 1e-10))
    t_crit      = t_dist.ppf((1.0 + confidence) / 2.0, df=dof)

    return BetaBinomialFit(
        rho              = float(rho_fit),
        p                = float(p),
        q                = float(q),
        rho_se           = float(rho_se),
        rho_se_corrected = float(rho_se_corr),
        rho_lo           = float(rho_fit - t_crit * rho_se_corr),
        rho_hi           = float(rho_fit + t_crit * rho_se_corr),
        wmse             = float(wmse),
        dof              = dof,
        Ns               = Ns_f,
        vars_Sn          = vars_Sn[mask],
        n_blocks         = k_f,
        n_min            = n_min,
    )


def fit_beta_binomial_from_stats(
    stats_entry: dict,
    p: float,
    n_min: int = 0,
    confidence: float = 0.95,
) -> BetaBinomialFit:
    """
    Convenience wrapper for a single entry from the ``stats`` dict returned by
    :func:`~analysis.visualization.plot_Sn_variance`.

    Parameters
    ----------
    stats_entry : dict
        One value from the stats dict, e.g. ``stats['raw']``.
        Must contain keys ``'Ns'``, ``'vars'``, and ``'n_blocks'``.
    p : float
        Empirical mean bit probability.
    n_min : int, optional
        Minimum block size to include.  Default: 0.
    confidence : float, optional
        Confidence level for CI.  Default: 0.95.

    Returns
    -------
    BetaBinomialFit
    """
    return fit_beta_binomial(
        Ns        = stats_entry["Ns"],
        vars_Sn   = stats_entry["vars"],
        n_blocks  = stats_entry["n_blocks"],
        p         = p,
        n_min     = n_min,
        confidence = confidence,
    )


def fit_beta_binomial_all_xor(
    binary_data: np.ndarray,
    xor_levels: List[int] = (2, 4, 8),
    Ns: Optional[np.ndarray] = None,
    n_min: int = 0,
    confidence: float = 0.95,
) -> Dict[str, BetaBinomialFit]:
    """
    Fit the Beta-Binomial model to raw binary data and each XOR-whitened
    version, returning a dict keyed by processing name.

    This is the primary entry point for multi-processing analysis.  The
    returned dict can be passed directly to the visualization functions
    :func:`~analysis.visualization.plot_beta_binomial_var` and
    :func:`~analysis.visualization.plot_icc_results`.

    Parameters
    ----------
    binary_data : np.ndarray
        Raw binary sequence (uint8, values 0 and 1).
    xor_levels : list of int, optional
        XOR whitening levels to include.  Default: (2, 4, 8).
    Ns : np.ndarray, optional
        Block sizes to use.  If None, chosen automatically by
        ``get_bernoulli_trials_statistics``.
    n_min : int, optional
        Minimum block size included in each fit.  Default: 0.
    confidence : float, optional
        Confidence level for CIs.  Default: 0.95.

    Returns
    -------
    dict[str, BetaBinomialFit]
        Keys: ``'raw'``, ``'xor02'``, ``'xor04'``, …

    Examples
    --------
    >>> fits = fit_beta_binomial_all_xor(data, xor_levels=[2, 4, 8], n_min=500)
    >>> for key, fit in fits.items():
    ...     print(key, fit)
    """
    from .processing import xorn_whitening

    results: Dict[str, BetaBinomialFit] = {}

    datasets = {"raw": binary_data}
    for n in xor_levels:
        datasets[f"xor{n:02d}"] = xorn_whitening(binary_data, n=n)

    for key, data in datasets.items():
        p_emp = float(np.mean(data))
        Ns_k, _, vars_k, _, _ = get_bernoulli_trials_statistics(data, Ns=Ns)
        n_blocks_k = len(data) // Ns_k
        results[key] = fit_beta_binomial(
            Ns        = Ns_k,
            vars_Sn   = vars_k,
            n_blocks  = n_blocks_k,
            p         = p_emp,
            n_min     = n_min,
            confidence = confidence,
        )

    return results