import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "get_bernoulli_trials_statistics",
    "binomial_var",
    "binomial_var_of_sample_var",
    "chisquare_binomial_sample_var",
    "compute_correlation_coefficients",
    "compute_binary_statistics",
    "load_binary_sequence",
    "acf_df_from_entropy_results",
    "acf_df_from_binary",
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
        vars.append(np.var(sum_array))
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