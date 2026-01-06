import numpy as np
from typing import Dict, List, Optional

__all__ = [
    "get_bernoulli_trials_statistics",
    "binomial_var",
    "binomial_var_of_sample_var",
    "chisquare_binomial_sample_var",
    "compute_correlation_coefficients",
    "compute_binary_statistics",
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
    max_correlation_len: int = 2**20
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
            max_len=max_correlation_len
        )
    }