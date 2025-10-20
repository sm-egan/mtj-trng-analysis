import numpy as np

__all__ = [
    "get_bernoulli_trials_statistics",
    'binomial_var',
    'binomial_var_of_sample_var',
    'chisquare_binomial_sample_var'
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