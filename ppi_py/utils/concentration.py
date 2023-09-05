import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq


def form_discrete_distribution(Yhat, sorted_highlow=False):
    # Construct the point estimate
    uq, uq_counts = np.unique(Yhat, return_counts=True)
    uq_freq = uq_counts / uq_counts.sum()
    if sorted_highlow:
        uq_sort = np.argsort(uq_freq)[::-1]
        uq_freq = uq_freq[uq_sort]
    return uq_freq


def binomial_iid(N, alpha, muhat):
    def invert_upper_tail(mu):
        return binom.cdf(N * muhat, N, mu) - (alpha / 2)

    def invert_lower_tail(mu):
        return binom.cdf(N * muhat, N, mu) - (1 - alpha / 2)

    u = brentq(invert_upper_tail, 0, 1)
    l = brentq(invert_lower_tail, 0, 1)
    return np.array([l, u])


def linfty_dkw(N, K, alpha):
    return np.sqrt(2 / N * np.log(2 / alpha))


def linfty_binom(N, K, alpha, qhat):
    epsilon = 0
    for k in np.arange(K):
        bci = binomial_iid(N, alpha / K, qhat[k])
        epsilon = np.maximum(epsilon, np.abs(bci - qhat[k]).max())
    return epsilon
