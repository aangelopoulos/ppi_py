import numpy as np
from numba import njit
from scipy.stats import binom
from scipy.optimize import brentq


def construct_weight_vector(n_obs, existing_weight, vectorized=False):
    res = (
        np.ones(n_obs)
        if existing_weight is None
        else existing_weight / existing_weight.sum() * n_obs
    )
    if vectorized and (len(res.shape) == 1):
        res = res[:, None]
    return res


def reshape_to_2d(x):
    """Reshapes a 1D array to a 2D array."""
    return x.reshape(-1, 1) if len(x.shape) == 1 else x.copy()


@njit
def safe_expit(x):
    """Computes the sigmoid function in a numerically stable way."""
    return np.exp(-np.logaddexp(0, -x))


def safe_log1pexp(x):
    """
    Compute log(1 + exp(x)) in a numerically stable way.
    """
    idxs = x > 10
    out = np.empty_like(x)
    out[idxs] = x[idxs]
    out[~idxs] = np.log1p(np.exp(x[~idxs]))
    return out


def compute_cdf(Y, grid, w=None):
    """Computes the empirical CDF of the data.

    Args:
        Y (ndarray): Data.
        grid (ndarray): Grid of values to compute the CDF at.
        w (ndarray, optional): Sample weights.

    Returns:
        tuple: Empirical CDF and its standard deviation at the specified grid points.
    """
    w = np.ones(Y.shape[0]) if w is None else w / w.sum() * Y.shape[0]
    if w is None:
        indicators = (Y[:, None] <= grid[None, :]).astype(float)
    else:
        indicators = ((Y[:, None] <= grid[None, :]) * w[:, None]).astype(float)
    return indicators.mean(axis=0), indicators.std(axis=0)


def compute_cdf_diff(Y, Yhat, grid, w=None):
    """Computes the difference between the empirical CDFs of the data and the predictions.

    Args:
        Y (ndarray): Data.
        Yhat (ndarray): Predictions.
        grid (ndarray): Grid of values to compute the CDF at.
        w (ndarray, optional): Sample weights.

    Returns:
        tuple: Difference between the empirical CDFs of the data and the predictions and its standard deviation at the specified grid points.
    """
    w = np.ones(Y.shape[0]) if w is None else w / w.sum() * Y.shape[0]
    indicators_Y = (Y[:, None] <= grid[None, :]).astype(float)
    indicators_Yhat = (Yhat[:, None] <= grid[None, :]).astype(float)
    if w is None:
        return (indicators_Y - indicators_Yhat).mean(axis=0), (
            indicators_Y - indicators_Yhat
        ).std(axis=0)
    else:
        return (w[:, None] * (indicators_Y - indicators_Yhat)).mean(axis=0), (
            w[:, None] * (indicators_Y - indicators_Yhat)
        ).std(axis=0)


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
