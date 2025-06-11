import numpy as np
from numba import njit
from scipy.stats import binom
from scipy.optimize import brentq


def bootstrap(data, statistic, n_resamples, paired="all", statistic_kwargs={}):
    """
    Bootstrap the given statistic on the data.

    Args:
        data (ndarray, list): Data to bootstrap. First dimension is the number of observations. Can be an array or a list of arrays, in which case the resamples will be paired.
        statistic (callable): Statistic to compute. Should take data as input if data is a single array, or *data if data is a list of arrays.
        n_resamples (int): Number of bootstrap resamples.
        paired (bool, optional): Whether to resample data in a paired way. Defaults to "all", in which case all the data is considered paired.

    Returns:
        ndarray: Bootstrap resamples of the statistic.
    """
    if not isinstance(data, list):
        data = [data]
    if paired == "all":
        paired = [list(range(len(data)))]
    # Input validation: check that all the paired data have the same number of observations
    for p in paired:
        assert len(set([d.shape[0] for d in [data[i] for i in p]])) == 1
    # Ensure that all the data is paired, even if only with itself
    for j in range(len(data)):
        if j not in [i for p in paired for i in p]:
            paired += [[j]]
    # Resample the data and compute the bootstrap samples
    bootstrap_distribution = []
    for i in range(n_resamples):
        resample_data = []
        for p in paired:
            resample_indexes = np.random.choice(
                data[p[0]].shape[0], data[p[0]].shape[0], replace=True
            )
            resample_data += [data[i][resample_indexes] for i in p]
        bootstrap_distribution.append(
            statistic(*resample_data, **statistic_kwargs)
        )
    return bootstrap_distribution


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


@njit
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


def _calc_lam_glm(
    grads,
    grads_hat,
    grads_hat_unlabeled,
    inv_hessian,
    coord=None,
    clip=False,
    optim_mode="overall",
):
    """
    Calculates the optimal value of lam for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lam`, when `optim_mode="overall"`.
        If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lam to be non-negative. Defaults to `False`.
        optim_mode (ndarray, optional): Mode for which to optimize `lam`, either `overall` or `element`.
        If `overall`, it optimizes the total variance over all coordinates, and the function returns a scalar.
        If `element`, it optimizes the variance for each coordinate separately, and the function returns a vector.


    Returns:
        float: Optimal value of `lam`. Lies in [0,1].
    """
    grads = reshape_to_2d(grads)
    grads_hat = reshape_to_2d(grads_hat)
    grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    if grads.shape[1] != d:
        raise ValueError(
            "Dimension mismatch between the gradient and the inverse Hessian."
        )

    grads_cent = grads - grads.mean(axis=0)
    grad_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * (
        grads_cent.T @ grad_hat_cent + grad_hat_cent.T @ grads_cent
    )

    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )
    var_grads_hat = var_grads_hat.reshape(d, d)

    vhat = inv_hessian if coord is None else inv_hessian[coord, :]
    if optim_mode == "overall":
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
        lam = num / denom
        lam = lam.item()
    elif optim_mode == "element":
        num = np.diag(vhat @ cov_grads @ vhat)
        denom = 2 * (1 + (n / N)) * np.diag(vhat @ var_grads_hat @ vhat)
        lam = num / denom
    else:
        raise ValueError(
            "Invalid value for optim_mode. Must be either 'overall' or 'element'."
        )
    if clip:
        lam = np.clip(lam, 0, 1)
    return lam
