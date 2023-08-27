import numpy as np
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from .utils import dataframe_decorator


def _rectified_mean(rectifier, imputed_mean):
    """
    Computes a rectified mean.

    Parameters
    ----------
    rectifier : float or ndarray
        The rectifier value.
    imputed_mean : float or ndarray
        The imputed mean.
    """
    return imputed_mean + rectifier


def _rectified_ci(
    rectifier,
    rectifier_std,
    imputed_mean,
    imputed_std,
    alpha,
    alternative="two-sided",
):
    """
    Computes a rectified confidence interval.

    Parameters
    ----------
    rectifier : float or ndarray
        The rectifier value.
    rectifier_std : float or ndarray
        The rectifier standard deviation.
    imputed_mean : float or ndarray
        The imputed mean.
    imputed_std : float or ndarray
        The imputed standard deviation.
    alpha : float
        The confidence level.
    """
    rectified_point_estimate = imputed_mean + rectifier
    rectified_std = np.sqrt(imputed_std**2 + rectifier_std**2)
    return _zconfint_generic(
        rectified_point_estimate, rectified_std, alpha, alternative
    )


def _rectified_p_value(
    rectifier,
    rectifier_std,
    imputed_mean,
    imputed_std,
    null=0,
    alternative="two-sided",
):
    """
    Computes a rectified p-value.

    Parameters
    ----------
    rectifier : float or ndarray
        The rectifier value.
    rectifier_std : float or ndarray
        The rectifier standard deviation.
    imputed_mean : float or ndarray
        The imputed mean.
    imputed_std : float or ndarray
        The imputed standard deviation.
    """
    rectified_point_estimate = imputed_mean + rectifier
    rectified_std = np.sqrt(imputed_std**2 + rectifier_std**2)
    return _zstat_generic(
        rectified_point_estimate, 0, rectified_std, alternative, null
    )[1]


"""
    MEAN ESTIMATION

"""


def ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled):
    return _rectified_mean((Y - Yhat).mean(), Yhat_unlabeled.mean())


def ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    return _rectified_ci(
        (Y - Yhat).mean(),
        (Y - Yhat).std() / np.sqrt(n),
        Yhat_unlabeled.mean(),
        Yhat_unlabeled.std() / np.sqrt(N),
        alpha,
        alternative,
    )


def ppi_mean_pval(Y, Yhat, Yhat_unlabeled, null=0, alternative="two-sided"):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    return _rectified_p_value(
        (Y - Yhat).mean(),
        (Y - Yhat).std() / np.sqrt(n),
        Yhat_unlabeled.mean(),
        Yhat_unlabeled.std() / np.sqrt(N),
        null,
        alternative,
    )


"""
    QUANTILE ESTIMATION

"""


def _compute_cdf(Y, grid):
    indicators = Y[:, None] <= grid[None, :]
    return indicators.mean(axis=0), indicators.std(axis=0)


def _compute_cdf_diff(Y, Yhat, grid):
    indicators_Y = Y[:, None] <= grid[None, :]
    indicators_Yhat = Y[:, None] <= grid[None, :]
    return (indicators_Y - indicators_Yhat).mean(axis=0), (
        indicators_Y - indicators_Yhat
    ).std(axis=0)


def _rectified_cdf(Y, Yhat, Yhat_unlabeled, grid):
    cdf_Yhat_unlabeled, _ = _compute_cdf(Yhat_unlabeled, grid)
    cdf_rectifier, _ = _compute_cdf_diff(Y, Yhat, grid)
    return cdf_Yhat_unlabeled + cdf_rectifier


def ppi_quantile_pointestimate(Y, Yhat, Yhat_unlabeled, q):
    assert len(Y.shape) == 1
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    grid = np.sort(grid)
    rectified_cdf = _rectified_cdf(Y, Yhat, Yhat_unlabeled, grid)
    return grid[np.argmin(np.abs(rectified_cdf - q))][
        0
    ]  # Find the intersection of the rectified CDF and the quantile


def ppi_quantile_ci(Y, Yhat, Yhat_unlabeled, q, alpha=0.1):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    grid = np.sort(grid)
    cdf_Yhat_unlabeled, cdf_Yhat_unlabeled_std = _compute_cdf(
        Yhat_unlabeled, grid
    )
    cdf_rectifier, cdf_rectifier_std = _compute_cdf_diff(Y, Yhat, grid)
    # Calculate rectified p-value for null that the rectified cdf is equal to q
    rectified_p_value = _rectified_p_value(
        cdf_rectifier,
        cdf_rectifier_std / np.sqrt(n),
        cdf_Yhat_unlabeled,
        cdf_Yhat_unlabeled_std / np.sqrt(N),
        null=q,
        alternative="two-sided",
    )
    # Return the min and max values of the grid where p > alpha
    return grid[rectified_p_value > alpha][[0, -1]]


"""
    ORDINARY LEAST SQUARES

"""


def _ols(X, Y, return_se=False):
    regression = OLS(Y, exog=X).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta

def ppi_ols_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled):
    imputed_theta = _ols(X_unlabeled, Yhat_unlabeled)
    rectifier = _ols(X, Y - Yhat)
    return imputed_theta + rectifier

def ppi_ols_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=0.1):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    imputed_theta, imputed_se = _ols(X_unlabeled, Yhat_unlabeled, return_se=True)
    rectifier, rectifier_se = _ols(X, Y - Yhat, return_se=True)
    return _rectified_ci(
        imputed_theta,
        imputed_se,
        rectifier,
        rectifier_se,
        alpha,
        alternative="two-sided",
    )

