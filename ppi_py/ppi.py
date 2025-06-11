import warnings
from collections.abc import Callable
from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import brentq, minimize
from scipy.stats import binom, norm
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import (
    _zconfint_generic,
    _zstat_generic,
    _zstat_generic2,
)

warnings.simplefilter("ignore")
from .multi_target_ppi import (
    ppi_multi_glm_ci,
    ppi_multi_glm_pointest,
    ppi_multi_glm_pval,
    _glm_get_stats,
)
from .utils import (
    bootstrap,
    calc_lam_glm,
    compute_cdf,
    compute_cdf_diff,
    construct_weight_vector,
    dataframe_decorator,
    form_discrete_distribution,
    linfty_binom,
    linfty_dkw,
    reshape_to_2d,
    safe_expit,
    safe_log1pexp,
)


def rectified_p_value(
    rectifier,
    rectifier_std,
    imputed_mean,
    imputed_std,
    null=0,
    alternative="two-sided",
):
    """Computes a rectified p-value.

    Args:
        rectifier (float or ndarray): Rectifier value.
        rectifier_std (float or ndarray): Rectifier standard deviation.
        imputed_mean (float or ndarray): Imputed mean.
        imputed_std (float or ndarray): Imputed standard deviation.
        null (float, optional): Value of the null hypothesis to be tested. Defaults to `0`.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.

    Returns:
        float or ndarray: P-value.
    """
    rectified_point_estimate = imputed_mean + rectifier
    rectified_std = np.maximum(
        np.sqrt(imputed_std**2 + rectifier_std**2), 1e-16
    )
    return _zstat_generic(
        rectified_point_estimate, 0, rectified_std, alternative, null
    )[1]


"""
    MEAN ESTIMATION

"""


def ppi_mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lam_optim_mode="overall",
):
    """Computes the prediction-powered point estimate of the d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the dimension of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.

    Returns:
        float or ndarray: Prediction-powered point estimate of the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Yhat.shape[1]

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lam is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean(0) + (
            w * (Y - Yhat)
        ).mean(0)
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lam = calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
            optim_mode=lam_optim_mode,
        )
        return ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return (w_unlabeled * lam * Yhat_unlabeled).mean(axis=0) + (
            w * (Y - lam * Yhat)
        ).mean(axis=0).squeeze()


def ppi_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lam_optim_mode="overall",
):
    """Computes the prediction-powered confidence interval for a d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Y.shape[1] if len(Y.shape) > 1 else 1

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lam is None:
        ppi_pointest = ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lam=1,
            w=w,
            w_unlabeled=w_unlabeled,
        )
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lam = calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
            optim_mode=lam_optim_mode,
        )
        return ppi_mean_ci(
            Y,
            Yhat,
            Yhat_unlabeled,
            alpha=alpha,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    imputed_std = (w_unlabeled * (lam * Yhat_unlabeled)).std(0) / np.sqrt(N)
    rectifier_std = (w * (Y - lam * Yhat)).std(0) / np.sqrt(n)

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(imputed_std**2 + rectifier_std**2),
        alpha,
        alternative,
    )


def ppi_mean_pval(
    Y,
    Yhat,
    Yhat_unlabeled,
    null=0,
    alternative="two-sided",
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lam_optim_mode="overall",
):
    """Computes the prediction-powered p-value for a 1D mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        null (float): Value of the null hypothesis to be tested.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        float or ndarray: Prediction-powered p-value for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)
    d = Y.shape[1]

    if lam is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean(0) + (
            w * (Y - Yhat)
        ).mean(0)
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lam = calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            optim_mode=lam_optim_mode,
        )

    return rectified_p_value(
        rectifier=(w * Y - lam * w * Yhat).mean(0),
        rectifier_std=(w * Y - lam * w * Yhat).std(0) / np.sqrt(n),
        imputed_mean=(w_unlabeled * lam * Yhat_unlabeled).mean(0),
        imputed_std=(w_unlabeled * lam * Yhat_unlabeled).std(0) / np.sqrt(N),
        null=null,
        alternative=alternative,
    )


"""
    QUANTILE ESTIMATION

"""


def _rectified_cdf(Y, Yhat, Yhat_unlabeled, grid, w=None, w_unlabeled=None):
    """Computes the rectified CDF of the data.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        grid (ndarray): Grid of values to compute the CDF at.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Rectified CDF of the data at the specified grid points.
    """
    w = np.ones(Y.shape[0]) if w is None else w / w.sum() * Y.shape[0]
    w_unlabeled = (
        np.ones(Yhat_unlabeled.shape[0])
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * Yhat_unlabeled.shape[0]
    )
    cdf_Yhat_unlabeled, _ = compute_cdf(Yhat_unlabeled, grid, w=w_unlabeled)
    cdf_rectifier, _ = compute_cdf_diff(Y, Yhat, grid, w=w)
    return cdf_Yhat_unlabeled + cdf_rectifier


def ppi_quantile_pointestimate(
    Y, Yhat, Yhat_unlabeled, q, exact_grid=False, w=None, w_unlabeled=None
):
    """Computes the prediction-powered point estimate of the quantile.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        q (float): Quantile to estimate.
        exact_grid (bool, optional): Whether to compute the exact solution (True) or an approximate solution based on a linearly spaced grid of 5000 values (False).
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        float: Prediction-powered point estimate of the quantile.
    """
    assert len(Y.shape) == 1
    w = np.ones(Y.shape[0]) if w is None else w / w.sum() * Y.shape[0]
    w_unlabeled = (
        np.ones(Yhat_unlabeled.shape[0])
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * Yhat_unlabeled.shape[0]
    )
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
    rectified_cdf = _rectified_cdf(
        Y, Yhat, Yhat_unlabeled, grid, w=w, w_unlabeled=w_unlabeled
    )
    minimizers = np.argmin(np.abs(rectified_cdf - q))
    minimizer = (
        minimizers
        if isinstance(minimizers, (int, np.int64))
        else minimizers[0]
    )
    return grid[
        minimizer
    ]  # Find the intersection of the rectified CDF and the quantile


def ppi_quantile_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    q,
    alpha=0.1,
    exact_grid=False,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the quantile.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        q (float): Quantile to estimate. Must be in the range (0, 1).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        exact_grid (bool, optional): Whether to use the exact grid of values or a linearly spaced grid of 5000 values.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the quantile.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
    cdf_Yhat_unlabeled, cdf_Yhat_unlabeled_std = compute_cdf(
        Yhat_unlabeled, grid, w=w_unlabeled
    )
    cdf_rectifier, cdf_rectifier_std = compute_cdf_diff(Y, Yhat, grid, w=w)
    # Calculate rectified p-value for null that the rectified cdf is equal to q
    rec_p_value = rectified_p_value(
        cdf_rectifier,
        cdf_rectifier_std / np.sqrt(n),
        cdf_Yhat_unlabeled,
        cdf_Yhat_unlabeled_std / np.sqrt(N),
        null=q,
        alternative="two-sided",
    )
    # Return the min and max values of the grid where p > alpha
    return grid[rec_p_value > alpha][[0, -1]]


"""
    ORDINARY LEAST SQUARES

"""


def _ols(X, Y, return_se=False):
    """Computes the ordinary least squares coefficients.

    Args:
        X (ndarray): Covariates.
        Y (ndarray): Labels.
        return_se (bool, optional): Whether to return the standard errors of the coefficients.

    Returns:
        theta (ndarray): Ordinary least squares estimate of the coefficients.
        se (ndarray): If return_se==True, return the standard errors of the coefficients.
    """
    regression = OLS(Y, exog=X).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta


def _wls(X, Y, w=None, return_se=False):
    """Computes the weighted least squares estimate of the coefficients.

    Args:
        X (ndarray): Covariates.
        Y (ndarray): Labels.
        w (ndarray, optional): Sample weights.
        return_se (bool, optional): Whether to return the standard errors.

    Returns:
        theta (ndarray): Weighted least squares estimate of the coefficients.
        se (ndarray): If return_se==True, returns the standard errors of the coefficients.
    """
    if w is None or np.all(w == 1):
        return _ols(X, Y, return_se=return_se)

    regression = WLS(Y, exog=X, weights=w).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta


@njit
def _ols_get_stats(
    pointest,
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    w=None,
    w_unlabeled=None,
    use_unlabeled=True,
):
    """Computes the statistics needed for the OLS-based prediction-powered inference.

    Args:
        pointest (ndarray): A point estimate of the coefficients.
        X (ndarray): Covariates for the labeled data set.
        Y (ndarray): Labels for the labeled data set.
        Yhat (ndarray): Predictions for the labeled data set.
        X_unlabeled (ndarray): Covariates for the unlabeled data set.
        Yhat_unlabeled (ndarray): Predictions for the unlabeled data set.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.
        use_unlabeled (bool, optional): Whether to use the unlabeled data set.

    Returns:
        grads (ndarray): Gradient of the loss function with respect to the coefficients.
        grads_hat (ndarray): Gradient of the loss function with respect to the coefficients, evaluated using the labeled predictions.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the coefficients, evaluated using the unlabeled predictions.
        inv_hessian (ndarray): Inverse Hessian of the loss function with respect to the coefficients.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = X.shape[1]
    w = np.ones(n) if w is None else w / np.sum(w) * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / np.sum(w_unlabeled) * N
    )

    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(X_unlabeled.shape)
    if use_unlabeled:
        for i in range(N):
            hessian += (
                w_unlabeled[i]
                / (N + n)
                * np.outer(X_unlabeled[i], X_unlabeled[i])
            )
            grads_hat_unlabeled[i, :] = (
                w_unlabeled[i]
                * X_unlabeled[i, :]
                * (np.dot(X_unlabeled[i, :], pointest) - Yhat_unlabeled[i])
            )

    grads = np.zeros(X.shape)
    grads_hat = np.zeros(X.shape)
    for i in range(n):
        hessian += (
            w[i] / (N + n) * np.outer(X[i], X[i])
            if use_unlabeled
            else w[i] / n * np.outer(X[i], X[i])
        )
        grads[i, :] = w[i] * X[i, :] * (np.dot(X[i, :], pointest) - Y[i])
        grads_hat[i, :] = (
            w[i] * X[i, :] * (np.dot(X[i, :], pointest) - Yhat[i])
        )

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    return grads, grads_hat, grads_hat_unlabeled, inv_hessian


def ppi_ols_pointestimate(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered point estimate of the OLS coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Prediction-powered point estimate of the OLS coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / np.sum(w) * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / np.sum(w_unlabeled) * N
    )
    use_unlabeled = lam != 0

    imputed_theta = (
        _wls(X_unlabeled, Yhat_unlabeled, w=w_unlabeled)
        if lam is None
        else _wls(X_unlabeled, lam * Yhat_unlabeled, w=w_unlabeled)
    )
    rectifier = (
        _wls(X, Y - Yhat, w=w) if lam is None else _wls(X, Y - lam * Yhat, w=w)
    )
    ppi_pointest = imputed_theta + rectifier

    if lam is None:
        grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
            ppi_pointest,
            X.astype(float),
            Y,
            Yhat,
            X_unlabeled.astype(float),
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=use_unlabeled,
        )
        lam = calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord,
            clip=True,
        )
        return ppi_ols_pointestimate(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return ppi_pointest


def ppi_ols_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the OLS coefficients using the PPI++ algorithm from `[ADZ23] <https://arxiv.org/abs/2311.01453>`__.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the OLS coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lam != 0  # If lam is 0, revert to classical estimation.

    ppi_pointest = ppi_ols_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        use_unlabeled=use_unlabeled,
    )

    if lam is None:
        lam = calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord,
            clip=True,
        )
        return ppi_ols_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            alternative=alternative,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lam * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lam * grads_hat.T).reshape(d, d)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )


"""
    LOGISTIC REGRESSION

"""

@njit
def logistic_loss(_theta, X, Y, w):
    mu = X @ _theta
    return np.sum(w * (-Y * (mu) + safe_log1pexp(mu)))

@njit
def logistic_gradient(_theta, X, Y, w):
    return X.T @ (w * (safe_expit(X @ _theta) - Y))

def logistic_initial_params(X, Y):
    # Initialize theta
    theta = (
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X, Y)
        .coef_.squeeze()
    )
    if len(theta.shape) == 0:
        theta = theta.reshape(1)
    return theta


def ppi_multiple_logistic_pointestimate(
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
    return_lam=False,
):
    return ppi_multi_glm_pointest(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        logistic_initial_params,
        logistic_loss,
        logistic_gradient,
        get_stats=_logistic_get_stats,
        lam=lam,
        coord=None,
        w=w,
        w_unlabeled=w_unlabeled,
        return_lam=return_lam,
        method="L-BFGS-B",
        tol=1e-15,
        options=optimizer_options,
    )


def ppi_logistic_pointestimate(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered point estimate of the logistic regression coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, optional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Prediction-powered point estimate of the logistic regression coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """

    return ppi_multiple_logistic_pointestimate(
        X,
        Y,
        X,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam,
        coord,
        optimizer_options,
        w,
        w_unlabeled,
        return_lam=False,
    )


@njit
def logistic_scalar_grad(mu: NDArray, X: NDArray, Y: float) -> NDArray:
    return X * (mu - Y)


@njit
def logistic_scalar_hessian(mu: NDArray, X: NDArray) -> NDArray:
    return mu * (1 - mu) * np.outer(X, X)

def _logistic_get_stats(
    pointest,
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    w=None,
    w_unlabeled=None,
    use_unlabeled=True,
):
    return _glm_get_stats(
        link=safe_expit,
        scalar_grad=logistic_scalar_grad,
        scalar_hessian=logistic_scalar_hessian,
        pointest=pointest,
        X=X,
        Y=Y,
        Xhat=Xhat,
        Yhat=Yhat,
        Xhat_unlabeled=Xhat_unlabeled,
        Yhat_unlabeled=Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        use_unlabeled=use_unlabeled,
    )


def ppi_multiple_logistic_pval(
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
    alternative="two-sided",
):
    """Computes the prediction-powered pvalues on the logistic regression coefficients for the null hypothesis that the coefficient is zero.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, optional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.

    Returns:
        ndarray: Prediction-powered point estimate of the logistic regression coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """

    return ppi_multi_glm_pval(
        X=X,
        Y=Y,
        Xhat=X,
        Yhat=Yhat,
        Xhat_unlabeled=Xhat_unlabeled,
        Yhat_unlabeled=Yhat_unlabeled,
        initial_params=logistic_initial_params,
        loss=logistic_loss,
        gradient=logistic_gradient,
        get_stats=_logistic_get_stats,
        alternative=alternative,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
        optimizer_options=optimizer_options,
    )


def ppi_logistic_pval(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
    alternative="two-sided",
):
    return ppi_multiple_logistic_pval(
        X,
        Y,
        X,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam,
        coord,
        optimizer_options,
        w,
        w_unlabeled,
        alternative,
    )


def ppi_multiple_logistic_ci(
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    return ppi_multi_glm_ci(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        logistic_initial_params,
        logistic_loss,
        logistic_gradient,
        _logistic_get_stats,
        alpha,
        alternative,
        lam,
        coord,
        w,
        w_unlabeled,
        optimizer_options=optimizer_options,
    )


def ppi_logistic_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the logistic regression coefficients using the PPI++ algorithm from `[ADZ23] <https://arxiv.org/abs/2311.01453>`__.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, ooptional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
        w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
        w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the logistic regression coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """

    return ppi_multiple_logistic_ci(
        X,
        Y,
        X,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        alpha=0.1,
        alternative="two-sided",
        lam=None,
        coord=None,
        optimizer_options=None,
        w=None,
        w_unlabeled=None,
    )


@njit
def poisson_loss(_theta, X, Y, w):
    mu = X @ _theta
    return np.sum(w * (np.exp(mu) - Y * (mu)))


@njit
def poisson_gradient(_theta, X, Y, w):
    return X.T @ (w * np.exp(X @ _theta) - Y)


@njit
def poisson_scalar_grad(mu, X, Y):
    return X * (mu - Y)


@njit
def poisson_scalar_hessian(mu, X):
    return 2 * mu * np.outer(X, X)

def poisson_initial_params(X, Y):
    theta = (
        PoissonRegressor(
            alpha=0,
            fit_intercept=False,
            max_iter=10000,
            tol=1e-15,
        )
        .fit(X, Y)
        .coef_
    )
    if len(theta.shape) == 0:
        theta = theta.reshape(1)
    return theta


def _poisson_get_stats(
    pointest,
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    w=None,
    w_unlabeled=None,
    use_unlabeled=True,
):
    return _glm_get_stats(
        link=np.exp,
        scalar_grad=poisson_scalar_grad,
        scalar_hessian=poisson_scalar_hessian,
        pointest=pointest,
        X=X,
        Y=Y,
        Xhat=Xhat,
        Yhat=Yhat,
        Xhat_unlabeled=Xhat_unlabeled,
        Yhat_unlabeled=Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        use_unlabeled=use_unlabeled,
    )


def ppi_multiple_poisson_pointestimate(
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
    return_lam=False,
):
    return ppi_multi_glm_pointest(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        poisson_initial_params,
        poisson_loss,
        poisson_gradient,
        get_stats=_poisson_get_stats,
        lam=lam,
        coord=None,
        w=w,
        w_unlabeled=w_unlabeled,
        return_lam=return_lam,
        method="L-BFGS-B",
        tol=1e-15,
        options=optimizer_options,
    )


def ppi_poisson_pointestimate(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    return ppi_multiple_poisson_pointestimate(
        X,
        Y,
        X,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam,
        coord,
        optimizer_options,
        w,
        w_unlabeled,
        return_lam=False,
    )


def ppi_multiple_poisson_ci(
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    return ppi_multi_glm_ci(
        X=X,
        Y=Y,
        Xhat=Xhat,
        Yhat=Yhat,
        Xhat_unlabeled=Xhat_unlabeled,
        Yhat_unlabeled=Yhat_unlabeled,
        initial_params=poisson_initial_params,
        loss=poisson_loss,
        gradient=poisson_gradient,
        get_stats=_poisson_get_stats,
        alpha=alpha,
        alternative=alternative,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
        optimizer_options=optimizer_options,
    )


def ppi_poisson_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    return ppi_multiple_poisson_ci(
        X,
        Y,
        X,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        alpha=0.1,
        alternative="two-sided",
        lam=None,
        coord=None,
        optimizer_options=None,
        w=None,
        w_unlabeled=None,
    )


"""
    PPBOOT

"""


def ppboot(
    estimator,
    Y,
    Yhat,
    Yhat_unlabeled,
    X=None,
    X_unlabeled=None,
    lam=None,
    n_resamples=1000,
    n_resamples_lam=50,
    alpha=0.1,
    alternative="two-sided",
    method="percentile",
):
    """Computes the prediction-powered bootstrap confidence interval for the estimator.

    Args:
        estimator (callable): Estimator function. Takes in (X,Y) or (Y) and returns a point estimate.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        X (ndarray, optional): Covariates corresponding to the gold-standard labels. Defaults to `None`. If `None`, the estimator is assumed to only take in `Y`.
        X_unlabeled (ndarray, optional): Covariates corresponding to the unlabeled data. Defaults to `None`. If `None`, the estimator is assumed to only take in `Y`. If `X` is not `None`, `X_unlabeled` must also be provided, and vice versa.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ in addition to `[Z24] <https://arxiv.org/abs/2405.18379>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPBoot with no power tuning, and setting `lam=0` recovers the classical bootstrap interval.
        n_resamples (int, optional): Number of bootstrap resamples. Defaults to `1000`.
        n_resamples_lam (int, optional): Number of bootstrap resamples for the power-tuning parameter. Defaults to `50`.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1). Defaults to `0.1`.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'. Defaults to `'two-sided'`.
        method (str, optional): Method to compute the confidence interval, either 'percentile' or 'basic'. Defaults to `'percentile'`.

    Returns:
        float or ndarray: Lower and upper bounds of the prediction-powered bootstrap confidence interval for the estimator.

    Notes:
        `[Z24] <https://arxiv.org/abs/2405.18379>`__ T. Zrnic. A Note on the Prediction-Powered Bootstrap. arxiv:2405.18379, 2024.
    """
    if (X is None) and (X_unlabeled is not None):
        raise ValueError(
            "Both X and X_unlabeled must be either None, or take on values."
        )

    if X is None:

        def lam_statistic(Y, Yhat, Yhat_unlabeled, estimator=None):
            return {
                "Y": estimator(Y),
                "Yhat": estimator(Yhat),
                "Yhat_unlabeled": estimator(Yhat_unlabeled),
            }

        if lam is None:
            estimator_dicts = bootstrap(
                [Y, Yhat, Yhat_unlabeled],
                lam_statistic,
                n_resamples=n_resamples_lam,
                paired=[[0, 1]],
                statistic_kwargs={"estimator": estimator},
            )
            Y_samples = np.stack(
                [est_dict["Y"] for est_dict in estimator_dicts], axis=0
            )
            Yhat_samples = np.stack(
                [est_dict["Yhat"] for est_dict in estimator_dicts], axis=0
            )
            Yhat_unlabeled_samples = np.stack(
                [est_dict["Yhat_unlabeled"] for est_dict in estimator_dicts],
                axis=0,
            )

            cov_Y_Yhat = (
                np.sum(
                    [
                        np.cov(Y_samples[:, j], Yhat_samples[:, j])[0, 1]
                        for j in range(Y_samples.shape[1])
                    ]
                )
                if len(Y_samples.shape) > 1
                else np.cov(Y_samples, Yhat_samples)[0, 1]
            )
            var_Yhat = (
                np.sum(
                    [
                        np.var(Yhat_samples[:, j])
                        for j in range(Yhat_samples.shape[1])
                    ]
                )
                if len(Yhat_samples.shape) > 1
                else np.var(Yhat_samples)
            )
            var_Yhat_unlabeled = (
                np.sum(
                    [
                        np.var(Yhat_unlabeled_samples[:, j])
                        for j in range(Yhat_unlabeled_samples.shape[1])
                    ]
                )
                if len(Yhat_unlabeled_samples.shape) > 1
                else np.var(Yhat_unlabeled_samples)
            )
            lam = cov_Y_Yhat / (var_Yhat + var_Yhat_unlabeled)

        def rectified_estimator(Y, Yhat, Yhat_unlabeled, lam=None):
            return (
                lam * estimator(Yhat_unlabeled)
                + estimator(Y)
                - lam * estimator(Yhat)
            )

        ppi_pointest = rectified_estimator(Y, Yhat, Yhat_unlabeled, lam=lam)

        ppi_bootstrap_distribution = np.array(
            bootstrap(
                [Y, Yhat, Yhat_unlabeled],
                rectified_estimator,
                n_resamples=n_resamples,
                paired=[[0, 1]],
                statistic_kwargs={"lam": lam},
            )
        )

    else:

        def lam_statistic(
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled, estimator=None
        ):
            return {
                "XY": estimator(X, Y),
                "XYhat": estimator(X, Yhat),
                "XYhat_unlabeled": estimator(X_unlabeled, Yhat_unlabeled),
            }

        if lam is None:
            estimator_dicts = bootstrap(
                [X, Y, Yhat, X_unlabeled, Yhat_unlabeled],
                lam_statistic,
                n_resamples=n_resamples_lam,
                paired=[[0, 1, 2], [3, 4]],
                statistic_kwargs={"estimator": estimator},
            )
            XY_samples = np.stack(
                [est_dict["XY"] for est_dict in estimator_dicts], axis=0
            )
            XYhat_samples = np.stack(
                [est_dict["XYhat"] for est_dict in estimator_dicts], axis=0
            )
            XYhat_unlabeled_samples = np.stack(
                [est_dict["XYhat_unlabeled"] for est_dict in estimator_dicts],
                axis=0,
            )

            cov_XY_XYhat = (
                np.sum(
                    [
                        np.cov(XY_samples[:, j], XYhat_samples[:, j])[0, 1]
                        for j in range(XY_samples.shape[1])
                    ]
                )
                if len(XY_samples.shape) > 1
                else np.cov(XY_samples, XYhat_samples)[0, 1]
            )
            var_XYhat = (
                np.sum(
                    [
                        np.var(XYhat_samples[:, j])
                        for j in range(XYhat_samples.shape[1])
                    ]
                )
                if len(XYhat_samples.shape) > 1
                else np.var(XYhat_samples)
            )
            var_XYhat_unlabeled = (
                np.sum(
                    [
                        np.var(XYhat_unlabeled_samples[:, j])
                        for j in range(XYhat_unlabeled_samples.shape[1])
                    ]
                )
                if len(XYhat_unlabeled_samples.shape) > 1
                else np.var(XYhat_unlabeled_samples)
            )

            lam = cov_XY_XYhat / (var_XYhat + var_XYhat_unlabeled)

        def rectified_estimator(
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled, lam=None
        ):
            return (
                lam * estimator(X_unlabeled, Yhat_unlabeled)
                + estimator(X, Y)
                - lam * estimator(X, Yhat)
            )

        ppi_pointest = rectified_estimator(
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled, lam=lam
        )

        ppi_bootstrap_distribution = np.array(
            bootstrap(
                [X, Y, Yhat, X_unlabeled, Yhat_unlabeled],
                rectified_estimator,
                n_resamples=n_resamples,
                paired=[[0, 1, 2], [3, 4]],
                statistic_kwargs={"lam": lam},
            )
        )

    # Deal with the different types of alternative hypotheses
    if alternative == "two-sided":
        alpha_lower = alpha / 2
        alpha_upper = alpha / 2
    elif alternative == "larger":
        alpha_lower = alpha
        alpha_upper = 0
    elif alternative == "smaller":
        alpha_lower = 0
        alpha_upper = alpha

    # Compute the lower and upper bounds depending on the method
    if method == "percentile":
        lower_bound = np.quantile(
            ppi_bootstrap_distribution, alpha_lower, axis=0
        )
        upper_bound = np.quantile(
            ppi_bootstrap_distribution, 1 - alpha_upper, axis=0
        )
    elif method == "basic":
        lower_bound = 2 * ppi_pointest - np.quantile(
            ppi_bootstrap_distribution, 1 - alpha_lower, axis=0
        )
        upper_bound = 2 * ppi_pointest - np.quantile(
            ppi_bootstrap_distribution, alpha_upper, axis=0
        )
    else:
        raise ValueError(
            "Method must be either 'percentile' or 'basic'. The others are not implemented yet... want to contribute? ;)"
        )

    if alternative == "two-sided":
        return lower_bound, upper_bound
    elif alternative == "larger":
        return -np.inf, upper_bound
    elif alternative == "smaller":
        return lower_bound, np.inf
    else:
        raise ValueError(
            "Alternative must be either 'two-sided', 'larger' or 'smaller'."
        )


"""
    DISCRETE DISTRIBUTION ESTIMATION UNDER LABEL SHIFT

"""


def ppi_distribution_label_shift_ci(
    Y, Yhat, Yhat_unlabeled, K, nu, alpha=0.1, delta=None, return_counts=True
):
    """Computes the prediction-powered confidence interval for nu^T f for a discrete distribution f, under label shift.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        K (int): Number of classes.
        nu (ndarray): Vector nu. Coordinates must be bounded within [0, 1].
        alpha (float, optional): Final error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        delta (float, optional): Error level of the intermediate confidence interval for the mean. Must be in (0, alpha). If return_counts == False, then delta is set equal to alpha and ignored.
        return_counts (bool, optional): Whether to return the number of samples in each class as opposed to the mean.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for nu^T f for a discrete distribution f, under label shift.
    """
    if not return_counts:
        delta = alpha
    if delta is None:
        delta = alpha * 0.95
    # Construct the confusion matrix
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]

    # Construct column-normalized confusion matrix Ahat
    C = np.zeros((K, K)).astype(int)
    for j in range(K):
        for l in range(K):
            C[j, l] = np.bitwise_and(Yhat == j, Y == l).astype(int).sum()
    Ahat = C / C.sum(axis=0)

    # Invert Ahat
    Ahatinv = np.linalg.inv(Ahat)
    qfhat = form_discrete_distribution(Yhat_unlabeled)

    # Calculate the bound
    point_estimate = nu @ Ahatinv @ qfhat

    nmin = C.sum(axis=0).min()

    def invert_budget_split(budget_split):
        return np.sqrt(1 / (4 * nmin)) * (
            norm.ppf(1 - (budget_split * delta) / (2 * K**2))
            - norm.ppf((budget_split * delta) / (2 * K**2))
        ) - np.sqrt(2 / N * np.log(2 / ((1 - budget_split) * delta)))

    try:
        budget_split = brentq(invert_budget_split, 1e-9, 1 - 1e-9)
    except:
        budget_split = 0.999999
    epsilon1 = max(
        [
            linfty_binom(C.sum(axis=0)[k], K, budget_split * delta, Ahat[:, k])
            for k in range(K)
        ]
    )
    epsilon2 = linfty_dkw(N, K, (1 - budget_split) * delta)

    qyhat_lb = np.clip(point_estimate - epsilon1 - epsilon2, 0, 1)
    qyhat_ub = np.clip(point_estimate + epsilon1 + epsilon2, 0, 1)

    if return_counts:
        count_lb = int(binom.ppf((alpha - delta) / 2, N, qyhat_lb))
        count_ub = int(binom.ppf(1 - (alpha - delta) / 2, N, qyhat_ub))
        return count_lb, count_ub
    else:
        return qyhat_lb, qyhat_ub
