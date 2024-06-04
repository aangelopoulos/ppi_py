import numpy as np
from numba import njit
from scipy.stats import norm, binom
from scipy.special import expit
from scipy.optimize import brentq, minimize
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
import warnings

warnings.simplefilter("ignore")
from .utils import (
    construct_weight_vector,
    safe_expit,
    safe_log1pexp,
    compute_cdf,
    compute_cdf_diff,
    dataframe_decorator,
    linfty_dkw,
    linfty_binom,
    form_discrete_distribution,
    reshape_to_2d,
    bootstrap,
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
        lam = _calc_lam_glm(
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
        lam = _calc_lam_glm(
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
    # w = np.ones(n) if w is None else w / w.sum() * n
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
        lam = _calc_lam_glm(
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
        lam = _calc_lam_glm(
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
        lam = _calc_lam_glm(
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
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    if optimizer_options is None:
        optimizer_options = {"ftol": 1e-15}
    if "ftol" not in optimizer_options.keys():
        optimizer_options["ftol"] = 1e-15

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

    lam_curr = 1 if lam is None else lam

    def rectified_logistic_loss(_theta):
        return (
            lam_curr
            / N
            * np.sum(
                w_unlabeled
                * (
                    -Yhat_unlabeled * (X_unlabeled @ _theta)
                    + safe_log1pexp(X_unlabeled @ _theta)
                )
            )
            - lam_curr
            / n
            * np.sum(w * (-Yhat * (X @ _theta) + safe_log1pexp(X @ _theta)))
            + 1
            / n
            * np.sum(w * (-Y * (X @ _theta) + safe_log1pexp(X @ _theta)))
        )

    def rectified_logistic_grad(_theta):
        return (
            lam_curr
            / N
            * X_unlabeled.T
            @ (
                w_unlabeled
                * (safe_expit(X_unlabeled @ _theta) - Yhat_unlabeled)
            )
            - lam_curr / n * X.T @ (w * (safe_expit(X @ _theta) - Yhat))
            + 1 / n * X.T @ (w * (safe_expit(X @ _theta) - Y))
        )

    ppi_pointest = minimize(
        rectified_logistic_loss,
        theta,
        jac=rectified_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options,
    ).x

    if lam is None:
        (
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
        ) = _logistic_get_stats(
            ppi_pointest,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            w,
            w_unlabeled,
        )
        lam = _calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        return ppi_logistic_pointestimate(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            optimizer_options=optimizer_options,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return ppi_pointest


@njit
def _logistic_get_stats(
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
    """Computes the statistics needed for the logistic regression confidence interval.

    Args:
        pointest (ndarray): Point estimate of the logistic regression coefficients.
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        w (ndarray, optional): Standard errors of the gold-standard labels.
        w_unlabeled (ndarray, optional): Standard errors of the unlabeled data.
        use_unlabeled (bool, optional): Whether to use the unlabeled data.

    Returns:
        grads (ndarray): Gradient of the loss function on the labeled data.
        grads_hat (ndarray): Gradient of the loss function on the labeled predictions.
        grads_hat_unlabeled (ndarray): Gradient of the loss function on the unlabeled predictions.
        inv_hessian (ndarray): Inverse Hessian of the loss function on the unlabeled data.
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

    mu = safe_expit(X @ pointest)
    mu_til = safe_expit(X_unlabeled @ pointest)

    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(X_unlabeled.shape)
    if use_unlabeled:
        for i in range(N):
            hessian += (
                w_unlabeled[i]
                / (N + n)
                * mu_til[i]
                * (1 - mu_til[i])
                * np.outer(X_unlabeled[i], X_unlabeled[i])
            )
            grads_hat_unlabeled[i, :] = (
                w_unlabeled[i]
                * X_unlabeled[i, :]
                * (mu_til[i] - Yhat_unlabeled[i])
            )

    grads = np.zeros(X.shape)
    grads_hat = np.zeros(X.shape)
    for i in range(n):
        hessian += (
            w[i] / (N + n) * mu[i] * (1 - mu[i]) * np.outer(X[i], X[i])
            if use_unlabeled
            else w[i] / n * mu[i] * (1 - mu[i]) * np.outer(X[i], X[i])
        )
        grads[i, :] = w[i] * X[i, :] * (mu[i] - Y[i])
        grads_hat[i, :] = w[i] * X[i, :] * (mu[i] - Yhat[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    return grads, grads_hat, grads_hat_unlabeled, inv_hessian


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
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lam != 0

    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options=optimizer_options,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _logistic_get_stats(
        ppi_pointest,
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w,
        w_unlabeled,
        use_unlabeled=use_unlabeled,
    )
    if lam is None:
        lam = _calc_lam_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        return ppi_logistic_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            optimizer_options=optimizer_options,
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

    vhat = inv_hessian if coord is None else inv_hessian[coord, coord]
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
    qfhat = form_discrete_distribution(Yhat_unlabeled, sorted_highlow=True)

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
