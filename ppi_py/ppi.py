import numpy as np
from numba import njit
from scipy.stats import norm, binom
from scipy.special import expit
from scipy.optimize import brentq, minimize
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
from .utils import (
    compute_cdf,
    compute_cdf_diff,
    dataframe_decorator,
    linfty_dkw,
    linfty_binom,
    form_discrete_distribution,
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
    Y, Yhat, Yhat_unlabeled, lhat=None, coord=None, w=None, w_unlabeled=None
):
    """Computes the prediction-powered point estimate of the d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the dimension of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.

    Returns:
        float or ndarray: Prediction-powered point estimate of the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Yhat.shape[1] if len(Yhat.shape) > 1 else 1
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    if lhat is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean() + (
            w * (Y - Yhat)
        ).mean()
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
        )
        return ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return (w_unlabeled * lhat * Yhat_unlabeled).mean(axis=0) + (
            w * (Y - lhat * Yhat)
        ).mean(axis=0)


def ppi_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for a d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
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
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    if lhat is None:
        ppi_pointest = ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=1,
            w=w,
            w_unlabeled=w_unlabeled,
        )
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
        )
        return ppi_mean_ci(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    imputed_std = (w_unlabeled * (lhat * Yhat_unlabeled)).std() / np.sqrt(N)
    rectifier_std = (w * (Y - lhat * Yhat)).std() / np.sqrt(n)

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
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered p-value for a 1D mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        null (float): Value of the null hypothesis to be tested.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        float or ndarray: Prediction-powered p-value for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    if lhat is None:
        if len(Y.shape) > 1 and Y.shape[1] > 1:
            lhat = 1
        else:
            ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean() + (
                w * (Y - Yhat)
            ).mean()
            grads = w * (Y - ppi_pointest)
            grads_hat = w * (Yhat - ppi_pointest)
            grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
            inv_hessian = np.ones((1, 1))
            lhat = _calc_lhat_glm(
                grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None
            )

    return _rectified_p_value(
        (w * Y - lhat * w * Yhat).mean(),
        (w * Y - lhat * w * Yhat).std() / np.sqrt(n),
        (w_unlabeled * lhat * Yhat_unlabeled).mean(),
        (w_unlabeled * lhat * Yhat_unlabeled).std() / np.sqrt(N),
        null,
        alternative,
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
    cdf_Yhat_unlabeled, _ = _compute_cdf(Yhat_unlabeled, grid, w=w_unlabeled)
    cdf_rectifier, _ = _compute_cdf_diff(Y, Yhat, grid, w=w)
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
    cdf_Yhat_unlabeled, cdf_Yhat_unlabeled_std = _compute_cdf(
        Yhat_unlabeled, grid, w=w_unlabeled
    )
    cdf_rectifier, cdf_rectifier_std = _compute_cdf_diff(Y, Yhat, grid, w=w)
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
    lhat=None,
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
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
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
    use_unlabeled = lhat != 0

    imputed_theta = (
        _wls(X_unlabeled, Yhat_unlabeled, w=w_unlabeled)
        if lhat is None
        else _wls(X_unlabeled, lhat * Yhat_unlabeled, w=w_unlabeled)
    )
    rectifier = (
        _wls(X, Y - Yhat, w=w)
        if lhat is None
        else _wls(X, Y - lhat * Yhat, w=w)
    )
    ppi_pointest = imputed_theta + rectifier

    if lhat is None:
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
        lhat = _calc_lhat_glm(
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
            lhat=lhat,
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
    lhat=None,
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
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
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
    use_unlabeled = lhat != 0  # If lhat is 0, revert to classical estimation.

    ppi_pointest = ppi_ols_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lhat=lhat,
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

    if lhat is None:
        lhat = _calc_lhat_glm(
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
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(d, d)

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
    lhat=None,
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
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
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
    if "ftol" not in optimizer_options.keys():
        optimizer_options = {"ftol": 1e-15}

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

    lhat_curr = 1 if lhat is None else lhat

    def rectified_logistic_loss(_theta):
        return (
            lhat_curr
            / N
            * np.sum(
                w_unlabeled
                * (
                    -Yhat_unlabeled * (X_unlabeled @ _theta)
                    + safe_log1pexp(X_unlabeled @ _theta)
                )
            )
            - lhat_curr
            / n
            * np.sum(w * (-Yhat * (X @ _theta) + safe_log1pexp(X @ _theta)))
            + 1
            / n
            * np.sum(w * (-Y * (X @ _theta) + safe_log1pexp(X @ _theta)))
        )

    def rectified_logistic_grad(_theta):
        return (
            lhat_curr
            / N
            * X_unlabeled.T
            @ (
                w_unlabeled
                * (safe_expit(X_unlabeled @ _theta) - Yhat_unlabeled)
            )
            - lhat_curr / n * X.T @ (w * (safe_expit(X @ _theta) - Yhat))
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

    if lhat is None:
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
        lhat = _calc_lhat_glm(
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
            lhat=lhat,
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
    lhat=None,
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
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
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
    use_unlabeled = lhat != 0

    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options=optimizer_options,
        lhat=lhat,
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
    if lhat is None:
        lhat = _calc_lhat_glm(
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
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(d, d)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )


def _calc_lhat_glm(
    grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None, clip=False
):
    """
    Calculates the optimal value of lhat for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lhat to be non-negative. Defaults to `False`.

    Returns:
        float: Optimal value of `lhat`. Lies in [0,1].
    """
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    cov_grads = np.zeros((d, d))

    for i in range(n):
        cov_grads += (1 / n) * (
            np.outer(
                grads[i] - grads.mean(axis=0),
                grads_hat[i] - grads_hat.mean(axis=0),
            )
            + np.outer(
                grads_hat[i] - grads_hat.mean(axis=0),
                grads[i] - grads.mean(axis=0),
            )
        )
    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )

    if coord is None:
        vhat = inv_hessian
    else:
        vhat = inv_hessian @ np.eye(d)[coord]

    if d > 1:
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
    else:
        num = vhat * cov_grads * vhat
        denom = 2 * (1 + (n / N)) * vhat * var_grads_hat * vhat

    lhat = num / denom
    if clip:
        lhat = np.clip(lhat, 0, 1)
    return lhat.item()


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
