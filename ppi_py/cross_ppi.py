import numpy as np
from numba import njit
from scipy.stats import norm, binom
from scipy.special import expit
from scipy.optimize import brentq, minimize
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
from . import ppi_mean_pointestimate, ppi_ols_pointestimate, rectified_p_value
from .utils import compute_cdf, compute_cdf_diff, safe_expit, safe_log1pexp

"""
    MEAN ESTIMATION

"""


def crossppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled):
    """Computes the cross-prediction-powered point estimate of the mean.

    Args:
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).

    Returns:
        float or ndarray: Cross-prediction-powered point estimate of the mean.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    return ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled.mean(axis=1), lam=1)


def crossppi_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    bootstrap_data=None,
):
    """Computes the cross-prediction-powered confidence interval for the mean.

    Args:
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        bootstrap_data (dict, optional): Bootstrap data used to estimate the variance of the point estimate. Assumes keys "Y", "Yhat", "Yhat_unlabeled".
    Returns:
        tuple: Lower and upper bounds of the cross-prediction-powered confidence interval for the mean.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    crossppi_pointest = crossppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled)
    if bootstrap_data == None:
        Y_bstrap = Y
        Yhat_bstrap = Yhat
        Yhat_unlabeled_bstrap = Yhat_unlabeled
    else:
        Y_bstrap = bootstrap_data["Y"]
        Yhat_bstrap = bootstrap_data["Yhat"]
        Yhat_unlabeled_bstrap = bootstrap_data["Yhat_unlabeled"]

    imputed_var = Yhat_unlabeled_bstrap.mean(axis=1).var() / N
    rectifier_var = (Yhat_bstrap - Y_bstrap).var() / n

    return _zconfint_generic(
        crossppi_pointest,
        np.sqrt(imputed_var + rectifier_var),
        alpha,
        alternative=alternative,
    )


"""
    QUANTILE ESTIMATION

"""


def _cross_rectified_cdf(Y, Yhat, Yhat_unlabeled, grid):
    """Computes the cross-prediction estimate of the CDF of the data.

    Args:
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        grid (ndarray): Grid of values to compute the CDF at.

    Returns:
        ndarray: Cross-prediction estimate of the CDF of the data at the specified grid points.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    K = Yhat_unlabeled.shape[1]
    cdf_Yhat_unlabeled = np.zeros(len(grid))
    for j in range(K):
        cdf_Yhat_unlabeled_temp, _ = compute_cdf(Yhat_unlabeled[:, j], grid)
        cdf_Yhat_unlabeled += cdf_Yhat_unlabeled_temp / K
    cdf_rectifier, _ = compute_cdf_diff(Y, Yhat, grid)
    return cdf_Yhat_unlabeled + cdf_rectifier


def crossppi_quantile_pointestimate(
    Y, Yhat, Yhat_unlabeled, q, exact_grid=False
):
    """Computes the cross-prediction-powered point estimate of the quantile.

    Args:
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        q (float): Quantile to estimate.
        exact_grid (bool, optional): Whether to compute the exact solution (True) or an approximate solution based on a linearly spaced grid of 5000 values (False).

    Returns:
        float: Cross-prediction-powered point estimate of the quantile.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    if len(Y.shape) != 1:
        raise ValueError(
            "Quantiles are only implemented for 1-dimensional arrays."
        )
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled.flatten()], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
    rectified_cdf = _cross_rectified_cdf(Y, Yhat, Yhat_unlabeled, grid)
    minimizers = np.argmin(np.abs(rectified_cdf - q))
    minimizer = (
        minimizers
        if isinstance(minimizers, (int, np.int64))
        else minimizers[0]
    )
    return grid[
        minimizer
    ]  # Find the intersection of the rectified CDF and the quantile


def crossppi_quantile_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    q,
    alpha=0.1,
    alternative="two-sided",
    bootstrap_data=None,
    exact_grid=False,
):
    """Computes the cross-prediction-powered confidence interval for the quantile.

    Args:
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        q (float): Quantile to estimate. Must be in the range (0, 1).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        bootstrap_data (dict, optional): Bootstrap data used to estimate the variance of the point estimate. Assumes keys "Y", "Yhat", "Yhat_unlabeled".
        exact_grid (bool, optional): Whether to use the exact grid of values or a linearly spaced grid of 5000 values.

    Returns:
        tuple: Lower and upper bounds of the cross-prediction-powered confidence interval for the quantile.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    K = Yhat_unlabeled.shape[1]

    grid = np.concatenate([Y, Yhat, Yhat_unlabeled.flatten()], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)

    cdf_Yhat_unlabeled = np.zeros(len(grid))
    for j in range(K):
        cdf_Yhat_unlabeled_temp, _ = compute_cdf(Yhat_unlabeled[:, j], grid)
        cdf_Yhat_unlabeled += cdf_Yhat_unlabeled_temp / K
    cdf_rectifier, _ = compute_cdf_diff(Y, Yhat, grid)

    if bootstrap_data == None:
        Y_bstrap = Y
        Yhat_bstrap = Yhat
        Yhat_unlabeled_bstrap = Yhat_unlabeled
    else:
        Y_bstrap = bootstrap_data["Y"]
        Yhat_bstrap = bootstrap_data["Yhat"]
        Yhat_unlabeled_bstrap = bootstrap_data["Yhat_unlabeled"]

    Yhat_unlabeled_bstrap_mean = Yhat_unlabeled_bstrap.mean(axis=1)
    indicators_Yhat_unlabeled_bstrap = (
        Yhat_unlabeled_bstrap_mean[:, None] <= grid[None, :]
    ).astype(float)
    indicators_Y_bstrap = (Y_bstrap[:, None] <= grid[None, :]).astype(float)
    indicators_Yhat_bstrap = (Yhat_bstrap[:, None] <= grid[None, :]).astype(
        float
    )
    imputed_std = (indicators_Yhat_unlabeled_bstrap).std(axis=0)
    rectifier_std = (indicators_Y_bstrap - indicators_Yhat_bstrap).std(axis=0)

    # Calculate rectified p-value for null that the rectified cdf is equal to q
    rectified_p_val = rectified_p_value(
        cdf_rectifier,
        rectifier_std / np.sqrt(n),
        cdf_Yhat_unlabeled,
        imputed_std / np.sqrt(N),
        null=q,
        alternative=alternative,
    )
    # Return the min and max values of the grid where p > alpha
    return grid[rectified_p_val > alpha][[0, -1]]


"""
     ORDINARY LEAST SQUARES

"""


def crossppi_ols_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled):
    """Computes the cross-prediction-powered point estimate of the OLS coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels. Shape (n, d).
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data. Shape (N, d).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).

    Returns:
        ndarray: Cross-prediction-powered point estimate of the OLS coefficients.

    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    return ppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled.mean(axis=1), lam=1
    )


def crossppi_ols_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    bootstrap_data=None,
):
    """Computes the cross-prediction-powered point estimate of the OLS coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels. Shape (n, d).
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data. Shape (N, d).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        bootstrap_data (dict, optional): Bootstrap data used to estimate the variance of the point estimate. Assumes keys "X", "Y", "Yhat", "Yhat_unlabeled".

    Returns:
        tuple: Lower and upper bounds of the cross-prediction-powered confidence interval for the OLS coefficients.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]

    crossppi_pointest = crossppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled
    )

    if bootstrap_data == None:
        X_bstrap = X
        Y_bstrap = Y
        Yhat_bstrap = Yhat
        Yhat_unlabeled_bstrap = Yhat_unlabeled.mean(axis=1)
    else:
        X_bstrap = bootstrap_data["X"]
        Y_bstrap = bootstrap_data["Y"]
        Yhat_bstrap = bootstrap_data["Yhat"]
        Yhat_unlabeled_bstrap = bootstrap_data["Yhat_unlabeled"].mean(axis=1)

    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(X_unlabeled.shape)
    for i in range(N):
        hessian += 1 / (N + n) * np.outer(X_unlabeled[i], X_unlabeled[i])
        grads_hat_unlabeled[i, :] = X_unlabeled[i, :] * (
            np.dot(X_unlabeled[i, :], crossppi_pointest)
            - Yhat_unlabeled_bstrap[i]
        )

    for i in range(n):
        hessian += 1 / (N + n) * np.outer(X[i], X[i])

    grads_diff = np.zeros(X_bstrap.shape)
    for i in range(X_bstrap.shape[0]):
        grads_diff[i, :] = X_bstrap[i, :] * (Yhat_bstrap[i] - Y_bstrap[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    var_unlabeled = np.cov(grads_hat_unlabeled.T).reshape(d, d)
    var = np.cov(grads_diff.T).reshape(d, d)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    return _zconfint_generic(
        crossppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )


"""
    LOGISTIC REGRESSION

"""


def crossppi_logistic_pointestimate(
    X, Y, Yhat, X_unlabeled, Yhat_unlabeled, optimizer_options=None
):
    """Computes the cross-prediction-powered point estimate of the logistic regression coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels. Shape (n, d).
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data. Shape (N, d).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        optimizer_options (dict, optional): Options to pass to the optimizer. See scipy.optimize.minimize for details.

    Returns:
        ndarray: Cross-prediction-powered point estimate of the logistic regression coefficients.
    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
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

    def rectified_logistic_loss(_theta):
        return (
            1
            / N
            * np.sum(
                -Yhat_unlabeled.mean(axis=1) * (X_unlabeled @ _theta)
                + safe_log1pexp(X_unlabeled @ _theta)
            )
            - 1 / n * np.sum(-Yhat * (X @ _theta) + safe_log1pexp(X @ _theta))
            + 1 / n * np.sum(-Y * (X @ _theta) + safe_log1pexp(X @ _theta))
        )

    def rectified_logistic_grad(_theta):
        return (
            1
            / N
            * X_unlabeled.T
            @ (safe_expit(X_unlabeled @ _theta) - Yhat_unlabeled.mean(axis=1))
            - 1 / n * X.T @ (safe_expit(X @ _theta) - Yhat)
            + 1 / n * X.T @ (safe_expit(X @ _theta) - Y)
        )

    crossppi_pointest = minimize(
        rectified_logistic_loss,
        theta,
        jac=rectified_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options,
    ).x

    return crossppi_pointest


def crossppi_logistic_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    bootstrap_data=None,
    optimizer_options=None,
):
    """Computes the cross-prediction-powered confidence interval for the logistic regression coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels. Shape (n, d).
        Y (ndarray): Gold-standard labels. Shape (n,).
        Yhat (ndarray): Predictions corresponding to the gold-standard labels. Shape (n,).
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data. Shape (N, d).
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data. Columns contain predictions from different models. Shape (N, K).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        bootstrap_data (dict, optional): Bootstrap data used to estimate the variance of the point estimate. Assumes keys "X", "Y", "Yhat", "Yhat_unlabeled".
        optimizer_options (dict, ooptional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
    Returns:
        tuple: Lower and upper bounds of the cross-prediction-powered confidence interval for the logistic regression coefficients.

    """
    if len(Yhat_unlabeled.shape) != 2:
        raise ValueError(
            "Yhat_unlabeled must be a 2-dimensional array with shape (N, K)."
        )
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    if bootstrap_data == None:
        X_bstrap = X
        Y_bstrap = Y
        Yhat_bstrap = Yhat
        Yhat_unlabeled_bstrap = Yhat_unlabeled.mean(axis=1)
    else:
        X_bstrap = bootstrap_data["X"]
        Y_bstrap = bootstrap_data["Y"]
        Yhat_bstrap = bootstrap_data["Yhat"]
        Yhat_unlabeled_bstrap = bootstrap_data["Yhat_unlabeled"].mean(axis=1)

    crossppi_pointest = crossppi_logistic_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled, optimizer_options
    )

    mu = safe_expit(X @ crossppi_pointest)
    mu_til = safe_expit(X_unlabeled @ crossppi_pointest)

    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(X_unlabeled.shape)
    for i in range(N):
        hessian += (
            1
            / (N + n)
            * mu_til[i]
            * (1 - mu_til[i])
            * np.outer(X_unlabeled[i], X_unlabeled[i])
        )
        grads_hat_unlabeled[i, :] = X_unlabeled[i, :] * (
            mu_til[i] - Yhat_unlabeled_bstrap[i]
        )

    for i in range(n):
        hessian += 1 / (N + n) * mu[i] * (1 - mu[i]) * np.outer(X[i], X[i])

    grads_diff = np.zeros(X_bstrap.shape)
    for i in range(X_bstrap.shape[0]):
        grads_diff[i, :] = X_bstrap[i, :] * (Yhat_bstrap[i] - Y_bstrap[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    var_unlabeled = np.cov(grads_hat_unlabeled.T).reshape(d, d)
    var = np.cov(grads_diff.T).reshape(d, d)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    return _zconfint_generic(
        crossppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )
