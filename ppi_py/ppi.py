import numpy as np
from scipy.stats import norm, binom
from scipy.special import expit
from scipy.optimize import brentq
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
from .utils import (
    dataframe_decorator,
    linfty_dkw,
    linfty_binom,
    form_discrete_distribution,
)
import pdb


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
    rectified_std = np.maximum(
        np.sqrt(imputed_std**2 + rectifier_std**2), 1e-16
    )
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
    indicators = (Y[:, None] <= grid[None, :]).astype(float)
    return indicators.mean(axis=0), indicators.std(axis=0)


def _compute_cdf_diff(Y, Yhat, grid):
    indicators_Y = (Y[:, None] <= grid[None, :]).astype(float)
    indicators_Yhat = (Yhat[:, None] <= grid[None, :]).astype(float)
    return (indicators_Y - indicators_Yhat).mean(axis=0), (
        indicators_Y - indicators_Yhat
    ).std(axis=0)


def _rectified_cdf(Y, Yhat, Yhat_unlabeled, grid):
    cdf_Yhat_unlabeled, _ = _compute_cdf(Yhat_unlabeled, grid)
    cdf_rectifier, _ = _compute_cdf_diff(Y, Yhat, grid)
    return cdf_Yhat_unlabeled + cdf_rectifier


def ppi_quantile_pointestimate(Y, Yhat, Yhat_unlabeled, q, exact_grid=False):
    assert len(Y.shape) == 1
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
    rectified_cdf = _rectified_cdf(Y, Yhat, Yhat_unlabeled, grid)
    minimizers = np.argmin(np.abs(rectified_cdf - q))
    minimizer = (
        minimizers
        if isinstance(minimizers, (int, np.int64))
        else minimizers[0]
    )
    return grid[
        minimizer
    ]  # Find the intersection of the rectified CDF and the quantile


def ppi_quantile_ci(Y, Yhat, Yhat_unlabeled, q, alpha=0.1, exact_grid=False):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
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
    imputed_theta, imputed_se = _ols(
        X_unlabeled, Yhat_unlabeled, return_se=True
    )
    rectifier, rectifier_se = _ols(X, Y - Yhat, return_se=True)
    return _rectified_ci(
        imputed_theta,
        imputed_se,
        rectifier,
        rectifier_se,
        alpha,
        alternative="two-sided",
    )


def eff_ppi_ols_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=0.1, alternative='two-sided'):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]

    ppi_pointest = ppi_ols_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled)

    Hessian = np.zeros((d,d))
    grads_til = np.zeros(X_unlabeled.shape)
    for i in range(N):
        Hessian += 1/N * np.outer(X_unlabeled[i], X_unlabeled[i])
        grads_til[i,:] = X_unlabeled[i,:]*(np.dot(X_unlabeled[i,:], ppi_pointest) - Yhat_unlabeled[i])

    inv_Hessian = np.linalg.inv(Hessian)
    var_unlabeled = np.cov(grads_til.T)

    pred_error = Yhat - Y
    grad_diff = np.diag(pred_error) @ X
    var = np.cov(grad_diff.T)

    Sigma_hat = inv_Hessian @ (n/N * var_unlabeled + var) @ inv_Hessian

    return _zconfint_generic(ppi_pointest, np.sqrt(np.diag(Sigma_hat)/n), alpha=alpha, alternative=alternative)


"""
    LOGISTIC REGRESSION

"""


# Todo: Numba accel
def ppi_logistic_pointestimate(
    X, Y, Yhat, X_unlabeled, Yhat_unlabeled, step_size=1e-3, grad_tol=5e-16
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    rectifier = 1 / n * X.T @ (Yhat - Y)
    theta = (
        LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X, Y)
        .coef_.squeeze()
    )
    mu_theta = expit(X_unlabeled @ theta)
    grad = grad_tol * np.ones(d) + 1  # Initialize to enter while loop
    while np.linalg.norm(grad) > grad_tol:
        mu_theta = expit(X_unlabeled @ theta)
        grad = 1 / N * X_unlabeled.T @ (mu_theta - Yhat_unlabeled) + rectifier
        theta -= step_size * grad
    return theta


def ppi_logistic_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    grid_size=200,
    grid_limit=800,
    max_refinements=10,
    grid_radius=1,
    grid_relative=False,
    step_size=1e-3,  # Optimizer step size
    grad_tol=5e-16,  # Optimizer grad tol
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        step_size=step_size,
        grad_tol=grad_tol,
    )
    if grid_relative:
        grid_radius *= ppi_pointest
    rectifier = 1 / n * X.T @ (Yhat - Y)
    rectifier_std = np.std(X * (Yhat - Y)[:, None], axis=0)
    confset = []
    grid_edge_accepted = True
    refinements = -1
    while (len(confset) == 0) or (grid_edge_accepted == True):
        refinements += 1
        if refinements > max_refinements:
            return np.array([-np.infty] * d), np.array([np.infty] * d)
        grid_radius *= 2
        grid_size *= 2
        grid_size = min(grid_size, grid_limit)
        theta_grid = np.concatenate(
            [
                np.linspace(
                    ppi_pointest - grid_radius * np.ones(d),
                    ppi_pointest,
                    grid_size // 2,
                ),
                np.linspace(
                    ppi_pointest,
                    ppi_pointest + grid_radius * np.ones(d),
                    grid_size // 2,
                )[1:],
            ]
        )
        mu_theta = expit(X_unlabeled @ theta_grid.T)
        grad = 1 / N * X_unlabeled.T @ (mu_theta - Yhat_unlabeled[:, None])
        prederr_std = np.std(
            X_unlabeled[:, :, None]
            * (mu_theta - Yhat_unlabeled[:, None])[:, None, :],
            axis=0,
        )
        w = norm.ppf(1 - alpha / (2 * d)) * np.sqrt(
            rectifier_std[:, None] ** 2 / n + prederr_std**2 / N
        )
        accept = np.all(np.abs(grad + rectifier[:, None]) <= w, axis=0)
        confset = theta_grid[accept]
        grid_edge_accepted = accept[0] or accept[-1]
    return confset.min(axis=0), confset.max(axis=0)


def eff_ppi_logistic_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    step_size=1e-3,  # Optimizer step size
    grad_tol=5e-16,  # Optimizer grad tol
    alternative='two-sided'
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]

    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        step_size=step_size,
        grad_tol=grad_tol,
    )

    mu_til = expit(X_unlabeled@ppi_pointest)

    Hessian = np.zeros((d,d))
    grads_til = np.zeros(X_unlabeled.shape)
    for i in range(N):
        Hessian += 1/N * mu_til[i] * (1-mu_til[i]) * np.outer(X_unlabeled[i], X_unlabeled[i])
        grads_til[i,:] = X_unlabeled[i,:]*(mu_til[i] - Yhat_unlabeled[i])

    inv_Hessian = np.linalg.inv(Hessian)
    var_unlabeled = np.cov(grads_til.T)

    pred_error = Yhat - Y
    grad_diff = np.diag(pred_error) @ X
    var = np.cov(grad_diff.T)

    Sigma_hat = inv_Hessian @ (n/N * var_unlabeled + var) @ inv_Hessian

    return _zconfint_generic(ppi_pointest, np.sqrt(np.diag(Sigma_hat)/n), alpha=alpha, alternative=alternative)

"""
    ORDINARY LEAST SQUARES UNDER COVARIATE SHIFT

"""


def _wls(X, Y, w, return_se=False):
    regression = WLS(Y, exog=X, weights=w).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta


def ppi_ols_covshift_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, w):
    imputed_theta = _wls(X_unlabeled, Yhat_unlabeled)
    rectifier = _wls(X, Y - Yhat, w)
    return imputed_theta + rectifier


def ppi_ols_covshift_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, w, alpha=0.1):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    imputed_theta, imputed_se = _ols(
        X_unlabeled, Yhat_unlabeled, return_se=True
    )
    rectifier, rectifier_se = _wls(X, Y - Yhat, w, return_se=True)
    return _rectified_ci(
        imputed_theta,
        imputed_se,
        rectifier,
        rectifier_se,
        alpha,
        alternative="two-sided",
    )


"""
    DISCRETE DISTRIBUTION ESTIMATION UNDER LABEL SHIFT ʕ·ᴥ·ʔ

"""


def ppi_distribution_label_shift_ci(
    Y, Yhat, Yhat_unlabeled, K, nu, alpha, delta, return_counts=True
):
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
