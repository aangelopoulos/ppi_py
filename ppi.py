import numpy as np
from scipy.stats import norm
from scipy.special import expit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
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


def classical_mean_ci(Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    return _zconfint_generic(
         Y.mean(), Y.std / np.sqrt(n) , alpha, alternative
    )


def semisupervised_mean_ci(X, Y, X_unlabeled, K, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    N = X_unlabeled.shape[0]
    fold_size = int(n/K)
    Yhat = np.zeros(n)
    Yhat_unlabeled = np.zeros(N)
    for j in range(K):
        fold_indices = range(j*fold_size,(j+1)*fold_size)
        train_indices = np.delete(range(n), fold_indices)
        X_train = X[train_indices,:]
        Y_train = Y[train_indices]
        beta_fold = _ols(X_train, Y_train)
        X_fold = X[fold_indices,:]
        Y_fold = Y[fold_indices]
        Yhat[fold_indices] = X_fold @ beta_fold
        Yhat_unlabeled += (X_unlabeled @ beta_fold)/K
    semisupervised_pointest = Yhat_unlabeled.mean() + (Y - Yhat).mean()
    se = ((Y - Yhat)**2).mean()/np.sqrt(n)
    return _zconfint_generic(
          semisupervised_pointest, se, alpha, alternative
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


 def classical_quantile_ci(Y, q, alpha=0.1):
    n = Y.shape[0]
    lower, upper = _zconfint_generic(
          q*n, np.sqrt(q*(1-q)*n), alpha, "two-sided"
      )
    sorted_Y = np.sort(Y)
    return sorted_Y[int(lower)], sorted_Y[int(upper)]


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


def classical_ols_ci(X, Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    pointest, se = _ols(X, Y, return_se=True)
    return _zconfint_generic(
         pointest, se, alpha, alternative
     )


 """
     LOGISTIC REGRESSION

 """

def _logistic(X, Y):
    regression = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000, tol=1e-15, fit_intercept=False).fit(X,Y)
    return regression.coef_.squeeze()


 def ppi_logistic_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, step_size=1,  grad_steps=5000):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    rectifier = 1/n * X.T @ (Yhat - Y)
    theta = np.zeros(d)
    mu_theta = expit(X_unlabeled@theta)
    for i in range(grad_steps):
        mu_theta = expit(X_unlabeled@theta)
        grad = 1/N * X_unlabeled.T @ (mu_theta - Yhat_unlabeled) + rectifier
        theta -=  step_size * grad
    return theta


 def ppi_logistic_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=0.1, grid_size=10000, grid_radius=1):
    n = Y.shape[0]
    d = X_labeled.shape[1]
    N = Yhat_unlabeled.shape[0]
    ppi_pointest = ppi_logistic_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled)
    rectifier = 1/n * X.T @ (Yhat - Y)
    rectifier_std = np.std(X * (Yhat - Y)[:,None], axis=0)
    confset = []
    grid_edge_accepted = True
    while len(confset) == 0 or grid_edge_accepted = True:
        theta_grid = np.concatenate([
            np.linspace(ppi_pointest - grid_radius*np.ones(d), ppi_pointest, grid_size//2),
            np.linspace(ppi_pointest, ppi_pointest + grid_radius*np.ones(d), grid_size//2)[1:]
        ])
        mu_theta = expit(X_unlabeled@theta_grid.T)
        grad = 1/N * X_unlabeled.T@(mu_theta - Yhat_unlabeled[:, None])
        prederr_std = np.std(X_unlabeled[:,:,None]*(mu_theta - Yhat_unlabeled[:,None])[:,None,:], axis=0)
        w = norm.ppf(1-alpha/(2*d)) * np.sqrt(rectifier_std[:,None]**2/n + prederr_std**2/N)
        accept = np.all( np.abs(grad + rectifier[:,None]) <= w, axis=0)
        confset = theta_grid[accept]
        grid_edge_accepted = accept[0] or accept[-1]
        grid_radius *= 2
        grid_size *= 2
    return confset.min(axis=0), confset.max(axis=0)


def classical_logistic_ci(X, Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    pointest = _logistic(X,Y)
    mu = expit(X @ pointest)
    D = np.diag(np.multiply(mu, 1 - mu))
    V = 1/n * X.T @ D @ X
    V_inv = np.linalg.inv(V)
    grads = np.diag(mu - Y) @ X
    cov_mat = V_inv @ np.cov(grads.T) @ V_inv
    return  _zconfint_generic(
          pointest, np.sqrt(np.diag(cov_mat)/n), alpha, alternative
      )


