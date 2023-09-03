import numpy as np
from scipy.stats import norm
from scipy.special import expit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
from .utils import dataframe_decorator
from .ppi import _ols

"""
    MEAN ESTIMATION

"""


def classical_mean_ci(Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    return _zconfint_generic(Y.mean(), Y.std / np.sqrt(n), alpha, alternative)


def semisupervised_mean_ci(
    X, Y, X_unlabeled, K, alpha=0.1, alternative="two-sided"
):
    n = Y.shape[0]
    N = X_unlabeled.shape[0]
    fold_size = int(n / K)
    Yhat = np.zeros(n)
    Yhat_unlabeled = np.zeros(N)
    for j in range(K):
        fold_indices = range(j * fold_size, (j + 1) * fold_size)
        train_indices = np.delete(range(n), fold_indices)
        X_train = X[train_indices, :]
        Y_train = Y[train_indices]
        beta_fold = _ols(X_train, Y_train)
        X_fold = X[fold_indices, :]
        Y_fold = Y[fold_indices]
        Yhat[fold_indices] = X_fold @ beta_fold
        Yhat_unlabeled += (X_unlabeled @ beta_fold) / K
    semisupervised_pointest = Yhat_unlabeled.mean() + (Y - Yhat).mean()
    se = ((Y - Yhat) ** 2).mean() / np.sqrt(n)
    return _zconfint_generic(semisupervised_pointest, se, alpha, alternative)


"""
    QUANTILE ESTIMATION

"""


def classical_quantile_ci(Y, q, alpha=0.1):
    n = Y.shape[0]
    lower, upper = _zconfint_generic(
        q * n, np.sqrt(q * (1 - q) * n), alpha, "two-sided"
    )
    sorted_Y = np.sort(Y)
    return sorted_Y[int(lower)], sorted_Y[int(upper)]


"""
    ORDINARY LEAST SQUARES

"""


def classical_ols_ci(X, Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    pointest, se = _ols(X, Y, return_se=True)
    return _zconfint_generic(pointest, se, alpha, alternative)


"""
    LOGISTIC REGRESSION

"""


def logistic(X, Y):
    regression = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False,
    ).fit(X, Y)
    return regression.coef_.squeeze()


def classical_logistic_ci(X, Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    pointest = logistic(X, Y)
    mu = expit(X @ pointest)
    D = np.diag(np.multiply(mu, 1 - mu))
    V = 1 / n * X.T @ D @ X
    V_inv = np.linalg.inv(V)
    grads = np.diag(mu - Y) @ X
    cov_mat = V_inv @ np.cov(grads.T) @ V_inv
    return _zconfint_generic(
        pointest, np.sqrt(np.diag(cov_mat) / n), alpha, alternative
    )
