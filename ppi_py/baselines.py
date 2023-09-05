import numpy as np
from scipy.stats import norm
from scipy.special import expit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.isotonic import IsotonicRegression
from .utils import dataframe_decorator
from .ppi import _ols
import pdb

"""
    MEAN ESTIMATION

"""


def classical_mean_ci(Y, alpha=0.1, alternative="two-sided"):
    n = Y.shape[0]
    return _zconfint_generic(
        Y.mean(), Y.std() / np.sqrt(n), alpha, alternative
    )


def semisupervised_mean_ci(
    X,
    Y,
    X_unlabeled,
    K,
    alpha=0.1,
    alternative="two-sided",
    add_intercept=True,
):
    if add_intercept:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        X_unlabeled = np.concatenate(
            [np.ones((X_unlabeled.shape[0], 1)), X_unlabeled], axis=1
        )
    n = Y.shape[0]
    N = X_unlabeled.shape[0]
    fold_size = int(n / K)
    Yhat = np.zeros(n)
    Yhat_unlabeled = np.zeros(N)
    muhat = X_unlabeled.mean(axis=0)
    Vhat = X_unlabeled - muhat[None, :]
    Chat = Vhat.T.dot(Vhat) / N
    epsilon_hats = np.zeros(n)
    bhat_squareds = np.zeros(K)
    betahat_transpose_Vhat_fold_avg = np.zeros((K,))
    bhat_squared_fold = np.zeros((K,))
    for j in range(K):
        fold_indices = range(j * fold_size, (j + 1) * fold_size)
        train_indices = np.delete(range(n), fold_indices)
        X_train = X[train_indices, :]
        Y_train = Y[train_indices]
        beta_fold = _ols(X_train, Y_train)
        X_fold = X[fold_indices, :]
        Y_fold = Y[fold_indices]
        Vhat_fold = Vhat[fold_indices]
        Yhat[fold_indices] = X_fold @ beta_fold
        Yhat_unlabeled += (X_unlabeled @ beta_fold) / K
        epsilon_hats[fold_indices] = Y_fold - beta_fold.dot(Vhat_fold.T)
        bhat_squared_fold[j] = (
            beta_fold.dot(Chat).dot(beta_fold)
            + 2
            * (beta_fold.dot(Vhat_fold.T) * epsilon_hats[fold_indices]).mean()
        )
        betahat_transpose_Vhat_fold_avg[j] = (
            2 * beta_fold.dot(Vhat_fold.T).mean()
        )
    semisupervised_pointest = Yhat_unlabeled.mean() + (Y - Yhat).mean()
    epsilon_hats -= semisupervised_pointest
    bhat_squared_fold -= (
        betahat_transpose_Vhat_fold_avg * semisupervised_pointest
    )
    bhat_squared = bhat_squared_fold.mean()
    sigmahat_squared_epsilon = (epsilon_hats**2).mean()
    se = np.sqrt(sigmahat_squared_epsilon + n / N * bhat_squared)
    return _zconfint_generic(
        semisupervised_pointest, se / np.sqrt(n), alpha, alternative
    )


# Make a conformal interval for each unlabeled sample and average. Only valid with bonferroni=True
def conformal_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.1, bonferroni=True):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    scores = np.abs(Y - Yhat)
    level = (
        (1 - alpha / N) * (1 + 1 / n)
        if bonferroni
        else (1 - alpha) * (1 + 1 / n)
    )
    if level >= 1:
        return -np.infty, np.infty
    conformal_quantile = np.quantile(scores, level, method="higher")
    imputed_estimate = Yhat_unlabeled.mean()
    return (
        imputed_estimate - conformal_quantile,
        imputed_estimate + conformal_quantile,
    )


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


def postprediction_ols_ci(
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    bootstrap_samples=50,
    alpha=0.1,
    alternative="two-sided",
):
    N, d = X_unlabeled.shape
    # fit map to debias predictions
    regression = IsotonicRegression(out_of_bounds='clip').fit(Yhat, Y)#LinearRegression().fit(Yhat[:,None], Y)
    # debias predictions on unlabeled data
    Yhat_unlabeled_debiased = regression.predict(Yhat_unlabeled[:,None])
    # obtain beta and std err via bootstrap
    bootstrap_betas = np.zeros((bootstrap_samples, d))
    bootstrap_ses = np.zeros((bootstrap_samples, d))
    for b in range(bootstrap_samples):
        idx = np.random.choice(range(N), N)
        X_ols = X_unlabeled[idx, :]
        Y_ols = Yhat_unlabeled_debiased[idx]
        bootstrap_betas[b, :], bootstrap_ses[b, :] = _ols(
            X_ols, Y_ols, return_se=True
        )
    pointest = np.median(bootstrap_betas, axis=0)
    se = np.median(bootstrap_ses, axis=0)
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
    d = X.shape[1]
    pointest = logistic(X, Y)
    mu = expit(X @ pointest)
    V = np.zeros((d, d))
    grads = np.zeros((n, d))
    for i in range(n):
        V += 1 / n * mu[i] * (1 - mu[i]) * X[i : i + 1, :].T @ X[i : i + 1, :]
        grads[i] += (mu[i] - Y[i]) * X[i]
    V_inv = np.linalg.inv(V)
    cov_mat = V_inv @ np.cov(grads.T) @ V_inv
    return _zconfint_generic(
        pointest, np.sqrt(np.diag(cov_mat) / n), alpha, alternative
    )
