import numpy as np
from scipy.stats import norm
from scipy.special import expit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.isotonic import IsotonicRegression
from .utils import dataframe_decorator, bootstrap
from .ppi import _ols, _wls
import pdb

"""
    MEAN ESTIMATION

"""


def classical_mean_ci(Y, w=None, alpha=0.1, alternative="two-sided"):
    """Classical mean confidence interval using the central limit theorem.

    Args:
        Y (ndarray): Array of observations.
        w (ndarray, optional): Sample weights for the data set. Must be positive and will be normalized to sum to the size of the dataset.
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        alternative (str, optional): One of "two-sided", "larger", or "smaller". Defaults to "two-sided".

    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
    n = Y.shape[0]
    if w is None:
        return _zconfint_generic(
            Y.mean(), Y.std() / np.sqrt(n), alpha, alternative
        )
    else:
        w = w / w.sum() * n
        return _zconfint_generic(
            (w * Y).mean(), (w * Y).std() / np.sqrt(n), alpha, alternative
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
    """Semisupervised mean confidence interval from `[ZB22] <https://arxiv.org/abs/1902.00772>`__.

    Args:
        X (ndarray): Labeled covariates.
        Y (ndarray): Labeled responses.
        X_unlabeled (ndarray): Unlabeled covariates.
        K (int): Number of folds for cross-fitting.
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        alternative (str, optional): One of "two-sided", "larger", or "smaller". Defaults to "two-sided".
        add_intercept (bool, optional): Whether to add an intercept to the covariates. Defaults to True.

    Returns:
        tuple: (lower, upper) confidence interval bounds.

    Notes:
        `[ZB22] <https://arxiv.org/abs/1902.00772>`__ Y. Zhang and J. Bradic, High-dimensional semi-supervised learning: in search of optimal inference of the mean. arxiv:1902.00772, 2022.
    """
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
    """Confidence interval for the mean using conformal inference.

    This method has distribution-free coverage guarantees with bonferroni=True. It tends to be extremely conservative.
    The method works by making a conformal interval for each unlabeled sample and averaging the endpoints.
    In order to get a valid interval, the individual conformal intervals are made at a level of 1 - alpha / N, where N is the number of unlabeled samples (this is a Bonferroni correction required for simultaneous inference).
    Of course, the intervals can be made less conservative by setting bonferroni=False, but this will result in invalid coverage guarantees.
    In practice, this method is not recommended.

    Args:
        Y (ndarray): Labeled responses.
        Yhat (ndarray): Predicted responses for labeled samples.
        Yhat_unlabeled (ndarray): Predicted responses for unlabeled samples.
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        bonferroni (bool, optional): Whether to use a Bonferroni correction for simultaneous inference. Defaults to True.

    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
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
    """Confidence interval for a quantile using the classical method.

    Args:
        Y (ndarray): Labeled responses.
        q (float): Quantile to estimate. Must be in (0, 1).
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).

    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
    n = Y.shape[0]
    lower, upper = _zconfint_generic(
        q * n, np.sqrt(q * (1 - q) * n), alpha, "two-sided"
    )
    sorted_Y = np.sort(Y)
    return sorted_Y[int(lower)], sorted_Y[int(upper)]


"""
    ORDINARY LEAST SQUARES

"""


def classical_ols_ci(X, Y, w=None, alpha=0.1, alternative="two-sided"):
    """Confidence interval for the OLS coefficients using the classical method.

    Args:
        X (ndarray): Labeled features.
        Y (ndarray): Labeled responses.
        w (ndarray, optional): Sample weights for the data set. Must be positive and will be normalized to sum to the size of the dataset.
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        alternative (str, optional): One of "two-sided", "less", or "greater". Defaults to "two-sided".

    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
    n = Y.shape[0]
    if w is None:
        pointest, se = _ols(X, Y, return_se=True)
    else:
        w = w / w.sum() * Y.shape[0]
        pointest, se = _wls(X, Y, w, return_se=True)
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
    """Confidence interval for the OLS coefficients using the PostPI method from `[WML20] <https://www.pnas.org/doi/full/10.1073/pnas.2001238117>`__.

    This method does not possess any coverage guarantees unless the model is perfect, but predates Prediction-Powered Inference.
    It is included for comparison purposes.

    Args:
        Y (ndarray): Labeled responses.
        Yhat (ndarray): Predicted responses for labeled samples.
        X_unlabeled (ndarray): Unlabeled features.
        Yhat_unlabeled (ndarray): Predicted responses for unlabeled samples.
        bootstrap_samples (int, optional): Number of bootstrap samples to use. Defaults to 50.
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        alternative (str, optional): One of "two-sided", "less", or "greater". Defaults to "two-sided".

    Returns:
        tuple: (lower, upper) confidence interval bounds.

    Notes:
        `[WML20] <https://www.pnas.org/doi/full/10.1073/pnas.2001238117>`__ S. Wang, T. H. McCormick, and J. T. Leek, Methods for correcting inference based on outcomes predicted by machine learning. Proceedings of the National Academy of Sciences, 117(48): 30266-30275, 2020.
    """
    N, d = X_unlabeled.shape
    # fit map to debias predictions
    regression = IsotonicRegression(out_of_bounds="clip").fit(
        Yhat, Y
    )  # LinearRegression().fit(Yhat[:,None], Y)
    # debias predictions on unlabeled data
    Yhat_unlabeled_debiased = regression.predict(Yhat_unlabeled[:, None])
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
    """Compute the logistic regression coefficients.

    Args:
        X (ndarray): Labeled features.
        Y (ndarray): Labeled responses.

    Returns:
        ndarray: Logistic regression coefficients.
    """
    regression = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False,
    ).fit(X, Y)
    return regression.coef_.squeeze()


def classical_logistic_ci(X, Y, alpha=0.1, alternative="two-sided"):
    """Confidence interval for the logistic regression coefficients using the classical method.

    Args:
        X (ndarray): Labeled
        Y (ndarray): Labeled responses.
        alpha (float, optional): Error level. Confidence interval will target a coverage of 1 - alpha. Defaults to 0.1. Must be in (0, 1).
        alternative (str, optional): One of "two-sided", "less", or "greater". Defaults to "two-sided".

    Returns:
        tuple: (lower, upper) confidence interval bounds.
    """
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


"""
    BOOTSTRAP CI

"""


def classical_bootstrap_ci(
    estimator,
    Y,
    X=None,
    n_resamples=1000,
    alpha=0.1,
    alternative="two-sided",
    method="percentile",
):
    """Classical bootstrap confidence interval for the estimator.

    Args:
        estimator (callable): Estimator function. Takes in (X,Y) or (Y) and returns a point estimate.
        Y (ndarray): Gold-standard labels.
        X (ndarray, optional): Covariates corresponding to the gold-standard labels. Defaults to `None`. If `None`, the estimator is assumed to only take in `Y`.
        n_resamples (int, optional): Number of bootstrap resamples. Defaults to `1000`.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1). Defaults to `0.1`.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'. Defaults to `'two-sided'`.
        method (str, optional): Method to compute the confidence interval, either 'percentile' or 'basic'. Defaults to `'percentile'`.

    Returns:
        float or ndarray: Lower and upper bounds of the bootstrap confidence interval for the estimator.
    """

    if X is None:

        pointest = estimator(Y)

        bootstrap_distribution = np.array(
            bootstrap([Y], estimator, n_resamples=n_resamples)
        )

    else:

        pointest = estimator(X, Y)

        bootstrap_distribution = np.array(
            bootstrap(
                [X, Y], estimator, n_resamples=n_resamples, paired=[[0, 1]]
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
        lower_bound = np.quantile(bootstrap_distribution, alpha_lower, axis=0)
        upper_bound = np.quantile(
            bootstrap_distribution, 1 - alpha_upper, axis=0
        )
    elif method == "basic":
        lower_bound = 2 * pointest - np.quantile(
            bootstrap_distribution, 1 - alpha_lower, axis=0
        )
        upper_bound = 2 * pointest - np.quantile(
            bootstrap_distribution, alpha_upper, axis=0
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
