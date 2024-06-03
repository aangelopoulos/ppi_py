import numpy as np
import statsmodels.api as sm
from ppi_py import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
    PPBoot tests for the mean
"""


def mean_estimator(Y):
    return np.mean(Y, axis=0)


def test_ppboot_mean_ci():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    n = 150
    N = 1000
    methods = ["percentile", "basic"]
    includeds = {method: np.zeros_like(alphas) for method in methods}
    for i in range(trials):
        print(i)
        Y = np.random.normal(0, 1, n)
        Yhat = Y + np.random.normal(-0.5, 1, n)
        Yhat_unlabeled = np.random.normal(-0.5, np.sqrt(2), N)
        for j in range(alphas.shape[0]):
            for method in methods:
                ci = ppboot(
                    mean_estimator,
                    Y,
                    Yhat,
                    Yhat_unlabeled,
                    lam=None,
                    alpha=alphas[j],
                    n_resamples=1000,
                    method=method,
                )
                if ci[0] <= 0 and ci[1] >= 0:
                    includeds[method][j] += 1
        print({method: includeds[method] / (i + 1) for method in methods})
    faileds = {
        method: np.any(includeds[method] / trials < 1 - alphas - epsilon)
        for method in methods
    }
    print({method: includeds[method] / trials for method in methods})
    assert not np.any(list(faileds.values()))


def test_ppboot_mean_multid():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    n_alphas = alphas.shape[0]
    n_dims = 5
    epsilon = 0.1
    n = 150
    N = 1000
    methods = ["percentile", "basic"]
    includeds = {method: np.zeros((n_alphas,)) for method in methods}
    for i in range(trials):
        print(i)
        Y = np.random.normal(0, 1, (n, n_dims))
        Yhat = np.random.normal(-2, 1, (n, n_dims))
        Yhat_unlabeled = np.random.normal(-2, 1, (N, n_dims))
        for j in range(alphas.shape[0]):
            for method in methods:
                ci = ppboot(
                    mean_estimator,
                    Y,
                    Yhat,
                    Yhat_unlabeled,
                    alpha=alphas[j],
                    n_resamples=1000,
                    method=method,
                )
                if (ci[0][0] <= 0) and (ci[1][0] >= 0):
                    includeds[method][j] += 1
    faileds = {
        method: np.any(includeds[method] / trials < 1 - alphas - epsilon)
        for method in methods
    }
    print({method: includeds[method] / trials for method in methods})
    assert not np.any(list(faileds.values()))


"""
    PPBoot tests for logistic regression
"""


def ppboot_logistic_ci_subtest(i, alphas, n, N, d, epsilon, method):
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = np.random.binomial(1, expit(X.dot(beta)))
    Yhat = expit(X.dot(beta_prediction))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(X_unlabeled.dot(beta_prediction))

    # Compute the confidence interval
    def estimator(X, Y):
        LR = sm.GLM(
            Y, X, family=sm.families.Binomial(link=sm.families.links.logit())
        )
        LR_fit = LR.fit()
        return LR_fit.params.squeeze()

    for j in range(len(alphas)):
        beta_ppi_ci = ppboot(
            estimator,
            Y,
            Yhat,
            Yhat_unlabeled,
            X=X,
            X_unlabeled=X_unlabeled,
            lam=None,
            alpha=alphas[j],
            n_resamples=1000,
            method=method,
        )
        # Check that the confidence interval contains the true beta
        solution_labeled = estimator(X, Y)
        solution_unlabeled = estimator(X_unlabeled, Yhat_unlabeled)
        # print(beta[0], beta_ppi_ci[0][0], beta_ppi_ci[1][0], solution_labeled[0], solution_unlabeled[0])
        includeds[j] += int(
            (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
        )
    return includeds


def test_ppboot_logistic_ci_parallel():
    n = 100
    N = 1000
    d = 3
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    num_trials = 100
    methods = ["percentile", "basic"]

    total_includeds = {method: np.zeros(len(alphas)) for method in methods}

    for method in methods:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    ppboot_logistic_ci_subtest,
                    i,
                    alphas,
                    n,
                    N,
                    d,
                    epsilon,
                    method,
                )
                for i in range(num_trials)
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                total_includeds[method] += future.result()

    print({method: total_includeds[method] / num_trials for method in methods})
    faileds = [
        np.any(total_includeds[method] / num_trials < 1 - alphas - epsilon)
        for method in methods
    ]
    assert not np.any(faileds)
