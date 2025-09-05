import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from ppi_py import *

"""
    PTD test for logistic regression
"""


def ptd_logistic_ci_subtest(alphas, n, N, d):
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

    for j in range(len(alphas)):
        _, _, beta_ppi_ci = ptd_logistic_regression(
            X,
            X,
            X_unlabeled,
            Y,
            Yhat,
            Yhat_unlabeled,
            B=1000,
            alpha=alphas[j],
        )
        includeds[j] += int(
            (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
        )
    return includeds


def test_ptd_logistic_ci_parallel():
    n = 100
    N = 1000
    d = 1
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.1
    num_trials = 100

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                ptd_logistic_ci_subtest,
                alphas,
                n,
                N,
                d,
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print(total_includeds / num_trials)
    faileds = [
        np.any(total_includeds / num_trials < 1 - alphas - epsilon)
    ]
    assert not np.any(faileds)

"""
    PTD test for linear regression
"""
def test_ptd_ols_ci():
    n = 100
    N = 1000
    d = 1
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 100
    includeds = np.zeros_like(alphas)
    for i in tqdm(range(num_trials)):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        Xhat = X + np.random.randn(n, d)
        beta = np.random.randn(d)
        beta_prediction = beta + np.random.randn(d) + 2
        Y = X.dot(beta) + np.random.randn(n)
        Yhat = X.dot(beta_prediction) + np.random.randn(n)
        # Make a synthetic unlabeled data set with predictions Xhat_unlabeled and Yhat_unlabeled
        X_unlabeled = np.random.randn(N, d)
        Xhat_unlabeled = X_unlabeled + np.random.randn(N, d)
        Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            _, _, beta_ppi_ci = ptd_linear_regression(
                X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, B=200, alpha=alphas[j]
            )
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
            )
    print((includeds / num_trials))
    failed = np.any((includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed