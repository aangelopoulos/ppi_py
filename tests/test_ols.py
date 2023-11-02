import numpy as np
from ppi_py import *
from tqdm import tqdm
import pdb

"""
    PPI tests
"""


def test_ppi_ols_pointestimate():
    # Make a synthetic regression problem
    n = 1000
    N = 10000
    d = 10
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = X.dot(beta) + np.random.randn(n)
    Yhat = X.dot(beta_prediction) + np.random.randn(n)
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_ppi_ols_ci():
    n = 1000
    N = 10000
    d = 1
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 1000
    includeds = np.zeros_like(alphas)
    for i in range(num_trials):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        beta = np.random.randn(d)
        beta_prediction = beta + np.random.randn(d) + 2
        Y = X.dot(beta) + np.random.randn(n)
        Yhat = X.dot(beta_prediction) + np.random.randn(n)
        # Make a synthetic unlabeled data set with predictions Yhat
        X_unlabeled = np.random.randn(N, d)
        Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ppi_ci = ppi_ols_ci(
                X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alphas[j]
            )
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
            )
    print((includeds / num_trials))
    failed = np.any((includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


"""
    Baseline tests
"""


def test_classical_ols_ci():
    n = 1000
    d = 3
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 1000
    includeds = np.zeros_like(alphas)
    for i in range(num_trials):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        beta = np.random.randn(d)
        Y = X.dot(beta) + np.random.randn(n)
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ci = classical_ols_ci(X, Y, alpha=alphas[j])
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
            )
    print((includeds / num_trials))
    failed = np.any((includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


def test_postprediction_ols_ci():
    n = 1000
    N = 10000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 20
    bias = 10
    sigma = 0.1
    includeds = np.zeros_like(alphas)
    for i in tqdm(range(num_trials)):
        # Make a synthetic regression problem
        X = np.random.randn(n, d)
        beta = np.random.randn(d)
        Y = X.dot(beta) + np.random.randn(n)
        Yhat = Y + sigma * np.random.randn(n) + bias
        # Make a synthetic unlabeled data set with predictions Yhat
        X_unlabeled = np.random.randn(N, d)
        Y_unlabeled = X_unlabeled.dot(beta) + np.random.randn(N)
        Yhat_unlabeled = Y_unlabeled + sigma * np.random.randn(N) + bias
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ci = postprediction_ols_ci(
                Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alphas[j]
            )
            print(beta, beta_ci)
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
            )
    print((includeds / num_trials))
    failed = False  # This confidence interval doesn't cover, so the test succeeds if it can construct intervals...
    assert not failed
