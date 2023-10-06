import numpy as np
from ppi_py import *
from tqdm import tqdm
import pdb

"""
    PPI tests
"""


def test_eff_ppi_ols_ci():
    n = 1000
    N = 10000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.03
    num_trials = 200
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
            beta_ppi_ci = eff_ppi_ols_ci(
                X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alphas[j]
            )
            # Check that the confidence interval contains the true beta
            includeds[j] += int(
                (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
            )
    print((includeds / num_trials))
    failed = np.any((includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed