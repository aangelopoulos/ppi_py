import numpy as np
from ppi_py import *
import pdb

def test_ols_pointestimate():
    # Make a synthetic regression problem
    n = 1000
    N = 10000
    p = 10
    X = np.random.randn(n, p)
    beta = np.random.randn(p)
    beta_prediction = beta + np.random.randn(p) + 2
    Y = X.dot(beta) + np.random.randn(n)
    Yhat = X.dot(beta_prediction) + np.random.randn(n)
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, p)
    Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_ols_pointestimate(X, Y, Yhat, X_unlabeled, Yhat_unlabeled)
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(beta_prediction - beta) # Makes it less biased

def test_ols_ci():
    n = 1000
    N = 10000
    p = 1
    alphas = [0.05, 0.1, 0.2]
    epsilon = 0.02
    num_trials = 1000
    failed = False
    for alpha in alphas:
        included = 0
        for i in range(num_trials):
            # Make a synthetic regression problem
            X = np.random.randn(n, p)
            beta = np.random.randn(p)
            beta_prediction = beta + np.random.randn(p) + 2
            Y = X.dot(beta) + np.random.randn(n)
            Yhat = X.dot(beta_prediction) + np.random.randn(n)
            # Make a synthetic unlabeled data set with predictions Yhat
            X_unlabeled = np.random.randn(N, p)
            Yhat_unlabeled = X_unlabeled.dot(beta_prediction) + np.random.randn(N)
            # Compute the confidence interval
            beta_ppi_ci = ppi_ols_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alpha)
            # Check that the confidence interval contains the true beta
            included += int((beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0]))
        print( (included / num_trials) )
        failed = failed | ( (included / num_trials) < (1 - alpha - epsilon) )
    assert not failed
