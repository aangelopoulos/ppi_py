import numpy as np
from ppi_py import *

"""
    PPI tests
"""


def test_ppi_mean_pointestimate():
    Y = np.random.normal(0, 1, 100)
    Yhat = Y + 2
    Yhat_unlabeled = np.ones(10000) * 2
    assert ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled) == 0


def test_ppi_mean_ci():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        Yhat = np.random.normal(-2, 1, 10000)
        Yhat_unlabeled = np.random.normal(-2, 1, 10000)
        for j in range(alphas.shape[0]):
            ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alphas[j])
            if ci[0] <= 0 and ci[1] >= 0:
                includeds[j] += 1
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


def test_ppi_mean_pval():
    trials = 1000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.05
    failed = False
    rejected = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        Yhat = np.random.normal(-2, 1, 10000)
        Yhat_unlabeled = np.random.normal(-2, 1, 10000)
        pval = ppi_mean_pval(Y, Yhat, Yhat_unlabeled, null=0)
        rejected += pval < alphas
    failed = rejected / trials > alphas + epsilon
    assert not np.any(failed)


"""
    Classical tests
"""


def test_classical_mean_ci():
    trials = 1000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.05
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        for j in range(alphas.shape[0]):
            ci = classical_mean_ci(Y, alpha=alphas[j])
            if ci[0] <= 0 and ci[1] >= 0:
                includeds[j] += 1
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


def test_semisupervised_mean_ci():
    trials = 100
    K = 5
    d = 2  # Dimension of _features_ (not mean)
    n = 1000
    N = 10000
    sigma = 1
    mu = 10
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        X = np.random.normal(0, 1, size=(n, d))
        X_unlabeled = np.random.normal(0, 1, size=(N, d))
        theta = np.random.normal(0, 1, size=d)
        theta /= np.linalg.norm(theta)
        Y = X.dot(theta) + sigma * np.random.normal(0, 1, n) + mu
        for j in range(alphas.shape[0]):
            ci = semisupervised_mean_ci(X, Y, X_unlabeled, K, alpha=alphas[j])
            if ci[0] <= mu and ci[1] >= mu:
                includeds[j] += 1
    print(includeds / trials)
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


def test_conformal_mean_ci():
    trials = 100
    n = 1000
    N = 10000
    bias = 5
    sigma = 10
    mu = 10
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(mu, 1, n)
        Yhat = Y + sigma * np.random.normal(0, 1, n) + bias
        Y_unlabeled = np.random.normal(mu, 1, N)
        Yhat_unlabeled = Y_unlabeled + sigma * np.random.normal(0, 1, N) + bias
        for j in range(alphas.shape[0]):
            ci = conformal_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alphas[j])
            if ci[0] <= mu and ci[1] >= mu:
                includeds[j] += 1
    print(includeds / trials)
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed
