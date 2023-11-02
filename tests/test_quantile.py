import numpy as np
from ppi_py import *
from tqdm import tqdm
import pdb

"""
    PPI tests
"""


def test_ppi_quantile_pointestimate():
    q = 0.75
    n = 1000
    N = 10000
    epsilon = 0.1
    bias = 1
    sigma = 0.1
    true_quantile = np.random.normal(0, 1)
    binary_vector = 2 * np.random.binomial(1, 1 - q, n) - 1
    binary_vector_unlabeled = 2 * np.random.binomial(1, 1 - q, N) - 1
    Y = true_quantile + np.abs(np.random.normal(0, 1, n)) * binary_vector
    Yhat = Y + sigma * np.random.normal(0, 1, n) + bias
    Y_unlabeled = (
        true_quantile
        + np.abs(np.random.normal(0, 1, N)) * binary_vector_unlabeled
    )
    Yhat_unlabeled = Y_unlabeled + sigma * np.random.normal(0, 1, N) + bias
    thetahat = ppi_quantile_pointestimate(Y, Yhat, Yhat_unlabeled, q)
    print(thetahat, true_quantile)
    assert np.abs(thetahat - true_quantile) <= epsilon


def test_ppi_quantile_ci():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    n = 1000
    N = 1000
    bias = 2
    sigma = 0.1
    q = 0.75
    includeds = np.zeros_like(alphas)
    true_quantile = np.random.normal(0, 1)
    for i in tqdm(range(trials)):
        binary_vector = 2 * np.random.binomial(1, 1 - q, n) - 1
        binary_vector_unlabeled = 2 * np.random.binomial(1, 1 - q, N) - 1
        Y = true_quantile + np.abs(np.random.normal(0, 1, n)) * binary_vector
        Yhat = Y + sigma * np.random.normal(0, 1, n) + bias
        Y_unlabeled = (
            true_quantile
            + np.abs(np.random.normal(0, 1, N)) * binary_vector_unlabeled
        )
        Yhat_unlabeled = Y_unlabeled + sigma * np.random.normal(0, 1, N) + bias
        for j in range(alphas.shape[0]):
            ci = ppi_quantile_ci(Y, Yhat, Yhat_unlabeled, q, alpha=alphas[j])
            if ci[0] <= true_quantile and ci[1] >= true_quantile:
                includeds[j] += 1
    print(includeds / trials)
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


"""
    Classical tests
"""


def test_classical_quantile_ci():
    trials = 1000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    n = 1000
    q = 0.75
    includeds = np.zeros_like(alphas)
    true_quantile = np.random.normal(0, 1)
    for i in range(trials):
        binary_vector = 2 * np.random.binomial(1, 1 - q, n) - 1
        Y = true_quantile + np.abs(np.random.normal(0, 1, n)) * binary_vector
        for j in range(alphas.shape[0]):
            ci = classical_quantile_ci(Y, q, alpha=alphas[j])
            if ci[0] <= true_quantile and ci[1] >= true_quantile:
                includeds[j] += 1
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed
