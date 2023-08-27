import numpy as np
from ppi_py import *

def test_ppi_mean_pointestimate():
    Y = np.random.normal(0, 1, 100)
    Yhat = Y + 2
    Yhat_unlabeled = np.ones(10000) * 2
    assert ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled) == 0

def test_ppi_mean_ci():
    trials = 10000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.02
    failed = False
    for alpha in alphas:
        included = 0
        for i in range(trials):
            Y = np.random.normal(0, 1, 10000)
            Yhat = np.random.normal(-2, 1, 10000)
            Yhat_unlabeled = np.random.normal(-2, 1, 10000)
            ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alpha)
            if ci[0] <= 0 and ci[1] >= 0:
                included += 1
        failed = failed | (included / trials < 1 - alpha - epsilon)
    assert not failed

def test_ppi_mean_pval():
    trials = 10000
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.02
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
