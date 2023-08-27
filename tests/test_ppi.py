import numpy as np
from ppi_py import *


def test_ppi_mean_ci():
    trials = 10000
    alphas = [0.5, 0.2, 0.1, 0.05, 0.01]
    epsilon = 0.02
    failed = False
    for alpha in alphas:
        included = 0
        for i in range(trials):
            Y = np.random.normal(0, 1, 10000)
            Yhat = np.random.normal(0, 1, 10000)
            Yhat_unlabeled = np.random.normal(0, 1, 10000)
            ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alpha)
            if ci[0] <= 0 and ci[1] >= 0:
                included += 1
        failed = failed | (included / trials < 1 - alpha - epsilon)
    assert not failed
