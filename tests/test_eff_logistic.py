import numpy as np
from scipy.special import expit
from ppi_py import *
import pdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
    PPI tests
"""


def eff_ppi_logistic_ci_subtest(i, alphas, n=1000, N=10000, d=1):
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
    print("Trial", i)
    for j in range(len(alphas)):
        beta_ppi_ci = eff_ppi_logistic_ci(
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=alphas[j], grad_tol=1e-0
        )
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
        )
    print(f"Trial {i} done")
    return includeds


def test_eff_ppi_logistic_ci_parallel():
    n = 10000
    N = 100000
    d = 3
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.03
    num_trials = 200

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                eff_ppi_logistic_ci_subtest, i, alphas, n, N, d
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed
