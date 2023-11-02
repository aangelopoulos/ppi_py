import numpy as np
from scipy.special import expit
from ppi_py import *
import pdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
    PPI tests
"""


def test_ppi_logistic_pointestimate_debias():
    # Make a synthetic regression problem
    n = 100
    N = 1000
    d = 2
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = expit(X.dot(beta) + np.random.randn(n))
    Yhat = expit(X.dot(beta_prediction) + np.random.randn(n))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(
        X_unlabeled.dot(beta_prediction) + np.random.randn(N)
    )
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_logistic_pointestimate(
        X,
        (Y > 0.5).astype(int),
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_ppi_logistic_pointestimate_recovers():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.binomial(1, expit(X.dot(beta)))
    Yhat = expit(X.dot(beta))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(X_unlabeled.dot(beta))
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < 0.2


def ppi_logistic_ci_subtest(i, alphas, n=1000, N=10000, d=1, epsilon=0.02):
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
    for j in range(len(alphas)):
        beta_ppi_ci = ppi_logistic_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alphas[j],
            optimizer_options={"gtol": 1e-3},
        )
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
        )
    return includeds


def test_ppi_logistic_ci_parallel():
    n = 1000
    N = 10000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.4
    num_trials = 10

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                ppi_logistic_ci_subtest, i, alphas, n, N, d, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


"""
    Baseline tests
"""


def classical_logistic_ci_subtest(i, alphas, n, d, epsilon):
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.binomial(1, expit(X.dot(beta)))
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ci = classical_logistic_ci(X, Y, alpha=alphas[j])
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
        )
    return includeds


def test_classical_logistic_ci_parallel():
    n = 1000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 200

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                classical_logistic_ci_subtest, i, alphas, n, d, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed
