import numpy as np
from scipy.special import expit
from ppi_py import *
import pdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
    Mean tests
"""


def test_crossppi_mean_pointestimate():
    Y = np.random.normal(0, 1, 100)
    Yhat = Y + 2
    K = 5
    Yhat_unlabeled = np.ones((10000, K)) * 2
    assert crossppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled) == 0


def test_crossppi_mean_ci():
    trials = 100
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.1
    K = 5
    includeds = np.zeros_like(alphas)
    for i in range(trials):
        Y = np.random.normal(0, 1, 10000)
        Yhat = np.random.normal(-2, 1, 10000)
        Yhat_unlabeled = np.random.normal(-2, 1, size=(10000, K))
        for j in range(alphas.shape[0]):
            ci = crossppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alphas[j])
            if ci[0] <= 0 and ci[1] >= 0:
                includeds[j] += 1
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed


"""
    OLS tests
"""


def test_crossppi_ols_pointestimate():
    # Make a synthetic regression problem
    n = 1000
    N = 10000
    d = 10
    K = 5
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = X.dot(beta) + np.random.randn(n)
    Yhat = X.dot(beta_prediction) + np.random.randn(n)
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = X_unlabeled.dot(beta_prediction)[
        :, None
    ] + np.random.randn(N, K)
    # Compute the point estimate
    beta_ppi_pointestimate = crossppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_crossppi_ols_ci():
    n = 1000
    N = 10000
    d = 1
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 50
    K = 5
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
        Yhat_unlabeled = X_unlabeled.dot(beta_prediction)[
            :, None
        ] + np.random.randn(N, K)
        for j in range(alphas.shape[0]):
            # Compute the confidence interval
            beta_ppi_ci = crossppi_ols_ci(
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
    Logistic tests
"""


def test_crossppi_logistic_pointestimate_debias():
    # Make a synthetic regression problem
    n = 100
    N = 1000
    d = 2
    K = 5
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = expit(X.dot(beta) + np.random.randn(n))
    Yhat = expit(X.dot(beta_prediction) + np.random.randn(n))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(
        X_unlabeled.dot(beta_prediction)[:, None] + np.random.randn(N, K)
    )
    # Compute the point estimate
    beta_ppi_pointestimate = crossppi_logistic_pointestimate(
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


def test_crossppi_logistic_pointestimate_recovers():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    K = 5
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.binomial(1, expit(X.dot(beta)))
    Yhat = expit(X.dot(beta))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = expit(X_unlabeled.dot(beta))
    # Repeat the same vector K times
    Yhat_unlabeled = np.repeat(Yhat_unlabeled[:, None], K, axis=1)
    # Compute the point estimate
    beta_ppi_pointestimate = crossppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < 0.2


def crossppi_logistic_ci_subtest(
    i, alphas, n=1000, N=10000, d=1, K=5, epsilon=0.02
):
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
    # Repeat the same vector K times
    Yhat_unlabeled = np.repeat(Yhat_unlabeled[:, None], K, axis=1)
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ppi_ci = crossppi_logistic_ci(
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


def test_crossppi_logistic_ci_parallel():
    n = 1000
    N = 10000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.4
    num_trials = 10
    K = 5

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                crossppi_logistic_ci_subtest, i, alphas, n, N, d, K, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print((total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


"""
    Quantile tests
"""


def test_crossppi_quantile_pointestimate():
    q = 0.75
    n = 1000
    N = 10000
    epsilon = 0.25
    K = 5
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
    Yhat_unlabeled = (
        Y_unlabeled[:, None]
        + sigma * np.random.normal(0, 1, size=(N, K))
        + bias
    )
    thetahat = crossppi_quantile_pointestimate(Y, Yhat, Yhat_unlabeled, q)
    print(thetahat, true_quantile)
    assert np.abs(thetahat - true_quantile) <= epsilon


def test_ppi_quantile_ci():
    trials = 20
    alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    epsilon = 0.3
    n = 1000
    N = 1000
    bias = 2
    sigma = 0.1
    q = 0.75
    K = 5
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
        Yhat_unlabeled = (
            Y_unlabeled[:, None]
            + sigma * np.random.normal(0, 1, size=(N, K))
            + bias
        )
        for j in range(alphas.shape[0]):
            ci = crossppi_quantile_ci(
                Y, Yhat, Yhat_unlabeled, q, alpha=alphas[j]
            )
            if ci[0] <= true_quantile and ci[1] >= true_quantile:
                includeds[j] += 1
    print(includeds / trials)
    failed = np.any(includeds / trials < 1 - alphas - epsilon)
    assert not failed
