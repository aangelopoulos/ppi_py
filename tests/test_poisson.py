import numpy as np
from ppi_py import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import random_correlation, invwishart

"""
    PPI tests for Poisson regression
"""


def test_ppi_poisson_pointestimate_debias():
    # Make a synthetic regression problem
    n = 100
    N = 1000
    d = 2
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = np.random.poisson(np.exp(X.dot(beta)))
    Yhat = np.exp(X.dot(beta_prediction))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = np.exp(X_unlabeled.dot(beta_prediction))
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_poisson_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_ppi_poisson_pval_makesense():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    X = np.random.randn(n, d)
    beta = np.array([0, 0, 1.0])
    beta_prediction = np.array([-0.5, 0.5, 0.7])
    Y = np.random.poisson(np.exp(X.dot(beta)))
    Yhat = np.exp(X.dot(beta_prediction))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = np.exp(X_unlabeled.dot(beta_prediction))
    # Compute the point estimate
    beta_ppi_pval = ppi_poisson_pval(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    assert beta_ppi_pval[-1] < 0.1


def test_ppi_multiple_poisson_pointestimate_debias():
    # Make a synthetic regression problem
    n = 100
    N = 1000
    d = 2
    A = np.random.randn(d, d)
    sigma = A.T @ A
    X = np.random.multivariate_normal(
        np.random.randn(d), invwishart.rvs(d + 1, sigma, 1), N + n
    )  # true value of X
    beta = np.random.randn(d)
    Y = np.random.poisson(np.exp(X.dot(beta)))  # true value of Y

    X_error = (
        np.random.multivariate_normal(
            np.random.randn(d), invwishart.rvs(d + 1, sigma, 1), N + n
        )
        / 2.5
    )  # for ~ 20% error
    X_predicted = X + X_error
    Y_predicted = np.clip(
        Y
        + np.random.choice([-1, 1])
        * np.random.poisson(np.exp(X_predicted.dot(beta))),
        a_min=0,
        a_max=None,
    )

    beta_prediction = poisson(X_predicted, Y_predicted)

    beta_ppi_pointestimate = ppi_multiple_poisson_pointestimate(
        X[0:n, :],
        Y[0:n],
        X_predicted[0:n, :],
        Y_predicted[0:n],
        X_predicted[n:, :],
        Y_predicted[n:],
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < np.linalg.norm(
        beta_prediction - beta
    )  # Makes it less biased


def test_ppi_multiple_poisson_pval_makesense():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    X = np.random.randn(n, d)
    beta = np.array([0, 0, 1.0])
    A = np.random.randn(d, d)
    sigma = A.T @ A
    X = np.random.multivariate_normal(
        np.random.randn(d), invwishart.rvs(d + 1, sigma, 1), N + n
    )  # true value of X
    Y = np.random.poisson(np.exp(X.dot(beta)))  # true value of Y
    X_error = (
        np.random.multivariate_normal(
            np.random.randn(d), invwishart.rvs(d + 1, sigma, 1), N + n
        )
        / 2.5
    )  # for ~ 20% error
    X_predicted = X + X_error
    Y_predicted = np.clip(
        Y
        + np.random.choice([-1, 1])
        * np.random.poisson(np.exp(X_predicted.dot(beta))),
        a_min=0,
        a_max=None,
    )
    beta_ppi_pval = ppi_multiple_poisson_pval(
        X[0:n, :],
        Y[0:n],
        X_predicted[0:n, :],
        Y_predicted[0:n],
        X_predicted[n:, :],
        Y_predicted[n:],
        optimizer_options={"gtol": 1e-3},
    )
    assert beta_ppi_pval[-1] < 0.1


def test_ppi_multiple_poisson_pointestimate_recovers():
    # Make a synthetic regression problem
    n = 1000
    N = 100000
    d = 3
    A = np.random.randn(d, d)
    sigma = A.T @ A
    X = np.random.multivariate_normal(
        np.random.randn(d), invwishart.rvs(d + 1, sigma, 1), N + n
    )  # true value of X
    beta = np.random.randn(d)
    Y = np.random.poisson(np.exp(X.dot(beta)))  # true value of Y

    X_error = (
        np.random.multivariate_normal(
            np.random.randn(d), invwishart.rvs(d + 1, sigma, 1), N + n
        )
        / 2.5
    )  # for ~ 20% error
    X_predicted = X + X_error
    Y_predicted = np.clip(
        Y
        + np.random.choice([-1, 1])
        * np.random.poisson(np.exp(X_predicted.dot(beta))),
        a_min=0,
        a_max=None,
    )

    Y_true = Y[0:n]
    X_true = X[0:n, :]

    beta_ppi_pointestimate = ppi_multiple_poisson_pointestimate(
        X_true,
        Y_true,
        X_predicted[0:n, :],
        Y_predicted[0:n],
        X_predicted[n:, :],
        Y_predicted[n:],
        optimizer_options={"gtol": 1e-3},
    )

    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < 0.2


def test_ppi_poisson_pointestimate_recovers():
    # Make a synthetic regression problem
    n = 10000
    N = 100000
    d = 3
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.poisson(np.exp(X.dot(beta)))
    Yhat = np.exp(X.dot(beta))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = np.exp(X_unlabeled.dot(beta))
    # Compute the point estimate
    beta_ppi_pointestimate = ppi_poisson_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options={"gtol": 1e-3},
    )
    # Check that the point estimate is close to the true beta
    assert np.linalg.norm(beta_ppi_pointestimate - beta) < 0.2


def ppi_poisson_ci_subtest(i, alphas, n=1000, N=10000, d=1, epsilon=0.02):
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    beta_prediction = beta + np.random.randn(d) + 2
    Y = np.random.poisson(np.exp(X.dot(beta)))
    Yhat = np.exp(X.dot(beta_prediction))
    # Make a synthetic unlabeled data set with predictions Yhat
    X_unlabeled = np.random.randn(N, d)
    Yhat_unlabeled = np.exp(X_unlabeled.dot(beta_prediction))
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ppi_ci = ppi_poisson_ci(
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


def test_ppi_poisson_ci_parallel():
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
                ppi_poisson_ci_subtest, i, alphas, n, N, d, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print("PPI:", (total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


def test_ppi_multiple_poisson_ci_parallel():
    n = 1000
    N = 10000
    d = 3
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.4
    num_trials = 10

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                ppi_multiple_poisson_ci_subtest,
                i,
                alphas,
                n,
                N,
                d,
                epsilon,
                i + 525,
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print("PPI:", (total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed


def ppi_multiple_poisson_ci_subtest(
    i, alphas, n=1000, N=10000, d=1, epsilon=0.02, seed=1
):
    rng = np.random.default_rng(seed=seed)
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    # Test case with correlation among covariates.
    A = rng.standard_normal((d, d))
    sigma = A.T @ A
    X = rng.multivariate_normal(
        rng.standard_normal(d), invwishart.rvs(d + 1, sigma, 1), N + n
    )  # true value of X
    beta = rng.standard_normal(d)
    Y = rng.poisson(np.exp(X.dot(beta)))  # true value of Y
    A = rng.standard_normal((d, d))
    sigma = A.T @ A
    X_error = rng.multivariate_normal(
        rng.standard_normal(d), invwishart.rvs(d + 1, sigma, 1), N + n
    )
    X_predicted = X + X_error / 4
    Y_predicted = rng.poisson(
        np.exp(X_predicted.dot(beta) + rng.standard_normal(N + n) / 3)
    )  # true value of Y

    # np.clip(Y + rng.choice([-1,1]) * rng.poisson(np.exp(X_predicted.dot(beta))), a_min=0, a_max=None)
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ppi_ci = ppi_multiple_poisson_ci(
            X[0:n, :],
            Y[0:n],
            X_predicted[0:n, :],
            Y_predicted[0:n],
            X_predicted[n:, :],
            Y_predicted[n:],
            alpha=alphas[j],
            optimizer_options={"gtol": 1e-3},
        )
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ppi_ci[0][0] <= beta[0]) & (beta[0] <= beta_ppi_ci[1][0])
        )
    return includeds


"""
    Baseline tests
"""


def classical_poisson_ci_subtest(i, alphas, n, d, epsilon):
    includeds = np.zeros(len(alphas))
    # Make a synthetic regression problem
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    Y = np.random.poisson(np.exp(X.dot(beta)))
    # Compute the confidence interval
    for j in range(len(alphas)):
        beta_ci = classical_poisson_ci(X, Y, alpha=alphas[j])
        # Check that the confidence interval contains the true beta
        includeds[j] += int(
            (beta_ci[0][0] <= beta[0]) & (beta[0] <= beta_ci[1][0])
        )
    return includeds


def test_classical_poisson_ci_parallel():
    n = 1000
    d = 2
    alphas = np.array([0.05, 0.1, 0.2])
    epsilon = 0.05
    num_trials = 200

    total_includeds = np.zeros(len(alphas))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                classical_poisson_ci_subtest, i, alphas, n, d, epsilon
            )
            for i in range(num_trials)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            total_includeds += future.result()

    print("Classical: ", (total_includeds / num_trials))
    failed = np.any((total_includeds / num_trials) < (1 - alphas - epsilon))
    assert not failed
