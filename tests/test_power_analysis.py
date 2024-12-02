import numpy as np
import statsmodels.api as sm
from ppi_py.ppi_power_analysis import *
from ppi_py import *
from ppi_py.baselines import *
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from ppi_py.utils.statistics_utils import safe_expit

"""
    Power analysis tests for most powerful pair
"""


def ppi_effective_n(ppi_corr, n, N):
    return n * (n + N) / (n + (1 - ppi_corr**2) * N)


def check_optimality(result, cost_X, cost_Y, cost_Yhat=0.01, n_max=np.inf):
    n_star = result["n"]
    cost = result["cost"]
    ppi_corr = result["ppi_corr"]

    n_upper = np.min([2 * n_star, n_max])
    n_upper = int(n_upper)
    ns = np.arange(1, n_upper + 1)
    cost_n = cost_Y + cost_Yhat + cost_X
    cost_N = cost_X + cost_Yhat

    Ns = (cost - ns * cost_n) / cost_N
    valid = (Ns > 0) * (ns + Ns <= n_max)
    ns = ns[valid]
    Ns = Ns[valid]
    Ns = Ns.astype(int)
    effective_ns = ppi_effective_n(ppi_corr, ns, Ns)

    return effective_ns.max()


## Test with high ppi_corr, low costs of unlabeled data
def test_ppi_poweranalysis_powerful():
    ppi_corr = np.random.uniform(0.8, 0.9)
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    budget = 1000

    powerful_pair = ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, budget=budget
    )

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        < 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"


## Test with low ppi_corr, high costs of unlabeled data


def test_ppi_poweranalysis_powerful2():
    ppi_corr = np.random.uniform(0.1, 0.2)
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    budget = 1000

    powerful_pair = ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, budget=budget
    )

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        < 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check that classical inference is being used
    assert powerful_pair["N"] == 0, powerful_pair["N"]

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"


# Test n_max constraint
def test_ppi_poweranalysis_powerful3():
    ppi_corr = np.random.uniform(0.8, 0.9)
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01
    budget = 1000
    n_max = 1500

    powerful_pair = ppi_power(
        ppi_corr,
        cost_X,
        cost_Y,
        cost_Yhat,
        budget=budget,
        n_max=n_max,
    )

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        < 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check that the total number of samples is  n_max
    assert (
        powerful_pair["n"] + powerful_pair["N"] == n_max
    ), f"{powerful_pair['n']}, {powerful_pair['N']}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(
        powerful_pair, cost_X, cost_Y, cost_Yhat, n_max=n_max
    )
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"


"""
    Power analysis test for cheapest pair
"""


# Test with high ppi_corr
def test_ppi_poweranalysis_cheapest():
    ppi_corr = np.random.uniform(0.8, 0.9)
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    effective_n = 1000

    cheapest_pair = ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, effective_n=effective_n
    )

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = (
        np.abs(cheapest_pair["effective_n"] - effective_n)
        <= (effective_n) * epsilon
    )
    assert (
        correct_effective_sample
    ), f"{cheapest_pair['effective_n']}, {effective_n}"

    # Check optimality of the cheapest pair
    optimal_n = check_optimality(cheapest_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= cheapest_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {cheapest_pair['effective_n']}"


# Test with low ppi_corr
def test_ppi_poweranalysis_cheapest2():
    ppi_corr = np.random.uniform(0, 0.2)
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    effective_n = 1000

    cheapest_pair = ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, effective_n=effective_n
    )

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = (
        np.abs(cheapest_pair["effective_n"] - effective_n)
        <= (effective_n) * epsilon
    )
    assert (
        correct_effective_sample
    ), f"{cheapest_pair['effective_n']}, {effective_n}"

    # Check optimality of the cheapest pair
    optimal_n = check_optimality(cheapest_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= cheapest_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {cheapest_pair['effective_n']}"


# Check n_max constraint
def test_ppi_poweranalysis_cheapest3():
    ppi_corr = np.random.uniform(0.8, 0.9)
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    effective_n = 1000
    n_max = 15000

    cheapest_pair = ppi_power(
        ppi_corr,
        cost_X,
        cost_Y,
        cost_Yhat,
        effective_n=effective_n,
        n_max=n_max,
    )

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = (
        np.abs(cheapest_pair["effective_n"] - effective_n)
        <= (effective_n) * epsilon
    )
    assert (
        correct_effective_sample
    ), f"{cheapest_pair['effective_n']}, {effective_n}"

    # Check if the total number of samples is at most n_max
    assert (
        cheapest_pair["n"] + cheapest_pair["N"] <= n_max
    ), f"{cheapest_pair['n']}, {cheapest_pair['N']}"

    # Check optimality of the cheapest pair
    optimal_n = check_optimality(cheapest_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= cheapest_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {cheapest_pair['effective_n']}"


"""
    Power analysis for mean estimation
"""


def simulate_ses_mean(n_star, N_star, ppi_corr_0, reps=100):
    ses = np.zeros(reps)
    if N_star == 0:
        for i in range(reps):
            Y = np.random.normal(0, 1, n_star)
            se = np.std(Y) / np.sqrt(n_star)
            ses[i] = se
    else:
        for i in range(reps):
            Z1 = np.random.normal(0, 1, n_star)
            Z2 = np.random.normal(0, 1, n_star)
            Z3 = np.random.normal(0, 1, n_star)

            Y = ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z2
            Yhat = ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z3
            Yhat_unlabelled = np.random.normal(0, 1, N_star)
            CI = ppi_mean_ci(Y, Yhat, Yhat_unlabelled, 0.05)
            ses[i] = (CI[1][0] - CI[0][0]) / (2 * norm.ppf(0.975))

    return ses


## Test with high ppi_corr
def test_ppi_poweranalysis_mean():
    ppi_corr_0 = np.random.uniform(0.8, 0.9)
    Z1 = np.random.normal(0, 1, 1000)
    Z2 = np.random.normal(0, 1, 1000)
    Z3 = np.random.normal(0, 1, 1000)

    Y = ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z2
    Yhat = ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z3

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0
    budget = 1000

    epsilon = 0.02

    powerful_pair = ppi_mean_power(Y, Yhat, cost_Y, cost_Yhat, budget=budget)

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        <= 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"

    ## Check if the estimated effective sample size is close to the true effective sample size
    reps = 100
    ppi_ses = simulate_ses_mean(
        powerful_pair["n"], powerful_pair["N"], ppi_corr_0, reps=reps
    )
    classical_ses = simulate_ses_mean(
        powerful_pair["effective_n"], 0, ppi_corr_0, reps=reps
    )
    ppi_se = ppi_ses.mean()
    classical_se = classical_ses.mean()

    mean_close = np.abs(ppi_se - classical_se) <= epsilon
    assert mean_close, f"{ppi_se}, {classical_se}"


## Test with low ppi_corr
def test_ppi_poweranalysis_mean2():
    ppi_corr_0 = np.random.uniform(0.1, 0.2)
    Z1 = np.random.normal(0, 1, 1000)
    Z2 = np.random.normal(0, 1, 1000)
    Z3 = np.random.normal(0, 1, 1000)

    Y = ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z2
    Yhat = ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z3

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0
    budget = 100

    epsilon = 0.02

    powerful_pair = ppi_mean_power(Y, Yhat, cost_Y, cost_Yhat, budget=budget)

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        <= 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"

    ## Check that classical inference is being used
    assert powerful_pair["N"] == 0, powerful_pair["N"]


"""
    Power analysis for OLS regression
"""


def simulate_ses_OLS(n_star, N_star, ppi_corr_0, beta, coord, reps=100):

    ses = np.zeros(reps)
    if N_star > 0:
        for i in range(reps):
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_linear_model(
                n_star, N_star, ppi_corr_0, beta
            )
            CI = ppi_ols_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, 0.05)
            ses[i] = np.linalg.norm(CI[1][coord] - CI[0][coord]) / (
                2 * norm.ppf(0.975)
            )
    else:
        for i in range(reps):
            X, Y, _, _, _ = simulate_linear_model(
                n_star, N_star, ppi_corr_0, beta
            )
            CI = classical_ols_ci(X, Y, alpha=0.05)
            ses[i] = (CI[1][coord] - CI[0][coord]) / (2 * norm.ppf(0.975))

    return ses


def simulate_linear_model(n_star, N_star, ppi_corr_0, beta):
    d = len(beta)
    X = np.random.normal(0, 1, (n_star, d))
    X_unlabeled = np.random.normal(0, 1, (N_star, d))

    Z1 = np.random.normal(0, 1, n_star)
    Z2 = np.random.normal(0, 1, n_star)
    Z3 = np.random.normal(0, 1, n_star)

    Y = X @ beta + ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z2
    Yhat = X @ beta + ppi_corr_0**0.5 * Z1 + (1 - ppi_corr_0) ** 0.5 * Z3
    Yhat_unlabeled = X_unlabeled @ beta + np.random.normal(0, 1, N_star)
    return X, Y, Yhat, X_unlabeled, Yhat_unlabeled


def test_ppi_poweranalysis_OLS():
    ppi_corr_0 = np.random.uniform(0, 1)
    d = 3
    beta = np.random.uniform(-1, 1, d)
    coord = 0

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0.5
    budget = 200
    epsilon = 0.02

    X, Y, Yhat, _, _ = simulate_linear_model(5000, 0, ppi_corr_0, beta)

    powerful_pair = ppi_ols_power(
        X,
        Y,
        Yhat,
        cost_X,
        cost_Y,
        cost_Yhat,
        coord,
        budget=budget,
    )

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        <= 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"

    ## Check if the estimated effective sample size is close to the true effective sample size
    reps = 100
    ppi_ses = simulate_ses_OLS(
        powerful_pair["n"],
        powerful_pair["N"],
        ppi_corr_0,
        beta,
        coord,
        reps=reps,
    )
    classical_ses = simulate_ses_OLS(
        powerful_pair["effective_n"], 0, ppi_corr_0, beta, coord, reps=reps
    )
    ppi_se = ppi_ses.mean()
    classical_se = classical_ses.mean()

    mean_close = np.abs(ppi_se - classical_se) <= epsilon
    assert mean_close, f"{ppi_se}, {classical_se}"


"""
    Power analysis for logistic regression
"""


def simulate_se_logisitic(n_star, N_star, ppi_corr_0, beta, coord, reps=100):
    ses = np.zeros(reps)

    if N_star > 0:
        for i in range(reps):
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_logistic_model(
                n_star, N_star, ppi_corr_0, beta
            )
            CI = ppi_logistic_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, 0.05)
            ses[i] = np.linalg.norm(CI[1][coord] - CI[0][coord]) / (
                2 * norm.ppf(0.975)
            )
    else:
        for i in range(reps):
            X, Y, _, _, _ = simulate_logistic_model(
                n_star, N_star, ppi_corr_0, beta
            )
            CI = classical_logistic_ci(X, Y, alpha=0.05)
            ses[i] = (CI[1][coord] - CI[0][coord]) / (2 * norm.ppf(0.975))

    return ses


def simulate_logistic_model(n_star, N_star, ppi_corr_0, beta):
    d = len(beta)
    p = 1 - ppi_corr_0**2
    X = np.random.normal(0, 1, (n_star, d))
    X_unlabeled = np.random.normal(0, 1, (N_star, d))

    Y = np.random.binomial(1, safe_expit(X @ beta))
    flips = np.random.binomial(1, p, n_star)
    Yhat = Y.copy()
    Yhat[flips == 1] = np.random.binomial(1, 0.5, np.sum(flips))

    Y_unlabeled = np.random.binomial(1, safe_expit(X_unlabeled @ beta))
    flips_unlabeled = np.random.binomial(1, p, N_star)
    Yhat_unlabeled = Y_unlabeled.copy()
    Yhat_unlabeled[flips_unlabeled == 1] = np.random.binomial(
        1, 0.5, np.sum(flips_unlabeled)
    )
    return X, Y, Yhat, X_unlabeled, Yhat_unlabeled


def test_ppi_poweranalysis_logistic():
    ppi_corr_0 = np.random.uniform(0.1, 0.9)
    d = 3
    beta = np.random.uniform(-1, 1, d)
    coord = 0

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0.1
    budget = 200
    epsilon = 0.02

    X, Y, Yhat, _, _ = simulate_logistic_model(10000, 0, ppi_corr_0, beta)

    powerful_pair = ppi_logistic_power(
        X,
        Y,
        Yhat,
        cost_X,
        cost_Y,
        cost_Yhat,
        coord,
        budget=budget,
    )

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        <= 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"

    ## Check if the estimated standard error is close to the true standard error
    ppi_ses = simulate_se_logisitic(
        powerful_pair["n"], powerful_pair["N"], ppi_corr_0, beta, coord
    )
    classical_ses = simulate_se_logisitic(
        powerful_pair["effective_n"], 0, ppi_corr_0, beta, coord
    )
    ppi_se = ppi_ses.mean()
    classical_se = classical_ses.mean()

    mean_close = np.abs(ppi_se - classical_se) <= 0.1 * ppi_se
    assert (
        mean_close
    ), f"{ppi_se}, {classical_se}, {np.std(ppi_ses)}, {np.std(classical_ses)}"


"""
    Power analysis for Poisson regression
"""


def simulate_se_poisson(n_star, N_star, ppi_corr_0, beta, coord, reps=100):
    ses = np.zeros(reps)
    if N_star > 0:
        for i in range(reps):
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_poisson_model(
                n_star, N_star, ppi_corr_0, beta
            )
            CI = ppi_poisson_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, 0.05)
            ses[i] = np.linalg.norm(CI[1][coord] - CI[0][coord]) / (
                2 * norm.ppf(0.975)
            )
    else:
        for i in range(reps):
            X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_poisson_model(
                n_star, N_star, ppi_corr_0, beta
            )
            CI = classical_poisson_ci(X, Y, alpha=0.05)
            ses[i] = (CI[1][coord] - CI[0][coord]) / (2 * norm.ppf(0.975))

    return ses


def simulate_poisson_model(n_star, N_star, ppi_corr_0, beta):
    d = len(beta)

    X = np.random.normal(0, 1, (n_star, d)) / np.sqrt(d)
    X_unlabeled = np.random.normal(0, 1, (N_star, d)) / np.sqrt(d)
    Y = np.random.poisson(np.exp(X @ beta))

    c = (1 - ppi_corr_0**2) / ppi_corr_0**2
    Z = np.random.poisson(c * np.exp(X @ beta))
    Yhat = Y + Z

    Y_unlabeled = np.random.poisson(np.exp(X_unlabeled @ beta))
    Z_unlabeled = np.random.poisson(c * np.exp(X_unlabeled @ beta))
    Yhat_unlabeled = Y_unlabeled + Z_unlabeled

    return X, Y, Yhat, X_unlabeled, Yhat_unlabeled


def test_ppi_poweranalysis_poisson():
    ppi_corr_0 = np.random.uniform(0.1, 0.9)
    d = 3
    beta = np.random.uniform(-1, 1, d)
    coord = 0

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0.5
    budget = 200
    epsilon = 0.02

    X, Y, Yhat, _, _ = simulate_poisson_model(10000, 0, ppi_corr_0, beta)

    powerful_pair = ppi_poisson_power(
        X,
        Y,
        Yhat,
        cost_X,
        cost_Y,
        cost_Yhat,
        coord,
        budget=budget,
    )

    ## Check if the most powerful pair achieves the budget
    achieves_budget = (
        np.abs(powerful_pair["cost"] - budget)
        <= 2 * (cost_X + cost_Yhat) + cost_Y
    )
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal_n = check_optimality(powerful_pair, cost_X, cost_Y, cost_Yhat)
    assert optimal_n <= powerful_pair["effective_n"] * (
        1 + epsilon
    ), f"{optimal_n}, {powerful_pair['effective_n']}"

    ## Check if the estimated
    ppi_ses = simulate_se_poisson(
        powerful_pair["n"], powerful_pair["N"], ppi_corr_0, beta, coord
    )
    classical_ses = simulate_se_poisson(
        powerful_pair["effective_n"], 0, ppi_corr_0, beta, coord
    )
    ppi_se = ppi_ses.mean()
    classical_se = classical_ses.mean()

    mean_close = np.abs(ppi_se - classical_se) <= epsilon
    assert (
        mean_close
    ), f"{ppi_se}, {classical_se}, {np.std(ppi_ses)}, {np.std(classical_ses)}"
