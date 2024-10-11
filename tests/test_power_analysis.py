import numpy as np
import statsmodels.api as sm
from ppi_py.power_ppi import *
from ppi_py.ppi import *
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
    Power analysis tests for most powerful pair
"""

def ppi_se(sigma_sq, rho, n, N):
    return np.sqrt(sigma_sq / n *(1 - rho**2 * N/(n+N)))

def check_optimal(result, cost_Y, cost_Yhat, cost_X, epsilon = 0.01):
    n_star = result["n"]
    N_star = result["N"]
    cost = result["cost"]
    rho = result["rho"]

    se_star = ppi_se(1, rho, n_star, N_star)

    ns = np.arange(1, n_star*2)
    cost_n = cost_Y + cost_Yhat + cost_X
    cost_N = cost_X + cost_Yhat

    Ns = (cost - ns*cost_n) / cost_N
    ns = ns[Ns > 0]
    Ns = Ns[Ns > 0]
    Ns = Ns.astype(int)
    ses = ppi_se(1, rho, ns, Ns)


    if N_star > 0:
        optimal = np.abs(se_star - ses.min()) < epsilon*se_star
    else:
        optimal = se_star*(1 - epsilon) <= ses.min()
    return optimal
    


## Test with high rho, low costs of unlabeled data
def test_ppi_poweranalysis_powerful():
    rho = np.random.uniform(0.8, 0.9)
    sigma_sq = 1
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    budget = 1000
    
    powerful_pair = ppi_power(sigma_sq,
                              rho,
                              cost_Y,
                              cost_Yhat,
                              cost_X,
                              budget = budget)
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) < epsilon*budget
    assert achieves_budget

    ## Check optimality of the most powerful pair    
    optimal = check_optimal(powerful_pair, cost_Y, cost_Yhat, cost_X)
    assert optimal

## Test with low rho, high costs of unlabeled data

def test_ppi_poweranalysis_powerful2():
    rho = np.random.uniform(0.1, 0.2)
    sigma_sq = 1
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    budget = 1000
    
    powerful_pair = ppi_power(sigma_sq,
                              rho,
                              cost_Y,
                              cost_Yhat,
                              cost_X,
                              budget = budget)
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) < epsilon*budget
    assert achieves_budget

    ## Check that classical inference is being used
    assert powerful_pair["N"] == 0

    ## Check optimality of the most powerful pair
    optimal = check_optimal(powerful_pair, cost_Y, cost_Yhat, cost_X)
    assert optimal

test_ppi_poweranalysis_powerful()
test_ppi_poweranalysis_powerful2()

"""
    Power analysis test for cheapest pair
"""

# Test with high rho
def test_ppi_poweranalysis_cheapest():
    rho = np.random.uniform(0.8, 0.9)
    sigma_sq = 1
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    se = 0.01
    
    cheapest_pair = ppi_power(sigma_sq,
                                 rho,
                                 cost_Y,
                                 cost_Yhat,
                                 cost_X,
                                 se = se)
    
    # Check if the cheapest pair achieves the desired se
    achieves_se = np.abs(cheapest_pair["se"] - se) < epsilon*se
    assert achieves_se

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = np.abs(cheapest_pair["effective_n"] - 100**2)*0.5 <= 100**2 * epsilon
    assert correct_effective_sample

    # Check optimality of the cheapest pair
    optimal = check_optimal(cheapest_pair, cost_Y, cost_Yhat, cost_X)
    assert optimal

# Test with low rho
def test_ppi_poweranalysis_cheapest2():
    rho = np.random.uniform(0, 0.2)
    sigma_sq = 1
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    se = 0.01
    
    cheapest_pair = ppi_power(sigma_sq,
                                 rho,
                                 cost_Y,
                                 cost_Yhat,
                                 cost_X,
                                 se = se)
    
    # Check if the cheapest pair achieves the desired se
    achieves_se = np.abs(cheapest_pair["se"] - se) < epsilon*se
    assert achieves_se

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = np.abs(cheapest_pair["effective_n"] - 100**2) <= 100**2 * epsilon
    assert correct_effective_sample

    # Check optimality of the cheapest pair
    optimal = check_optimal(cheapest_pair, cost_Y, cost_Yhat, cost_X)
    assert optimal

test_ppi_poweranalysis_cheapest()
test_ppi_poweranalysis_cheapest2()
    
"""
    Power analysis for mean estimation
"""

def simulate_ses(n_star, N_star, rho_0, reps = 100):
    ses = np.zeros(reps)
    if N_star == 0:
        for i in range(reps):
            Y = np.random.normal(0, 1, n_star)
            se = np.std(Y)/np.sqrt(n_star)
            ses[i] = se
    else:
        for i in range(reps):
            Z1 = np.random.normal(0, 1, n_star)
            Z2 = np.random.normal(0, 1, n_star)
            Z3 = np.random.normal(0, 1, n_star)

            Y = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z2
            Yhat = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z3
            Yhat_unlabelled = np.random.normal(0, 1, N_star)
            CI = ppi_mean_ci(Y, Yhat, Yhat_unlabelled, 0.05)
            ses[i] = (CI[1] - CI[0])/(2*norm.ppf(0.975))

    return ses

## Test with high rho
def test_ppi_poweranalysis_mean():
    rho_0 = 0.9
    Z1 = np.random.normal(0, 1, 1000)
    Z2 = np.random.normal(0, 1, 1000)
    Z3 = np.random.normal(0, 1, 1000)

    Y = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z2
    Yhat = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z3
    Yhat_unlabelled = np.random.normal(0, 1, 1)

    cost_Y = 1
    cost_Yhat = 0.1
    budget = 100

    epsilon = 0.01

    powerful_pair = ppi_mean_power( Y,
                                    Yhat,
                                    Yhat_unlabelled,
                                    cost_Y,
                                    cost_Yhat,
                                    budget = budget)
    
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= epsilon*budget
    assert achieves_budget

    ## Check optimality of the most powerful pair
    optimal = check_optimal(powerful_pair, cost_Y, cost_Yhat, 0)
    assert optimal

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_ses(powerful_pair["n"], powerful_pair["N"], rho_0)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()

    assert np.abs(se_star - se_sim) <= epsilon, f"{se_star}, {se_sim}"
    assert np.quantile(ses, 0.3) < se_star < np.quantile(ses, 0.7)

## Test with low rho
def test_ppi_poweranalysis_mean2():
    rho_0 = 0.1
    Z1 = np.random.normal(0, 1, 1000)
    Z2 = np.random.normal(0, 1, 1000)
    Z3 = np.random.normal(0, 1, 1000)

    Y = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z2
    Yhat = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z3
    Yhat_unlabelled = np.random.normal(0, 1, 1)

    cost_Y = 1
    cost_Yhat = 0.1
    budget = 100

    epsilon = 0.01

    powerful_pair = ppi_mean_power( Y,
                                    Yhat,
                                    Yhat_unlabelled,
                                    cost_Y,
                                    cost_Yhat,
                                    budget = budget)
    
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= epsilon*budget
    assert achieves_budget

    ## Check optimality of the most powerful pair
    optimal = check_optimal(powerful_pair, cost_Y, cost_Yhat, 0)
    assert optimal

    ## Check that classical inference is being used
    assert powerful_pair["N"] == 0

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_ses(powerful_pair["n"], powerful_pair["N"], rho_0)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()
    assert np.abs(se_star - se_sim) <= epsilon, f"{se_star}, {se_sim}"
    assert np.quantile(ses, 0.3) < se_star < np.quantile(ses, 0.7)
    


test_ppi_poweranalysis_mean()
test_ppi_poweranalysis_mean2()

"""
    Power analysis for regression
"""


## Run tests





print("All tests passed!")