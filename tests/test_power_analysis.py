import numpy as np
import statsmodels.api as sm
from ppi_py.power_ppi import *
from ppi_py.ppi import *
from ppi_py.baselines import *
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import expit

"""
    Power analysis tests for most powerful pair
"""

def ppi_se(sigma_sq, rho, n, N):
    return np.sqrt(sigma_sq / n *(1 - rho**2 * N/(n+N)))

def check_optimality(result, cost_Y, cost_Yhat, cost_X, epsilon = 0.01, n_max = np.inf):
    n_star = result["n"]
    N_star = result["N"]
    cost = result["cost"]
    rho = result["rho"]

    se_star = ppi_se(1, rho, n_star, N_star)
    n_upper = np.min([2*n_star, n_max])
    n_upper = int(n_upper)
    ns = np.arange(1, n_upper+1)
    cost_n = cost_Y + cost_Yhat + cost_X
    cost_N = cost_X + cost_Yhat

    Ns = (cost - ns*cost_n) / cost_N
    valid = (Ns > 0)*(ns + Ns <= n_max)
    ns = ns[valid]
    Ns = Ns[valid]
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
    achieves_budget = np.abs(powerful_pair["cost"] - budget) < 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair    
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, cost_X, epsilon)
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
    achieves_budget = np.abs(powerful_pair["cost"] - budget) < 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check that classical inference is being used
    assert powerful_pair["N"] == 0, powerful_pair["N"]

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, cost_X, epsilon)
    assert optimal

# Test n_max constraint
def test_ppi_poweranalysis_powerful3():
    rho = np.random.uniform(0.8, 0.9)
    sigma_sq = 1
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01
    budget = 1000
    n_max = 1500
    
    powerful_pair = ppi_power(sigma_sq,
                              rho,
                              cost_Y,
                              cost_Yhat,
                              cost_X,
                              budget = budget,
                              n_max = n_max)
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) < 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check that the total number of samples is  n_max
    assert powerful_pair["n"] + powerful_pair["N"] == n_max, f"{powerful_pair['n']}, {powerful_pair['N']}"

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, cost_X, epsilon, n_max)
    assert optimal


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
    assert achieves_se, f"{cheapest_pair['se']}, {se}"

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = np.abs(cheapest_pair["effective_n"] - 1/se**2)*0.5 <= (1/se**2) * epsilon
    assert correct_effective_sample, f"{cheapest_pair['effective_n']}, {1/se**2}"

    # Check optimality of the cheapest pair
    optimal = check_optimality(cheapest_pair, cost_Y, cost_Yhat, cost_X, epsilon)
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
    assert achieves_se, f"{cheapest_pair['se']}, {se}"

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = np.abs(cheapest_pair["effective_n"] - 1/se**2) <= (1/se**2) * epsilon
    assert correct_effective_sample, f"{cheapest_pair['effective_n']}, {1/se**2}"

    # Check optimality of the cheapest pair
    optimal = check_optimality(cheapest_pair, cost_Y, cost_Yhat, cost_X, epsilon)
    assert optimal

# Check n_max constraint
def test_ppi_poweranalysis_cheapest3():
    rho = np.random.uniform(0.8, 0.9)
    sigma_sq = 1
    cost_Y = np.random.uniform(1, 2)
    cost_Yhat = np.random.uniform(0.1, 0.2)
    cost_X = 0

    epsilon = 0.01

    se = 0.01
    n_max = 15000
    
    cheapest_pair = ppi_power(sigma_sq,
                                 rho,
                                 cost_Y,
                                 cost_Yhat,
                                 cost_X,
                                 se = se,
                                 n_max = n_max)
    
    # Check if the cheapest pair achieves the desired se
    achieves_se = np.abs(cheapest_pair["se"] - se) < epsilon*se
    assert achieves_se, f"{cheapest_pair['se']}, {se}"

    # Check if the cheapest pair has the correct effective sample size
    correct_effective_sample = np.abs(cheapest_pair["effective_n"] - 1/se**2) <= (1/se**2) * epsilon
    assert correct_effective_sample, f"{cheapest_pair['effective_n']}, {1/se**2}"

    # Check if the total number of samples is n_max
    assert cheapest_pair["n"] + cheapest_pair["N"] == n_max, f"{cheapest_pair['n']}, {cheapest_pair['N']}"

    # Check optimality of the cheapest pair
    optimal = check_optimality(cheapest_pair, cost_Y, cost_Yhat, cost_X, epsilon, n_max)
    assert optimal


"""
    Power analysis for mean estimation
"""

def simulate_ses_mean(n_star, N_star, rho_0, reps = 100):
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
            ses[i] = (CI[1][0] - CI[0][0])/(2*norm.ppf(0.975))

    return ses

## Test with high rho
def test_ppi_poweranalysis_mean():
    rho_0 = np.random.uniform(0.8, 0.9)
    Z1 = np.random.normal(0, 1, 1000)
    Z2 = np.random.normal(0, 1, 1000)
    Z3 = np.random.normal(0, 1, 1000)

    Y = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z2
    Yhat = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z3
    Yhat_unlabelled = np.random.normal(0, 1, 1)

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0
    budget = 100

    epsilon = 0.02

    powerful_pair = ppi_mean_power( Y,
                                    Yhat,
                                    Yhat_unlabelled,
                                    cost_Y,
                                    cost_Yhat,
                                    budget = budget)
    
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, 0, epsilon)
    assert optimal

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_ses_mean(powerful_pair["n"], powerful_pair["N"], rho_0)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()

    mean_close = np.abs(se_star - se_sim) <= 2*np.std(ses)
    assert mean_close, f"{se_star}, {se_sim}, {np.std(ses)}"


## Test with low rho
def test_ppi_poweranalysis_mean2():
    rho_0 = np.random.uniform(0.1, 0.2)
    Z1 = np.random.normal(0, 1, 1000)
    Z2 = np.random.normal(0, 1, 1000)
    Z3 = np.random.normal(0, 1, 1000)

    Y = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z2
    Yhat = rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z3
    Yhat_unlabelled = np.random.normal(0, 1, 1)

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0
    budget = 100

    epsilon = 0.02

    powerful_pair = ppi_mean_power( Y,
                                    Yhat,
                                    Yhat_unlabelled,
                                    cost_Y,
                                    cost_Yhat,
                                    budget = budget)
    
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, 0, epsilon)
    assert optimal

    ## Check that classical inference is being used
    assert powerful_pair["N"] == 0, powerful_pair["N"]

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_ses_mean(powerful_pair["n"], powerful_pair["N"], rho_0)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()

    mean_close = np.abs(se_star - se_sim) <= 2*np.std(ses)
    assert mean_close, f"{se_star}, {se_sim}, {np.std(ses)}"

    


"""
    Power analysis for OLS regression
"""

def simulate_ses_OLS(n_star, N_star, rho_0, beta, coord, reps = 100):
    

    ses = np.zeros(reps)
    if N_star > 0:
        for i in range(reps):
            X, Y, Yhat,  X_unlabeled, Yhat_unlabeled = simulate_linear_model(n_star, N_star, rho_0, beta)
            CI = ppi_ols_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, 0.05)
            ses[i] = np.linalg.norm(CI[1][coord] - CI[0][coord])/(2*norm.ppf(0.975))
    else:
        for i in range(reps):
            X, Y, _, _, _ = simulate_linear_model(n_star, N_star, rho_0, beta)
            CI = classical_ols_ci(X, Y, alpha = 0.05)
            ses[i] = (CI[1][coord] - CI[0][coord])/(2*norm.ppf(0.975))

    return ses

def simulate_linear_model(n_star, N_star, rho_0, beta):
    d = len(beta)
    X = np.random.normal(0, 1, (n_star, d))
    X_unlabeled = np.random.normal(0, 1, (N_star, d))

    Z1 = np.random.normal(0, 1, n_star)
    Z2 = np.random.normal(0, 1, n_star)
    Z3 = np.random.normal(0, 1, n_star)

    Y = X@beta + rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z2
    Yhat = X@beta + rho_0**0.5*Z1 + (1 - rho_0)**0.5*Z3
    Yhat_unlabeled = X_unlabeled@beta + np.random.normal(0, 1, N_star)
    return X, Y, Yhat, X_unlabeled, Yhat_unlabeled

def test_ppi_poweranalysis_OLS():
    rho_0 = np.random.uniform(0,1)
    d = 3
    beta = np.random.uniform(-1, 1, d)
    coord = 0

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0.5
    budget = 200
    epsilon = 0.02

    X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_linear_model(5000, 5000, rho_0, beta)

    powerful_pair = ppi_ols_power(X, 
                                  Y, 
                                  Yhat,
                                  X_unlabeled, 
                                  Yhat_unlabeled, 
                                  cost_Y, 
                                  cost_Yhat, 
                                  cost_X, 
                                  coord, 
                                  budget = budget)
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, cost_X, epsilon)
    assert optimal

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_ses_OLS(powerful_pair["n"], powerful_pair["N"], rho_0, beta, coord)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()

    mean_close = np.abs(se_star - se_sim) <= 2*np.std(ses)
    assert mean_close, f"{se_star}, {se_sim}, {np.std(ses)}"

test_ppi_poweranalysis_OLS()

"""
    Power analysis for logistic regression
"""
    
def simulate_se_logisitic(n_star, N_star, rho_0, beta, coord, reps = 100):
    ses = np.zeros(reps)
  
    if N_star > 0:
        for i in range(reps):
            X, Y, Yhat,  X_unlabeled, Yhat_unlabeled = simulate_logistic_model(n_star, N_star, rho_0, beta)
            CI = ppi_logistic_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, 0.05)
            ses[i] = np.linalg.norm(CI[1][coord] - CI[0][coord])/(2*norm.ppf(0.975))
    else:
        for i in range(reps):
            X, Y, _, _, _ = simulate_logistic_model(n_star, N_star, rho_0, beta)
            CI = classical_logistic_ci(X, Y, alpha = 0.05)
            ses[i] = (CI[1][coord] - CI[0][coord])/(2*norm.ppf(0.975))

    return ses

def simulate_logistic_model(n_star, N_star, rho_0, beta):
    d = len(beta)
    p = 1 - rho_0**2
    X = np.random.normal(0, 1, (n_star, d))
    X_unlabeled = np.random.normal(0, 1, (N_star, d))


    Y = np.random.binomial(1, expit(X@beta))
    flips = np.random.binomial(1, p, n_star)
    Yhat = Y.copy()
    Yhat[flips == 1] = np.random.binomial(1, 0.5, np.sum(flips))

    Y_unlabeled = np.random.binomial(1, expit(X_unlabeled@beta))
    flips_unlabeled = np.random.binomial(1, p, N_star)
    Yhat_unlabeled = Y_unlabeled.copy()
    Yhat_unlabeled[flips_unlabeled == 1] = np.random.binomial(1, 0.5, np.sum(flips_unlabeled))
    return X, Y, Yhat, X_unlabeled, Yhat_unlabeled

def test_ppi_poweranalysis_logistic():
    rho_0 = np.random.uniform(0.1, 0.9)
    d = 3
    beta = np.random.uniform(-1, 1, d)
    coord = 0

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0.5
    budget = 200
    epsilon = 0.02

    X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_logistic_model(10000, 10000, rho_0, beta)

    powerful_pair = ppi_logistic_power(X, 
                                      Y, 
                                      Yhat,
                                      X_unlabeled, 
                                      Yhat_unlabeled, 
                                      cost_Y, 
                                      cost_Yhat, 
                                      cost_X, 
                                      coord, 
                                      budget = budget)
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, cost_X, epsilon)
    assert optimal

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_se_logisitic(powerful_pair["n"], powerful_pair["N"], rho_0, beta, coord)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()

    mean_close = np.abs(se_star - se_sim) <= 2*np.std(ses)
    assert mean_close, f"{se_star}, {se_sim}, {np.std(ses)}"


"""
    Power analysis for Poisson regression
"""



def simulate_se_poisson(n_star, N_star, rho_0, beta, coord, reps = 100):
    ses = np.zeros(reps)
    if N_star > 0:
        for i in range(reps):
            X, Y, Yhat,  X_unlabeled, Yhat_unlabeled = simulate_poisson_model(n_star, N_star, rho_0, beta)
            CI = ppi_poisson_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, 0.05)
            ses[i] = np.linalg.norm(CI[1][coord] - CI[0][coord])/(2*norm.ppf(0.975))
    else:
        for i in range(reps):
            X, Y, Yhat,  X_unlabeled, Yhat_unlabeled = simulate_poisson_model(n_star, N_star, rho_0, beta)
            CI = classical_poisson_ci(X, Y, alpha = 0.05)
            ses[i] = (CI[1][coord] - CI[0][coord])/(2*norm.ppf(0.975))

    return ses


def simulate_poisson_model(n_star, N_star, rho_0, beta):
    d = len(beta)

    X = np.random.normal(0, 1, (n_star, d))/np.sqrt(d)
    X_unlabeled = np.random.normal(0, 1, (N_star, d))/np.sqrt(d)
    Y = np.random.poisson(np.exp(X@beta))

    c = (1 - rho_0**2)/rho_0**2
    Z = np.random.poisson(c*np.exp(X@beta))
    Yhat = Y + Z 

    Y_unlabeled = np.random.poisson(np.exp(X_unlabeled@beta))
    Z_unlabeled = np.random.poisson(c*np.exp(X_unlabeled@beta))
    Yhat_unlabeled = Y_unlabeled + Z_unlabeled

    return X, Y, Yhat, X_unlabeled, Yhat_unlabeled


def test_ppi_poweranalysis_poisson():
    rho_0 = np.random.uniform(0.1, 0.9)
    d = 3
    beta = np.random.uniform(-1, 1, d)
    coord = 0

    cost_Y = 1
    cost_Yhat = 0.1
    cost_X = 0.5
    budget = 200
    epsilon = 0.02

    X, Y, Yhat, X_unlabeled, Yhat_unlabeled = simulate_poisson_model(10000, 10000, rho_0, beta)

    powerful_pair = ppi_poisson_power(X, 
                                      Y, 
                                      Yhat,
                                      X_unlabeled, 
                                      Yhat_unlabeled, 
                                      cost_Y, 
                                      cost_Yhat, 
                                      cost_X, 
                                      coord, 
                                      budget = budget)
    
    ## Check if the most powerful pair achieves the budget
    achieves_budget = np.abs(powerful_pair["cost"] - budget) <= 2*(cost_X + cost_Yhat) + cost_Y
    assert achieves_budget, f"{powerful_pair['cost']}, {budget}"

    ## Check optimality of the most powerful pair
    optimal = check_optimality(powerful_pair, cost_Y, cost_Yhat, cost_X, epsilon)
    assert optimal

    ## Check if the estimated standard error is close to the true standard error
    ses = simulate_se_poisson(powerful_pair["n"], powerful_pair["N"], rho_0, beta, coord)
    se_star = powerful_pair["se"]
    se_sim = ses.mean()

    mean_close = np.abs(se_star - se_sim) <= 2*np.std(ses)
    assert mean_close, f"{se_star}, {se_sim}, {np.std(ses)}"



"""
    Run all tests
"""


reps = 10
for i in tqdm(range(reps)):
    test_ppi_poweranalysis_powerful()
    test_ppi_poweranalysis_powerful2()
    test_ppi_poweranalysis_powerful3()

    test_ppi_poweranalysis_cheapest()
    test_ppi_poweranalysis_cheapest2()
    test_ppi_poweranalysis_cheapest3()

    test_ppi_poweranalysis_mean()
    test_ppi_poweranalysis_mean2()

    test_ppi_poweranalysis_OLS()
    test_ppi_poweranalysis_logistic()
    test_ppi_poweranalysis_poisson()





print("All tests passed!")