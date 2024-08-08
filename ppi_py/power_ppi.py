import numpy as np
import pandas as pd
from ppi_py.utils import reshape_to_2d, construct_weight_vector
from ppi_py.ppi import (
    ppi_mean_pointestimate, 
    ppi_ols_pointestimate, 
    _ols_get_stats, 
    ppi_logistic_pointestimate, 
    _logistic_get_stats,
    # ppi_poisson_pointestimate,
    # _poisson_get_stats
)


"""
    MEAN POWER CALCULATION

"""

def ppi_mean_power(
        Y,
        Yhat,
        Yhat_unlabeled,
        cost_Y,
        cost_Yhat,
        budget=None,
        se_tol=None,
        w=None,
        w_unlabeled=None     
):
    """
    Computes the optimal pair of sample sizes for estimating the mean with ppi.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        cost_Y (float) : Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se_tol (float, optional): Tolerance for the standard error. Used to compute the cheapest pair achieving a desired standard error.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.

    Returns:
        Pandas data frame: containing the following columns
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator. 
            sigma_sq (float): Asymptotic variance of the classical point estimate.
            rho_sq (float): PPI correlation squared.

    Notes:
        At least one of `budget` and `se_tol` must be provided. If both are provided, `budget` will be used.
    """
    if budget is None and se_tol is None:
        raise ValueError(
            "At least one of `budget` and `se_tol` must be provided."
        )
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        raise ValueError("Y must be a 1D array.")
    if len(Yhat.shape) > 1 and Yhat.shape[1] > 1:
        raise ValueError("Yhat must be a 1D array.")
    if len(Yhat_unlabeled.shape) > 1 and Yhat_unlabeled.shape[1] > 1:
        raise ValueError("Yhat_unlabeled must be a 1D array.")

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = 1

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled
    )

    grads = w * (Y - ppi_pointest)
    grads_hat = w * (Yhat - ppi_pointest)
    grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
    inv_hessian = np.eye(d) 

    sigma_sq, rho_sq = _get_power_analysis_params(
        grads,
        grads_hat,
        grads_hat_unlabeled,
        inv_hessian
    )

    gamma, ppi_cost, classical_cost = _get_costs(
        rho_sq,
        cost_Y,
        cost_Yhat,
        cost_X = 0
    )

    if budget is not None:
        return _get_powerful_pair(
            sigma_sq, 
            rho_sq, 
            gamma, 
            ppi_cost, 
            classical_cost, 
            cost_Y,
            cost_Yhat,
            cost_X = 0,
            budget = budget
        )
    else:
        return _get_cheap_pair(
            sigma_sq,
            rho_sq,
            gamma,
            ppi_cost,
            classical_cost,
            cost_Y,
            cost_Yhat,
            cost_X = 0,
            se_tol = se_tol
        )



def _get_power_analysis_params(
        grads, 
        grads_hat, 
        grads_hat_unlabeled, 
        inv_hessian,
        coord=None
):
    """
    Calculates the parameters needed for power analysis.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lam`, when `optim_mode="overall"`. If `None`, it optimizes the total variance  over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.

    Returns:
        float or ndarray: Variance of the classical point estimate.
        float or ndarray: PPI correlation
    """
    grads = reshape_to_2d(grads)
    grads_hat = reshape_to_2d(grads_hat)
    grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)

    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]

    if grads.shape[1] != d:
        raise ValueError(
            "Dimension mismatch between the gradient and the inverse Hessian."
    )
    grads_cent = grads - grads.mean(axis=0)
    grad_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * grads_cent.T @ grad_hat_cent

    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )
    var_grads_hat = var_grads_hat.reshape(d, d)
    var_grads = grads_cent.T @ grads_cent / n

    sigma_sq = np.diag(inv_hessian @ var_grads @ inv_hessian)

    num = np.diag(inv_hessian @ cov_grads @ inv_hessian)**2
    denom = sigma_sq * np.diag(inv_hessian @ var_grads_hat @ inv_hessian)
    rho_sq = num / denom
    rho_sq = np.minimum(rho_sq, 1-1/n)

    if coord is not None:
        return sigma_sq[[coord]], rho_sq[[coord]]
    else:
        return sigma_sq, rho_sq

def _get_costs(rho_sq,
               cost_Y,
               cost_Yhat,
               cost_X,
):
    """
    Computes the cost of the most efficient PPI and classical estimators per classical sample.
    
    Args:
        rho_sq (ndarray): PPI correlation.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        cost_X (float): Cost per unlabeled data point.

    Returns:
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample.
    """
    gamma = (cost_Yhat + cost_X) / cost_Y
    ppi_cost = cost_Y * (1 - rho_sq + gamma * rho_sq + 2 * np.sqrt(gamma * rho_sq * (1 - rho_sq)))
    classical_cost = (cost_Y + cost_X)
    return gamma, ppi_cost, classical_cost
    
def _get_powerful_pair(
        sigma_sq,
        rho_sq,
        gamma,
        ppi_cost,
        classical_cost,
        cost_Y,
        cost_Yhat,
        cost_X,
        budget
):
    """
    Computes the most powerful pair of sample sizes given a budget.
    
    Args:
        sigma_sq (ndarray): Variance of the classical point estimate.
        rho_sq (ndarray): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample.

    Returns:
        Pandas data frame: containing the following columns
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator. 
            sigma_sq (float): Asymptotic variance of the classical point estimate.
            rho_sq (float): PPI correlation squared.  
    """
    if ppi_cost < classical_cost:
        n0 = budget / ppi_cost
        return(_optimal_pair(n0, sigma_sq, rho_sq, gamma, cost_Y, cost_Yhat, cost_X))
    else:
        n = budget / classical_cost
        N = np.zeros_like(n)
        return(pd.DataFrame({"n" : n, 
                             "N" : N, 
                             "cost" : n * classical_cost, 
                             "se" : (sigma_sq / n)**0.5
                             }))


def _get_cheap_pair(
        sigma_sq,
        rho_sq,
        gamma,
        ppi_cost,
        classical_cost,
        cost_Y,
        cost_Yhat,
        cost_X,
        se_tol
):
    """
    Computes the most powerful pair of sample sizes given a budget.
    
    Args:
        sigma_sq (ndarray): Variance of the classical point estimate.
        rho_sq (ndarray): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample
        se_tol (float): Tolerance for the standard error.

    Returns:
        Pandas data frame: containing the following columns
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator. 
            sigma_sq (float): Asymptotic variance of the classical point estimate.
            rho_sq (float): PPI correlation squared.   
    """
    

    if ppi_cost < classical_cost:
        n0 = sigma_sq / se_tol**2
        return(_optimal_pair(n0, sigma_sq, rho_sq, gamma, cost_Y, cost_Yhat, cost_X))
    else:
        n = sigma_sq / se_tol**2
        N = np.zeros_like(n)
        return(pd.DataFrame({"n" : n, 
                             "N" : N, 
                             "cost" : n * classical_cost, 
                             "se" : (sigma_sq / n)**0.5
                            }))

            

def _optimal_pair(
        n0,
        sigma_sq,
        rho_sq,
        gamma,
        cost_Y,
        cost_Yhat,
        cost_X
):
    """"
    Compute the optimal pair of PPI samples achieving the same standard error as a classical estimator with n0 samples.

    Args:
        n0 (float): Number of samples for the classical estimator.
        sigma_sq (float): Variance of the classical point estimate.
        rho_sq (float): PPI correlation squared.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        cost_Y (float): Cost per gold-standard label.
    Returns:
        Pandas data frame: containing the following columns
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator. 
            sigma_sq (float): Asymptotic variance of the classical point estimate.
            rho_sq (float): PPI correlation squared.        
    """
    n = n0 * (1 - rho_sq + np.sqrt(gamma * rho_sq * (1 - rho_sq)))
    N = n * (n0 - n) / (n - (1 - rho_sq) * n0)
    n = n.astype(int)
    N = N.astype(int)

    cost = n * cost_Y + (n + N) * (cost_Yhat + cost_X)
    se = np.sqrt(sigma_sq / n)*np.sqrt(1 - rho_sq * N/(n + N))

    return pd.DataFrame({"n": n, 
                         "N": N, 
                         "cost": cost, 
                         "se": se, 
                         "sigma_sq": sigma_sq, 
                         "rho_sq": rho_sq
                         })

"""
    ORDINARY LEAST SQUARES POWER CALCULATION

"""


def ppi_ols_power(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    cost_Y,
    cost_Yhat,
    cost_X,
    coord,
    budget=None,
    se_tol=None,
    w=None,
    w_unlabeled=None, 
):
    """
    Computes the optimal pair of sample sizes for estimating OLS coefficients with PPI.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        cost_X (float): Cost per unlabeled data point.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se_tol (float, optional): Tolerance for the standard error. Used to compute the cheapest pair achieving a desired standard error.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        Pandas data frame: containing the following columns
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator. 
            sigma_sq (float): Asymptotic variance of the classical point estimate.
            rho_sq (float): PPI correlation squared.
    
    Notes:
        At least one of `budget` and `se_tol` must be provided. If both are provided, `budget` will be used.
    """
    if budget is None and se_tol is None:
        raise ValueError(
            "At least one of `budget` and `se_tol` must be provided."
        )
    
    ppi_pointest = ppi_ols_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled
    )

    sigma_sq, rho_sq = _get_power_analysis_params(
        grads,
        grads_hat,
        grads_hat_unlabeled,
        inv_hessian,
        coord=coord
    )


    gamma, ppi_cost, classical_cost = _get_costs(
        rho_sq,
        cost_Y,
        cost_Yhat,
        cost_X
    )

    if budget is not None:
        return _get_powerful_pair(
            sigma_sq, 
            rho_sq,
            gamma,
            ppi_cost,
            classical_cost, 
            cost_Y, 
            cost_Yhat, 
            cost_X, 
            budget
        )
    else:
        return _get_cheap_pair(
            sigma_sq, 
            rho_sq,
            gamma,
            ppi_cost,
            classical_cost, 
            cost_Y, 
            cost_Yhat, 
            cost_X, 
            se_tol
        )

"""
    LOGISTIC REGRESSION POWER CALCULATION
"""

def ppi_logistic_power(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    cost_Y,
    cost_Yhat,
    cost_X,
    coord,
    budget=None,
    se_tol=None,
    w=None,
    w_unlabeled=None, 
):
    """
    Computes the optimal pair of sample sizes for estimating logistic regression coefficients with PPI.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        cost_X (float): Cost per unlabeled data point.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se_tol (float, optional): Tolerance for the standard error. Used to compute the cheapest pair achieving a desired standard error.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        Pandas data frame: containing the following columns
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator. 
            sigma_sq (float): Asymptotic variance of the classical point estimate.
            rho_sq (float): PPI correlation squared.
    
    Notes:
        At least one of `budget` and `se_tol` must be provided. If both are provided, `budget` will be used.
    """
    if budget is None and se_tol is None:
        raise ValueError(
            "At least one of `budget` and `se_tol` must be provided."
        )
    
    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _logistic_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled
    )

    sigma_sq, rho_sq = _get_power_analysis_params(
        grads,
        grads_hat,
        grads_hat_unlabeled,
        inv_hessian,
        coord=coord
    )

    gamma, ppi_cost, classical_cost = _get_costs(
        rho_sq,
        cost_Y,
        cost_Yhat,
        cost_X
    )

    if budget is not None:
        return _get_powerful_pair(
            sigma_sq, 
            rho_sq,
            gamma,
            ppi_cost,
            classical_cost, 
            cost_Y, 
            cost_Yhat, 
            cost_X, 
            budget
        )
    else:
        return _get_cheap_pair(
            sigma_sq, 
            rho_sq,
            gamma,
            ppi_cost,
            classical_cost, 
            cost_Y, 
            cost_Yhat, 
            cost_X, 
            se_tol
        )
    
"""
    POISSON REGRESSION POWER CALCULATION
"""

# def ppi_poisson_power(
#         X,
#         Y,
#         Yhat,
#         X_unlabeled,
#         Yhat_unlabeled,
#         cost_Y,
#         cost_Yhat,
#         cost_X,
#         coord,
#         budget=None,
#         se_tol=None,
#         w=None,
#         w_unlabeled=None,
# ):
#     """
#     Computes the optimal pair of sample sizes for estimating Poisson regression coefficients with PPI.
    
#     Args:
#         X (ndarray): Covariates corresponding to the gold-standard labels.
#         Y (ndarray): Gold-standard labels.
#         Yhat (ndarray): Predictions corresponding to the gold-standard labels.
#         X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
#         Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
#         cost_Y (float): Cost per gold-standard label.
#         cost_Yhat (float): Cost per prediction.
#         cost_X (float): Cost per unlabeled data point.
#         coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
#         budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
#         se_tol (float, optional): Tolerance for the standard error. Used to compute the cheapest pair achieving a desired standard error.
#         w (ndarray, optional): Sample weights for the labeled data set.
#         w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.
        
#     Returns:
#         Pandas data frame: containing the following columns
#             n (int): Optimal number of gold-labeled samples.
#             N (int): Optimal number of unlabeled samples.
#             cost (float): Total cost.
#             se (float): Estimated standard error of the PPI estimator. 
#             sigma_sq (float): Asymptotic variance of the classical point estimate.
#             rho_sq (float): PPI correlation squared.
#     """
#     if budget is None and se_tol is None:
#         raise ValueError(
#             "At least one of `budget` and `se_tol` must be provided."
#         )

#     ppi_pointest = ppi_poisson_pointestimate(
#         X,
#         Y,
#         Yhat,
#         X_unlabeled,
#         Yhat_unlabeled,
#         w=w,
#         w_unlabeled=w_unlabeled
#     )

#     grads, grads_hat, grads_hat_unlabeled, inv_hessian = _poisson_get_stats(
#         ppi_pointest,
#         X.astype(float),
#         Y,
#         Yhat,
#         X_unlabeled.astype(float),
#         Yhat_unlabeled,
#         w=w,
#         w_unlabeled=w_unlabeled
#     )

#     sigma_sq, rho_sq = _get_power_analysis_params(
#         grads,
#         grads_hat,
#         grads_hat_unlabeled,
#         inv_hessian
#     )

#     gamma, ppi_cost, classical_cost = _get_costs(
#         rho_sq,
#         cost_Y,
#         cost_Yhat,
#         cost_X
#     )

#     if budget is not None:
#         return _get_powerful_pair(
#             sigma_sq, 
#             rho_sq,
#             gamma,
#             ppi_cost,
#             classical_cost, 
#             cost_Y, 
#             cost_Yhat, 
#             cost_X, 
#             budget
#         )
#     else:
#         return _get_cheap_pair(
#             sigma_sq, 
#             rho_sq,
#             gamma,
#             ppi_cost,
#             classical_cost, 
#             cost_Y, 
#             cost_Yhat, 
#             cost_X, 
#             se_tol
#         )    