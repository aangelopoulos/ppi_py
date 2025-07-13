import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import WLS, RegressionResults
from statsmodels.stats.weightstats import _zconfint_generic
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
import statsmodels
from tqdm import tqdm

'''
HELPER FUNCTIONS
'''

def resample_datapoints(data_truth, data_pred, data_pred_unlabeled, w, w_unlabeled):
    '''
    Resamples datasets and weights with replacement (to be used in bootstrap step).
    
    Args:
        data_truth (List[ndarray]): ground truth labeled data (each ndarray has n rows)
        data_pred (List[ndarray]): predicted labeled data (each ndarray has n rows)
        data_pred_unlabeled (List[ndarray]): predicted unlabeled data (each ndarray has N rows)
        w (ndarray, optional): sample weights for labeled data (length n)
        w_unlabeled (ndarray, optional): sample weights for unlabeled data (length N)
        
    Returns:
        resampled version of each of the inputs
    '''
    n = len(data_truth[0])
    N = len(data_pred_unlabeled[0])
    resampled_indices = np.random.choice(np.arange(0, n+N), size=n+N, replace=True)
    
    calibration_indices = resampled_indices[resampled_indices < n]
    data_truth_b = []
    for data in data_truth:
        data_truth_b.append(data[calibration_indices])
    data_pred_b = []
    for data in data_pred:
        data_pred_b.append(data[calibration_indices])
    
    pred_indices = resampled_indices[resampled_indices >= n] - n
    data_pred_unlabeled_b = []
    for data in data_pred_unlabeled:
        data_pred_unlabeled_b.append(data[pred_indices])
    
    if w is None:
        w_b = None
    else:
        w_b = w[calibration_indices]
        
    if w_unlabeled is None:
        w_unlabeled_b = None
    else:
        w_unlabeled_b = w_unlabeled[pred_indices]
    
    return data_truth_b, data_pred_b, data_pred_unlabeled_b, w_b, w_unlabeled_b

'''
MAIN PTD BOOTSTRAP FUNCTION
'''

def ptd_bootstrap(algorithm, data_truth, data_pred, data_pred_unlabeled, w=None, w_unlabeled=None, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    """
    Computes tuning matrix, point estimates, and confidence intervals for regression coefficients using the Predict-then-Debias bootstrap algorithm from Kluger et al. (2025), 'Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling,' <https://arxiv.org/abs/2501.18577>.
    
    Args:
        algorithm (function): python function that takes in data and weights, and returns array containing parameters of interest (e.g., a function that computes linear regression or logistic regression coefficients)
        data_truth (List[ndarray]): ground truth labeled data (each ndarray has n rows)
        data_pred (List[ndarray]): predicted labeled data (each ndarray has n rows)
        data_pred_unlabeled (List[ndarray]): predicted unlabeled data (each ndarray has N rows)
        w (ndarray, optional): sample weights for labeled data (length n)
        w_unlabeled (ndarray, optional): sample weights for unlabeled data (length N)
        B (int, optional): number of bootstrap steps
        alpha (float, optional): error level (must be in the range (0, 1)). The PTD confidence interval will target a coverage of 1 - alpha. 
        tuning_method (str, optional): method used to create the tuning matrix: "optimal_diagonal", "optimal", or None. (If tuning_method is None, the identity matrix is used.) 
        
    Returns:
        ndarray: the tuning matrix (dimensions d x d) computed from the selected tuning method
        ndarray: PTD point estimate of the parameters of interest (length d)
        tuple: lower and upper bounds of PTD confidence intervals with (1-alpha) coverage
    """
    coeff_calibration_list = []
    coeff_pred_calibration_list = []
    coeff_pred_unlabeled_list = []
    
    # compute bootstrap coefficient estimates 
    for b in range(B):
        data_truth_b, data_pred_b, data_pred_unlabeled_b, w_b, w_unlabeled_b = resample_datapoints(data_truth, data_pred, data_pred_unlabeled, w, w_unlabeled)

        coeff_calibration = algorithm(data_truth_b, w_b)
        coeff_calibration_list.append(coeff_calibration)

        coeff_pred_calibration = algorithm(data_pred_b, w_b)
        coeff_pred_calibration_list.append(coeff_pred_calibration)
        
        coeff_pred_unlabeled = algorithm(data_pred_unlabeled_b, w_unlabeled_b)
        coeff_pred_unlabeled_list.append(coeff_pred_unlabeled)

    coeff_calibration_list = np.array(coeff_calibration_list)
    coeff_pred_calibration_list = np.array(coeff_pred_calibration_list)
    coeff_pred_unlabeled_list = np.array(coeff_pred_unlabeled_list)
    
    # compute tuning matrix
    d = coeff_calibration_list.shape[1]
    if tuning_method is None:
        tuning_matrix = np.identity(d)
    else:
        cross_cov_calibration = np.atleast_2d(np.cov(np.concatenate((coeff_calibration_list.T, coeff_pred_calibration_list.T)))[:d, d:])
        cov_pred_calibration = np.atleast_2d(np.cov(coeff_pred_calibration_list.T))
        cov_pred_unlabeled = np.atleast_2d(np.cov(coeff_pred_unlabeled_list.T))
        if tuning_method == "optimal":
            tuning_matrix = cross_cov_calibration @ np.linalg.inv(cov_pred_calibration + cov_pred_unlabeled)
        elif tuning_method == "optimal_diagonal":
            tuning_matrix = np.diag(np.diag(cross_cov_calibration)/(np.diag(cov_pred_calibration) + np.diag(cov_pred_unlabeled)))
            
    # PTD point estimate
    coeff_calibration = algorithm(data_truth, w)
    coeff_pred_calibration = algorithm(data_pred, w)
    coeff_pred_unlabeled = algorithm(data_pred_unlabeled, w_unlabeled)
    ptd_pointestimate = coeff_pred_unlabeled @ tuning_matrix.T + (coeff_calibration - coeff_pred_calibration @ tuning_matrix.T)
    
    # PTD confidence interval
    # compute B point estimates using the bootstrap coefficient estimates and tuning matrix 
    pointestimates = coeff_pred_unlabeled_list @ tuning_matrix.T + (coeff_calibration_list - coeff_pred_calibration_list @ tuning_matrix.T)
    # compute lower and upper bounds for PTD confidence interval with (1-alpha) coverage
    lo = np.percentile(pointestimates, 100*alpha/2, axis=0)
    hi = np.percentile(pointestimates, 100*(1-alpha/2), axis=0)
    ptd_ci = (lo, hi)
    
    return tuning_matrix, ptd_pointestimate, ptd_ci

'''
LINEAR REGRESSION
'''

def algorithm_linear_regression(data, w):
    X, Y = data
    if w is None:
        regression = WLS(endog=Y, exog=X).fit()
    else:
        regression = WLS(endog=Y, exog=X, weights=w).fit()
    coeff = regression.params
    return coeff

def ptd_linear_regression(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, w=None, w_unlabeled=None, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    """
    Computes tuning matrix, point estimates, and confidence intervals for linear regression coefficients using the Predict-then-Debias bootstrap algorithm. 
    
    Args:
        X (ndarray): ground truth covariates in labeled data (dimensions n x p)
        Xhat (ndarray): predicted covariates in labeled data (dimensions n x p)
        Xhat_unlabeled (ndarray): predicted covariates in unlabeled data (dimensions N x p)
        Y (ndarray): ground truth response variable in labeled data (dimensions n x 1)
        Yhat (ndarray): predicted response variable in labeled data (dimensions n x 1)
        Yhat_unlabeled (ndarray): predicted response variable in unlabeled data (dimensions N x 1)
        w (ndarray, optional): sample weights for labeled data (length n)
        w_unlabeled (ndarray, optional): sample weights for unlabeled data (length N)
        B (int, optional): number of bootstrap steps
        alpha (float, optional): error level (must be in the range (0, 1)). The PTD confidence interval will target a coverage of 1 - alpha. 
        tuning_method (str, optional): method used to create the tuning matrix: "optimal_diagonal", "optimal", or None. (If tuning_method is None, the identity matrix is used.) 
        
    Returns:
        ndarray: the tuning matrix (dimensions d x d) computed from the selected tuning method
        ndarray: PTD point estimate of the regression coefficients (length d)
        tuple: lower and upper bounds of PTD confidence intervals with (1-alpha) coverage
    """
    data_truth = [X, Y]
    data_pred = [Xhat, Yhat]
    data_pred_unlabeled = [Xhat_unlabeled, Yhat_unlabeled]
    return ptd_bootstrap(algorithm_linear_regression, data_truth, data_pred, data_pred_unlabeled, w=w, w_unlabeled=w_unlabeled, B=B, alpha=alpha, tuning_method=tuning_method)

def classical_linear_regression_ci(X, Y, w=None, alpha=0.05):
    """
    Computes confidence intervals for linear regression coefficients using the classical method.

    Args:
        X (ndarray): labeled covariates (dimensions n x p)
        Y (ndarray): labeled responses (length n)
        w (ndarray, optional): sample weights for the labeled dataset (length n)
        alpha (float, optional): error level (must be in the range (0, 1)). Confidence interval will target a coverage of 1 - alpha.

    Returns:
        tuple: lower and upper bounds of classical confidence intervals for the coefficients
    """
    if w is None:
        regression = WLS(endog=Y, exog=X).fit()
    else:
        regression = WLS(endog=Y, exog=X, weights=w).fit()
    coeff = regression.params
    se = regression.HC0_se
    ci = _zconfint_generic(coeff, se, alpha, alternative="two-sided")
    return (ci[0], ci[1])

'''
LOGISTIC REGRESSION
'''

def algorithm_logistic_regression(data, w):
    X, Y = data
    regression = GLM(endog=Y, exog=X, freq_weights=w, family=Binomial(link=Logit())).fit()
    coeff = regression.params
    return coeff

def ptd_logistic_regression(X, Xhat, Xhat_unlabeled, Y, Yhat, Yhat_unlabeled, w=None, w_unlabeled=None, B=2000, alpha=0.05, tuning_method='optimal_diagonal'):
    """
    Computes tuning matrix, point estimates, and confidence intervals for logistic regression coefficients using the Predict-then-Debias bootstrap algorithm. 
    
    Args:
        X (ndarray): ground truth covariates in labeled data (dimensions n x p)
        Xhat (ndarray): predicted covariates in labeled data (dimensions n x p)
        Xhat_unlabeled (ndarray): predicted covariates in unlabeled data (dimensions N x p)
        Y (ndarray): ground truth response variable in labeled data (dimensions n x 1)
        Yhat (ndarray): predicted response variable in labeled data (dimensions n x 1)
        Yhat_unlabeled (ndarray): predicted response variable in unlabeled data (dimensions N x 1)
        w (ndarray, optional): sample weights for labeled data (length n)
        w_unlabeled (ndarray, optional): sample weights for unlabeled data (length N)
        B (int, optional): number of bootstrap steps
        alpha (float, optional): error level (must be in the range (0, 1)). The PTD confidence interval will target a coverage of 1 - alpha. 
        tuning_method (str, optional): method used to create the tuning matrix: "optimal_diagonal", "optimal", or None. (If tuning_method is None, the identity matrix is used.) 
        
    Returns:
        ndarray: the tuning matrix (dimensions d x d) computed from the selected tuning method
        ndarray: PTD point estimate of the regression coefficients (length d)
        tuple: lower and upper bounds of PTD confidence intervals with (1-alpha) coverage
    """
    data_truth = [X, Y]
    data_pred = [Xhat, Yhat]
    data_pred_unlabeled = [Xhat_unlabeled, Yhat_unlabeled]
    return ptd_bootstrap(algorithm_logistic_regression, data_truth, data_pred, data_pred_unlabeled, w=w, w_unlabeled=w_unlabeled, B=B, alpha=alpha, tuning_method=tuning_method)

def classical_logistic_regression_ci(X, Y, w=None, alpha=0.05):
    """
    Computes confidence intervals for logistic regression coefficients using the classical method.

    Args:
        X (ndarray): labeled covariates (dimensions n x p)
        Y (ndarray): labeled responses (length n)
        w (ndarray, optional): sample weights for the labeled dataset (length n)
        alpha (float, optional): error level (must be in the range (0, 1)). Confidence interval will target a coverage of 1 - alpha.

    Returns:
        tuple: lower and upper bounds of classical confidence intervals for the coefficients
    """
    regression = GLM(endog=Y, exog=X, freq_weights=w, family=Binomial(link=Logit())).fit()
    ci = regression.conf_int(alpha=alpha).T
    return ci