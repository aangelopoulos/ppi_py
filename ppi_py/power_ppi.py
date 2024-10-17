import numpy as np
import warnings
from .utils import reshape_to_2d, construct_weight_vector
from .ppi import (
    ppi_mean_pointestimate,
    ppi_ols_pointestimate,
    _ols_get_stats,
    ppi_logistic_pointestimate,
    _logistic_get_stats,
    ppi_poisson_pointestimate,
    _poisson_get_stats,
)


"""
    PPI POWER ANALYSIS

"""


def ppi_power(
    ppi_corr,
    sigma_sq,
    cost_X,
    cost_Y,
    cost_Yhat,
    budget=None,
    se=None,
    n_max=None,
):
    """
    Computes the optimal pair of sample sizes for PPI when the asymptotic variance sigma_sq and the PPI correlation are known.

    Args:
        ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
        sigma_sq (float): Asymptotic variance of the classical point estimate.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se (float, optional): Desired standard error. Used to compute the cheapest pair achieving a desired standard error.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy n + N <= n_max.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation as defined in.
            effective_n (int): Effective number of samples as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.

    Notes:
        At least one of `budget` and `se` must be provided. If both are provided, `budget` will be used and the most powerful pair will be returned.
        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt

    """
    if budget is None and se is None:
        raise ValueError("At least one of `budget` and `se` must be provided.")

    if ppi_corr >= 1 or ppi_corr <= -1:
        raise ValueError("`ppi_corr` must be strictly between -1 and 1.")

    if sigma_sq <= 0:
        raise ValueError("`sigma_sq` must be positive")

    gamma, ppi_cost, classical_cost = _get_costs(
        ppi_corr,
        cost_X,
        cost_Y,
        cost_Yhat,
    )

    if budget is not None:
        return _get_powerful_pair(
            sigma_sq,
            ppi_corr,
            gamma,
            ppi_cost,
            classical_cost,
            cost_X,
            cost_Y,
            cost_Yhat,
            budget=budget,
            n_max=n_max,
        )
    else:
        return _get_cheap_pair(
            sigma_sq,
            ppi_corr,
            gamma,
            ppi_cost,
            classical_cost,
            cost_X,
            cost_Y,
            cost_Yhat,
            se=se,
            n_max=n_max,
        )


def _get_costs(
    ppi_corr,
    cost_X,
    cost_Y,
    cost_Yhat,
):
    """
    Computes the cost of the most efficient PPI and classical estimators per classical sample.

    Args:
        ppi_corr (ndarray): PPI correlation.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.

    Returns:
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample.
    """
    gamma = (cost_Yhat + cost_X) / cost_Y
    ppi_corr_sq = ppi_corr**2
    ppi_cost = cost_Y * (
        1
        - ppi_corr_sq
        + gamma * ppi_corr_sq
        + 2 * (gamma * ppi_corr_sq * (1 - ppi_corr_sq)) ** 0.5
    )
    classical_cost = cost_Y + cost_X
    return gamma, ppi_cost, classical_cost


def _get_powerful_pair(
    sigma_sq,
    ppi_corr,
    gamma,
    ppi_cost,
    classical_cost,
    cost_X,
    cost_Y,
    cost_Yhat,
    budget,
    n_max=None,
):
    """
    Computes the most powerful pair of sample sizes given a budget.

    Args:
        sigma_sq (ndarray): Variance of the classical point estimate.
        ppi_corr (ndarray): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        budget (float): Total budget.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy n + N <= n_max.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation.
            effective_n (int): Effective number of samples.
    """

    n0 = budget / ppi_cost
    result = _optimal_pair(
        n0, ppi_corr, sigma_sq, gamma, cost_X, cost_Y, cost_Yhat
    )

    if classical_cost < ppi_cost or result["N"] < 0:
        n = int(budget / classical_cost)
        result = {
            "n": n,
            "N": 0,
            "cost": n * classical_cost,
            "se": (sigma_sq / n) ** 0.5,
            "ppi_corr": ppi_corr,
            "effective_n": n,
        }

    if n_max is None:
        return result
    if result["n"] + result["N"] <= n_max:
        return result

    if n_max * (cost_Y + cost_X) <= budget:
        return {
            "n": n_max,
            "N": 0,
            "cost": n_max * (cost_Y + cost_X),
            "se": (sigma_sq / n_max) ** 0.5,
            "ppi_corr": ppi_corr,
            "effective_n": n_max,
        }

    n = int(budget / cost_Y - n_max * gamma)
    N = n_max - n
    se = (sigma_sq / n * (1 - ppi_corr**2 * N / (n + N))) ** 0.5
    return {
        "n": n,
        "N": N,
        "cost": n * (cost_Y + cost_Yhat + cost_X) + N * (cost_Yhat + cost_X),
        "se": se,
        "ppi_corr": ppi_corr,
        "effective_n": int(sigma_sq / se**2),
    }


def _get_cheap_pair(
    sigma_sq,
    ppi_corr,
    gamma,
    ppi_cost,
    classical_cost,
    cost_X,
    cost_Y,
    cost_Yhat,
    se,
    n_max=None,
):
    """
    Computes the most powerful pair of sample sizes given a budget.

    Args:
        sigma_sq (ndarray): Variance of the classical point estimate.
        ppi_corr (ndarray): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        se (float): Desired standard error.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy n + N <= n_max.


    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation.
            effective_n (int): Effective number.

    Notes:
        If sigma_sq / n_max > se**2, then there is no pair of sample sizes (n, N) with n + N <= n_max that has a standard error of se or smaller. In this case, the function will give a warning and will return n = n_max and N = 0. This is the most powerful pair of sample sizes that can be achieved with n_max unlabeled samples.
    """

    n0 = sigma_sq / se**2
    result = _optimal_pair(
        n0, ppi_corr, sigma_sq, gamma, cost_X, cost_Y, cost_Yhat
    )

    if classical_cost < ppi_cost or result["N"] < 0:
        n = int(sigma_sq / se**2)
        result = {
            "n": n,
            "N": 0,
            "cost": n * classical_cost,
            "se": (sigma_sq / n) ** 0.5,
            "ppi_corr": ppi_corr,
            "effective_n": n,
        }

    if n_max is None:
        return result
    if result["n"] + result["N"] <= n_max:
        return result

    if sigma_sq / n_max > se**2:
        warnings.warn(
            "The desired standard error is too small for the given number of unlabeled samples. \nReturning n = n_max and N = 0. To achieve the desired standard error, increase n_max or decrease se.",
            UserWarning,
        )

        n = n_max
        return {
            "n": n,
            "N": 0,
            "cost": n * classical_cost,
            "se": (sigma_sq / n) ** 0.5,
            "ppi_corr": ppi_corr,
            "effective_n": n,
        }

    else:
        n = int(
            n_max
            * sigma_sq
            * (1 - ppi_corr**2)
            / (n_max * se**2 - ppi_corr**2 * sigma_sq)
        )
        N = n_max - n
        se = (sigma_sq / n * (1 - ppi_corr**2 * N / (n + N))) ** 0.5
        return {
            "n": n,
            "N": N,
            "cost": n * (cost_Y + cost_Yhat + cost_X)
            + N * (cost_Yhat + cost_X),
            "se": se,
            "ppi_corr": ppi_corr,
            "effective_n": int(sigma_sq / se**2),
        }


def _optimal_pair(n0, ppi_corr, sigma_sq, gamma, cost_X, cost_Y, cost_Yhat):
    """ "
    Compute the optimal pair of PPI samples achieving the same standard error as a classical estimator with n0 samples.

    Args:
        n0 (float): Number of samples for the classical estimator.
        sigma_sq (float): Variance of the classical point estimate.
        ppi_corr (float): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation as defined.
            effective_n (int): Effective number of samples.
    """
    ppi_corr_sq = ppi_corr**2
    n = n0 * (
        1 - ppi_corr_sq + np.sqrt(gamma * ppi_corr_sq * (1 - ppi_corr_sq))
    )
    if ppi_corr != 0:
        N = n * (n0 - n) / (n - (1 - ppi_corr_sq) * n0)
    else:
        N = 0

    n = int(n)
    N = int(N)

    cost = n * cost_Y + (n + N) * (cost_Yhat + cost_X)
    se = ((sigma_sq / n) ** 0.5) * ((1 - ppi_corr_sq * N / (n + N))) ** 0.5

    return {
        "n": n,
        "N": N,
        "cost": cost,
        "se": se,
        "ppi_corr": ppi_corr,
        "effective_n": int(sigma_sq / se**2),
    }


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
    se=None,
    n_max=None,
    w=None,
    w_unlabeled=None,
):
    """
    Computes the optimal pair of sample sizes for estimating the mean with ppi.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se (float, optional): Desired standard error. Used to compute the cheapest pair achieving a desired standard error.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
            effective_n (int): Effective number of samples as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.

    Notes:
        At least one of `budget` and `se` must be provided. If both are provided, `budget` will be used and the most powerful pair will be returned.
        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and se is None:
        raise ValueError("At least one of `budget` and `se` must be provided.")
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
        Y, Yhat, Yhat_unlabeled, w=w, w_unlabeled=w_unlabeled
    )

    grads = w * (Y - ppi_pointest)
    grads_hat = w * (Yhat - ppi_pointest)
    grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
    inv_hessian = np.eye(d)

    sigma_sq, ppi_corr = _get_power_analysis_params(
        grads, grads_hat, grads_hat_unlabeled, inv_hessian
    )

    return ppi_power(
        ppi_corr,
        sigma_sq,
        cost_X=0,
        cost_Y=cost_Y,
        cost_Yhat=cost_Yhat,
        budget=budget,
        se=se,
        n_max=n_max,
    )


def _get_power_analysis_params(
    grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None
):
    """
    Calculates the parameters needed for power analysis.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for regression coefficients. Must be in {1, ..., d} where d is the shape of the estimand.

    Returns:
        float: Variance of the classical point estimate.
        float: PPI correlation
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

    num = np.diag(inv_hessian @ cov_grads @ inv_hessian)
    denom = np.sqrt(
        sigma_sq * np.diag(inv_hessian @ var_grads_hat @ inv_hessian)
    )
    ppi_corr_sq = num / denom
    ppi_corr_sq = np.minimum(ppi_corr_sq, 1 - 1 / n)

    if coord is not None:
        return float(sigma_sq[coord]), float(ppi_corr_sq[coord])
    else:
        return float(sigma_sq[0]), float(ppi_corr_sq[0])


"""
    ORDINARY LEAST SQUARES POWER CALCULATION

"""


def ppi_ols_power(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    cost_X,
    cost_Y,
    cost_Yhat,
    coord,
    budget=None,
    se=None,
    n_max=None,
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
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se (float, optional): Desired standard error. Used to compute the cheapest pair achieving a desired standard error.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__
            effective_n (int): Effective number of samples as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__

    Notes:
        At least one of `budget` and `se` must be provided. If both are provided, `budget` will be used.
        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and se is None:
        raise ValueError("At least one of `budget` and `se` must be provided.")

    ppi_pointest = ppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled, w=w, w_unlabeled=w_unlabeled
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    sigma_sq, ppi_corr = _get_power_analysis_params(
        grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=coord
    )

    return ppi_power(
        ppi_corr, sigma_sq, cost_X, cost_Y, cost_Yhat, budget, se, n_max
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
    cost_X,
    cost_Y,
    cost_Yhat,
    coord,
    budget=None,
    se=None,
    n_max=None,
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
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se (float, optional): Desired standard error. Used to compute the cheapest pair achieving a desired standard error.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__
            effective_n (int): Effective number of samples as defined in`[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__

    Notes:
        At least one of `budget` and `se` must be provided. If both are provided, `budget` will be used.
        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and se is None:
        raise ValueError("At least one of `budget` and `se` must be provided.")

    ppi_pointest = ppi_logistic_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled, w=w, w_unlabeled=w_unlabeled
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _logistic_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    sigma_sq, ppi_corr = _get_power_analysis_params(
        grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=coord
    )

    return ppi_power(
        ppi_corr, sigma_sq, cost_X, cost_Y, cost_Yhat, budget, se, n_max
    )


"""
    POISSON REGRESSION POWER CALCULATION
"""


def ppi_poisson_power(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    cost_X,
    cost_Y,
    cost_Yhat,
    coord,
    budget=None,
    se=None,
    n_max=None,
    w=None,
    w_unlabeled=None,
):
    """
    Computes the optimal pair of sample sizes for estimating Poisson regression coefficients with PPI.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        se (float, optional): Desired standard error. Used to compute the cheapest pair achieving a desired standard error.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        Dictionary: containing the following items
            n (int): Optimal number of gold-labeled samples.
            N (int): Optimal number of unlabeled samples.
            cost (float): Total cost.
            se (float): Estimated standard error of the PPI estimator.
            ppi_corr (float): PPI correlation `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
            effective_n (int): Effective number of samples as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.

    Notes:
        At least one of `budget` and `se` must be provided. If both are provided, `budget` will be used.
        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and se is None:
        raise ValueError("At least one of `budget` and `se` must be provided.")

    ppi_pointest = ppi_poisson_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled, w=w, w_unlabeled=w_unlabeled
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _poisson_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    sigma_sq, ppi_corr = _get_power_analysis_params(
        grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=coord
    )

    return ppi_power(
        ppi_corr, sigma_sq, cost_X, cost_Y, cost_Yhat, budget, se, n_max
    )
