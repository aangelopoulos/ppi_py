import numpy as np
import warnings
from .utils import reshape_to_2d, construct_weight_vector
from .ppi import _ols_get_stats, _logistic_get_stats, _poisson_get_stats, _wls
from sklearn.linear_model import LogisticRegression, PoissonRegressor


"""
    PPI POWER ANALYSIS

"""


def ppi_power(
    ppi_corr,
    cost_X,
    cost_Y,
    cost_Yhat,
    budget=None,
    effective_n=None,
    n_max=None,
):
    """
    Computes the optimal pair of sample sizes for PPI when the PPI correlation is known.

    Args:
        ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        effective_n (int, optional): Effective sample size. Used to compute the cheapest pair.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy n + N <= n_max.

    Returns:
        dict: Dictionary containing the following items:
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective number of samples as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
            - ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.

    Notes:
        At least one of `budget` and `effective_n` must be provided. If both are provided, `budget` will be used and the most powerful pair will be returned.

        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt

    """
    if budget is None and effective_n is None:
        raise ValueError(
            "At least one of `budget` and `effective_n` must be provided."
        )

    if ppi_corr >= 1 or ppi_corr <= -1:
        raise ValueError("`ppi_corr` must be strictly between -1 and 1.")

    gamma, ppi_cost, classical_cost = _get_costs(
        ppi_corr,
        cost_X,
        cost_Y,
        cost_Yhat,
    )

    if budget is not None:
        return _get_powerful_pair(
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
            ppi_corr,
            gamma,
            ppi_cost,
            classical_cost,
            cost_X,
            cost_Y,
            cost_Yhat,
            effective_n=effective_n,
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
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size.
            - ppi_corr (float): PPI correlation.
    """

    n0 = budget / ppi_cost
    result = _optimal_pair(n0, ppi_corr, gamma, cost_X, cost_Y, cost_Yhat)

    if classical_cost < ppi_cost or result["N"] < 0:
        n = round(budget / classical_cost)
        result = {
            "n": n,
            "N": 0,
            "cost": n * classical_cost,
            "effective_n": n,
            "ppi_corr": ppi_corr,
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
            "effective_n": n_max,
            "ppi_corr": ppi_corr,
        }

    n = round(budget / cost_Y - n_max * gamma)
    N = n_max - n
    effective_n = round(n * (n + N) / (n + (1 - ppi_corr**2) * N))
    return {
        "n": n,
        "N": N,
        "cost": n * (cost_Y + cost_Yhat + cost_X) + N * (cost_Yhat + cost_X),
        "effective_n": effective_n,
        "ppi_corr": ppi_corr,
    }


def _get_cheap_pair(
    ppi_corr,
    gamma,
    ppi_cost,
    classical_cost,
    cost_X,
    cost_Y,
    cost_Yhat,
    effective_n,
    n_max=None,
):
    """
    Computes the most powerful pair of sample sizes given a budget.

    Args:
        ppi_corr (ndarray): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        ppi_cost (float): Cost of the most efficient PPI estimator per classical sample.
        classical_cost (float): Cost of the classical estimator per classical sample.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        effective_n (int): Effective sample size.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy n + N <= n_max.


    Returns:
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size.
            - ppi_corr (float): PPI correlation.

    Notes:
        If effective_n > n_max, then there is no pair of sample sizes (n, N) with n + N <= n_max that has a standard error of se or smaller. In this case, the function will give a warning and will return n = n_max and N = 0. This is the most powerful pair of sample sizes that can be achieved with n_max unlabeled samples.
    """

    n0 = effective_n
    result = _optimal_pair(n0, ppi_corr, gamma, cost_X, cost_Y, cost_Yhat)

    if classical_cost < ppi_cost or result["N"] < 0:
        n = round(n0)
        result = {
            "n": n,
            "N": 0,
            "cost": n * classical_cost,
            "effective_n": n,
            "ppi_corr": ppi_corr,
        }

    if n_max is None:
        return result
    if result["n"] + result["N"] <= n_max:
        return result

    if effective_n > n_max:
        warnings.warn(
            "The desired effective sample size is too large for the given number of unlabeled samples. \nReturning n = n_max and N = 0. To achieve the desired effective sample size, increase n_max or decrease effective_n.",
            UserWarning,
        )

        n = n_max
        return {
            "n": n,
            "N": 0,
            "cost": n * classical_cost,
            "effective_n": n,
            "ppi_corr": ppi_corr,
        }

    else:
        n = round(n0 * n_max * (1 - ppi_corr**2) / (n_max - ppi_corr**2 * n0))
        N = n_max - n
        effective_n = round(n * (n + N) / (n + (1 - ppi_corr**2) * N))
        return {
            "n": n,
            "N": N,
            "cost": n * (cost_Y + cost_Yhat + cost_X)
            + N * (cost_Yhat + cost_X),
            "effective_n": effective_n,
            "ppi_corr": ppi_corr,
        }


def _optimal_pair(n0, ppi_corr, gamma, cost_X, cost_Y, cost_Yhat):
    """ "
    Compute the optimal pair of PPI samples achieving the same standard error as a classical estimator with n0 samples.

    Args:
        n0 (float): Number of samples for the classical estimator.
        ppi_corr (float): PPI correlation.
        gamma (float): Ratio of the cost of a prediction plus unlabled data to the cost of a gold-standard label.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.

    Returns:
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size.
            - ppi_corr (float): PPI correlation.
    """
    ppi_corr_sq = ppi_corr**2
    n = n0 * (
        1 - ppi_corr_sq + np.sqrt(gamma * ppi_corr_sq * (1 - ppi_corr_sq))
    )
    if ppi_corr != 0:
        N = n * (n0 - n) / (n - (1 - ppi_corr_sq) * n0)
    else:
        N = 0

    n = round(n)
    N = round(N)

    cost = n * cost_Y + (n + N) * (cost_Yhat + cost_X)
    effective_n = round(n * (n + N) / (n + (1 - ppi_corr_sq) * N))

    return {
        "n": n,
        "N": N,
        "cost": cost,
        "effective_n": effective_n,
        "ppi_corr": ppi_corr,
    }


"""
    MEAN POWER CALCULATION

"""


def ppi_mean_power(
    Y,
    Yhat,
    cost_Y,
    cost_Yhat,
    budget=None,
    effective_n=None,
    n_max=None,
    w=None,
):
    """
    Computes the optimal pair of sample sizes for estimating the mean with ppi.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        effective_n (int, optional): Effective sample size. Used to compute the cheapest pair.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.

    Returns:
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
            - ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.

    Notes:
        At least one of `budget` and `effective_n` must be provided. If both are provided, `budget` will be used and the most powerful pair will be returned.

        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and effective_n is None:
        raise ValueError(
            "At least one of `budget` and `effective_n` must be provided."
        )
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        raise ValueError("Y must be a 1D array.")
    if len(Yhat.shape) > 1 and Yhat.shape[1] > 1:
        raise ValueError("Yhat must be a 1D array.")

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    n = Y.shape[0]
    d = 1

    w = construct_weight_vector(n, w, vectorized=True)

    pointest = np.sum(w * Y) / np.sum(w)

    grads = w * (Y - pointest)
    grads_hat = w * (Yhat - pointest)
    inv_hessian = np.eye(d)

    ppi_corr = _get_ppi_corr(grads, grads_hat, inv_hessian)

    return ppi_power(
        ppi_corr,
        cost_X=0,
        cost_Y=cost_Y,
        cost_Yhat=cost_Yhat,
        budget=budget,
        effective_n=effective_n,
        n_max=n_max,
    )


def _get_ppi_corr(grads, grads_hat, inv_hessian, coord=None):
    """
    Calculates the parameters needed for power analysis.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for regression coefficients. Must be in {1, ..., d} where d is the shape of the estimand.

    Returns:
        float: Variance of the classical point estimate.
        float: PPI correlation
    """
    grads = reshape_to_2d(grads)
    grads_hat = reshape_to_2d(grads_hat)

    n = grads.shape[0]
    d = inv_hessian.shape[0]

    if grads.shape[1] != d:
        raise ValueError(
            "Dimension mismatch between the gradient and the inverse Hessian."
        )
    grads_cent = grads - grads.mean(axis=0)
    grads_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * grads_cent.T @ grads_hat_cent

    var_grads_hat = grads_hat_cent.T @ grads_hat_cent / n
    var_grads = grads_cent.T @ grads_cent / n

    sigma_sq = np.diag(inv_hessian @ var_grads @ inv_hessian)

    num = np.diag(inv_hessian @ cov_grads @ inv_hessian)
    denom = np.sqrt(
        sigma_sq * np.diag(inv_hessian @ var_grads_hat @ inv_hessian)
    )
    ppi_corr = num / denom
    ppi_corr = np.minimum(ppi_corr, 1 - 1 / n)

    if coord is not None:
        return float(ppi_corr[coord])
    else:
        return float(ppi_corr[0])


"""
    ORDINARY LEAST SQUARES POWER CALCULATION

"""


def ppi_ols_power(
    X,
    Y,
    Yhat,
    cost_X,
    cost_Y,
    cost_Yhat,
    coord,
    budget=None,
    effective_n=None,
    n_max=None,
    w=None,
):
    """
    Computes the optimal pair of sample sizes for estimating OLS coefficients with PPI.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        effective_n (int, optional): Effective sample size. Used to compute the cheapest pair.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set.

    Returns:
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__
            - ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__

    Notes:
        At least one of `budget` and `effective_n` must be provided. If both are provided, `budget` will be used.

        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and effective_n is None:
        raise ValueError(
            "At least one of `budget` and `effective_n` must be provided."
        )

    pointest = _wls(X, Y, w=w)

    grads, grads_hat, _, inv_hessian = _ols_get_stats(
        pointest,
        X.astype(float),
        Y,
        Yhat,
        X.astype(float),
        Yhat,
        w=w,
        use_unlabeled=False,
    )

    ppi_corr = _get_ppi_corr(grads, grads_hat, inv_hessian, coord=coord)

    return ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, budget, effective_n, n_max
    )


"""
    LOGISTIC REGRESSION POWER CALCULATION
"""


def ppi_logistic_power(
    X,
    Y,
    Yhat,
    cost_X,
    cost_Y,
    cost_Yhat,
    coord,
    budget=None,
    effective_n=None,
    n_max=None,
    w=None,
):
    """
    Computes the optimal pair of sample sizes for estimating logistic regression coefficients with PPI.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        effective_n (int, optional): Effective sample size. Used to compute the cheapest pair.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set.

    Returns:
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size as defined in`[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__
            - ppi_corr (float): PPI correlation as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__

    Notes:
        At least one of `budget` and `effective_n` must be provided. If both are provided, `budget` will be used.

        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and effective_n is None:
        raise ValueError(
            "At least one of `budget` and `effective_n` must be provided."
        )

    pointest = (
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X, Y)
        .coef_.squeeze()
    )

    grads, grads_hat, _, inv_hessian = _logistic_get_stats(
        pointest,
        X.astype(float),
        Y,
        Yhat,
        X.astype(float),
        Yhat,
        w=w,
        use_unlabeled=False,
    )

    ppi_corr = _get_ppi_corr(grads, grads_hat, inv_hessian, coord=coord)

    return ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, budget, effective_n, n_max
    )


"""
    POISSON REGRESSION POWER CALCULATION
"""


def ppi_poisson_power(
    X,
    Y,
    Yhat,
    cost_X,
    cost_Y,
    cost_Yhat,
    coord,
    budget=None,
    effective_n=None,
    n_max=None,
    w=None,
):
    """
    Computes the optimal pair of sample sizes for estimating Poisson regression coefficients with PPI.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        cost_X (float): Cost per unlabeled data point.
        cost_Y (float): Cost per gold-standard label.
        cost_Yhat (float): Cost per prediction.
        coord (int): Coordinate to perform power analysis on. Must be in {0, ..., d-1} where d is the shape of the estimand.
        budget (float, optional): Total budget. Used to compute the most powerful pair given the budget.
        effective_n (int, optional): Effective sample size. Used to compute the cheapest pair.
        n_max (int, optional): Maximum number of samples allowed. If provided, the optimal pair will satisfy the additional constraint that n + N <= n_max.
        w (ndarray, optional): Sample weights for the labeled data set.

    Returns:
        dict: Dictionary containing the following items
            - n (int): Optimal number of gold-labeled samples.
            - N (int): Optimal number of unlabeled samples.
            - cost (float): Total cost.
            - effective_n (int): Effective sample size as defined in `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.
            - ppi_corr (float): PPI correlation `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__.

    Notes:
        At least one of `budget` and `effective_n` must be provided. If both are provided, `budget` will be used.

        `[BHvL24] <https://osf.io/preprints/socarxiv/j3bnt>`__ Broska, D., Howes, M., & van Loon, A. (2024, August 22). The Mixed Subjects Design: Treating Large Language Models as  (Potentially) Informative Observations. https://doi.org/10.31235/osf.io/j3bnt
    """
    if budget is None and effective_n is None:
        raise ValueError(
            "At least one of `budget` and `effective_n` must be provided."
        )

    pointest = (
        PoissonRegressor(
            alpha=0,
            fit_intercept=False,
            max_iter=10000,
            tol=1e-15,
        )
        .fit(X, Y)
        .coef_
    )

    grads, grads_hat, _, inv_hessian = _poisson_get_stats(
        pointest,
        X.astype(float),
        Y,
        Yhat,
        X.astype(float),
        Yhat,
        w=w,
        use_unlabeled=False,
    )

    ppi_corr = _get_ppi_corr(grads, grads_hat, inv_hessian, coord=coord)

    return ppi_power(
        ppi_corr, cost_X, cost_Y, cost_Yhat, budget, effective_n, n_max
    )
