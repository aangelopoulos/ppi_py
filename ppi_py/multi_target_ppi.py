#!/usr/bin/env python3
"""
   PPI++ with support for several predicted variables.
"""

from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union, get_origin

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from statsmodels.stats.weightstats import (
    _zconfint_generic,
    _zstat_generic,
    _zstat_generic2,
)

from .utils import reshape_to_2d

from numba import njit


def _ppi_glm_init(
    X,
    Y,
    Xhat,
    Yhat,
    Xhat_unlabeled,
    Yhat_unlabeled,
    initial_params,
    w=None,
    w_unlabeled=None,
    optimizer_options=None,
):
    X, Y, Xhat, Yhat, Xhat_unlabeled, Yhat_unlabeled = [
        np.array(v)
        for v in [
            X,
            Y,
            Xhat,
            Yhat,
            Xhat_unlabeled,
            Yhat_unlabeled,
        ]
    ]
    try:
        initial_params = initial_params(X, Y)
    except TypeError:
        initial_params = np.array(initial_params)

    n = X.shape[0]
    d = initial_params.shape[0]
    N = Xhat_unlabeled.shape[0]

    if w is not None:
        w = np.array(w)
        w = w / w.sum() * n
    else:
        w = np.ones(n)

    if w_unlabeled is not None:
        w_unlabeled = np.array(w_unlabeled)
        w_unlabeled / w_unlabeled.sum() * N
    else:
        w_unlabeled = np.ones(N)

    if optimizer_options is None:
        optimizer_options = {"ftol": 1e-15}
    if "ftol" not in optimizer_options.keys():
        optimizer_options["ftol"] = 1e-15

    return (
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
        n,
        d,
        N,
    )


def ppi_multi_glm_pointest(
    X: ArrayLike,
    Y: ArrayLike,
    Xhat: ArrayLike,
    Yhat: ArrayLike,
    Xhat_unlabeled: ArrayLike,
    Yhat_unlabeled: ArrayLike,
    initial_params: Union[ArrayLike, Callable[[NDArray, NDArray], NDArray]],
    loss: Callable[[NDArray, NDArray, Optional[NDArray]], float],
    gradient: Callable[
        [NDArray, Union[NDArray, float], Union[NDArray, float]], float
    ],
    get_stats: Callable[
        [
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            Optional[NDArray],
            Optional[NDArray],
        ],
        [Tuple[NDArray, NDArray, NDArray, NDArray]],
    ],
    lam: Optional[float] = None,
    coord: Optional[int] = None,
    w: Optional[ArrayLike] = None,
    w_unlabeled: Optional[ArrayLike] = None,
    return_lam: Optional[bool] = False,
    optimizer_options: Optional[dict] = None,
    **kwargs,
) -> Union[NDArray, Tuple[NDArray, float]]:
    """Computes the prediction-powered point estimate for a model with the given loss, gradient, and  using the PPI++ algorithm.

    Args:
    X (ArrayLike): Gold-standard covariate observations (columns are variables; rows are observations).
    Y (ArrayLike): Gold-standard response observations.
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels for covariates
    Yhat (ArrayLike): Predictions corresponding to gold-standard labels for response
    Xhat_unlabeled (ArrayLike): Covariate data without labels, only predictions
    Yhat_unlabeled (ArrayLike): Response data without labels, only predictions
    initial_params (ArrayLike): Initial parameters or a function(X,Y) to estimate them.
    loss (function NDArray, NDArray, Optional[NDArray] -> float): Function calculating the loss, optionally accepting weights
    gradient (function NDArray, NDArray, Optional[NDArray] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    get_stats: Function for getting statistics needed to calculate a CI.
    lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
    coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
    w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
    w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.
    return_lam (bool, optional): whether to return a tuple including lam
    ** kwargs: passed through to minimize

    Returns:
       ndarray or tuple(ndarray, float): (point estimate for the parameters, lam used)

    """

    (
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
        n,
        d,
        N,
    ) = _ppi_glm_init(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
    )

    lam_curr = 1 if lam is None else lam

    ## make contiguous arrays for GLM.
    ## memory inefficient.

    # Initialize theta with the gold-standard data
    theta = minimize(
        loss, args=(X, Y, w), x0=initial_params, jac=gradient, **kwargs
    ).x

    def rectified_loss(_theta):
        # Correct the loss
        return (
            lam_curr
            / N
            * loss(_theta, Xhat_unlabeled, Yhat_unlabeled, w_unlabeled)
            - lam_curr / n * loss(_theta, Xhat, Yhat, w)
            + 1 / n * loss(_theta, X, Y, w)
        )

    def rectified_grad(_theta):
        # Correct the gradient
        return (
            lam_curr
            / N
            * gradient(_theta, Xhat_unlabeled, Yhat_unlabeled, w_unlabeled)
            - lam_curr / n * gradient(_theta, Xhat, Yhat, w)
            + 1 / n * gradient(_theta, X, Y, w)
        )

    pointest = minimize(
        rectified_loss, x0=theta, jac=rectified_grad, **kwargs
    ).x

    if lam is None:
        lam = _calc_lam_multi(
            X=X,
            Y=Y,
            Xhat=Xhat,
            Yhat=Yhat,
            Xhat_unlabeled=Xhat_unlabeled,
            Yhat_unlabeled=Yhat_unlabeled,
            pointest=pointest,
            gradient=gradient,
            get_stats=get_stats,
            coord=coord,
            clip=True,
            w=w,
            w_unlabeled=w_unlabeled,
        )

        return ppi_multi_glm_pointest(
            X=X,
            Y=Y,
            Xhat=Xhat,
            Yhat=Yhat,
            Xhat_unlabeled=Xhat_unlabeled,
            Yhat_unlabeled=Yhat_unlabeled,
            initial_params=pointest,
            loss=loss,
            gradient=gradient,
            get_stats=get_stats,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
            return_lam=return_lam,
            optimizer_options=optimizer_options,
            **kwargs,
        )

    else:
        if return_lam:
            return pointest, lam
        else:
            return pointest


def ppi_multi_glm_pval(
    X: ArrayLike,
    Y: ArrayLike,
    Xhat: ArrayLike,
    Yhat: ArrayLike,
    Xhat_unlabeled: ArrayLike,
    Yhat_unlabeled: ArrayLike,
    initial_params: ArrayLike,
    loss: Callable[[NDArray, NDArray, Optional[NDArray]], float],
    gradient: Callable[[NDArray, NDArray, Optional[NDArray]], float],
    get_stats: Callable[
        [
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            Optional[NDArray],
            Optional[NDArray],
        ],
        [Tuple[NDArray, NDArray, NDArray, NDArray]],
    ],
    alternative: Optional[str] = "two-sided",
    lam: Optional[float] = None,
    coord: Optional[int] = None,
    w: Optional[ArrayLike] = None,
    w_unlabeled: Optional[ArrayLike] = None,
    optimizer_options: Optional[dict] = None,
    **kwargs,
):
    """Computes the pvalues for a model with the given gradient and hessian using the PPI++ algorithm.


    Args:
    X (ArrayLike): Gold-standard covariate observations (columns are variables; rows are observations).
    Y (ArrayLike): Gold-standard response observations.
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels for covariates
    Yhat (ArrayLike): Predictions corresponding to gold-standard labels for response
    Xhat_unlabeled (ArrayLike): Covariate data without labels, only predictions
    Yhat_unlabeled (ArrayLike): Response data without labels, only predictions
    initial_params (ArrayLike): Initial parameters.
    loss (function NDArray, NDArray, Optional[NDArray] -> float): Function calculating the loss, optionally accepting weights
    gradient (function NDArray, NDArray, Optional[NDArray] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    get_stats: Function for getting statistics needed to calculate a CI.
    alternative (string, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
    lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
    coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
    w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
    w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.
    return_lam (bool, optional): whether to return a tuple including lam
    ** kwargs: passed through to minimize

    Returns:
       tuple(ndarray, ndarray, ndarray): (point estimate, CI upper bound, CI lower bound) for the parameters

    """

    (
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
        n,
        d,
        N,
    ) = _ppi_glm_init(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
    )

    ### This is the correct pointest with chosen or optimal lam
    ppi_pointest, lam = ppi_multi_glm_pointest(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        loss,
        gradient,
        get_stats,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
        return_lam=True,
        optimizer_options=optimizer_options,
        **kwargs,
    )

    # Note that we need a new inv_hessian with the correct point estimate, even though we used a hessian to choose lam.
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = get_stats(
        pointest=ppi_pointest,
        X=X,
        Y=Y,
        Xhat=Xhat,
        Yhat=Yhat,
        Xhat_unlabeled=Xhat_unlabeled,
        Yhat_unlabeled=Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    var_unlabeled = np.cov(lam * grads_hat_unlabeled.T).reshape(d, d)
    var = np.cov(grads.T - lam * grads_hat.T).reshape(d, d)
    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian
    var_diag = np.sqrt(np.diag(Sigma_hat) / n)
    pvals = _zstat_generic2(
        ppi_pointest,
        var_diag,
        alternative=alternative,
    )[1]
    return pvals


def ppi_multi_glm_ci(
    X: NDArray,
    Y: NDArray,
    Xhat: NDArray,
    Yhat: NDArray,
    Xhat_unlabeled: NDArray,
    Yhat_unlabeled: NDArray,
    initial_params: NDArray,
    loss: Callable[[NDArray, NDArray, Optional[NDArray]], float],
    gradient: Callable[[NDArray, NDArray, NDArray], float],
    get_stats: Callable[
        [
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            NDArray,
            Optional[NDArray],
            Optional[NDArray],
        ],
        [Tuple[NDArray, NDArray, NDArray, NDArray]],
    ],
    alpha: Optional[float] = 0.95,
    alternative: Optional[str] = "two-sided",
    lam: Optional[float] = None,
    coord: Optional[int] = None,
    w: Optional[ArrayLike] = None,
    w_unlabeled: Optional[ArrayLike] = None,
    optimizer_options: Optional[dict] = None,
    **kwargs,
):
    """Computes the confidence interval for a model with the given gradient and hessian using the PPI++ algorithm.

    Args:
    X (ArrayLike): Gold-standard label data (columns are variables; rows are observations).
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels
    Xhat_unlabeled (ArrayLike): Data without labels, only predictions
    pointest (ArrayLike): the point estimate.
    loss (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the loss, optionally accepting weights
    gradient (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    get_stats: Function for calculating statistics for the confidence
    alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
    alternative (string, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
    lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
    coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
    w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
    w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.
    * args: passed through to optimizer (e.g., to support constrained optimization).
    ** kwargs: passed through to minimize

    Returns:
       tuple(ndarray, ndarray, ndarray): (point estimate, CI upper bound, CI lower bound) for the parameters

    """

    (
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
        n,
        d,
        N,
    ) = _ppi_glm_init(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        w,
        w_unlabeled,
        optimizer_options,
    )

    ### This is the correct pointest with chosen or optimal lam
    ppi_pointest, lam = ppi_multi_glm_pointest(
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        initial_params,
        loss,
        gradient,
        get_stats,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
        return_lam=True,
        optimizer_options=optimizer_options,
        **kwargs,
    )

    # Note that we need a new inv_hessian with the correct point estimate, even though we used a hessian to choose lam.
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = get_stats(
        ppi_pointest,
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        w,
        w_unlabeled,
    )
    var_unlabeled = np.cov(lam * grads_hat_unlabeled.T).reshape(d, d)
    var = np.cov(grads.T - lam * grads_hat.T).reshape(d, d)
    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian
    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )


def _calc_lam_glm(
    grads,
    grads_hat,
    grads_hat_unlabeled,
    inv_hessian,
    coord=None,
    clip=False,
    optim_mode="overall",
):
    """
    Calculates the optimal value of lam for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lam`, when `optim_mode="overall"`.
        If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lam to be non-negative. Defaults to `False`.
        optim_mode (ndarray, optional): Mode for which to optimize `lam`, either `overall` or `element`.
        If `overall`, it optimizes the total variance over all coordinates, and the function returns a scalar.
        If `element`, it optimizes the variance for each coordinate separately, and the function returns a vector.


    Returns:
        float: Optimal value of `lam`. Lies in [0,1].
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
    cov_grads = (1 / n) * (
        grads_cent.T @ grad_hat_cent + grad_hat_cent.T @ grads_cent
    )

    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )
    var_grads_hat = var_grads_hat.reshape(d, d)

    vhat = inv_hessian if coord is None else inv_hessian[coord, :]
    if optim_mode == "overall":
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
        lam = num / denom
        lam = lam.item()
    elif optim_mode == "element":
        num = np.diag(vhat @ cov_grads @ vhat)
        denom = 2 * (1 + (n / N)) * np.diag(vhat @ var_grads_hat @ vhat)
        lam = num / denom
    else:
        raise ValueError(
            "Invalid value for optim_mode. Must be either 'overall' or 'element'."
        )
    if clip:
        lam = np.clip(lam, 0, 1)
    return lam


def _calc_lam_multi(
    X: NDArray,
    Y: NDArray,
    Xhat: NDArray,
    Yhat: NDArray,
    Xhat_unlabeled: NDArray,
    Yhat_unlabeled: NDArray,
    pointest: NDArray,
    gradient: Callable[[ArrayLike, ArrayLike, ArrayLike], float],
    get_stats,
    coord,
    clip,
    w: NDArray,
    w_unlabeled: NDArray,
) -> float:
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = get_stats(
        pointest,
        X,
        Y,
        Xhat,
        Yhat,
        Xhat_unlabeled,
        Yhat_unlabeled,
        w,
        w_unlabeled,
    )

    return _calc_lam_glm(
        grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord, clip
    )


@njit
def _glm_get_stats(
    link: Callable[[NDArray], NDArray],
    scalar_grad: Callable[[NDArray, NDArray, float], [NDArray]],
    scalar_hessian: Callable[[NDArray, NDArray], [NDArray]],
    pointest: NDArray,
    X: NDArray,
    Y: NDArray,
    Xhat: NDArray,
    Yhat: NDArray,
    Xhat_unlabeled: NDArray,
    Yhat_unlabeled: NDArray,
    w: Optional[NDArray] = None,
    w_unlabeled: Optional[NDArray] = None,
    use_unlabeled: Optional[bool] = True,
):
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]

    mu = link(X @ pointest)
    mu_til = link(Xhat_unlabeled @ pointest)
    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(Xhat_unlabeled.shape)

    if use_unlabeled:
        for i in range(N):
            hessian += (
                w_unlabeled[i]
                / (N + n)
                * scalar_hessian(
                    mu_til[i],
                    Xhat_unlabeled[i],
                )
            )

            grads_hat_unlabeled[i, :] = w_unlabeled[i] * scalar_grad(
                mu_til[i], Xhat_unlabeled[i, :], Yhat_unlabeled[i]
            )
    grads = np.zeros(X.shape)
    grads_hat = np.zeros(X.shape)
    for i in range(n):
        hessian += (
            w[i] / (N + n) * scalar_hessian(mu[i], X[i])
            if use_unlabeled
            else w[i] / n * scalar_hessian(mu[i], X[i])
        )
        grads[i, :] = w[i] * scalar_grad(mu[i], X[i, :], Y[i])
        grads_hat[i, :] = w[i] * scalar_grad(mu[i], X[i, :], Yhat[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    return grads, grads_hat, grads_hat_unlabeled, inv_hessian
