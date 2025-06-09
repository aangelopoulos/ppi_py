#!/usr/bin/env python3
"""
   PPI++ with support for several predicted variables.
"""

from typing import Optional, Union, Tuple
import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable
from enum import Enum
from scipy.optimize import minimize
from .ppi import _calc_lam_glm
from .utils import reshape_to_2d

def _ppi_multi_convex_pointest(X: ArrayLike,
                               Xhat: ArrayLike,
                               Xhat_unlabeled: ArrayLike,
                               initial_params: ArrayLike,
                               loss: Callable[[ArrayLike,
                                               ArrayLike,
                                               Optional[ArrayLike]],
                                              float],
                               gradient: Callable[[ArrayLike,
                                                   ArrayLike,
                                                   Optional[ArrayLike]],
                                                  float],
                               hessian: Optional[Callable[[ArrayLike,
                                                           ArrayLike,
                                                           Optional[ArrayLike]],
                                                          np.ndarray]] = None,
                               lam: Optional[float] = None,
                               coord: Optional[int] = None,
                               w: Optional[ArrayLike] = None,
                               w_unlabeled: Optional[ArrayLike] = None,
                               *args,
                               **kwargs,
                               ) -> Tuple[ndarray, float]:



    """Computes the prediction-powered point estimate for a model with the given gradient using the PPI++ algorithm.

    Args:
    X (ArrayLike): Gold-standard label data (columns are variables; rows are observations).
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels
    Xhat_unlabeled (ArrayLike): Data without labels, only predictions
    initial_params (ArrayLike): Initial parameters.
    loss (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the loss, optionally accepting weights
    gradient (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    hessian (function ArrayLike, ArrayLike, Optional[ArrayLike] -> ndarray): Function for evaluating the hessian of the loss; used for choosing power tuning parameter.
    alternative (string, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
    lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
    coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
    w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
    w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.
    * args: passed through to optimizer (e.g., to support constrained optimization).
    ** kwargs: passed through to minimize

    Returns:
       tuple(ndarray, float): (point estimate for the parameters, lam used)

    """

    if hessian is None and lam is None:
        raise ValueError("""Either hessian or lam is required.
        If lam isn't provided, hessian is used to choose it""")

    X = np.array(X)
    Xhat_unlabeled = np.array(Xhat_unlabeled)
    Xhat = np.array(Xhat)
    w = np.array(w)
    w_unlabeled = np.array(w_unlabeled)
    
    n = X.shape[0] # No. labels
    d = X.shape[1] # No. variables
    N = Xhat_unlabeled.shape[0] # No. unlabeled
    w = np.ones(n) if w is None else w / w.sum() * n

    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    lam_curr = 1 if lam is None else lam

    # Initialize theta with the gold-standard data

    theta = minimize(loss,
                     initial_params,
                     jac=gradient,
                     args=args,
                     **kwargs).x

    def ppi_loss(_theta):
        # Correct the loss
        return (
            lam_curr
            / N
            * loss(Xhat_unlabeled, _theta, w_unlabeled)
            - lam_curr
            / n
            * loss(Xhat, _theta, w)
            + 1 / n * loss(X, _theta, w))

    def ppi_grad(_theta):
        # Correct the gradient
        return (
            lam_curr
            / N
            * gradient(Xhat_unlabeled, _theta, w_unlabeled)
            - lam_curr / n * gradient(Xhat, _theta, w)
            + 1 / n * gradient(X, _theta, w)
        )

    pointest = minimize(ppi_loss,
                        theta,
                        jac=ppi_grad,
                        args=args,
                        **kwargs).x

    if lam is None:
        lam = _calc_lam_multi(X=X,
                              Xhat=Xhat,
                              Xhat_unlabeled=Xhat_unlabeled,
                              gradient=gradient,
                              hessian=hessian,
                              coord=coord,
                              clip=True,
                              w=w,
                              w_unlabeled=w_unlabeled)

        return _ppi_multi_convex_pointest(X=X,
                                          Xhat=Xhat,
                                          Xhat_unlabeled=Xhat_unlabeled,
                                          initial_params=pointest,
                                          loss=loss,
                                          gradient=gradient,
                                          hessian=hessian,
                                          lam=lam,
                                          coord=coord,
                                          w=w,
                                          w_unlabeled=w_unlabeled,
                                          *args,
                                          **kwargs), lam

    else:
        return pointest, lam

def ppi_multi_convex_ci(X: ArrayLike,
                        Xhat: ArrayLike,
                        Xhat_unlabeled: ArrayLike,
                        pointest: ArrayLike,
                        gradient: Callable[[ArrayLike,
                                            ArrayLike,
                                            Optional[ArrayLike]],
                                           float],
                        hessian: Optional[Callable[[ArrayLike,
                                                    ArrayLike,
                                                    Optional[ArrayLike]],
                                                   np.ndarray]] = None,
                        alpha: Optional[float] = 0.95,
                        alternative: Optional[str] = "two-sided",
                        lam: Optional[float] = None,
                        coord: Optional[int] = None,
                        w: Optional[ArrayLike] = None,
                        w_unlabeled: Optional[ArrayLike] = None,
                        *args,
                        **kwargs):
    """Computes the confidence interval for a model with the given gradient using the PPI++ algorithm.

    Args:
    X (ArrayLike): Gold-standard label data (columns are variables; rows are observations).
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels
    Xhat_unlabeled (ArrayLike): Data without labels, only predictions
    initial_params (ArrayLike): Initial parameters.
    loss (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the loss, optionally accepting weights
    gradient (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    hessian (function ArrayLike, ArrayLike, Optional[ArrayLike] -> ndarray): Function for evaluating the hessian of the loss; used for choosing power tuning parameter.
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

    pointest = np.ndarray(pointest)
    X = np.ndarray(X)
    Xhat = np.ndarray(Xhat)
    Xhat_unlabeled = np.ndarray(Xhat_unlabeled)
    n = X.shape[0]
    d = X.shape[1]
    N = Xhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    ### This is the correct pointest with chosen or optimal lam
    pointest, lam = _ppi_multi_convex_pointest(X,
                                               Xhat,
                                               Xhat_unlabeled,
                                               initial_params,
                                               loss,
                                               gradient,
                                               hessian, 
                                               lam=lam,
                                               coord=coord,
                                               w=w,
                                               w_unlabeled=w_unlabeled,
                                               *args,
                                               **kwargs
                                               )

    # Note that we need a new inv_hessian with the correct point estimate, even though we used a hessian to choose lam.
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = multi_get_stats(X,
                                                                         Xhat,
                                                                         Xhat_unlabeled,
                                                                         pointest,
                                                                         gradient,
                                                                         hessian,
                                                                         w,
                                                                         w_unlabeled)
    var_unlabeled = np.cov(lam * grads_hat_unlabeled.T).reshape(d, d)
    var = np.cov(grads.T - lam * grads_hat.T).reshape(d, d)
    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian
    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )

def multi_get_stats(X: np.ndarray,
                    Xhat: np.ndarray,
                    Xhat_unlabeled: np.ndarray,
                    pointest: np.ndarray,
                    gradient: Callable[[ArrayLike,
                                        ArrayLike,
                                        Optional[ArrayLike]],
                                       float],
                    hessian: Callable[[ArrayLike,
                                       ArrayLike,
                                       Optional[ArrayLike]],
                                      np.ndarray],
                    w: np.ndarray,
                    w_unlabeled: np.ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    grads = gradient(X, pointest, w)
    grads_hat = gradient(Xhat, pointest, w)
    grads_hat_unlabeled = gradient(Xhat_unlabeled, pointest, w_unlabeled)
    hess = hessian(X, pointest, w)  # type: ignore
    inv_hessian = np.linalg.inv(hess).reshape(d, d)
    return grads, grads_hat, grads_hat_unlabeled, inv_hessian


def _calc_lam_multi(X: np.ndarray,
                    Xhat: np.ndarray,
                    Xhat_unlabeled: np.ndarray,
                    pointest: np.ndarray,
                    gradient: Callable[[ArrayLike,
                                        ArrayLike,
                                        Optional[ArrayLike]],
                                       float],
                    hessian: Callable[[ArrayLike,
                                       ArrayLike,
                                       Optional[ArrayLike]],
                                      np.ndarray],
                    coord = coord,
                    clip = True,
                    w: np.ndarray,
                    w_unlabeled: np.ndarray) -> float:

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = multi_get_stats(X,
                                                                         Xhat,
                                                                         Xhat_unlabeled,
                                                                         pointest,
                                                                         gradient,
                                                                         hessian,
                                                                         w,
                                                                         w_unlabeled)

    return _calc_lam_glm(grads,
                         grads_hat,
                         grads_hat_unlabeled,
                         inv_hessian,
                         coord=coord,
                         clip=clip)
