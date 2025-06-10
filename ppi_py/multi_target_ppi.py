#!/usr/bin/env python3
"""
   PPI++ with support for several predicted variables.
"""

from typing import Optional, Union, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable
from enum import Enum
from scipy.optimize import minimize
from .utils import reshape_to_2d, calc_lam_glm
from functools import partial
from statsmodels.stats.weightstats import (_zconfint_generic, _zstat_generic,
                                           _zstat_generic2)


def ppi_multi_glm_pointest(Y: NDArray,
                           X: NDArray,
                           Yhat: NDArray,
                           Xhat: NDArray,
                           Yhat_unlabeled: NDArray,
                           Xhat_unlabeled: NDArray,
                           initial_params: NDArray,
                           loss: Callable[[NDArray,
                                           NDArray,
                                           Optional[NDArray]],
                                          float],
                           gradient: Callable[[NDArray,
                                               Union[NDArray, float],
                                               Union[NDArray,float]],
                                              float],
                           get_stats: Callable[[NDArray,
                                                NDArray,
                                                NDArray,
                                                NDArray,
                                                NDArray,
                                                NDArray,
                                                Optional[NDArray],
                                                Optional[NDArray]],
                                               [Tuple[NDArray, NDArray, NDArray, NDArray]]
                                               ],
                           lam: Optional[float] = None,
                           coord: Optional[int] = None,
                           w: Optional[ArrayLike] = None,
                           w_unlabeled: Optional[ArrayLike] = None,
                           return_lam: Optional[bool] = False,
                           **kwargs,
                           ) -> Tuple[NDArray, float]:
    """Computes the prediction-powered point estimate for a model with the given loss, gradient, and  using the PPI++ algorithm.

    Args:
    X (ArrayLike): Gold-standard covariate observations (columns are variables; rows are observations).
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels
    Xhat_unlabeled (ArrayLike): Data without labels, only predictions
    initial_params (ArrayLike): Initial parameters.
    loss (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the loss, optionally accepting weights
    gradient (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    get_stats: Function for getting statistics needed to calculate a CI. 
    alternative (string, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
    lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
    coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
    w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
    w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.
    return_lam (bool, optional): whether to return a tuple including lam
    ** kwargs: passed through to minimize

    Returns:
       tuple(ndarray, float): (point estimate for the parameters, lam used)

    """
    n = X.shape[0] # No. labels
    d = initial_params.shape[0] # No. variables
    N = Xhat_unlabeled.shape[0] # No. unlabeled
    w = np.ones(n) if w is None else w / w.sum() * n

    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    lam_curr = 1 if lam is None else lam

    ## make contiguous arrays for GLM.
    ## memory inefficient.

    # Initialize theta with the gold-standard data
    theta = minimize(loss,
                     args=(Y, X, w),
                     x0=initial_params,
                     jac=gradient,
                     **kwargs).x

    def rectified_loss(_theta):
        # Correct the loss
        return (
            lam_curr
            / N
            * loss(_theta, Yhat_unlabeled, Xhat_unlabeled, w_unlabeled)
            - lam_curr
            / n
            * loss(_theta, Yhat, Xhat, w)
            + 1 / n * loss(_theta, Y, X, w))

    def rectified_grad(_theta):
        # Correct the gradient
        return (
            lam_curr
            / N
            * gradient(_theta, Yhat_unlabeled, Xhat_unlabeled, w_unlabeled)
            - lam_curr / n * gradient(_theta, Yhat, Xhat, w)
            + 1 / n * gradient(_theta, Y, X, w)
        )

    pointest = minimize(rectified_loss,
                        x0=theta,
                        jac=rectified_grad,
                        **kwargs).x

    if lam is None:
        lam = _calc_lam_multi(Y=Y,
                              X=X,
                              Yhat=Yhat,
                              Xhat=Xhat,
                              Yhat_unlabeled=Yhat_unlabeled,
                              Xhat_unlabeled=Xhat_unlabeled,
                              pointest=pointest,
                              gradient=gradient,
                              get_stats=get_stats,
                              coord=coord,
                              clip=True,
                              w=w,
                              w_unlabeled=w_unlabeled)

        return ppi_multi_glm_pointest(Y=Y,
                                      X=X,
                                      Yhat=Yhat,
                                      Xhat=Xhat,
                                      Yhat_unlabeled=Yhat_unlabeled,
                                      Xhat_unlabeled=Xhat_unlabeled,
                                      initial_params=pointest,
                                      loss=loss,
                                      gradient=gradient,
                                      get_stats=get_stats,
                                      lam=lam,
                                      coord=coord,
                                      w=w,
                                      w_unlabeled=w_unlabeled,
                                      return_lam=return_lam,
                                      **kwargs)

    else:
        if return_lam:
            return pointest, lam
        else:
            return pointest


def ppi_multi_glm_pval(Y: NDArray,
                       X: NDArray,
                       Yhat: NDArray,
                       Xhat: NDArray,
                       Yhat_unlabeled: NDArray,
                       Xhat_unlabeled: NDArray,
                       initial_params: NDArray,
                       loss: Callable[[NDArray,
                                       NDArray,
                                       Optional[NDArray]],
                                      float],
                       gradient: Callable[[NDArray,
                                            NDArray,
                                            Optional[NDArray]],
                                           float],
                       get_stats: Callable[[NDArray,
                                             NDArray,
                                             NDArray,
                                             NDArray,
                                             NDArray,
                                             NDArray,
                                             Optional[NDArray],
                                             Optional[NDArray]],
                                            [Tuple[NDArray, NDArray, NDArray, NDArray]]
                                            ],                        
                        alternative: Optional[str] = "two-sided",
                        lam: Optional[float] = None,
                        coord: Optional[int] = None,
                        w: Optional[ArrayLike] = None,
                        w_unlabeled: Optional[ArrayLike] = None,
                        **kwargs):

    """Computes the pvalues for a model with the given gradient and hessian using the PPI++ algorithm.

    Args:
    X (ArrayLike): Gold-standard label data (columns are variables; rows are observations).
    Xhat (ArrayLike): Predictions corresponding to gold-standard labels
    Xhat_unlabeled (ArrayLike): Data without labels, only predictions
    initial_params (ArrayLike): Initial parameters.
    loss (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the loss, optionally accepting weights
    gradient (function ArrayLike, ArrayLike, Optional[ArrayLike] -> float): Function calculating the subgradient of the loss with respect to the parameters, optionally accepting weights.
    get_stats: Function for getting statistics needed to calculate a CI. 
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

    n = X.shape[0]
    d = initial_params.shape[0]
    N = Xhat_unlabeled.shape[0]

    ### This is the correct pointest with chosen or optimal lam
    ppi_pointest, lam = ppi_multi_glm_pointest(Y,
                                               X,
                                               Yhat,
                                               Xhat,
                                               Yhat_unlabeled,
                                               Xhat_unlabeled,
                                               initial_params,
                                               loss,
                                               gradient,
                                               get_stats,
                                               lam=lam,
                                               coord=coord,
                                               w=w,
                                               w_unlabeled=w_unlabeled,
                                               return_lam=True,
                                               **kwargs
                                               )

    # Note that we need a new inv_hessian with the correct point estimate, even though we used a hessian to choose lam.
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = get_stats(pointest=ppi_pointest,
                                                                   Y=Y,
                                                                   X=X,
                                                                   Yhat=Yhat,
                                                                   Xhat=Xhat,
                                                                   Yhat_unlabeled=Yhat_unlabeled,
                                                                   Xhat_unlabeled=Xhat_unlabeled,
                                                                   w=w,
                                                                   w_unlabeled=w_unlabeled)
                                                                   

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


def ppi_multi_glm_ci(Y: NDArray,
                     X: NDArray,
                     Yhat: NDArray,
                     Xhat: NDArray,
                     Yhat_unlabeled: NDArray,
                     Xhat_unlabeled: NDArray,
                     initial_params: NDArray,
                     loss: Callable[[NDArray,
                                     NDArray,
                                     Optional[NDArray]],
                                    float],
                     gradient: Callable[[NDArray,
                                         NDArray,
                                         NDArray],
                                        float],
                     get_stats: Callable[[NDArray,
                                          NDArray,
                                          NDArray,
                                          NDArray,
                                          NDArray,
                                          NDArray,
                                          Optional[NDArray],
                                          Optional[NDArray]],
                                         [Tuple[NDArray, NDArray, NDArray, NDArray]]
                                         ],
                     alpha: Optional[float] = 0.95,
                     alternative: Optional[str] = "two-sided",
                     lam: Optional[float] = None,
                     coord: Optional[int] = None,
                     w: Optional[ArrayLike] = None,
                     w_unlabeled: Optional[ArrayLike] = None,
                     **kwargs):
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
    n = X.shape[0]
    d = initial_params.shape[0]
    N = Xhat_unlabeled.shape[0]

    ### This is the correct pointest with chosen or optimal lam
    ppi_pointest, lam = ppi_multi_glm_pointest(Y,
                                               X,
                                               Yhat,
                                               Xhat,
                                               Yhat_unlabeled,
                                               Xhat_unlabeled,
                                               initial_params,
                                               loss,
                                               gradient,
                                               get_stats,
                                               lam=lam,
                                               coord=coord,
                                               w=w,
                                               w_unlabeled=w_unlabeled,
                                               return_lam = True,
                                               **kwargs
                                           )

    # Note that we need a new inv_hessian with the correct point estimate, even though we used a hessian to choose lam.
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = get_stats(ppi_pointest,
                                                                   Y,
                                                                   X,
                                                                   Yhat,
                                                                   Xhat,
                                                                   Yhat_unlabeled,
                                                                   Xhat_unlabeled,
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

def _calc_lam_multi(Y: NDArray,
                    X: NDArray,
                    Yhat: NDArray,
                    Xhat: NDArray,
                    Yhat_unlabeled: NDArray,
                    Xhat_unlabeled: NDArray,
                    pointest: NDArray,
                    gradient: Callable[[ArrayLike,
                                        ArrayLike,
                                        ArrayLike],
                                       float],
                    get_stats,
                    coord,
                    clip,
                    w: NDArray,
                    w_unlabeled: NDArray) -> float:

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = get_stats(pointest,
                                                                   Y,
                                                                   X,
                                                                   Yhat,
                                                                   Xhat,
                                                                   Yhat_unlabeled,
                                                                   Xhat_unlabeled,
                                                                   w,
                                                                   w_unlabeled)

    return calc_lam_glm(grads,
                        grads_hat,
                        grads_hat_unlabeled,
                        inv_hessian,
                        coord,
                        clip)
