import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import Callable, Optional, Union, Dict
import warnings
import inspect
from pprint import pprint


def _validate_args(
    Y,
    Yhat_unlabeled,
    X,
    X_unlabeled,
    loss,
    dp_alpha,
    n_samples,
    m,
    n_jobs,
    optimizer_kwargs,
    base_measure,
    theta0,
):
    """
    Validate and process arguments for sample_ai_posterior.

    Returns:
        tuple: Processed arguments
    """
    if (Yhat_unlabeled is None) == (base_measure is None):
        raise ValueError(
            "Exactly one of Yhat_unlabeled or base_measure must be provided. "
            "Got Yhat_unlabeled={} and base_measure={}".format(
                "provided" if Yhat_unlabeled is not None else "None",
                "provided" if base_measure is not None else "None",
            )
        )

    if loss is None:
        raise ValueError(
            "Loss function is required. Please provide a loss function with signature:\n"
            "loss(y: ndarray, theta: scalar or ndarray) -> ndarray\n"
            "where y is the data array and theta is the parameter(s) to optimize.\n"
            "Example: lambda y, theta: (y - theta) ** 2  # for squared loss"
        )

    # When using base_measure, X_unlabeled is not required since base_measure generates it
    if Yhat_unlabeled is not None and (X is None) != (X_unlabeled is None):
        raise ValueError(
            "When using Yhat_unlabeled, X and X_unlabeled must both be provided or both be None."
        )

    Y = np.asarray(Y)
    if Yhat_unlabeled is not None:
        Yhat_unlabeled = np.asarray(Yhat_unlabeled)
    if X is not None:
        X = np.asarray(X)
        X_unlabeled = np.asarray(X_unlabeled)

    n = len(Y)
    N = len(Yhat_unlabeled) if Yhat_unlabeled is not None else None

    if X is not None and len(X) != n:
        raise ValueError(
            f"X must have same length as Y. Got len(X)={len(X)}, len(Y)={n}"
        )

    if (
        X_unlabeled is not None
        and Yhat_unlabeled is not None
        and len(X_unlabeled) != N
    ):
        raise ValueError(
            f"X_unlabeled must have same length as Yhat_unlabeled. Got len(X_unlabeled)={len(X_unlabeled)}, len(Yhat_unlabeled)={N}"
        )

    if dp_alpha <= 0:
        raise ValueError(f"dp_alpha must be positive. Got {dp_alpha}")

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got {n_samples}")

    if Yhat_unlabeled is not None:
        if m is None:
            m = N
        elif m > N:
            warnings.warn(
                f"Truncation size m={m} exceeds unlabeled data size N={N}. Using m=N."
            )
            m = N
        elif m <= 0:
            raise ValueError(f"m must be positive. Got {m}")
    else:
        # With base_measure, m is required
        if m is None:
            raise ValueError("m must be specified when using base_measure")
        elif m <= 0:
            raise ValueError(f"m must be positive. Got {m}")

    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count()
    elif n_jobs <= 0:
        raise ValueError(f"n_jobs must be positive or -1. Got {n_jobs}")

    if theta0 is None:
        raise ValueError(
            "theta0 (initial parameter guess) is required. "
            "Please provide an appropriate initial value for optimization."
        )

    theta0 = np.asarray(theta0)

    # Default optimizer kwargs
    if optimizer_kwargs is None:
        optimizer_kwargs = {
            "method": "Nelder-Mead",
            "options": {"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
        }

    return (
        Y,
        Yhat_unlabeled,
        X,
        X_unlabeled,
        loss,
        dp_alpha,
        n_samples,
        m,
        n_jobs,
        optimizer_kwargs,
        theta0,
    )


def _validate_loss(loss, has_covariates=False):
    """
    Validate loss function signature.

    Args:
        loss (Callable): Loss function to validate
        has_covariates (bool): Whether covariates are provided

    Raises:
        ValueError: If loss function signature is incorrect
    """
    try:
        sig = inspect.signature(loss)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

        actual_count = len(params)

        if has_covariates and actual_count == 2:
            raise ValueError(
                "Loss function must accept 3 parameters (y, theta, X) when covariates are provided. "
                f"Got {actual_count} parameters."
            )
        elif not has_covariates and actual_count == 3:
            raise ValueError(
                "Loss function accepts 3 parameters (y, theta, X) but no covariates were provided. "
                "Either provide covariates or use a loss function with 2 parameters (y, theta)."
            )
        elif actual_count not in [2, 3]:
            raise ValueError(
                f"Loss function must accept either 2 parameters (y, theta) or 3 parameters (y, theta, X). "
                f"Got {actual_count} parameters."
            )

    except (ValueError, TypeError) as e:
        if isinstance(e, ValueError):
            raise
        else:
            warnings.warn(
                "Could not validate loss function signature. "
                "Potential signature errors will be caught at runtime."
            )


def _validate_base_measure(base_measure, Y, X=None):
    """
    Validate that base_measure returns outputs compatible with provided data.

    Args:
        base_measure (Callable): Function to validate
        Y (ndarray): Labeled data to check compatibility against
        X (ndarray, optional): Covariates to check compatibility against

    Raises:
        ValueError: If base_measure returns incompatible types/shapes
    """
    if base_measure is None:
        return

    has_covariates = X is not None

    try:
        # Test call with size=2
        test_output = base_measure(size=2)
    except Exception as e:
        raise ValueError(f"base_measure failed with size=2: {e}")

    if has_covariates:
        # Must return (X, Y) tuple
        if not isinstance(test_output, tuple) or len(test_output) != 2:
            raise ValueError(
                "base_measure must return (X, Y) tuple when covariates are provided. "
                f"Got {type(test_output)}"
            )

        X_test, Y_test = test_output

        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)

        if X_test.shape[0] != 2:
            raise ValueError(
                f"base_measure X output has wrong first dimension. Expected 2, got {X_test.shape[0]}"
            )

        if X.ndim == 1 and X_test.ndim == 2 and X_test.shape[1] != 1:
            raise ValueError(
                f"base_measure X output has wrong shape. Expected (2,) or (2,1), got {X_test.shape}"
            )
        elif X.ndim == 2 and X_test.shape != (2, X.shape[1]):
            raise ValueError(
                f"base_measure X output has wrong shape. Expected (2, {X.shape[1]}), got {X_test.shape}"
            )

    else:
        # Must return Y only (not a tuple)
        if isinstance(test_output, tuple):
            raise ValueError(
                "base_measure must return Y array only when no covariates are provided, not a tuple"
            )

        Y_test = np.asarray(test_output)

    if Y_test.shape[0] != 2:
        raise ValueError(
            f"base_measure Y output has wrong first dimension. Expected 2, got {Y_test.shape[0]}"
        )

    if Y.ndim == 1 and Y_test.ndim == 2 and Y_test.shape[1] != 1:
        raise ValueError(
            f"base_measure Y output has wrong shape. Expected (2,) or (2,1), got {Y_test.shape}"
        )
    elif Y.ndim == 2 and Y_test.shape != (2, Y.shape[1]):
        raise ValueError(
            f"base_measure Y output has wrong shape. Expected (2, {Y.shape[1]}), got {Y_test.shape}"
        )

    # Check for numeric types
    if not np.issubdtype(Y_test.dtype, np.number):
        raise ValueError(
            f"base_measure Y output must be numeric. Got dtype {Y_test.dtype}"
        )

    if has_covariates and not np.issubdtype(X_test.dtype, np.number):
        raise ValueError(
            f"base_measure X output must be numeric. Got dtype {X_test.dtype}"
        )


def sample_ai_posterior(
    Y,
    loss,
    dp_alpha,
    theta0,
    Yhat_unlabeled=None,
    n_samples=1000,
    m=None,
    n_jobs=1,
    verbose=False,
    optimizer_kwargs=None,
    X=None,
    X_unlabeled=None,
    base_measure=None,
):
    """
    Samples from the posterior on the risk minimizer with an AI prior.

    The AI prior is DP(dp_alpha, F_AI) where the base measure F_AI
    is the empirical distribution of Yhat_unlabeled.

    Args:
        Y (ndarray): Gold-standard labels of shape (n,) or (n, d).
        loss (Callable): Loss function with signature loss(y, theta) -> ndarray.
            Must accept vectorized inputs where y is an array and theta is a scalar or array.
        dp_alpha (float): Concentration parameter for Dirichlet Process.
        theta0 (ndarray): Initial guess for optimization. Must be provided.
        Yhat_unlabeled (ndarray, optional): AI predictions on unlabeled data of shape (N,) or (N, d).
            Either Yhat_unlabeled or base_measure must be provided. Defaults to None.
        n_samples (int): Number of posterior samples to draw. Defaults to 1000.
        m (int, optional): Truncation size for F_AI (not used by default). If None, uses all unlabeled data.
        n_jobs (int): Number of parallel jobs. Use -1 for all CPUs. Defaults to 1.
        verbose (bool): Whether to show progress bar. Defaults to True.
        optimizer_kwargs (dict, optional): Arguments for scipy.optimize.minimize.
            Defaults to {"method": "Nelder-Mead"}.
        X (ndarray, optional): Covariates for labeled data of shape (n,) or (n, p). Defaults to None.
        X_unlabeled (ndarray, optional): Covariates for unlabeled data of shape (N,) or (N, p). Defaults to None.
        base_measure (Callable, optional): Custom base measure for DP prior. If None, uses empirical distribution of Yhat_unlabeled.
            When X is None: should have signature base_measure(size) -> Y array of shape (size,) or (size, d).
            When X is provided: should have signature base_measure(size) -> (X, Y) tuple. Defaults to None.

    Returns:
        ndarray: Array of shape (n_samples,) or (n_samples, p) containing posterior samples.

    Notes:
        This implements the posterior bootstrap algorithm from the AI-priors paper,
        where we sample from F|D_n ~ DP(alpha+n, G_n) by:
        1. Using all points from F_AI (no truncation by default)
        2. Sampling weights from Dirichlet distribution
        3. Minimizing weighted loss
    """

    (
        Y,
        Yhat_unlabeled,
        X,
        X_unlabeled,
        loss,
        dp_alpha,
        n_samples,
        m,
        n_jobs,
        optimizer_kwargs,
        theta0,
    ) = _validate_args(
        Y,
        Yhat_unlabeled,
        X,
        X_unlabeled,
        loss,
        dp_alpha,
        n_samples,
        m,
        n_jobs,
        optimizer_kwargs,
        base_measure,
        theta0,
    )

    _validate_loss(loss, has_covariates=(X is not None))

    _validate_base_measure(base_measure, Y, X)

    n = len(Y)
    N = len(Yhat_unlabeled) if Yhat_unlabeled is not None else None

    if n_jobs == 1:
        # sequential execution
        samples = []
        for i in tqdm(
            range(n_samples), desc="Posterior sampling", disable=not verbose
        ):
            samples.append(
                _sample_one_helper(
                    (
                        i,
                        Y,
                        Yhat_unlabeled,
                        n,
                        m,
                        dp_alpha,
                        loss,
                        optimizer_kwargs,
                        X,
                        X_unlabeled,
                        theta0,
                        base_measure,
                    )
                )
            )
    else:
        # parallel execution
        samples = list(
            tqdm(
                Parallel(return_as="generator", n_jobs=n_jobs)(
                    delayed(_sample_one_helper)(
                        (
                            i,
                            Y,
                            Yhat_unlabeled,
                            n,
                            m,
                            dp_alpha,
                            loss,
                            optimizer_kwargs,
                            X,
                            X_unlabeled,
                            theta0,
                            base_measure,
                        )
                    )
                    for i in range(n_samples)
                ),
                total=n_samples,
                desc="Posterior samples",
                disable=not verbose,
            )
        )

    return np.array(samples)


def _sample_weights(n, m, alpha):
    """
    Sample weights from Dirichlet distribution for n labeled and m unlabeled points.

    Args:
        n (int): Number of labeled observations.
        m (int): Number of unlabeled observations.
        alpha (float): Concentration parameter.

    Returns:
        tuple: (w_y, w_ystar) weights for labeled and unlabeled data.
    """
    alpha_vec = np.array([1] * n + [alpha / m] * m)
    all_weights = np.random.dirichlet(alpha_vec, size=1).flatten()
    w_y = all_weights[:n]
    w_ystar = all_weights[n:]
    return w_y, w_ystar


def _sample_one_helper(args):
    """Helper function for parallel sampling."""
    (
        seed,
        Y,
        Yhat_unlabeled,
        n,
        m,
        dp_alpha,
        loss,
        optimizer_kwargs,
        X,
        X_unlabeled,
        theta0,
        base_measure,
    ) = args

    if seed is not None:
        np.random.seed(seed)

    # sample from base measure if provided, else use Yhat_unlabeled
    if base_measure is None:
        ystar = Yhat_unlabeled
        xstar = X_unlabeled
    else:
        if X is None:
            ystar = base_measure(size=m)
            ystar = np.asarray(ystar)
            xstar = None
        else:
            xstar, ystar = base_measure(size=m)
            xstar = np.asarray(xstar)
            ystar = np.asarray(ystar)

    w_Y, w_Y_star = _sample_weights(n, m, dp_alpha)
    return _minimize_weighted_loss(
        Y, ystar, w_Y, w_Y_star, loss, optimizer_kwargs, X, xstar, theta0
    )


def _minimize_weighted_loss(
    y,
    y_star,
    w_y,
    w_y_star,
    loss,
    optimizer_kwargs,
    x=None,
    x_star=None,
    theta0=None,
):
    """
    Internal function to minimize the weighted loss criterion.

    Args:
        y (ndarray): Labeled data of shape (n,) or (n, d).
        y_star (ndarray): Unlabeled data of shape (m,) or (m, d).
        w_y (ndarray): Weights for labeled data of shape (n,).
        w_y_star (ndarray): Weights for unlabeled data of shape (m,).
        loss (Callable): Loss function with signature loss(y, theta) -> ndarray.
        optimizer_kwargs (dict): Arguments for scipy.optimize.minimize.
        x (ndarray, optional): Covariates for labeled data of shape (n,) or (n, p).
        x_star (ndarray, optional): Covariates for unlabeled data of shape (m,) or (m, p).
        theta0 (ndarray): Initial guess for optimization.

    Returns:
        ndarray: Optimal parameter value.
    """

    def criterion(theta):

        # call user-provided loss function with appropriate signature
        if x is None:
            loss_y = loss(y, theta)
            loss_y_star = loss(y_star, theta)
        else:
            loss_y = loss(y, theta, x)
            loss_y_star = loss(y_star, theta, x_star)

        weighted_loss_y = w_y * loss_y
        weighted_loss_y_star = w_y_star * loss_y_star

        return np.sum(weighted_loss_y) + np.sum(weighted_loss_y_star)

    result = minimize(criterion, theta0, **optimizer_kwargs)

    if not result.success:
        warnings.warn(
            f"Optimization procedure did not succeed: {result.message}"
        )

    return result.x


def calibrate_dp_alpha_empirical_coverage_estimate(
    Y,
    loss,
    theta0,
    target_coverage=0.9,
    Yhat_unlabeled=None,
    m=None,
    n_jobs=1,
    verbose=False,
    optimizer_kwargs=None,
    X=None,
    X_unlabeled=None,
    calibration_kwargs=None,
    base_measure=None,
    coordinate=None,
):
    """
    Calibrate dp_alpha via empirical coverage using bootstrap.

    This function tests multiple alpha values and selects the largest alpha
    that achieves coverage within tolerance of the target coverage.

    Args:
        Y (ndarray): Gold-standard labels of shape (n,) or (n, d).
        loss (Callable): Loss function with signature loss(y, theta) -> ndarray.
        theta0 (ndarray): Initial guess for optimization. Must be provided.
        target_coverage (float): Target coverage level for credible intervals. Defaults to 0.9.
        Yhat_unlabeled (ndarray, optional): AI predictions on unlabeled data.
        m (int, optional): Truncation size for F_AI.
        n_jobs (int): Number of parallel jobs. Use -1 for all CPUs. Defaults to 1.
        verbose (bool): Whether to show progress. Defaults to False.
        optimizer_kwargs (dict, optional): Arguments for scipy.optimize.minimize.
        X (ndarray, optional): Covariates for labeled data.
        X_unlabeled (ndarray, optional): Covariates for unlabeled data.
        calibration_kwargs (dict, optional): Calibration parameters with keys:
            - num_bootstrap_samples: Number of bootstrap samples. Defaults to 20.
            - num_posterior_samples: Number of posterior samples per bootstrap. Defaults to 200.
            - tolerance: Acceptable deviation from target coverage. Defaults to 0.02.
            - alpha_grid: Grid of alpha values to test. Defaults to n-scaled grid.
        base_measure (Callable, optional): Custom base measure for DP prior.
        coordinate (int): if theta is a vector, specify which coordinate of theta to empirical validate coverage

    Returns:
        float: Calibrated dp_alpha value.
    """

    if calibration_kwargs is None:
        calibration_kwargs = {}

    num_bootstrap_samples = calibration_kwargs.get(
        "num_bootstrap_samples", 100
    )
    num_posterior_samples = calibration_kwargs.get(
        "num_posterior_samples", 100
    )
    tolerance = calibration_kwargs.get("tolerance", 0.02)

    n = len(Y)
    default_grid = 0.01 * n * (2 ** np.arange(10))
    alpha_grid = calibration_kwargs.get("alpha_grid", default_grid)

    (
        Y,
        Yhat_unlabeled,
        X,
        X_unlabeled,
        loss,
        _,
        num_posterior_samples,
        m,
        n_jobs,
        optimizer_kwargs,
        theta0,
    ) = _validate_args(
        Y,
        Yhat_unlabeled,
        X,
        X_unlabeled,
        loss,
        1.0,
        num_posterior_samples,
        m,
        n_jobs,
        optimizer_kwargs,
        base_measure,
        theta0,
    )

    _validate_loss(loss, has_covariates=(X is not None))

    theta0 = np.asarray(theta0)
    if theta0.ndim > 0:
        if coordinate is None:
            raise ValueError(
                "coordinate must be specified when theta0 is not scalar. "
                f"theta0 has shape {theta0.shape}, please specify which coordinate to validate coverage for."
            )
        if (
            not isinstance(coordinate, int)
            or coordinate < 0
            or coordinate >= len(theta0)
        ):
            raise ValueError(
                f"coordinate must be a valid index for theta0. "
                f"Got coordinate={coordinate} but theta0 has length {len(theta0)}"
            )

    if verbose:
        tqdm.write(
            f"Calibrating dp_alpha using {len(alpha_grid)} candidates: {alpha_grid}"
        )

    best_alpha = None
    alpha_pbar = tqdm(
        alpha_grid, desc="Testing alphas", position=0, disable=not verbose
    )

    coverage_results = {}

    for alpha in alpha_pbar:
        coverage_count = 0

        alpha_pbar.set_description(f"Testing alpha={alpha:.2f}")

        bootstrap_pbar = tqdm(
            range(num_bootstrap_samples),
            desc=f"Bootstrap (alpha={alpha:.2f})",
            position=1,
            leave=False,
            disable=not verbose,
        )

        for b in bootstrap_pbar:
            # bootstrap sample from Y (and X if provided)
            boot_idx = np.random.choice(n, size=n, replace=True)
            Y_boot = Y[boot_idx]
            X_boot = X[boot_idx] if X is not None else None

            # compute "true" parameter on bootstrap sample
            def bootstrap_loss(theta):
                if X_boot is None:
                    return np.mean(loss(Y_boot, theta))
                else:
                    return np.mean(loss(Y_boot, theta, X_boot))

            result = minimize(bootstrap_loss, theta0, **optimizer_kwargs)
            theta_boot = result.x

            posterior_samples = sample_ai_posterior(
                Y_boot,
                loss,
                alpha,
                theta0,
                Yhat_unlabeled=Yhat_unlabeled,
                n_samples=num_posterior_samples,
                m=m,
                n_jobs=n_jobs,
                verbose=False,
                optimizer_kwargs=optimizer_kwargs,
                X=X,
                X_unlabeled=X_unlabeled,
                base_measure=base_measure,
            )

            lower_q = (1 - target_coverage) / 2 * 100
            upper_q = (1 + target_coverage) / 2 * 100

            # extract the relevant coordinate if theta is vector
            if theta0.ndim > 0:
                posterior_samples_coord = posterior_samples[:, coordinate]
                ci_lower = np.percentile(posterior_samples_coord, lower_q)
                ci_upper = np.percentile(posterior_samples_coord, upper_q)
                theta_boot_coord = theta_boot[coordinate]
            else:
                ci_lower = np.percentile(posterior_samples, lower_q)
                ci_upper = np.percentile(posterior_samples, upper_q)
                theta_boot_coord = theta_boot

            if ci_lower <= theta_boot_coord <= ci_upper:
                coverage_count += 1

        bootstrap_pbar.close()

        empirical_coverage = coverage_count / num_bootstrap_samples

        coverage_results[alpha] = empirical_coverage

        if empirical_coverage >= target_coverage - tolerance:
            best_alpha = alpha

    alpha_pbar.close()

    if verbose:
        print("Coverage results (dp_alpha: estimated coverage):\n")
        pprint(coverage_results)

    if best_alpha is None:
        if verbose:
            tqdm.write(
                "Warning: No alpha achieved target coverage. Using smallest alpha."
            )
        best_alpha = alpha_grid[0]

    if verbose:
        tqdm.write(f"\nSelected alpha={best_alpha:.2f}")

    return best_alpha


# Example losses


def squared_loss(y, theta):
    """Squared loss for mean estimation"""
    return (y - theta) ** 2


def absolute_loss(y, theta):
    """Absolute loss for median estimation"""
    return np.abs(y - theta)


def quantile_loss(y, theta, tau=0.5):
    """Check loss for quantile estimation"""
    diff = y - theta
    return diff * (tau - (diff < 0).astype(float))
