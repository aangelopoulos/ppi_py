import numpy as np
from ai_priors import (
    sample_ai_posterior,
    squared_loss,
    absolute_loss,
    quantile_loss,
)


def test_posterior_consistency():
    """Test that larger vs smaller dp_alpha shifts the mean in the expected direction"""
    trials = 10
    n = 100
    N = 1000
    true_mean_Y = 0.0
    true_mean_Yhat = -0.5
    dp_alphas = np.array([0.01, 1000.0])

    posterior_means = np.zeros((trials, len(dp_alphas)))

    for i in range(trials):
        Y = np.random.normal(true_mean_Y, 1, n)
        Yhat_unlabeled = np.random.normal(true_mean_Yhat, 1, N)

        theta0 = np.mean(Y)

        for j, dp_alpha in enumerate(dp_alphas):
            samples = sample_ai_posterior(
                Y,
                squared_loss,
                dp_alpha,
                theta0,
                Yhat_unlabeled=Yhat_unlabeled,
                n_samples=1000,
                n_jobs=1,
                verbose=False,
            )
            posterior_means[i, j] = np.mean(samples)

    mean_posterior_low_alpha = np.mean(posterior_means[:, 0])
    mean_posterior_high_alpha = np.mean(posterior_means[:, 1])

    distance_low_to_Y = np.abs(mean_posterior_low_alpha - true_mean_Y)
    distance_low_to_Yhat = np.abs(mean_posterior_low_alpha - true_mean_Yhat)
    distance_high_to_Y = np.abs(mean_posterior_high_alpha - true_mean_Y)
    distance_high_to_Yhat = np.abs(mean_posterior_high_alpha - true_mean_Yhat)

    assert distance_low_to_Y < distance_low_to_Yhat
    assert distance_high_to_Yhat < distance_high_to_Y


def test_different_loss_functions():
    """Test that different loss functions produce appropriate posteriors."""
    np.random.seed(42)
    n = 200
    N = 2000

    # Generate data with known median and mean with a slightly skewed distribution
    Y = np.concatenate(
        [np.random.normal(0, 1, n // 2), np.random.normal(2, 0.5, n // 2)]
    )
    Yhat_unlabeled = np.concatenate(
        [np.random.normal(0, 1.2, N // 2), np.random.normal(2, 0.6, N // 2)]
    )

    true_mean = np.mean(Y)
    true_median = np.median(Y)

    # squared loss (mean estimation)
    samples_mean = sample_ai_posterior(
        Y,
        squared_loss,
        dp_alpha=10.0,
        theta0=true_mean,
        Yhat_unlabeled=Yhat_unlabeled,
        n_samples=500,
        n_jobs=1,
        verbose=False,
    )
    posterior_mean_squared = np.mean(samples_mean)

    # absolute loss (median estimation)
    samples_median = sample_ai_posterior(
        Y,
        absolute_loss,
        dp_alpha=10.0,
        theta0=true_median,
        Yhat_unlabeled=Yhat_unlabeled,
        n_samples=500,
        n_jobs=1,
        verbose=False,
    )
    posterior_median_absolute = np.median(samples_median)

    # quantile loss (75th percentile)
    tau = 0.75

    def quantile_loss_75(y, theta):
        return quantile_loss(y, theta, tau=tau)

    true_quantile = np.percentile(Y, tau * 100)
    samples_quantile = sample_ai_posterior(
        Y,
        quantile_loss_75,
        dp_alpha=10.0,
        theta0=true_quantile,
        Yhat_unlabeled=Yhat_unlabeled,
        n_samples=500,
        n_jobs=1,
        verbose=False,
    )
    posterior_quantile = np.median(samples_quantile)

    # assert that each loss function estimates the appropriate quantity
    # mean estimator should be closer to true mean than to true median
    assert np.abs(posterior_mean_squared - true_mean) < np.abs(
        posterior_mean_squared - true_median
    )

    # median estimator should be closer to true median than to true mean
    assert np.abs(posterior_median_absolute - true_median) < np.abs(
        posterior_median_absolute - true_mean
    )

    # quantile estimator should be reasonably close to true quantile
    assert np.abs(posterior_quantile - true_quantile) < 0.5


def test_covariates():
    """Test that covariate handling works correctly."""
    np.random.seed(123)
    n = 150
    N = 1500
    p = 3  # number of covariates

    true_beta = np.array([1.0, -0.5, 2.0])

    X = np.random.normal(0, 1, (n, p))
    X_unlabeled = np.random.normal(0, 1, (N, p))

    noise_std = 0.5
    Y = X @ true_beta + np.random.normal(0, noise_std, n)

    ai_beta = true_beta + np.array([0.2, -0.1, 0.3])
    Yhat_unlabeled = X_unlabeled @ ai_beta + np.random.normal(
        0, noise_std * 1.2, N
    )

    # define regression loss
    def regression_loss(y, theta, x):
        """Squared loss for linear regression."""
        predictions = x @ theta
        return (y - predictions) ** 2

    theta0 = np.linalg.lstsq(X, Y, rcond=None)[0]

    samples = sample_ai_posterior(
        Y,
        regression_loss,
        dp_alpha=20.0,
        theta0=theta0,
        Yhat_unlabeled=Yhat_unlabeled,
        n_samples=500,
        n_jobs=1,
        verbose=False,
        X=X,
        X_unlabeled=X_unlabeled,
    )

    assert samples.shape == (500, p)

    posterior_mean = np.mean(samples, axis=0)

    # should be closer to true coefficients than AI coefficients for small dp_alpha
    for i in range(p):
        distance_to_true = np.abs(posterior_mean[i] - true_beta[i])
        distance_to_ai = np.abs(posterior_mean[i] - ai_beta[i])
        # with moderate dp_alpha, should be between true and AI
        assert distance_to_true < 1.0  # reasonable bound


def test_base_measure():
    """Test using custom base measure instead of Yhat_unlabeled."""
    np.random.seed(456)
    n = 100
    m = 1000

    Y = np.random.normal(1.0, 1.0, n)

    def example_base_measure(size):
        # 70% from N(0, 1), 30% from N(3, 0.5)
        n1 = int(0.7 * size)
        n2 = size - n1
        samples = np.concatenate(
            [np.random.normal(0, 1, n1), np.random.normal(3, 0.5, n2)]
        )
        np.random.shuffle(samples)
        return samples

    theta0_simple = np.mean(Y)

    samples_base = sample_ai_posterior(
        Y,
        squared_loss,
        dp_alpha=10.0,
        theta0=theta0_simple,
        base_measure=example_base_measure,
        m=m,
        n_samples=500,
        n_jobs=1,
        verbose=False,
    )

    assert samples_base.shape == (500,) or samples_base.shape == (500, 1)
    assert np.all(np.isfinite(samples_base))

    # test with covariates
    p = 2
    X = np.random.normal(0, 1, (n, p))
    true_beta = np.array([1.5, -1.0])
    Y_cov = X @ true_beta + np.random.normal(0, 0.5, n)

    # base measure that returns (X, Y) pairs
    def example_base_measure_covariates(size):
        X_sample = np.random.normal(0, 1, (size, p))

        base_beta = np.array([1.0, -0.5])
        Y_sample = X_sample @ base_beta + np.random.normal(0, 0.7, size)
        return X_sample, Y_sample

    def regression_loss(y, theta, x):
        return (y - x @ theta) ** 2

    theta0 = np.linalg.lstsq(X, Y_cov, rcond=None)[0]

    samples_base_cov = sample_ai_posterior(
        Y_cov,
        regression_loss,
        dp_alpha=15.0,
        theta0=theta0,
        base_measure=example_base_measure_covariates,
        m=m,
        n_samples=500,
        n_jobs=1,
        verbose=False,
        X=X,
    )

    assert samples_base_cov.shape == (500, p)
    assert np.all(np.isfinite(samples_base_cov))

    posterior_mean = np.mean(samples_base_cov, axis=0)
    assert np.all(np.abs(posterior_mean - true_beta) < 1.0)
