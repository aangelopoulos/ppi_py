import numpy as np
from ppi_py import *

def test_ppi_distribution_label_shift_ci_sorted():
    """Test ppi_distribution_label_shift_ci with various sorting scenarios."""
    n = 1000
    N = 10000
    num_classes = 10

    # Generate multi-class predictions (integer labels)
    Y = np.random.randint(0, num_classes, n)
    Yhat = np.random.randint(0, num_classes, n)
    Yhat_unlabeled = np.random.randint(0, num_classes, N)

    # Determine K and nu
    unique_Yhat = np.unique(Yhat)
    K = len(unique_Yhat)  # Should be equal to num_classes if all classes are present in Yhat

    # Set nu (example: vector of ones)
    nu = np.ones(K)

    # Test 1: Unsorted Yhat_unlabeled (implicit in the current implementation)
    ci_unsorted = ppi_distribution_label_shift_ci(Y, Yhat, Yhat_unlabeled, K, nu, alpha=0.1)

    # Test 2: Explicitly sorted Yhat_unlabeled (to demonstrate no dependence on order)
    uq, uq_counts = np.unique(Yhat_unlabeled, return_counts=True)
    sort_indices = np.argsort(uq)
    Yhat_unlabeled_sorted = Yhat_unlabeled[np.argsort(np.argsort(Yhat_unlabeled))] # Stable sort to mimic previous behavior.
    ci_sorted = ppi_distribution_label_shift_ci(Y, Yhat, Yhat_unlabeled_sorted, K, nu, alpha=0.1)


    # Test 3: Reverse sorted Yhat_unlabeled
    uq, uq_counts = np.unique(Yhat_unlabeled, return_counts=True)
    sort_indices = np.argsort(uq)[::-1]
    Yhat_unlabeled_reverse_sorted = Yhat_unlabeled[np.argsort(np.argsort(Yhat_unlabeled))[::-1]]
    ci_reverse_sorted = ppi_distribution_label_shift_ci(Y, Yhat, Yhat_unlabeled_reverse_sorted, K, nu, alpha=0.1)


    # The confidence intervals should be identical regardless of the sorting of Yhat_unlabeled
    np.testing.assert_allclose(ci_unsorted, ci_sorted)
    np.testing.assert_allclose(ci_unsorted, ci_reverse_sorted)