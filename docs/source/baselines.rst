
Functions for Baseline Inference
================================

Documentation for functions implementing baseline inference strategies can be found here. These are functions that either use only gold-standard data or use gold-standard + unlabeled data in a way that is not consistent with or part of the PPI framework.

.. autofunction:: ppi_py.classical_mean_ci
.. autofunction:: ppi_py.semisupervised_mean_ci
.. autofunction:: ppi_py.conformal_mean_ci
.. autofunction:: ppi_py.classical_quantile_ci
.. autofunction:: ppi_py.classical_ols_ci
.. autofunction:: ppi_py.classical_ols_covshift_ci
.. autofunction:: ppi_py.postprediction_ols_ci
.. autofunction:: ppi_py.logistic
.. autofunction:: ppi_py.classical_logistic_ci
