Getting Started
===============

In order to install the package, run 

.. code-block:: python

   pip install ppi-python

Warmup: estimating the mean
---------------------------

To test your installation, you can try running the prediction-powered mean estimation algorithm on the ``galaxies`` dataset.
The gold-standard labels and model predictions from the dataset will be downloaded into a folder called ``./data/``.
The labels, :math:`Y`, are binary indicators of whether or not the galaxy is a spiral galaxy.
The model predictions, :math:`\hat{Y}`, are the model's estimated probability of whether the galaxy image has spiral arms.
The inference target is :math:`\theta^* = \mathbb{E}[Y]`, the fraction of spiral galaxies.
You will produce a confidence interval, :math:`\mathcal{C}^{\mathrm{PP}}_\alpha`, which contains :math:`\theta^*` with probability :math:`1-\alpha=0.9`, i.e.,

.. math::

       \mathbb{P}\left( \theta^* \in \mathcal{C}^{\mathrm{PP}}_\alpha\right) \geq 0.9.

The code for this is below. It can be copy-pasted directly into the Python REPL.

.. code-block:: python

   # Imports
   import numpy as np
   from ppi_py import ppi_mean_ci
   from ppi_py.datasets import load_dataset
   np.random.seed(0) # For reproducibility's sake
   # Download and load dataset
   data = load_dataset('./data/', "galaxies")
   Y_total = data["Y"]; Yhat_total = data["Yhat"]
   # Set up the inference problem
   alpha = 0.1 # Error rate
   n = 1000 # Number of labeled data points
   rand_idx = np.random.permutation(Y_total.shape[0])
   Yhat = Yhat_total[rand_idx[:n]]
   Y = Y_total[rand_idx[:n]]
   Yhat_unlabeled = Yhat_total[n:]
   # Produce the prediction-powered confidence interval
   ppi_ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alpha)
   # Print the results
   print(f"theta={Y_total.mean():.3f}, CPP={ppi_ci}")

The expected results look as below :math:`^*`: 

.. code-block::

   theta=0.259, CPP=(0.2322466630315982, 0.2626038799812829)

(:math:`^*` these results were produced with ``numpy=1.26.0``\ , and may differ slightly due to randomness in other environments.)

If you have reached this stage, congratulations! You have constructed a prediction-powered confidence interval.
See the documentation for more usages of prediction-powered inference.

Examples
========

The package somes with a suite of examples on real data:


* Proteomic Analysis with AlphaFold (`alphafold.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/alphafold.ipynb>`_)
* Galaxy Classification with Computer Vision (`galaxies.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/galaxies.ipynb>`_)
* Gene Expression Analysis with Transformers (`gene_expression.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/gene_expression.ipynb>`_)
* Deforestation Analysis with Computer Vision (`forest.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/forest.ipynb>`_)
* Health Insurance Analysis with Boosting Trees (`census_healthcare.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_healthcare.ipynb>`_)
* Income Analysis with Boosting Trees (covariate shift) (`census_income_covshift.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_income_covshift.ipynb>`_)
* Plankton Counting with Computer Vision (label shift) (`plankton.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/plankton.ipynb>`_)
* Ballot Counting with Computer Vision (`ballots.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/ballots.ipynb>`_)
* Income Analysis with Boosting Trees (`census_income.ipynb <https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_income.ipynb>`_)

Usage and Documentation
=======================

There is a common template that all PPI confidence intervals follow.

.. code-block:: python

   ppi_[ESTIMAND]_ci(X, Y, Yhat, X_unlabeled, Yhat_unlabeled, alpha=0.1)

You can replace ``[ESTIMAND]`` with the estimand of your choice. For certain estimands, not all the arguments are required, and in this case, they are omitted. For example, in the case of mean estimation, the function signature is:

.. code-block:: python

   ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=0.1)

All the prediction-powered point estimates and confidence intervals implemented so far can be imported by running ``from ppi_py import ppi_[ESTIMAND]_pointestimate, ppi_[ESTIMAND]_ci``. For the case of the mean, one can also import the p-value as ``from ppi import ppi_mean_pval``.

Full API documentation can be found by following the links on the left-hand sidebar of this page.

Papers
======

The repository currently implements the methods developed in the following papers:

`Prediction-Powered Inference <https://arxiv.org/abs/2106.06487>`_

`PPI++: Efficient Prediction-Powered Inference <https://arxiv.org/abs/2311.01453>`_
