.. ppi_py documentation master file, created by
   @aangelopoulos on Sat Sep 30 23:23:04 2023.

====================================
Welcome to the ppi_py documentation!
====================================

.. raw:: html

   <p align="center">
       <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2301.09633" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
       <a style="text-decoration:none !important;" href="https://pypi.org/project/ppi-python/" alt="package management"> <img src="https://img.shields.io/badge/pip-package-blue" /></a>
       <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-750014" /></a>
       <a style="text-decoration:none !important;" href="https://github.com/aangelopoulos/ppi_py" alt="License"><img src="https://img.shields.io/badge/GitHub-repo-black" /></a>
       <a style="text-decoration:none !important;" href="http://hits.dwyl.com/aangelopoulos/ppi_py" alt="hits"><img src="https://hits.dwyl.com/aangelopoulos/ppi_py.svg?style=flat-square" /></a>
   </p>

Prediction-powered inference (PPI) is a framework for statistically rigorous scientific discovery using machine learning.
Given a small amount of data with gold-standard labels and a large amount of unlabeled data, prediction-powered inference allows for the estimation of population parameters, such as the mean outcome, median outcome, linear and logistic regression coefficients.
Prediction-powered inference can be used both to produce better point estimates of these quantities as well as tighter confidence intervals and more powerful p-values.
The methods work both in the i.i.d. setting and for certain classes of distribution shifts.

This package is actively maintained, and contributions from the community are welcome.
To install the package, run the following code!

.. code-block:: python

   pip install ppi-python

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   README
   ppi
   baselines

Indices and tables:

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
