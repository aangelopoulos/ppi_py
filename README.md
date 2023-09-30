# ppi-py
## A Python package for scientific discovery using machine learning

Prediction-powered inference (PPI) is a framework for statistically rigorous scientific discovery using machine learning.
Given a small amount of data with gold-standard labels and a large amount of unlabeled data, prediction-powered inference allows for the estimation of population parameters, such as the mean outcome, median outcome, linear and logistic regression coefficients.
Prediction-powered inference can be used both to produce better point estimates of these quantities as well as tighter confidence intervals and more powerful p-values.
The methods work both in the i.i.d. setting and for certain classees of distribution shifts.

This package is actively maintained, and contributions from the community are welcome.

# Quickstart 
In order to install the package, run 
```python
python setup.py sdist bdist_wheel
pip install dist/ppi_py-0.1-py3-none-any.whl
```
This will build and install the most recent version of the package!

# Usage
Coming soon!

# Examples

The package somes with a suite of examples on real data:
- Proteomic Analysis with AlphaFold ([```alphafold.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/alphafold.ipynb))
- Galaxy Classification with Computer Vision ([```galaxies.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/galaxies.ipynb))
- Gene Expression Analysis with Transformers ([```gene_expression.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/gene_expression.ipynb))
- Deforestation Analysis with Computer Vision ([```forest.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/forest.ipynb))
- Health Insurance Analysis with Boosting Trees ([```census_healthcare.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_healthcare.ipynb))
- Income Analysis with Boosting Trees (covariate shift) ([```census_income_covshift.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_income_covshift.ipynb))
- Plankton Counting with Computer Vision (label shift) ([```plankton.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/plankton.ipynb))
- Ballot Counting with Computer Vision ([```ballots.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/ballots.ipynb))
- Income Analysis with Boosting Trees ([```census_income.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_income.ipynb))
 
# Documentation
Coming soon!

# Contributing
Coming soon!
