# Examples

This folder contains a suite of examples on real data:
- Proteomic Analysis with AlphaFold ([```alphafold.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/alphafold.ipynb))
- Galaxy Classification with Computer Vision ([```galaxies.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/galaxies.ipynb))
- Gene Expression Analysis with Transformers ([```gene_expression.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/gene_expression.ipynb))
- Deforestation Analysis with Computer Vision ([```forest.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/forest.ipynb))
- Health Insurance Analysis with Boosting Trees ([```census_healthcare.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_healthcare.ipynb))
- Income Analysis with Boosting Trees (covariate shift) ([```census_income_covshift.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_income_covshift.ipynb))
- Plankton Counting with Computer Vision (label shift) ([```plankton.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/plankton.ipynb))
- Ballot Counting with Computer Vision ([```ballots.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/ballots.ipynb))
- Income Analysis with Boosting Trees ([```census_income.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/census_income.ipynb))

Each notebook runs a simulation that forms a dataframe containing confidence intervals produced by different methods (PPI, classical, imputation), for different values of labeled data set size ```n``` and over different trials. Based on the computed dataframe, the notebook plots:
- five randomly chosen intervals for PPI and the classical method, and the imputed interval;
- the average interval width for PPI and the classical method, together with a scatterplot of the widths from the five random draws.

Each notebook also compares PPI and classical inference in terms of the number of labeled examples needed to reject a natural null hypothesis in the analyzed problem.

The notebook ([```tree_cover_ptd.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/tree_cover_ptd.ipynb)) shows how to use the Predict-Then-Debias (PTD) estimator from Kluger et al. (2025), 'Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling,' https://arxiv.org/abs/2501.18577.

Finally, there is a notebook that shows how to compute the optimal `n` and `N` given a cost constraint ([```power_analysis.ipynb```](https://github.com/aangelopoulos/ppi_py/blob/main/examples/power_analysis.ipynb)).