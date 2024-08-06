import numpy as np

"""
    MEAN POWER CALCULATION

"""

def ppi_mean_power(
        Y,
        Yhat,
        Yhat_unlabeled,
        cost_Y,
        cost_Yhat,
        budget=None,
        se_tol=None,
        lam=None,
        coord=None,
        w=None,
        w_unlabeled=None,
        lam_optim_mode="overall"       
):
    """
    """
