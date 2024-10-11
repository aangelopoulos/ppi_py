import os
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import pandas as pd
from ppi_py import ppi_ols_ci, classical_ols_ci, ppi_ols_pointestimate


def plot_interval(
    ax,
    lower,
    upper,
    height,
    color_face,
    color_stroke,
    linewidth=5,
    linewidth_modifier=1.1,
    offset=0.25,
    label=None,
):
    label = label if label is None else " " + label
    ax.plot(
        [lower, upper],
        [height, height],
        linewidth=linewidth,
        color=color_face,
        path_effects=[
            pe.Stroke(
                linewidth=linewidth * linewidth_modifier,
                offset=(-offset, 0),
                foreground=color_stroke,
            ),
            pe.Stroke(
                linewidth=linewidth * linewidth_modifier,
                offset=(offset, 0),
                foreground=color_stroke,
            ),
            pe.Normal(),
        ],
        label=label,
        solid_capstyle="butt",
    )


def make_plots(
    df,
    plot_savename,
    n_idx=-1,
    true_theta=None,
    true_label=r"$\theta^*$",
    intervals_xlabel="x",
    plot_classical=True,
    ppi_facecolor="#DAF3DA",
    ppi_strokecolor="#71D26F",
    classical_facecolor="#EEEDED",
    classical_strokecolor="#BFB9B9",
    imputation_facecolor="#FFEACC",
    imputation_strokecolor="#FFCD82",
    empty_panel=True,
):
    # Make plot
    num_intervals = 5
    num_scatter = 3
    ns = df.n.unique()
    ns = ns[~np.isnan(ns)].astype(int)
    n = ns[n_idx]
    num_trials = len(df[(df.n == n) * (df.method == "PPI")])

    ppi_intervals = df[(df.n == n) & (df.method == "PPI")].sample(
        n=num_intervals, replace=False
    )
    if plot_classical:
        classical_intervals = df[
            (df.n == n) & (df.method == "Classical")
        ].sample(n=num_intervals, replace=False)
    imputation_interval = df[df.method == "Imputation"]

    xlim = [None, None]
    ylim = [0, 1.15]

    if empty_panel:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    sns.set_theme(style="white", font_scale=1, font="DejaVu Sans")
    if true_theta is not None:
        axs[-2].axvline(
            true_theta,
            ymin=0.0,
            ymax=1,
            linestyle="dotted",
            linewidth=3,
            label=true_label,
            color="#F7AE7C",
        )

    for i in range(num_intervals):
        ppi_interval = ppi_intervals.iloc[i]
        if plot_classical:
            classical_interval = classical_intervals.iloc[i]

        if i == 0:
            plot_interval(
                axs[-2],
                ppi_interval.lower,
                ppi_interval.upper,
                0.7,
                ppi_facecolor,
                ppi_strokecolor,
                label="prediction-powered",
            )
            if plot_classical:
                plot_interval(
                    axs[-2],
                    classical_interval.lower,
                    classical_interval.upper,
                    0.25,
                    classical_facecolor,
                    classical_strokecolor,
                    label="classical",
                )
            plot_interval(
                axs[-2],
                imputation_interval.lower,
                imputation_interval.upper,
                0.1,
                imputation_facecolor,
                imputation_strokecolor,
                label="imputation",
            )
        else:
            lighten_factor = 0.8 / np.sqrt(num_intervals - i)
            yshift = (num_intervals - i) * 0.07
            plot_interval(
                axs[-2],
                ppi_interval.lower,
                ppi_interval.upper,
                0.7 + yshift,
                lighten_color(ppi_facecolor, lighten_factor),
                lighten_color(ppi_strokecolor, lighten_factor),
            )
            if plot_classical:
                plot_interval(
                    axs[-2],
                    classical_interval.lower,
                    classical_interval.upper,
                    0.25 + yshift,
                    lighten_color(classical_facecolor, lighten_factor),
                    lighten_color(classical_strokecolor, lighten_factor),
                )

    axs[-2].set_xlabel(intervals_xlabel, labelpad=10)
    axs[-2].set_yticks([])
    axs[-2].set_yticklabels([])
    axs[-2].set_ylim(ylim)
    axs[-2].set_xlim(xlim)

    sns.despine(ax=axs[-2], top=True, right=True, left=True)

    ppi_widths = [
        df[(df.n == _n) & (df.method == "PPI")].width.mean() for _n in ns
    ]
    if plot_classical:
        classical_widths = [
            df[(df.n == _n) & (df.method == "Classical")].width.mean()
            for _n in ns
        ]

    axs[-1].plot(
        ns,
        ppi_widths,
        label="prediction-powered",
        color=ppi_strokecolor,
        linewidth=3,
    )
    if plot_classical:
        axs[-1].plot(
            ns,
            classical_widths,
            label="classical",
            color=classical_strokecolor,
            linewidth=3,
        )

    n_list = []
    ppi_width_list = []
    if plot_classical:
        classical_width_list = []
    for _n in ns:
        trials = np.random.choice(
            num_trials, size=num_scatter, replace=False
        ).astype(int)
        ppi_width_list += df[
            (df.n == _n) & (df.method == "PPI") & df.trial.isin(trials)
        ].width.to_list()
        if plot_classical:
            classical_width_list += df[
                (df.n == _n)
                & (df.method == "Classical")
                & df.trial.isin(trials)
            ].width.to_list()
        n_list += [_n] * num_scatter

    axs[-1].scatter(n_list, ppi_width_list, color=ppi_strokecolor, alpha=0.5)

    if plot_classical:
        axs[-1].scatter(
            n_list,
            classical_width_list,
            color=classical_strokecolor,
            alpha=0.5,
        )

    axs[-1].locator_params(axis="y", tight=None, nbins=6)
    axs[-1].set_ylabel("width")
    axs[-1].set_xlabel("n", labelpad=10)
    sns.despine(ax=axs[-1], top=True, right=True)

    if empty_panel:
        sns.despine(ax=axs[0], top=True, right=True, left=True, bottom=True)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

    plt.tight_layout()
    os.makedirs("/".join(plot_savename.split("/")[:-1]), exist_ok=True)
    plt.savefig(plot_savename)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



# function to calculate amce with ppi for moral machine experiment
def compute_amce_ppi(n_data, N_data, x, y, alpha=0.05):

    # functions to calculate weights for conjoint experiment
    def CalcTheoreticalInt(r):
        # this function is applied to each row (r)
        if r["Intervention"]==0:
            if r["Barrier"]==0:
                if r["PedPed"]==1: p = 0.48
                else: p = 0.32
                
                if r["CrossingSignal"]==0:   p = p * 0.48
                elif r["CrossingSignal"]==1: p = p * 0.2
                else: p = p * 0.32
            else: p = 0.2

        else: 
            if r["Barrier"]==0:
                if r["PedPed"]==1: 
                    p = 0.48
                    if r["CrossingSignal"]==0: p = p * 0.48
                    elif r["CrossingSignal"]==1: p = p * 0.32
                    else: p = p * 0.2
                else: 
                    p = 0.2
                    if r["CrossingSignal"]==0: p = p * 0.48
                    elif r["CrossingSignal"]==1: p = p * 0.2
                    else: p = p * 0.32
            else: p = 0.32  
        
        return(p)  
            
    def calcWeightsTheoretical(profiles):
        
        p = profiles.apply(CalcTheoreticalInt, axis=1)

        weight = 1/p 

        return(weight) 


    # specify regression for swerve or stay in lane
    if x=="Intervention":
        
        # calculate weights
        n_data.loc[:,"weights"] = calcWeightsTheoretical(n_data)
        N_data.loc[:,"weights"] = calcWeightsTheoretical(N_data)
    
        # drop rows with missing values on dependent variable
        n_dd = n_data.dropna(subset=y)
        N_dd = N_data.dropna(subset=y)

        # if X=1 characters die if AV serves, if X=0 characters if AV stays
        n_X = n_dd["Intervention"]               
        N_X = N_dd["Intervention"]

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights



    # specify regression for relationship to vehicle
    if x=="Barrier":

        # consider only dilemmas without legality and only pedestrians vs passengers
        n_data_sub = n_data.loc[(n_data["CrossingSignal"]==0) & (n_data["PedPed"]==0), :].copy()
        N_data_sub = N_data.loc[(N_data["CrossingSignal"]==0) & (N_data["PedPed"]==0), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)
        
        # if X=1 passengers die and if X=0 pedestrians die
        n_X = n_dd["Barrier"]
        N_X = N_dd["Barrier"]

        # recode to estimate the preference for pedestrians over passengers 
        n_X = 1 - n_X
        N_X = 1 - N_X

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights

    

    # specify regression for legality
    if x=="CrossingSignal": 
        
        # consider dilemmas with legality and only pedestrians vs pedestrians
        n_data_sub = n_data.loc[(n_data["CrossingSignal"]!=0) & (n_data["PedPed"]==1), :].copy()
        N_data_sub = N_data.loc[(N_data["CrossingSignal"]!=0) & (N_data["PedPed"]==1), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)

        # if X=1 pedestrians cross on a green light, if X=2 pedestrians cross on a red light 
        n_X = n_dd["CrossingSignal"]
        N_X = N_dd["CrossingSignal"]

        # create dummy variable to estimate preference for pedestrians that cross legally (1) vs legally (0)
        n_X = 2 - n_X 
        N_X = 2 - N_X 

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights
    


    # Specify regressions for the remaining six attributes
    if x=="Utilitarian":
        
        # consider dilemmas that compare 'More' versus 'Less' characters
        n_data_sub = n_data.loc[(n_data["ScenarioType"]=="Utilitarian") & (n_data["ScenarioTypeStrict"]=="Utilitarian"), :].copy()
        N_data_sub = N_data.loc[(N_data["ScenarioType"]=="Utilitarian") & (N_data["ScenarioTypeStrict"]=="Utilitarian"), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)
        
        # rename column to extract coefficient from result
        n_dd = n_dd.rename(columns = {'AttributeLevel': 'Utilitarian'})
        N_dd = N_dd.rename(columns = {'AttributeLevel': 'Utilitarian'})

        # create dummy variable to estimate the preference for sparing more characters
        n_X = (n_dd.loc[:,"Utilitarian"]=="More").astype(int)
        N_X = (N_dd.loc[:,"Utilitarian"]=="More").astype(int)

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights



    if x=="Species":
        
        # consider dilemmas that compare humans versus animals 
        n_data_sub = n_data.loc[(n_data["ScenarioType"]=="Species") & (n_data["ScenarioTypeStrict"]=="Species"), :].copy()
        N_data_sub = N_data.loc[(N_data["ScenarioType"]=="Species") & (N_data["ScenarioTypeStrict"]=="Species"), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)

        # rename column to extract coefficient from result
        n_dd = n_dd.rename(columns = {'AttributeLevel': 'Species'})
        N_dd = N_dd.rename(columns = {'AttributeLevel': 'Species'})

        # create dummy variable to estimate the preference for sparing humans
        n_X = (n_dd.loc[:,"Species"]=="Hoomans").astype(int)
        N_X = (N_dd.loc[:,"Species"]=="Hoomans").astype(int)

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights

    

    if x=="Gender":
        
        # consider dilemmas that compare women versus men
        n_data_sub = n_data.loc[(n_data["ScenarioType"]=="Gender") & (n_data["ScenarioTypeStrict"]=="Gender"), :].copy()
        N_data_sub = N_data.loc[(N_data["ScenarioType"]=="Gender") & (N_data["ScenarioTypeStrict"]=="Gender"), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)

        # rename column to extract coefficient from result
        n_dd = n_dd.rename(columns = {'AttributeLevel': 'Gender'})
        N_dd = N_dd.rename(columns = {'AttributeLevel': 'Gender'})

        # create dummy variable to estimate the preference for sparing women
        n_X = (n_dd.loc[:,"Gender"]=="Female").astype(int)
        N_X = (N_dd.loc[:,"Gender"]=="Female").astype(int)

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights



    if x=="Fitness":
        
        # consider dilemmas that compare fit characters versus those that are not
        n_data_sub = n_data.loc[(n_data["ScenarioType"]=="Fitness") & (n_data["ScenarioTypeStrict"]=="Fitness"), :].copy()
        N_data_sub = N_data.loc[(N_data["ScenarioType"]=="Fitness") & (N_data["ScenarioTypeStrict"]=="Fitness"), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)

        # rename column to extract coefficient from result
        n_dd = n_dd.rename(columns = {'AttributeLevel': 'Fitness'})
        N_dd = N_dd.rename(columns = {'AttributeLevel': 'Fitness'})

        # create dummy variable to estimate the preference for sparing fit characters
        n_X = (n_dd.loc[:,"Fitness"]=="Fit").astype(int)
        N_X = (N_dd.loc[:,"Fitness"]=="Fit").astype(int)

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights



    if x=="Age":
        
        # consider dilemmas that compare younger versus older characters
        n_data_sub = n_data.loc[(n_data["ScenarioType"]=="Age") & (n_data["ScenarioTypeStrict"]=="Age"), :].copy()
        N_data_sub = N_data.loc[(N_data["ScenarioType"]=="Age") & (N_data["ScenarioTypeStrict"]=="Age"), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)

        # rename column to extract coefficient from result
        n_dd = n_dd.rename(columns = {'AttributeLevel': 'Age'})
        N_dd = N_dd.rename(columns = {'AttributeLevel': 'Age'})

        # create dummy variable to estimate the preference for sparing younger characters
        n_X = (n_dd.loc[:,"Age"]=="Young").astype(int)
        N_X = (N_dd.loc[:,"Age"]=="Young").astype(int)

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd["Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd[y].to_numpy()          # predicted outcomes
        n_weights = n_dd["weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd["weights"].to_numpy()    # define weights


    
    if x=="Social Status":
        
        # consider dilemmas that compare high status versus low status characters
        n_data_sub = n_data.loc[(n_data["ScenarioType"]=="Social Status") & (n_data["ScenarioTypeStrict"]=="Social Status"), :].copy()
        N_data_sub = N_data.loc[(N_data["ScenarioType"]=="Social Status") & (N_data["ScenarioTypeStrict"]=="Social Status"), :].copy()

        # calculate weights
        n_data_sub.loc[:,"weights"] = calcWeightsTheoretical(n_data_sub)
        N_data_sub.loc[:,"weights"] = calcWeightsTheoretical(N_data_sub)

        # drop rows with missing values on dependent variable
        n_dd = n_data_sub.dropna(subset=y)
        N_dd = N_data_sub.dropna(subset=y)

        # rename column to extract coefficient from result
        n_dd = n_dd.rename(columns = {'AttributeLevel': 'Social Status'})
        N_dd = N_dd.rename(columns = {'AttributeLevel': 'Social Status'})

        # create dummy variable to estimate the preference for sparing high status characters
        n_X = (n_dd.loc[:,"Social Status"]=="High").astype(int)
        N_X = (N_dd.loc[:,"Social Status"]=="High").astype(int)

        # add intercept
        n_X = np.column_stack((np.ones(n_X.shape[0]), n_X))
        N_X = np.column_stack((np.ones(N_X.shape[0]), N_X))

        # gold standard data
        n_Y_human   = n_dd.loc[:,"Saved"].to_numpy()    # observed outcomes
        n_Y_silicon = n_dd.loc[:,y].to_numpy()          # predicted outcomes
        n_weights = n_dd.loc[:,"weights"].to_numpy()    # define weights

        # unlabeled data
        N_Y_silicon = N_dd[y].to_numpy()          # predicted outcomes
        N_weights = N_dd.loc[:,"weights"].to_numpy()    # define weights


    # calculate point estimate
    pointest_ppi = ppi_ols_pointestimate(X=n_X, Y=n_Y_human, Yhat=n_Y_silicon, 
                                         X_unlabeled=N_X, Yhat_unlabeled=N_Y_silicon, 
                                         w=n_weights, w_unlabeled=N_weights)

    # calculate PPI confidence intervals
    lower_CI_ppi, upper_CI_ppi = ppi_ols_ci(X=n_X, Y=n_Y_human, Yhat=n_Y_silicon, 
                                            X_unlabeled=N_X, Yhat_unlabeled=N_Y_silicon, 
                                            w=n_weights, w_unlabeled=N_weights, alpha=alpha)
    
    # calculate OLS confidence intervals
    lower_CI_ols, upper_CI_ols = classical_ols_ci(X=n_X, Y=n_Y_human, w=n_weights, alpha=alpha)

    # create and return the output DataFrame
    output_df = pd.DataFrame({
        "y": y,                              
        "x": x,                              # Predictor variable (scenario attribute)
        "pointest_ppi": pointest_ppi[1],     # PPI point estimate
        "conf_low_ppi": lower_CI_ppi[1],     # The lower bound of the PPI confidence interval
        "conf_high_ppi": upper_CI_ppi[1],    # The upper bound of the PPI confidence interval
        "conf_low_ols": lower_CI_ols[1],     # The lower bound of the OLS confidence interval
        "conf_high_ols": upper_CI_ols[1],    # The upper bound of the OLS confidence interval
    },index=[0])
    
    return output_df 
