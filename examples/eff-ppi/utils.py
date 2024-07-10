import os
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
import pdb


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
