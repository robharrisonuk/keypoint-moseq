'''
Modified versions of analysis.py functions - less restrictive on project formats etc...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import math
from itertools import combinations
import pandas as pd
import seaborn as sns

import keypoint_moseq.analysis as analysis

# ----------------------------------------------------------

def visualize_transition_bigram_v2(group, trans_mats, syll_include, normalize="bigram", figsize=(12, 6)):
    """Visualize the transition matrices for each group.

    Parameters
    ----------
    group : list or np.ndarray
        the groups in the project
    trans_mats : list
        the list of transition matrices for each group
    normalize : str, optional
        the method to normalize the transition matrix, by default 'bigram'
    figsize : tuple, optional
        the figure size, by default (12,6)
    """

    syll_names = [f"{ix}" for ix in syll_include]

    # infer max_syllables
    max_syllables = trans_mats[0].shape[0]

    fig, ax = plt.subplots(1, len(group), figsize=figsize, sharex=False, sharey=True)
    title_map = dict(bigram="Bigram", columns="Incoming", rows="Outgoing")
    color_lim = max([x.max() for x in trans_mats])
    if len(group) == 1:
        axs = [ax]
    else:
        axs = ax.flat
    for i, g in enumerate(group):
        h = axs[i].imshow(
            trans_mats[i][:max_syllables, :max_syllables],
            cmap="cubehelix",
            vmax=color_lim,
        )
        if i == 0:
            axs[i].set_ylabel("Incoming syllable")
            plt.yticks(np.arange(len(syll_include)), syll_names)
        cb = fig.colorbar(h, ax=axs[i], fraction=0.046, pad=0.04)
        cb.set_label(f"{title_map[normalize]} transition probability")
        axs[i].set_xlabel("Outgoing syllable")
        axs[i].set_title(g)
        axs[i].set_xticks(np.arange(len(syll_include)), syll_names, rotation=90)

    plt.show()

# ----------------------------------------------------------

def plot_transition_graph_group_v2(
    groups,
    trans_mats,
    usages,
    syll_include,
    layout="circular",
    node_scaling=2000,
):
    """Plot the transition graph for each group.

    Parameters
    ----------
    groups : list
        the list of groups to plot
    trans_mats : list
        the list of transition matrices for each group
    usages : list
        the list of syllable usage for each group
    layout : str, optional
        the layout of the graph, by default 'circular'
    node_scaling : int, optional
        the scaling factor for the node size, by default 2000,
    """

    syll_names = [f"{ix}" for ix in syll_include]

    n_row = math.ceil(len(groups) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(20, 10 * n_row))
    ax = all_axes.flat

    for i in range(len(groups)):
        G = nx.from_numpy_array(trans_mats[i] * 100)
        widths = nx.get_edge_attributes(G, "weight")
        if layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        # get node list
        nodelist = G.nodes()
        # normalize the usage values
        sum_usages = sum(usages[i])
        normalized_usages = (
            np.array([u / sum_usages for u in usages[i]]) * node_scaling + 1000
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=normalized_usages,
            node_color="white",
            edgecolors="red",
            ax=ax[i],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=list(widths.values()),
            edge_color="black",
            ax=ax[i],
            alpha=0.6,
        )
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=dict(zip(nodelist, syll_names)),
            font_color="black",
            ax=ax[i],
        )
        ax[i].set_title(groups[i])
    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")

    plt.show()

# ----------------------------------------------------------

def plot_transition_graph_difference_v2(
    groups,
    trans_mats,
    usages,
    syll_include,
    layout="circular",
    node_scaling=3000
):
    """Plot the difference of transition graph between groups.

    Parameters
    ----------
    groups : list
        the list of groups to plot
    trans_mats : list
        the list of transition matrices for each group
    usages : list
        the list of syllable usage for each group
    layout : str, optional
        the layout of the graph, by default 'circular'
    node_scaling : int, optional
        the scaling factor for the node size, by default 3000
    """

    syll_names = [f"{ix}" for ix in syll_include]

    # find combinations
    group_combinations = list(combinations(groups, 2))

    # create group index dict
    group_idx_dict = {group: idx for idx, group in enumerate(groups)}

    # Figure out the number of rows for the plot
    n_row = math.ceil(len(group_combinations) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(16, 8 * n_row))
    ax = all_axes.flat

    for i, pair in enumerate(group_combinations):
        left_ind = group_idx_dict[pair[0]]
        right_ind = group_idx_dict[pair[1]]
        # left tm minus right tm
        tm_diff = trans_mats[left_ind] - trans_mats[right_ind]
        # left usage minus right usage
        usages_diff = np.array(list(usages[left_ind])) - np.array(
            list(usages[right_ind])
        )
        normlized_usg_abs_diff = (
            np.abs(usages_diff) / np.abs(usages_diff).sum()
        ) * node_scaling + 500

        G = nx.from_numpy_array(tm_diff * 1000)
        if layout == "circular":
            pos = nx.circular_layout(G)
        else:
            G_for_spring = nx.from_numpy_array(np.mean(trans_mats, axis=0))
            pos = nx.spring_layout(G_for_spring, iterations=500)

        nodelist = G.nodes()
        widths = nx.get_edge_attributes(G, "weight")

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=normlized_usg_abs_diff,
            node_color="white",
            edgecolors=["blue" if u > 0 else "red" for u in usages_diff],
            ax=ax[i],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=np.abs(list(widths.values())),
            edge_color=["blue" if u > 0 else "red" for u in widths.values()],
            ax=ax[i],
            alpha=0.6,
        )
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=dict(zip(nodelist, syll_names)),
            font_color="black",
            ax=ax[i],
        )
        ax[i].set_title(pair[0] + " - " + pair[1])

    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")
    # add legend
    legend_elements = [
        Line2D([0], [0], color="r", lw=2, label=f"Up-regulated transistion"),
        Line2D([0], [0], color="b", lw=2, label=f"Down-regulated transistion"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Up-regulated usage",
            markerfacecolor="w",
            markeredgecolor="r",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Down-regulated usage",
            markerfacecolor="w",
            markeredgecolor="b",
            markersize=10,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left", borderaxespad=0)
    plt.show()

# ----------------------------------------------------------

def compute_moseq_df_v2(result_dictionaries, group_names, *, fps=30, smooth_heading=True):
    """Compute moseq dataframe from results dict that contains all kinematic
    values by frame.

    Parameters
    ----------
    result_dictionaries : list of dictionaries of results from model fitting
    group_names : list of group names (one per dictionary).
    smooth_heading : bool, optional
        boolean flag whether smooth the computed heading, by default True

    Returns
    -------
    moseq_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    """

    recording_name = []
    centroid = []
    velocity = []
    heading = []
    angular_velocity = []
    syllables = []
    frame_index = []
    s_group = []

    for i in range(len(result_dictionaries)):

        results_dict = result_dictionaries[i]
        group_name = group_names[i]

        for k, v in results_dict.items():
            n_frame = v["centroid"].shape[0]
            recording_name.append([str(k)] * n_frame)
            centroid.append(v["centroid"])
            # velocity is pixel per second
            velocity.append(
                np.concatenate(
                    (
                        [0],
                        np.sqrt(np.square(np.diff(v["centroid"], axis=0)).sum(axis=1))
                        * fps,
                    )
                )
            )

            s_group.append([group_name] * n_frame)
            frame_index.append(np.arange(n_frame))

            if smooth_heading:
                recording_heading = analysis.filter_angle(v["heading"])
            else:
                recording_heading = v["heading"]

            # heading in radian
            heading.append(recording_heading)

            # compute angular velocity (radian per second)
            gaussian_smoothed_heading = analysis.filter_angle(
                recording_heading, size=3, method="gaussian"
            )
            angular_velocity.append(
                np.concatenate(([0], np.diff(gaussian_smoothed_heading) * fps))
            )

            # add syllable data
            syllables.append(v["syllable"])

    # construct dataframe
    moseq_df = pd.DataFrame(np.concatenate(recording_name), columns=["name"])
    column_names = (
        ["centroid_x", "centroid_y"]
        if centroid[0].shape[1] == 2
        else ["centroid_x", "centroid_y", "centroid_z"]
    )
    moseq_df = pd.concat(
        [
            moseq_df,
            pd.DataFrame(np.concatenate(centroid), columns=column_names),
        ],
        axis=1,
    )
    moseq_df["heading"] = np.concatenate(heading)
    moseq_df["angular_velocity"] = np.concatenate(angular_velocity)
    moseq_df["velocity_px_s"] = np.concatenate(velocity)
    moseq_df["syllable"] = np.concatenate(syllables)
    moseq_df["frame_index"] = np.concatenate(frame_index)
    moseq_df["group"] = np.concatenate(s_group)

    # compute syllable onset
    change = np.diff(moseq_df["syllable"]) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))

    onset = np.full(moseq_df.shape[0], False)
    onset[indices] = True
    moseq_df["onset"] = onset
    return moseq_df

# ----------------------------------------------------------

def compute_stats_df_v2(
    moseq_df,
    result_dictionaries,
    group_names,
    min_frequency=0.005,
    groupby=["group", "name"],
    fps=30,
):
    """Summary statistics for syllable frequencies and kinematic values.

    Parameters
    ----------
    moseq_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    result_dictionaries : list of dictionaries of results from model fitting
    group_names : list of group names (one per dictionary).
    threshold : float, optional
        usge threshold for the syllable to be included, by default 0.005
    groupby : list, optional
        the list of column names to group by, by default ['group', 'name']
    fps : int, optional
        frame per second information of the recording, by default 30

    Returns
    -------
    stats_df : pandas.DataFrame
        the summary statistics dataframe for syllable frequencies and kinematic values
    """

    # compute runlength encoding for syllables

    syllables = {}
    for i in range(len(result_dictionaries)):
        results_dict = result_dictionaries[i]
        recordings = results_dict.keys()
        for recording in recordings:
            syllables[recording] = results_dict[recording]["syllable"]

    # frequencies is array of frequencies for sorted syllables [syll_0, syll_1...]
    frequencies = analysis.get_frequencies(syllables)
    syll_include = np.where(frequencies > min_frequency)[0]

    # construct frequency dataframe
    # syllable frequencies within one session add up to 1
    frequency_df = []

    for i in range(len(result_dictionaries)):
        results_dict = result_dictionaries[i]
        group_name = group_names[i]

        for k, v in results_dict.items():
            syll_freq = analysis.get_frequencies(v["syllable"])
            df = pd.DataFrame(
                {
                    "name": k,
                    "group": group_name,
                    "syllable": np.arange(len(syll_freq)),
                    "frequency": syll_freq,
                }
            )
            frequency_df.append(df)

    frequency_df = pd.concat(frequency_df)
    if "name" not in groupby:
        frequency_df.drop(columns=["name"], inplace=True)

    # filter out syllables that are used less than threshold in all recordings
    filtered_df = moseq_df[moseq_df["syllable"].isin(syll_include)].copy()

    # TODO: hard-coded heading for now, could add other scalars
    features = filtered_df.groupby(groupby + ["syllable"])[
        ["heading", "angular_velocity", "velocity_px_s"]
    ].agg(["mean", "std", "min", "max"])

    features.columns = ["_".join(col).strip() for col in features.columns.values]
    features.reset_index(inplace=True)

    # get durations
    trials = filtered_df["onset"].cumsum()
    trials.name = "trials"
    durations = filtered_df.groupby(groupby + ["syllable"] + [trials])["onset"].count()
    # average duration in seconds
    durations = durations.groupby(groupby + ["syllable"]).mean() / fps
    durations.name = "duration"
    # only keep the columns we need
    durations = durations.fillna(0).reset_index()[groupby + ["syllable", "duration"]]

    stats_df = pd.merge(features, frequency_df, on=groupby + ["syllable"])
    stats_df = pd.merge(stats_df, durations, on=groupby + ["syllable"])
    return stats_df

# ----------------------------------------------------------

def _validate_and_order_syll_stats_params(
    complete_df,
    stat="frequency",
    order="stat",
    groups=None,
    ctrl_group=None,
    exp_group=None,
    colors=None,
    figsize=(10, 5),
):
    if not isinstance(figsize, (tuple, list)):
        print(
            "Invalid figsize. Input a integer-tuple or list of len(figsize) = 2. Setting figsize to (10, 5)"
        )
        figsize = (10, 5)

    unique_groups = complete_df["group"].unique()

    if groups is None or len(groups) == 0:
        groups = unique_groups
    elif isinstance(groups, str):
        groups = [groups]

    if isinstance(groups, (list, tuple, np.ndarray)):
        diff = set(groups) - set(unique_groups)
        if len(diff) > 0:
            groups = unique_groups

    if stat.lower() not in complete_df.columns:
        raise ValueError(
            f"Invalid stat entered: {stat}. Must be a column in the supplied dataframe."
        )

    if order == "stat":
        ordering, _ = analysis.sort_syllables_by_stat(complete_df, stat=stat)
    elif order == "diff":
        if (
            ctrl_group is None
            or exp_group is None
            or not np.all(np.isin([ctrl_group, exp_group], groups))
        ):
            raise ValueError(
                f"Attempting to sort by {stat} differences, but {ctrl_group} or {exp_group} not in {groups}."
            )
        ordering = analysis.sort_syllables_by_stat_difference(
            complete_df, ctrl_group, exp_group, stat=stat
        )
    if colors is None:
        colors = []
    if len(colors) == 0 or len(colors) != len(groups):
        colors = sns.color_palette(n_colors=len(groups))

    return np.array(ordering), groups, colors, figsize

# ----------------------------------------------------------

def plot_syll_stats_with_sem_v2(
    stats_df,
    plot_sig=True,
    thresh=0.05,
    stat="frequency",
    order="stat",
    groups=None,
    ctrl_group=None,
    exp_group=None,
    colors=None,
    join=False,
    figsize=(8, 4),
):
    """Plot syllable statistics with standard error of the mean.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        the dataframe that contains kinematic data and the syllable label
    plot_sig : bool, optional
        whether to plot the significant syllables, by default True
    thresh : float, optional
        the threshold for significance, by default 0.05
    stat : str, optional
        the statistic to plot, by default 'frequency'
    order : str, optional
        the ordering of the syllables, by default 'stat'
    groups : list, optional
        the list of groups to plot, by default None
    ctrl_group : str, optional
        the control group, by default None
    exp_group : str, optional
        the experimental group, by default None
    colors : list, optional
        the list of colors to use for each group, by default None
    join : bool, optional
        whether to join the points with a line, by default False
    figsize : tuple, optional
        the figure size, by default (8, 4)

    Returns
    -------
    fig : matplotlib.figure.Figure
        the figure object
    legend : matplotlib.legend.Legend
        the legend object
    """

    # get significant syllables
    sig_sylls = None
    if groups is None:
        groups = stats_df["group"].unique()

    if plot_sig and len(stats_df["group"].unique()) > 1:
        # run kruskal wallis and dunn's test
        _, _, sig_pairs = analysis.run_kruskal(stats_df, statistic=stat, thresh=thresh)
        # plot significant syllables for control and experimental group when user specify something
        if ctrl_group in groups and exp_group in groups:
            # check if the group pair is in the sig pairs dict
            if (ctrl_group, exp_group) in sig_pairs.keys():
                sig_sylls = sig_pairs.get((ctrl_group, exp_group))
            # flip the order of the groups
            else:
                sig_sylls = sig_pairs.get((exp_group, ctrl_group))
        else:
            # plot everything if no group pair is specified
            sig_sylls = sig_pairs

    xlabel = f"Syllables sorted by {stat}"
    if order == "diff":
        xlabel += " difference"
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(
        stats_df,
        stat=stat,
        order=order,
        groups=groups,
        ctrl_group=ctrl_group,
        exp_group=exp_group,
        colors=colors,
        figsize=figsize,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot each group's stat data separately, computes groupwise SEM, and orders data based on the stat/ordering parameters
    hue = "group" if groups is not None else None
    ax = sns.pointplot(
        data=stats_df,
        x="syllable",
        y=stat,
        hue=hue,
        order=ordering,
        errorbar=("ci", 68),
        ax=ax,
        hue_order=groups,
        palette=colors,
    )

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # add syllable labels if they exist

    labels = {ix: f"{ix}" for ix in ordering}
    syll_names = [labels[ix] for ix in ordering]

    plt.xticks(range(len(syll_names)), syll_names, rotation=90)

    # if a list of significant syllables is given, mark the syllables above the x-axis
    if sig_sylls is not None:
        init_y = -0.05
        # plot all sig syllables when no reasonable control and experimental group is specified
        if isinstance(sig_sylls, dict):
            for key in sig_sylls.keys():
                markings = []
                for s in sig_sylls[key]:
                    markings.append(np.where(ordering == s)[0])
                if len(markings) > 0:
                    markings = np.concatenate(markings)
                    plt.scatter(
                        markings, [init_y] * len(markings), color="r", marker="*"
                    )
                    plt.text(
                        plt.xlim()[1],
                        init_y,
                        f"{key[0]} vs. {key[1]} - Total {len(sig_sylls[key])} S.S.",
                    )
                    init_y += -0.05
                else:
                    print("No significant syllables found.")
        else:
            markings = []
            for s in sig_sylls:
                if s in ordering:
                    markings.append(np.where(ordering == s)[0])
                else:
                    continue
            if len(markings) > 0:
                markings = np.concatenate(markings)
                plt.scatter(markings, [-0.05] * len(markings), color="r", marker="*")
            else:
                print("No significant syllables found.")

        # manually define a new patch
        patch = Line2D(
            [],
            [],
            color="red",
            marker="*",
            markersize=9,
            label="Significant Syllable",
        )
        handles.append(patch)

    # add legend and axis labels
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel, fontsize=12)
    sns.despine()

    plt.show()

