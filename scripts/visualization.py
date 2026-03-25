# visualization.py — all figure-generating functions for the analysis.

import locale

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import stats
from scipy.spatial import ConvexHull

from config import CSV_ROOT, FOCUS_SITES, GROWING_SEASON_DAYLENGTH, TMP_DIR
from util import get_resampled, get_subdaily, single_slope


# ---- Fig. A1: climate classification ----


def plot_classification(df_classification):
    # scatter sites in TAir-PRECIP space, coloured by SLOPE, sized by AREA
    df_plot = df_classification[1]["data"]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 7))

    # scale AREA to marker size
    area_norm = (df_plot["AREA"] - df_plot["AREA"].min()) / (
        df_plot["AREA"].max() - df_plot["AREA"].min()
    )
    min_size = 50
    max_size = 900
    df_plot["AREA_size"] = min_size + area_norm * (max_size - min_size)

    # diverging colormap centred at zero
    vmin = -max(abs(df_plot["SLOPE"].min()), abs(df_plot["SLOPE"].max()))
    vmax = -vmin
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    sc = ax.scatter(
        df_plot["TAir"], df_plot["PRECIP"],
        c=df_plot["SLOPE"], cmap="RdBu", norm=norm,
        alpha=1, edgecolor="k", linewidth=1,
        s=df_plot["AREA_size"], zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, label=r"sSLOPE")
    cbar.ax.tick_params(labelsize=14)

    # highlight focus sites
    highlight_sites = FOCUS_SITES
    offsets = {"GUF_GUY_GUY": (-130, 0), "FRA_PUE": (-90, 40), "RUS_POG_VAR": (-10, 40)}

    for site in highlight_sites:
        if site in df_plot.index:
            row = df_plot.loc[site]
            site_color = sc.cmap(sc.norm(row["SLOPE"]))
            ax.scatter(
                row["TAir"], row["PRECIP"],
                s=row["AREA_size"], facecolors=site_color,
                edgecolors="red", linewidth=1.5, zorder=3,
            )
            dx, dy = offsets[site]
            ax.annotate(
                site, (row["TAir"], row["PRECIP"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=14, color="red", weight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                ha="left", va="bottom",
            )

    ax.set_ylabel(r"PRECIP [mm]", fontsize=16)
    ax.set_xlabel(r"TAir [$^\circ$C]", fontsize=16)
    ax.grid(True)

    # marker-size legend
    area_values = df_plot["AREA"].quantile([0.25, 0.5, 0.75]).round(2)
    handles = [
        ax.scatter(
            [], [],
            s=min_size + (
                (a - df_plot["AREA"].min())
                / (df_plot["AREA"].max() - df_plot["AREA"].min())
            ) * (max_size - min_size),
            facecolors="gray", edgecolors="k", alpha=0.6,
        )
        for a in area_values
    ]
    labels = [f"{a}" for a in area_values]
    ax.legend(
        handles, labels, title="nAREA", scatterpoints=1,
        frameon=True, fontsize=14, title_fontsize=15, loc="upper left",
    )

    fig.tight_layout()
    return fig


# ---- Fig. 3: seasonal cycles per focus site ----


def plot_environment(fig, spec, datafile_env, years_to_keep, site, ppfd_min, ppfd_max, panel_label):
    # plot PPFD heatbar, TAir, and TSM seasonal cycles
    df_env = pd.read_csv(datafile_env)
    df_env["solar_TIMESTAMP"] = pd.to_datetime(df_env["solar_TIMESTAMP"])
    df_env.index = df_env["solar_TIMESTAMP"]

    tsm_col = "swc_shallow"
    ppfd_col = "ppfd_in"
    tsm_label = r"TSM ($\frac{cm^{3}}{cm^{3}}$)"

    df_env = df_env[df_env.index.year.isin(years_to_keep)]

    tair_daily = df_env["ta"].resample("D").mean()
    ppfd_daily = df_env[ppfd_col].resample("D").mean()
    tsm_daily = df_env[tsm_col].resample("D").mean()

    tair_cycle = tair_daily.groupby(tair_daily.index.dayofyear).mean()
    ppfd_cycle = ppfd_daily.groupby(ppfd_daily.index.dayofyear).mean()
    tsm_cycle = tsm_daily.groupby(tsm_daily.index.dayofyear).mean()

    inner_spec = spec.subgridspec(nrows=2, ncols=1, height_ratios=[1, 4], hspace=0.05)

    ax_ppfd = fig.add_subplot(inner_spec[0])
    im = ax_ppfd.imshow(
        np.tile(ppfd_cycle.values, (10, 1)),
        aspect="auto", cmap="inferno", origin="lower",
        extent=[1, 365, 0, 1],
        vmin=ppfd_min, vmax=ppfd_max,
    )
    ax_ppfd.set_yticks([])
    ax_ppfd.set_xticks([])
    ax_ppfd.text(
        -0.02, 1.65, f"({panel_label})",
        transform=ax_ppfd.transAxes, fontsize=22, fontweight="bold",
        va="top", ha="right",
    )

    ax_temp = fig.add_subplot(inner_spec[1])
    ax_tsm = ax_temp.twinx()

    ax_temp.plot(tair_cycle.index, tair_cycle.values, color="tab:red")
    ax_temp.set_ylabel(r"TAir ($^\circ$C)", color="tab:red", fontsize=22)
    ax_temp.tick_params(axis="y", labelcolor="tab:red", labelsize=22)

    ax_tsm.plot(tsm_cycle.index, tsm_cycle.values, color="blue")
    ax_tsm.set_ylabel(tsm_label, color="blue", fontsize=22)
    ax_tsm.tick_params(axis="y", labelcolor="blue", labelsize=22)

    ax_temp.set_xlim(0, 364)
    ax_temp.set_xticks([])

    return tsm_cycle.reset_index(drop=True), ppfd_cycle.reset_index(drop=True), im


def plot_concept(fig, spec, concept, panel_label):
    # SF-VPD hysteresis loop with morning-branch regression line
    ax_concept = fig.add_subplot(spec)

    ax_concept.scatter(concept["VPD"], concept["SF"], color="k")
    ax_concept.plot(concept["VPD"], concept["SF"], ".k-", lw=4)

    vpd = concept["VPD"]
    sap = concept["SF"]
    reg_morning = single_slope(vpd, sap)

    slope_morning = reg_morning[0]
    intercept_morning = reg_morning[1]
    f_morning = lambda x: slope_morning * x + intercept_morning
    x_morning = np.array([vpd.min(), vpd.max()])

    morning_data = concept.between_time(
        vpd.idxmin().strftime("%H:%M"), sap.idxmax().strftime("%H:%M")
    )

    ax_concept.scatter(
        morning_data["VPD"], morning_data["SF"], c="#1f77b4", zorder=10, s=300,
    )
    ax_concept.plot(x_morning, f_morning(x_morning), c="#1f77b4", lw=4)

    xmin, xmax = vpd.min(), vpd.max()
    ymin, ymax = sap.min(), sap.max()

    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
    ax_concept.tick_params(axis="x", labelsize=22)
    ax_concept.set_xticks(
        [xmin, xmax],
        labels=[
            f"{round(xmin, 2)},\n{vpd.idxmin().strftime('%-I %p')}",
            f"{round(xmax, 2)},\n{vpd.idxmax().strftime('%-I %p')}",
        ],
    )

    ax_concept.tick_params(axis="y", labelsize=22)
    ax_concept.set_yticks(
        [ymin, ymax],
        labels=[
            f"{round(ymin, 2)},\n{sap.idxmin().strftime('%-I %p')}",
            f"{round(ymax, 2)},\n{sap.idxmax().strftime('%-I %p')}",
        ],
    )

    for value in [xmin, xmax]:
        ax_concept.axvline(value, color="gray", ls="--", lw=1)
    for value in [ymin, ymax]:
        ax_concept.axhline(value, color="gray", ls="--", lw=1)

    ax_concept.set_ylabel(r"Sapflux in ($\frac{m^{3}}{s}$)", fontsize=22)
    ax_concept.set_xlabel(r"Vapour Pressure Deficit in ($kPa$)", fontsize=22)

    # close the cycle
    first_point = (concept["VPD"].iloc[0], concept["SF"].iloc[0])
    last_point = (concept["VPD"].iloc[-1], concept["SF"].iloc[-1])
    ax_concept.plot(
        [last_point[0], first_point[0]],
        [last_point[1], first_point[1]],
        color="k", lw=4, linestyle="-",
    )

    ax_concept.text(
        -0.05, 1.05, f"({panel_label})",
        transform=ax_concept.transAxes, fontsize=22, fontweight="bold",
        verticalalignment="top", horizontalalignment="right",
    )

    return ax_concept


def plot_metrics(ax, metric_cycles, daylength, site):
    # seasonal SLOPE and AREA median with IQR shading
    if site == "FRA_PUE" or site == "FR-Pue":
        slope_ylim = (-0.65, 0.65)
        area_ylim = (0.55, -0.55)
    elif site == "RUS_POG_VAR":
        slope_ylim = (-2, 2)
        area_ylim = (1.2, -1.2)
    elif site == "GUF_GUY_GUY":
        slope_ylim = (-50, 50)
        area_ylim = (7, -7)
    else:
        raise ValueError(f"Unknown site: {site}")

    daylength.index = pd.to_datetime(daylength["date"])
    daylength = (
        daylength[["daylength"]]
        .groupby([daylength.index.month, daylength.index.day])
        .mean()
        .reset_index(drop=True)
    )
    dl_12 = daylength[daylength["daylength"] == GROWING_SEASON_DAYLENGTH].index
    first_12, last_12 = dl_12[0], dl_12[-1]

    slope_df, area_df = metric_cycles
    slope_med = slope_df.median(axis=1).rolling(7, center=True).mean()
    slope_q25 = slope_df.quantile(0.25, axis=1).rolling(7, center=True).mean()
    slope_q75 = slope_df.quantile(0.75, axis=1).rolling(7, center=True).mean()

    area_med = area_df.median(axis=1).rolling(7, center=True).mean()
    area_q25 = area_df.quantile(0.25, axis=1).rolling(7, center=True).mean()
    area_q75 = area_df.quantile(0.75, axis=1).rolling(7, center=True).mean()

    ax_area = ax
    ax_slope = ax.twinx()

    ax_slope.fill_between(range(len(slope_med)), slope_q25, slope_q75, color="#1f77b4", alpha=0.2)
    ax_area.fill_between(range(len(area_med)), area_q25, area_q75, color="black", alpha=0.2)

    ax_slope.plot(slope_med, color="#1f77b4", lw=3)
    ax_area.plot(area_med, color="black", lw=3)

    # month ticks
    dates = slope_df.index
    dates_full = pd.to_datetime(dates.astype(str) + "-2023", format="%m-%d-%Y")
    month_starts = pd.date_range(dates_full.min(), dates_full.max(), freq="MS")
    month_idx = [dates_full.get_loc(d) + 1 for d in month_starts]

    ax_area.set_xticks(month_idx, month_starts.month)
    ax_slope.set_xticks(month_idx, month_starts.month)
    ax_area.set_xlabel("month", fontsize=22)

    ax_area.axvline(first_12, color="gray", linestyle="--")
    ax_area.axvline(last_12, color="gray", linestyle="--")
    ax_area.axhline(0.0, color="gray")

    ax_area.set_ylabel(r"AREA ($\frac{m^{3}}{s}*kPa$)", fontsize=22, color="k")
    ax_slope.set_ylabel(r"SLOPE ($\frac{m^{3}}{s*kPa}$)", fontsize=22, color="#1f77b4")

    ax_area.set_ylim(area_ylim)
    ax_slope.set_ylim(slope_ylim)

    ax_area.tick_params(axis="both", labelsize=22)
    ax_slope.tick_params(axis="y", labelsize=22, labelcolor="#1f77b4")

    x_min = 0
    x_max = len(slope_med) - 1
    ax_area.set_xlim(x_min, x_max)
    ax_slope.set_xlim(x_min, x_max)


def plot_cycle(
    concept, metric_cycles, daylength, slopeframe, areaframe,
    site, datafile_env, ppfd_min, ppfd_max, panel_labels,
    years_to_keep=None, fig=None, spec=None,
):
    # combined panel: conceptual loop + environment + seasonal metrics for one site
    if years_to_keep is None:
        years_to_keep = sorted(set(areaframe.index.year) & set(slopeframe.index.year))

    areaframe = areaframe.loc[areaframe.index.year.isin(years_to_keep)]
    slopeframe = slopeframe.loc[slopeframe.index.year.isin(years_to_keep)]

    concept_label, env_label = panel_labels

    tsm, ppfd, im = plot_environment(
        fig, spec[0, 1], datafile_env, years_to_keep,
        site, ppfd_min, ppfd_max, env_label,
    )

    areaframe.index = pd.to_datetime(areaframe.index)
    slopeframe.index = pd.to_datetime(slopeframe.index)

    ax_concept = plot_concept(fig, spec[:, 0], concept, concept_label)
    ax_concept.text(
        0.5, -0.05, site,
        transform=ax_concept.transAxes, fontsize=22, fontweight="bold",
        verticalalignment="center", horizontalalignment="center",
    )

    ax_metrics = fig.add_subplot(spec[1, 1])
    plot_metrics(ax_metrics, metric_cycles, daylength, site)

    return im


# ---- Fig. 4: correlation heatmap ----


def plot_heatmap_summary_param(all_correlations, metadata_path):
    # heatmap of seasonal-cycle correlations aggregated by site prefix
    site_metadata_df = pd.read_csv(metadata_path)
    site_metadata_df["site_prefix"] = site_metadata_df["Site"].str[:7]

    # collect correlations per site prefix (aggregate across plots at the same site)
    r_records = {}
    p_records = {}
    plot_counts = {}

    for site, corr in all_correlations.items():
        prefix = site[:7]
        r_records.setdefault(prefix, {})
        p_records.setdefault(prefix, {})
        plot_counts[prefix] = plot_counts.get(prefix, 0) + 1

        if corr is None:
            continue

        for metric, rp in corr.items():
            if rp is None or not isinstance(rp, (tuple, list)) or len(rp) != 2:
                continue
            r, p = rp
            r_records[prefix].setdefault(metric, [])
            p_records[prefix].setdefault(metric, [])
            if r is not None and not np.isnan(r):
                r_records[prefix][metric].append(r)
            if p is not None and not np.isnan(p):
                p_records[prefix][metric].append(p)

    # average across plots (simple mean)
    r_data = {
        prefix: {m: np.mean(v) if len(v) > 0 else np.nan for m, v in metrics.items()}
        for prefix, metrics in r_records.items()
    }
    p_data = {
        prefix: {m: np.mean(v) if len(v) > 0 else np.nan for m, v in metrics.items()}
        for prefix, metrics in p_records.items()
    }

    r_df = pd.DataFrame(r_data).T
    p_df = pd.DataFrame(p_data).T

    valid_rows = r_df.notna().any(axis=1)
    r_df = r_df.loc[valid_rows]
    p_df = p_df.loc[valid_rows]

    r_df = r_df.sort_index()
    p_df = p_df.loc[r_df.index]

    # enforce metric order on x-axis
    base_vars = ["TSM", "TAir", "PPFD"]
    metric_order = (
        [f"SLOPE-{v}" for v in base_vars]
        + [f"AREA-{v}" for v in base_vars]
    )
    data_columns = [m for m in metric_order if m in r_df.columns]
    r_df_plot = r_df[data_columns]
    p_df_plot = p_df[data_columns]

    # significance annotations
    sig = p_df_plot.map(
        lambda p: "***" if p < 0.001 else
                  "**" if p < 0.01 else
                  "*" if p < 0.05 else ""
    )

    annot = r_df_plot.copy().astype(str)
    for r in annot.index:
        for c in annot.columns:
            v = r_df_plot.loc[r, c]
            annot.loc[r, c] = f"{v:.2f}{sig.loc[r, c]}" if not np.isnan(v) else ""

    # y-axis labels with plot count
    yticklabels = []
    for idx in r_df_plot.index:
        n = plot_counts.get(idx, 1)
        label = f"{idx} (n={n})" if n > 1 else idx
        yticklabels.append(label)

    fig, ax = plt.subplots(figsize=(9, max(4, len(r_df_plot) * 0.35)))

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("white")

    sns.heatmap(
        r_df_plot.astype(float),
        annot=annot, fmt="", cmap=cmap,
        vmin=-1, vmax=1,
        linewidths=0.8, linecolor="white",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Correlation", "shrink": 0.6, "aspect": 30},
        ax=ax,
    )

    ax.set_yticklabels(yticklabels, rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.18)

    return fig, ax


# ---- Fig. 5: percentile patterns and hysteresis fingerprints ----


def plot_percentiles(anomalies, code, fig=None, outer_spec=None):
    # scatter sites in TAir-TSM space, coloured by SLOPE, sized by AREA,
    # grouped into 4 percentile-based clusters
    sites = list(anomalies.keys())
    n_sites = len(sites)

    if fig is None or outer_spec is None:
        fig = plt.figure(figsize=(6 * n_sites, 6))
        spec = gridspec.GridSpec(ncols=n_sites + 1, nrows=1)
    else:
        spec = outer_spec.subgridspec(1, n_sites + 1)

    all_cluster_slopes = []
    all_cluster_areas = []
    cluster_summaries = {}

    for site in sites:
        df = anomalies[site]
        site_clusters = []
        labels = df["Cluster"].dropna().unique()

        for label in labels:
            cluster_df = df[df["Cluster"] == label]
            stddev_TAir, stddev_TSM = cluster_df[["TAir", "TSM"]].std().to_numpy()
            slope_mean = cluster_df["SLOPE"].mean()
            area_mean = cluster_df["AREA"].mean()
            centroid_x = cluster_df["TAir"].mean()
            centroid_y = cluster_df["TSM"].mean()

            all_cluster_slopes.append(slope_mean)
            all_cluster_slopes.extend(cluster_df["SLOPE"].values)
            all_cluster_areas.append(area_mean)
            all_cluster_areas.extend(cluster_df["AREA"].values)

            site_clusters.append({
                "label": label,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "stddev_x": stddev_TAir,
                "stddev_y": stddev_TSM,
                "slope_mean": slope_mean,
                "area_mean": area_mean,
                "points": cluster_df[["TSM", "TAir"]].values,
                "slopes": cluster_df["SLOPE"].values,
                "areas": cluster_df["AREA"].values,
            })

        cluster_summaries[site] = site_clusters

    # zero-centred colormap for SLOPE
    slope_min, slope_max = min(all_cluster_slopes), max(all_cluster_slopes)
    area_min, area_max = min(all_cluster_areas), max(all_cluster_areas)
    cmap = cm.get_cmap("RdBu")
    slope_abs_max = max(abs(slope_min), abs(slope_max))
    norm = mcolors.TwoSlopeNorm(vmin=-slope_abs_max, vcenter=0, vmax=slope_abs_max)

    # axis limits from all sites
    all_tair = np.concatenate([anomalies[site]["TAir"].values for site in sites])
    all_tsm = np.concatenate([anomalies[site]["TSM"].values for site in sites])
    xlim = (all_tair.min() - 2, all_tair.max() + 1)
    ylim = (all_tsm.min(), all_tsm.max() + 0.03)

    subplot_labels = ["(a)", "(b)", "(c)"]
    axes = []

    for idx, site in enumerate(sites):
        ax = fig.add_subplot(spec[0, idx + 1])
        axes.append(ax)
        if idx > 0:
            ax.tick_params(labelleft=False)

        df = anomalies[site]

        if code == "supp":
            ax.scatter(
                df["TAir"], df["TSM"], s=10, c="gray", marker="x", alpha=0.5, zorder=0,
            )

        for cluster in cluster_summaries[site]:
            points = cluster["points"]
            slopes = cluster["slopes"]
            areas = cluster["areas"]
            label = cluster["label"]

            sizes = 900 * (areas - area_min) / (area_max - area_min + 1e-6) + 90
            colors = [cmap(norm(s)) for s in slopes]
            ax.scatter(
                points[:, 1], points[:, 0],
                c=colors, s=sizes, alpha=0.6, marker="o",
                edgecolor="k", linewidths=2, zorder=2,
            )

            centroid_x = points[:, 1].mean()
            centroid_y = points[:, 0].max() + 0.02
            dx, dy = 0, 0
            if site == "GUF_GUY_GUY" and label == "cold & wet":
                dx, dy = -4, -0.04
            if site == "GUF_GUY_GUY" and label == "cold & dry":
                dx, dy = -4, -0.04

            ax.text(
                centroid_x + dx, centroid_y + dy, str(label),
                fontsize=16, weight="bold", ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"),
                zorder=3,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Site: {site}", fontsize=32)
        ax.tick_params(labelsize=22)

        if idx < len(subplot_labels):
            ax.text(
                -0.05, 1.1, subplot_labels[idx],
                transform=ax.transAxes, fontsize=22, fontweight="bold",
                va="top", ha="right",
            )

    axes[0].set_ylabel(r"TSM in $\frac{cm^{3}}{cm^{3}}$", fontsize=26)
    axes[1].set_xlabel(r"TAir in $^\circ$C", fontsize=26, labelpad=15)

    # AREA size legend
    area_legend_values = [area_min, (area_min + area_max) / 2, area_max]
    size_legend_values = [
        900 * (a - area_min) / (area_max - area_min + 1e-6) + 90
        for a in area_legend_values
    ]
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", label=f"{a:.2f}",
            markerfacecolor="gray", markersize=np.sqrt(s), markeredgecolor="black",
        )
        for a, s in zip(area_legend_values, size_legend_values)
    ]
    ax_legend = fig.add_subplot(spec[0, 0])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=legend_elements, title="nAREA",
        loc="center left", bbox_to_anchor=(0.25, 0.5),
        fontsize=20, title_fontsize=22, frameon=False,
    )

    return cmap, norm


def plot_dailycycle(groups_dicts, ax, quadrant_linestyles):
    # overlay convex hulls and morning-branch slopes for each cluster
    gr_data = []
    for groups_dict in groups_dicts:
        for cluster_id, group_df in groups_dict.items():
            mask = ~np.isnan(group_df["VPD"]) & ~np.isnan(group_df["SF"])
            group = group_df[mask]
            if group.empty:
                continue

            points = np.column_stack((group["VPD"], group["SF"]))
            linestyle = quadrant_linestyles[cluster_id]

            if len(points) > 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(
                        points[simplex, 0], points[simplex, 1],
                        color="black", linestyle=linestyle, lw=5,
                    )

            vpd = group["VPD"]
            sap = group["SF"]
            start_idx = vpd.idxmin()
            end_idx = sap.idxmax()

            if start_idx <= end_idx:
                mask_slice = (vpd.index >= start_idx) & (vpd.index <= end_idx)
            else:
                mask_slice = (vpd.index >= start_idx) | (vpd.index <= end_idx)

            vpd_slice = vpd[mask_slice]
            sap_slice = sap[mask_slice]
            reg = single_slope(vpd_slice, sap_slice)

            slope, intercept = reg[:2]
            x_vals = np.array([vpd_slice.min(), vpd_slice.max()])
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color="#1f77b4", linestyle=linestyle, lw=5)

            gr_data.append(group)

    return gr_data


def plot_hysteresis_main(mean_cycles, fig=None, outer_spec=None):
    # hysteresis loops per cluster per focus site
    quadrant_linestyles = {
        "hot & wet": "solid",
        "cold & wet": "dashed",
        "cold & dry": "dashdot",
        "hot & dry": "dotted",
    }
    site_initials = sorted(set(k[0] for k in mean_cycles if k != "all"))
    sites = sorted(set(name for name in mean_cycles if name != "all"))

    n_sites = len(site_initials)
    total_cols = max(n_sites, 4)

    if fig is None or outer_spec is None:
        fig = plt.figure(figsize=(6 * total_cols, 7))
        spec = gridspec.GridSpec(ncols=total_cols, nrows=1)
    else:
        spec = outer_spec.subgridspec(1, total_cols)

    subplot_labels = ["(d)", "(e)", "(f)"]
    axs = []

    for j in range(total_cols):
        if j < n_sites:
            site_initial = site_initials[j]
            site_clusters = {
                k: v for k, v in mean_cycles.items() if k.startswith(site_initial)
            }
            groups = [df for _, df in site_clusters.items()]

            ax = fig.add_subplot(spec[0, j + 1])
            plot_dailycycle(groups, ax, quadrant_linestyles=quadrant_linestyles)
            ax.tick_params(axis="x", labelsize=22)
            ax.tick_params(axis="y", labelsize=22)

            if j < len(subplot_labels):
                ax.text(
                    -0.05, 1.1, subplot_labels[j],
                    transform=ax.transAxes, fontsize=22, fontweight="bold",
                    va="top", ha="right",
                )
        else:
            # legend column
            ax = fig.add_subplot(spec[0, 0])
            ax.axis("off")
            quadrant_labels = ["hot & wet", "hot & dry", "cold & dry", "cold & wet"]
            legend_elements = [
                Line2D(
                    [0], [0], color="black",
                    linestyle=quadrant_linestyles[label], lw=3, label=label,
                )
                for label in quadrant_labels
            ]
            ax.legend(
                handles=legend_elements, loc="center",
                fontsize=32, handlelength=2, frameon=False, labelspacing=1.5,
            )

        axs.append(ax)

    axs[0].set_ylabel(r"Sapflux (SF) in ($\frac{m^{3}}{s}$)", fontsize=26)
    axs[n_sites // 2].set_xlabel(r"VPD in $kPa$", fontsize=26)


def plot_patterns(extreme_anomalies, mean_cycles, code):
    # combined figure: percentile clusters (top) + hysteresis loops (bottom)
    fig = plt.figure(figsize=(30, 15))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    gray_cmap, grayscale_norm = plot_percentiles(
        extreme_anomalies, code, fig=fig, outer_spec=outer[0],
    )
    plot_hysteresis_main(
        mean_cycles, fig=fig, outer_spec=outer[1],
    )

    # shared SLOPE colorbar
    sm = plt.cm.ScalarMappable(cmap=gray_cmap, norm=grayscale_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.54, 0.015, 0.35])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("sSLOPE", fontsize=26)
    cbar.ax.tick_params(labelsize=22)

    return fig


# ---- Fig. 7: sample-rate comparison ----


def plot_rates(site, fig, spec):
    # hysteresis loops at different temporal resolutions vs hourly reference
    environment = CSV_ROOT / f"plant/{site}_env_data.csv"
    sapflux = CSV_ROOT / f"plant/{site}_sapf_data.csv"

    subdaily = get_subdaily(sapflux, environment)
    subdaily.index = pd.to_datetime(subdaily.index)

    resamplings = get_resampled(subdaily)

    sample_rates = ["8TPD", "4TPD", "3TPDnight", "3TPDday"]
    positions = [
        spec[0, 0],  # 8TPD
        spec[1, 0],  # 4TPD
        spec[1, 1],  # 3TPDnight
        spec[0, 1],  # 3TPDday
    ]

    for i, samplerate in enumerate(sample_rates):
        resampling = resamplings[i]
        if resampling is None:
            continue

        concept = (
            resampling[["SF", "VPD"]]
            .groupby(resampling[["SF", "VPD"]].index.hour)
            .mean()
        )
        first_row = concept.iloc[[0]]
        concept = pd.concat([concept, first_row])

        ax_inner = fig.add_subplot(positions[i])

        # hourly reference in grey
        hourly_data = subdaily[["SF", "VPD"]].groupby(subdaily.index.hour).mean()
        ax_inner.plot(
            hourly_data["VPD"], hourly_data["SF"],
            color="grey", linestyle="-", lw=2, label="Hourly Reference",
        )

        hourly_vpdmin = hourly_data["VPD"].min()
        hourly_vpdmax = hourly_data["VPD"].max()
        hourly_sapmin = hourly_data["SF"].min()
        hourly_sapmax = hourly_data["SF"].max()

        ax_inner.scatter(concept["VPD"], concept["SF"], color="k", label=samplerate)
        ax_inner.plot(concept["VPD"], concept["SF"], color="k", lw=3)

        vpd = concept["VPD"]
        sap = concept["SF"]
        xmax, ymax = vpd.max(), sap.max()
        xmin, ymin = vpd.min(), sap.min()

        row = 0 if i in [0, 3] else 1
        col = 0 if i in [0, 1] else 1

        if row == 1:
            ax_inner.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=14)
            ax_inner.set_xlabel("VPD (kPa)", fontsize=22)
        else:
            ax_inner.tick_params(axis="x", bottom=False, labelbottom=False)

        if col == 0:
            ax_inner.tick_params(axis="y", left=True, labelleft=True, labelsize=14)
            ax_inner.set_ylabel(r"Sapflux ($\frac{m^{3}}{s}$)", fontsize=22)
        else:
            ax_inner.tick_params(axis="y", left=False, labelleft=False)

        ax_inner.axvline(xmin, ls="--", lw=1.5, color="black")
        ax_inner.axvline(xmax, ls="--", lw=1.5, color="black")
        ax_inner.axhline(ymin, ls="--", lw=1.5, color="black")
        ax_inner.axhline(ymax, ls="--", lw=1.5, color="black")

        ax_inner.axvline(hourly_vpdmin, ls="--", lw=1.5, color="gray")
        ax_inner.axvline(hourly_vpdmax, ls="--", lw=1.5, color="gray")
        ax_inner.axhline(hourly_sapmin, ls="--", lw=1.5, color="gray")
        ax_inner.axhline(hourly_sapmax, ls="--", lw=1.5, color="gray")

        ax_inner.set_title(samplerate, fontsize=22)


def plot_coefficients(slope_longterm, area_longterm, fig, spec):
    # boxplot of R-squared values for SLOPE and AREA across sample rates
    ax = fig.add_subplot(spec[:, 2])

    slope_melted = slope_longterm.melt(var_name="SampleRate", value_name="R2")
    slope_melted["Type"] = "Slope"

    area_melted = area_longterm.melt(var_name="SampleRate", value_name="R2")
    area_melted["Type"] = "Area"

    data = pd.concat([slope_melted, area_melted], ignore_index=True)
    data = data.dropna()

    sns.boxplot(
        data=data, x="SampleRate", y="R2", hue="Type",
        palette={"Slope": "#1f77b4", "Area": "#2c2c2c"},
        dodge=True, ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample Rate", fontsize=22)
    ax.set_ylabel("Pearson $R^2$", fontsize=22)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=45)

    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    ax.get_legend().remove()

    return ax


def plot_srs(slope_longterm, area_longterm):
    # combined figure: sample-rate loops (left) + R-squared boxplot (right)
    fig = plt.figure(figsize=(18, 10))

    spec = gridspec.GridSpec(
        nrows=2, ncols=3,
        left=0.05, right=0.95,
        height_ratios=[1, 1], width_ratios=[1, 1, 2],
        hspace=0.2, wspace=0.2,
    )

    plot_rates("FRA_PUE", fig, spec)
    plot_coefficients(slope_longterm, area_longterm, fig, spec)

    # panel labels
    fig.text(0.03, 0.97, "(a)", fontsize=26, fontweight="bold", ha="left", va="top")
    fig.text(0.53, 0.97, "(b)", fontsize=26, fontweight="bold", ha="left", va="top")


# ---- Fig. 6: SLOPE and AREA distributions ----


def plot_distributions_SLOPE_AREA(slope_area_distributions):
    # KDE distributions of SLOPE and AREA by TAir/TSM anomaly combination
    combination_colors = {
        "high TAir & low TSM": "#d62728",
        "low TAir & high TSM": "#1f77b4",
        "high TAir & high TSM": "#2c2c2c",
        "low TAir & low TSM": "#7f7f7f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    combinations = slope_area_distributions["combination"].unique()

    # SLOPE distribution (left panel)
    ax_slope = axes[0]
    for combination in combinations:
        data = slope_area_distributions[
            slope_area_distributions["combination"] == combination
        ]["SLOPE"]

        if len(data) > 1:
            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min() - 0.1, data.max() + 0.1, 300)
                density = kde(x_range)
                ax_slope.plot(
                    x_range, density,
                    color=combination_colors[combination], linewidth=2.5,
                    label=f"{combination}", alpha=0.8, zorder=3,
                )
                ax_slope.fill_between(
                    x_range, density, alpha=0.2,
                    color=combination_colors[combination], zorder=1,
                )
            except Exception:
                pass

    ax_slope.set_xlabel("sSLOPE", fontsize=14, fontweight="bold")
    ax_slope.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax_slope.legend(loc="best", fontsize=11, handlelength=1, handleheight=1.5)
    ax_slope.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_slope.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    ax_slope.text(
        -0.1, 1.05, "(a)", transform=ax_slope.transAxes,
        fontsize=18, fontweight="bold", va="top", ha="right",
    )

    # AREA distribution (right panel)
    ax_area = axes[1]
    for combination in combinations:
        data = slope_area_distributions[
            slope_area_distributions["combination"] == combination
        ]["AREA"]

        if len(data) > 1:
            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min() - 0.01, data.max() + 0.01, 300)
                density = kde(x_range)
                ax_area.plot(
                    x_range, density,
                    color=combination_colors[combination], linewidth=2.5,
                    label=f"{combination}", alpha=0.8, zorder=3,
                )
                ax_area.fill_between(
                    x_range, density, alpha=0.2,
                    color=combination_colors[combination], zorder=1,
                )
            except Exception:
                pass

    ax_area.set_xlabel("nAREA", fontsize=14, fontweight="bold")
    ax_area.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax_area.legend(loc="best", fontsize=11, handlelength=1, handleheight=1.5)
    ax_area.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_area.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    ax_area.text(
        -0.1, 1.05, "(b)", transform=ax_area.transAxes,
        fontsize=18, fontweight="bold", va="top", ha="right",
    )

    plt.tight_layout()
    return fig
