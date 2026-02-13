import locale
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable, coolwarm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import stats
from scipy.spatial import ConvexHull

from config import CSV_ROOT, FOCUS_SITES, GROWING_SEASON_DAYLENGTH, TMP_DIR
from util import get_resampled, get_subdaily, single_slope


# SAPFLUXNET data in climate space of temperature and precipitation
def plot_classification(df_classification):
    df_plot = df_classification[1]["data"]

    # Set global font size
    plt.rcParams.update({"font.size": 14})

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scale AREA to marker size
    area_norm = (df_plot["AREA"] - df_plot["AREA"].min()) / (
        df_plot["AREA"].max() - df_plot["AREA"].min()
    )
    min_size = 50
    max_size = 900
    df_plot["AREA_size"] = min_size + area_norm * (max_size - min_size)

    # Use diverging colormap centered at 0
    vmin = -max(abs(df_plot["SLOPE"].min()), abs(df_plot["SLOPE"].max()))
    vmax = -vmin
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    sc = ax.scatter(
        df_plot["TAir"],
        df_plot["PRECIP"],
        c=df_plot["SLOPE"],
        cmap="RdBu",
        norm=norm,
        alpha=1,
        edgecolor="k",
        linewidth=1,
        s=df_plot["AREA_size"],
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, label=r"sSLOPE")
    cbar.ax.tick_params(labelsize=14)

    # Highlight selected sites
    highlight_sites = FOCUS_SITES
    offsets = {"GUF_GUY_GUY": (-130, 0), "FRA_PUE": (-90, 40), "RUS_POG_VAR": (-10, 40)}

    # Highlight selected sites with same color as main scatter + red edge
    for site in highlight_sites:
        if site in df_plot.index:
            row = df_plot.loc[site]

            # Get exact color from the main scatter
            site_color = sc.cmap(sc.norm(row["SLOPE"]))

            ax.scatter(
                row["TAir"],
                row["PRECIP"],
                s=row["AREA_size"],
                facecolors=site_color,  # match scatter color
                edgecolors="red",  # red border
                linewidth=1.5,
                zorder=3,
            )

            dx, dy = offsets[site]
            ax.annotate(
                site,
                (row["TAir"], row["PRECIP"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=14,
                color="red",
                weight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                ha="left",
                va="bottom",
            )
        else:
            print("Fail")

    # Axis labels and title
    ax.set_ylabel(r"PRECIP [mm]", fontsize=16)
    ax.set_xlabel(r"TAir [°C]", fontsize=16)
    ax.grid(True)
    
    # Marker size legend using the same scaling as scatter points
    area_values = df_plot["AREA"].quantile([0.25, 0.5, 0.75]).round(2)
    handles = [
        ax.scatter(
            [],
            [],
            s=min_size
            + (
                (a - df_plot["AREA"].min())
                / (df_plot["AREA"].max() - df_plot["AREA"].min())
            )
            * (max_size - min_size),
            facecolors="gray",
            edgecolors="k",
            alpha=0.6,
        )
        for a in area_values
    ]
    labels = [f"{a}" for a in area_values]
    ax.legend(
        handles,
        labels,
        title="nAREA",
        scatterpoints=1,
        frameon=True,
        fontsize=14,
        title_fontsize=15,
        loc="upper left",
    )

    fig.tight_layout()
    return fig

# seasonal cycles
def plot_environment(fig, spec, datafile_env, years_to_keep, site, ppfd_min, ppfd_max, panel_label):
    df_env = pd.read_csv(datafile_env)
    df_env["solar_TIMESTAMP"] = pd.to_datetime(df_env["solar_TIMESTAMP"])
    df_env.index = df_env["solar_TIMESTAMP"]

    tsm_col = "swc_shallow"
    ppfd_col = "ppfd_in"
    tsm_label = r"TSM ($\frac{cm^{3}}{cm^{3}}$)"

    # Filter by year
    df_env = df_env[df_env.index.year.isin(years_to_keep)]

    # Resample
    tair_daily = df_env["ta"].resample("D").mean()
    ppfd_daily = df_env[ppfd_col].resample("D").mean()
    tsm_daily = df_env[tsm_col].resample("D").mean()

    # Mean seasonal cycle
    tair_cycle = tair_daily.groupby(tair_daily.index.dayofyear).mean()
    ppfd_cycle = ppfd_daily.groupby(ppfd_daily.index.dayofyear).mean()
    tsm_cycle = tsm_daily.groupby(tsm_daily.index.dayofyear).mean()

    # === Plotting ===
    inner_spec = spec.subgridspec(nrows=2, ncols=1, height_ratios=[1, 4], hspace=0.05)

    ax_ppfd = fig.add_subplot(inner_spec[0])
    im = ax_ppfd.imshow(
        np.tile(ppfd_cycle.values, (10, 1)),
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[1, 365, 0, 1],
        vmin=ppfd_min,
        vmax=ppfd_max,
    )
    ax_ppfd.set_yticks([])
    ax_ppfd.set_xticks([])
    
    # Add subplot label with panel_label
    ax_ppfd.text(-0.02, 1.65, f'{panel_label})', transform=ax_ppfd.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

    ax_temp = fig.add_subplot(inner_spec[1])
    ax_tsm = ax_temp.twinx()

    ax_temp.plot(tair_cycle.index, tair_cycle.values, color="tab:red")
    ax_temp.set_ylabel("TAir (°C)", color="tab:red", fontsize=22)
    ax_temp.tick_params(axis="y", labelcolor="tab:red", labelsize=22)

    ax_tsm.plot(tsm_cycle.index, tsm_cycle.values, color="blue")
    ax_tsm.set_ylabel(tsm_label, color="blue", fontsize=22)
    ax_tsm.tick_params(axis="y", labelcolor="blue", labelsize=22)

    ax_temp.set_xlim(0, 364)
    ax_temp.set_xticks([])

    return tsm_cycle.reset_index(drop=True), ppfd_cycle.reset_index(drop=True), im


def plot_concept(fig, spec, concept, panel_label):
    ax_concept = fig.add_subplot(spec)

    # Scatter plot of VPD vs SF
    ax_concept.scatter(concept["VPD"], concept["SF"], color="k")
    ax_concept.plot(concept["VPD"], concept["SF"], ".k-", lw=4)

    # Calculate regression for morning data
    vpd = concept["VPD"]
    sap = concept["SF"]
    reg_morning = single_slope(vpd, sap)

    # Extract slope and intercept from regression results
    slope_morning = reg_morning[0]
    intercept_morning = reg_morning[1]

    # Define regression function for plotting
    f_morning = lambda x: slope_morning * x + intercept_morning

    # Define x range for plotting the regression line
    x_morning = np.array([vpd.min(), vpd.max()])

    morning_data = concept.between_time(
        vpd.idxmin().strftime("%H:%M"), sap.idxmax().strftime("%H:%M")
    )

    ax_concept.scatter(
        morning_data["VPD"], morning_data["SF"], c="#1f77b4", zorder=10, s=300
    )
    # Plot the regression line
    ax_concept.plot(x_morning, f_morning(x_morning), c="#1f77b4", lw=4)

    # Set axis limits and labels
    xmin, xmax = vpd.min(), vpd.max()
    ymin, ymax = sap.min(), sap.max()

    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")  # ensures AM/PM shows
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

    # Draw dashed lines at min and max values
    for value in [xmin, xmax]:
        ax_concept.axvline(value, color="gray", ls="--", lw=1)

    for value in [ymin, ymax]:
        ax_concept.axhline(value, color="gray", ls="--", lw=1)

    # Set axis labels
    ax_concept.set_ylabel(r"Sapflux in ($\frac{m^{3}}{s}$)", fontsize=22)
    ax_concept.set_xlabel(r"Vapour Pressure Deficit in ($kPa$)", fontsize=22)

    # Close the cycle by connecting the last point to the first point
    first_point = (concept["VPD"].iloc[0], concept["SF"].iloc[0])
    last_point = (concept["VPD"].iloc[-1], concept["SF"].iloc[-1])

    ax_concept.plot(
        [last_point[0], first_point[0]],
        [last_point[1], first_point[1]],
        color="k",
        lw=4,
        linestyle="-",
    )
    
    # Add panel label
    ax_concept.text(
        -0.05,
        1.05,
        f'{panel_label})',
        transform=ax_concept.transAxes,
        fontsize=22,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="right",
    )
    
    return ax_concept


def plot_metrics(ax, metric_cycles, daylength, site):
    # === Site-specific Y-limits ===
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

    # Process daylength data
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

    ax_slope.fill_between(
        range(len(slope_med)), slope_q25, slope_q75, color="#1f77b4", alpha=0.2
    )
    ax_area.fill_between(
        range(len(area_med)), area_q25, area_q75, color="black", alpha=0.2
    )

    ax_slope.plot(slope_med, color="#1f77b4", lw=3)
    ax_area.plot(area_med, color="black", lw=3)

    # Month ticks
    dates = slope_df.index
    dates_full = pd.to_datetime(dates.astype(str) + "-2023", format="%m-%d-%Y")
    month_starts = pd.date_range(dates_full.min(), dates_full.max(), freq="MS")
    month_idx = [dates_full.get_loc(d) + 1 for d in month_starts]

    ax_area.set_xticks(month_idx, month_starts.month)
    ax_slope.set_xticks(month_idx, month_starts.month)
    ax_area.set_xlabel("month", fontsize=22)

    # Vertical markers
    ax_area.axvline(first_12, color="gray", linestyle="--")
    ax_area.axvline(last_12, color="gray", linestyle="--")
    ax_area.axhline(0.0, color="gray")

    ax_area.set_ylabel(r"AREA ($\frac{m^{3}}{s}*kPa$)", fontsize=22, color="k")
    ax_slope.set_ylabel(r"SLOPE ($\frac{m^{3}}{s*kPa}$)", fontsize=22, color="#1f77b4")

    ax_area.set_ylim(area_ylim)
    ax_slope.set_ylim(slope_ylim)

    ax_area.tick_params(axis="both", labelsize=22)
    ax_slope.tick_params(axis="y", labelsize=22, labelcolor="#1f77b4")

    # Remove white space on left and right
    x_min = 0
    x_max = len(slope_med) - 1
    ax_area.set_xlim(x_min, x_max)
    ax_slope.set_xlim(x_min, x_max)


def plot_cycle(
    concept,
    metric_cycles,
    daylength,
    slopeframe,
    areaframe,
    site,
    datafile_env,
    ppfd_min,
    ppfd_max,
    panel_labels,  # NEW: tuple of (concept_label, env_label)
    years_to_keep=None,
    fig=None,
    spec=None,
):
    if years_to_keep is None:
        # fallback: use all available years in both frames
        years_to_keep = sorted(set(areaframe.index.year) & set(slopeframe.index.year))

    areaframe = areaframe.loc[areaframe.index.year.isin(years_to_keep)]
    slopeframe = slopeframe.loc[slopeframe.index.year.isin(years_to_keep)]

    concept_label, env_label = panel_labels

    tsm, ppfd, im = plot_environment(
        fig,
        spec[0, 1],
        datafile_env,
        years_to_keep,
        site,
        ppfd_min,
        ppfd_max,
        env_label,  # Pass env label (b, d, or f)
    )

    areaframe.index = pd.to_datetime(areaframe.index)
    slopeframe.index = pd.to_datetime(slopeframe.index)

    # Axes for concept and metrics panels
    ax_concept = plot_concept(fig, spec[:, 0], concept, concept_label)  # Pass concept label (a, c, or e)
    
    ax_concept.text(
        0.5,
        -0.05,
        site,
        transform=ax_concept.transAxes,
        fontsize=22,
        fontweight="bold",
        verticalalignment="center",
        horizontalalignment="center",
    )

    ax_metrics = fig.add_subplot(spec[1, 1])
    plot_metrics(ax_metrics, metric_cycles, daylength, site)

    return im

def plot_heatmap_summary(all_correlations):
    selected_sites = FOCUS_SITES

    # convert correlations {site: {metric: (r,p)}} into two DataFrames
    r_df = pd.DataFrame(
        {
            site: {k: v[0] for k, v in corr.items()}
            for site, corr in all_correlations.items()
        }
    ).T

    p_df = pd.DataFrame(
        {
            site: {k: v[1] for k, v in corr.items()}
            for site, corr in all_correlations.items()
        }
    ).T

    r_df = r_df.replace({None: np.nan})
    p_df = p_df.replace({None: np.nan})

    # add average row
    r_df.loc["all"] = r_df.mean(skipna=True)
    # Add placeholder p-value row
    p_df.loc["all"] = np.nan

    # filter
    r_df = r_df.loc[selected_sites + ["all"]]
    p_df = p_df.loc[selected_sites + ["all"]]

    # significance markers
    sig = p_df.applymap(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    # build annotation matrix
    annot = r_df.copy().astype(str)
    for row in annot.index:
        for col in annot.columns:
            val = r_df.loc[row, col]
            star = sig.loc[row, col]
            annot.loc[row, col] = f"{val:.2f}{star}" if not np.isnan(val) else ""

    # plot
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(
        r_df.astype(float),
        annot=annot,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add subplot label
    ax.text(-0.05, 1.05, 'd)', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()

    # Add significance legend
    legend_text = "***  p < 0.001\n**    p < 0.01\n*      p < 0.05\nns    p ≥ 0.05"

    plt.gcf().text(0.04, 0.15, legend_text, fontsize=12, ha="left", va="top")

    return fig, ax

# TODO: leave it out, replaced by plot_heatmap_parameters
def plot_heatmap_summary_tax(all_correlations, metadata_path):
        # Load site metadata
    site_metadata_df = pd.read_csv(metadata_path)
    site_metadata_df['site_prefix'] = site_metadata_df['Site'].str[:7]

    # Aggregate metadata by site prefix
    metadata_agg = site_metadata_df.groupby('site_prefix').agg({
        'plant_group': 'first',
        'biome': 'first',
    }).reset_index()

    prefix_to_group = dict(zip(metadata_agg['site_prefix'], metadata_agg['plant_group']))
    prefix_to_biome = dict(zip(metadata_agg['site_prefix'], metadata_agg['biome']))

    # Aggregate correlations by prefix
    r_records = {}
    p_records = {}
    # for site, corr in all_correlations.items():
    #     prefix = site[:7]
    for site, corr in all_correlations.items():
        prefix = site   # keep full site / treatment ID
        if prefix not in r_records:
            r_records[prefix] = {}
            p_records[prefix] = {}
        for metric, (r, p) in corr.items():
            if metric not in r_records[prefix]:
                r_records[prefix][metric] = []
                p_records[prefix][metric] = []
            if r is not None:
                r_records[prefix][metric].append(r)
            if p is not None:
                p_records[prefix][metric].append(p)

    r_data = {}
    p_data = {}
    for prefix in r_records:
        r_data[prefix] = {m: np.mean(vals) if vals else np.nan for m, vals in r_records[prefix].items()}
        p_data[prefix] = {m: np.mean(vals) if vals else np.nan for m, vals in p_records[prefix].items()}

    r_df = pd.DataFrame(r_data).T.replace({None: np.nan})
    p_df = pd.DataFrame(p_data).T.replace({None: np.nan})

    # Add metadata
    r_df['plant_group'] = r_df.index.map(prefix_to_group)
    r_df['biome'] = r_df.index.map(prefix_to_biome)
    p_df['plant_group'] = p_df.index.map(prefix_to_group)
    p_df['biome'] = p_df.index.map(prefix_to_biome)

    # Get data columns
    data_columns = [c for c in r_df.columns if c not in ['plant_group', 'biome']]

    # Split by plant group
    groups = ['angiosperm', 'gymnosperm', 'mixed']
    group_data = {}

    for group in groups:
        mask = r_df['plant_group'] == group
        if mask.any():
            r_group = r_df.loc[mask].sort_values('biome')[data_columns]
            p_group = p_df.loc[mask].sort_values('biome')[data_columns]

            # Add group mean
            r_group.loc['MEAN'] = r_group.mean(skipna=True)
            p_group.loc['MEAN'] = np.nan

            group_data[group] = (r_group, p_group)

    # Calculate overall mean
    r_all = r_df[data_columns]
    r_overall_mean = r_all.mean(skipna=True)

    # Create subplots
    n_groups = len(group_data)
    fig, axes = plt.subplots(n_groups + 1, 1, figsize=(10, 2.5 * (n_groups + 1)),
                              gridspec_kw={'height_ratios': [len(group_data[g][0]) for g in group_data] + [1]})

    if n_groups + 1 == 1:
        axes = [axes]

    # Plot each group
    for idx, (group, (r_group, p_group)) in enumerate(group_data.items()):
        ax = axes[idx]

        # Build annotation matrix
        sig = p_group.map(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        )
        annot = r_group.copy().astype(str)
        for row in annot.index:
            for col in annot.columns:
                val = r_group.loc[row, col]
                star = sig.loc[row, col]
                annot.loc[row, col] = f'{val:.2f}{star}' if not np.isnan(val) else ''

        sns.heatmap(
            r_group.astype(float),
            annot=annot,
            fmt='',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            cbar=idx == 0,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8} if idx == 0 else {},
            ax=ax,
            linewidths=0.5,
            linecolor='white'
        )

        ax.set_title(group.upper(), fontsize=12, fontweight='bold', loc='left')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        if idx < n_groups - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Plot overall mean
    ax_mean = axes[-1]
    r_mean_df = pd.DataFrame([r_overall_mean], index=['ALL MEAN'])
    p_mean_df = pd.DataFrame([[np.nan] * len(data_columns)], index=['ALL MEAN'], columns=data_columns)

    # Build annotation for mean
    sig_mean = p_mean_df.map(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    annot_mean = r_mean_df.copy().astype(str)
    for row in annot_mean.index:
        for col in annot_mean.columns:
            val = r_mean_df.loc[row, col]
            star = sig_mean.loc[row, col]
            annot_mean.loc[row, col] = f'{val:.2f}{star}' if not np.isnan(val) else ''

    sns.heatmap(
        r_mean_df.astype(float),
        annot=annot_mean,
        fmt='',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        cbar=False,
        ax=ax_mean,
        linewidths=0.5,
        linecolor='white'
    )

    ax_mean.set_title('OVERALL', fontsize=12, fontweight='bold', loc='left')
    ax_mean.set_yticklabels(ax_mean.get_yticklabels(), rotation=0, fontsize=9)
    ax_mean.set_xticklabels(ax_mean.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()

    # Add significance legend
    legend_text = '*** p < 0.001\n**  p < 0.01\n*   p < 0.05'
    fig.text(0.01, 0.01, legend_text, fontsize=8, ha='left', va='bottom',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    return fig, axes

def plot_heatmap_summary_param(all_correlations, metadata_path):
    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    site_metadata_df = pd.read_csv(metadata_path)
    site_metadata_df["site_prefix"] = site_metadata_df["Site"].str[:7]

    # ------------------------------------------------------------------
    # Collect correlations per SITE PREFIX (aggregate plots)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Average across plots (simple mean)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Sort alphabetically
    # ------------------------------------------------------------------
    r_df = r_df.sort_index()
    p_df = p_df.loc[r_df.index]

    # ------------------------------------------------------------------
    # Enforce x-axis metric order
    # ------------------------------------------------------------------
    base_vars = ["TSM", "TAir", "PPFD"]
    metric_order = (
        [f"SLOPE-{v}" for v in base_vars] +
        [f"AREA-{v}" for v in base_vars]
    )

    data_columns = [m for m in metric_order if m in r_df.columns]
    r_df_plot = r_df[data_columns]
    p_df_plot = p_df[data_columns]

    # ------------------------------------------------------------------
    # Build annotations with significance asterisks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Y-axis labels WITH plot-count indicator
    # ------------------------------------------------------------------
    yticklabels = []
    for idx in r_df_plot.index:
        n = plot_counts.get(idx, 1)
        label = f"{idx} (n={n})" if n > 1 else idx
        yticklabels.append(label)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, max(4, len(r_df_plot) * 0.35)))

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("white")

    sns.heatmap(
        r_df_plot.astype(float),
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Correlation", "shrink": 0.6, "aspect": 30},
        ax=ax
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

def plot_percentiles(anomalies, code, fig=None, outer_spec=None):
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
            # if len(cluster_df) < 8:
            #     continue
            stddev_TAir, stddev_TSM = cluster_df[["TAir", "TSM"]].std().to_numpy()
            slope_mean = cluster_df["SLOPE"].mean()
            area_mean = cluster_df["AREA"].mean()
            centroid_x = cluster_df["TAir"].mean()
            centroid_y = cluster_df["TSM"].mean()

            all_cluster_slopes.append(slope_mean)
            all_cluster_slopes.extend(cluster_df["SLOPE"].values)
            all_cluster_areas.append(area_mean)
            all_cluster_areas.extend(cluster_df["AREA"].values)

            site_clusters.append(
                {
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
                }
            )

        cluster_summaries[site] = site_clusters

    # Color normalization: ensure zero-centered colormap
    slope_min, slope_max = min(all_cluster_slopes), max(all_cluster_slopes)
    area_min, area_max = min(all_cluster_areas), max(all_cluster_areas)
    cmap = cm.get_cmap("RdBu")

    slope_abs_max = max(abs(slope_min), abs(slope_max))
    norm = mcolors.TwoSlopeNorm(vmin=-slope_abs_max, vcenter=0, vmax=slope_abs_max)

    # Limits
    all_tair = np.concatenate([anomalies[site]["TAir"].values for site in sites])
    all_tsm = np.concatenate([anomalies[site]["TSM"].values for site in sites])
    xlim = (all_tair.min() - 2, all_tair.max() + 1)
    ylim = (all_tsm.min(), all_tsm.max() + 0.03)

    axes = []
    for idx, site in enumerate(sites):
        ax = fig.add_subplot(spec[0, idx + 1])
        axes.append(ax)
        if idx > 0:
            ax.tick_params(labelleft=False)

        df = anomalies[site]

        if code == "supp":
            # Background points as gray crosses
            ax.scatter(
                df["TAir"], df["TSM"], s=10, c="gray", marker="x", alpha=0.5, zorder=0
            )

        for cluster in cluster_summaries[site]:
            points = cluster["points"]
            slopes = cluster["slopes"]
            areas = cluster["areas"]
            label = cluster["label"]

            # Individual cluster point sizes and colors
            sizes = 900 * (areas - area_min) / (area_max - area_min + 1e-6) + 90
            colors = [cmap(norm(s)) for s in slopes]
            ax.scatter(
                points[:, 1],
                points[:, 0],
                c=colors,
                s=sizes,
                alpha=0.6,
                marker="o",
                edgecolor="k",
                linewidths=2,
                zorder=2,
            )

            # Annotate cluster label at the centroid of its points
            centroid_x = points[:, 1].mean()
            centroid_y = points[:, 0].max() + 0.02
            # Default offsets
            dx, dy = 0, 0

            # Special case
            if site == "GUF_GUY_GUY" and label == "cold & wet":
                dx, dy = -4, -0.04
            if site == "GUF_GUY_GUY" and label == "cold & dry":
                dx, dy = -4, -0.04

            ax.text(
                centroid_x + dx,
                centroid_y + dy,
                str(label),
                fontsize=16,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                ),
                zorder=3,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Site: {site}", fontsize=32)
        ax.tick_params(labelsize=22)
        
        # Add subplot labels
        subplot_labels = ['a)', 'b)', 'c)']
        if idx < len(subplot_labels):
            ax.text(-0.05, 1.1, subplot_labels[idx], transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

    axes[0].set_ylabel(r"TSM in $\frac{cm^{3}}{cm^{3}}$", fontsize=26)
    axes[1].set_xlabel(r"TAir in $^\circ$C", fontsize=26, labelpad=15)
    # Create custom legend for point sizes (area)
    # Choose a few representative area values (e.g., small, medium, large)
    area_legend_values = [area_min, (area_min + area_max) / 2, area_max]
    size_legend_values = [
        900 * (a - area_min) / (area_max - area_min + 1e-6) + 90
        for a in area_legend_values
    ]

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{a:.2f}",
            markerfacecolor="gray",
            markersize=np.sqrt(s),
            markeredgecolor="black",
        )  # use sqrt because size in scatter is area
        for a, s in zip(area_legend_values, size_legend_values)
    ]
    ax_legend = fig.add_subplot(spec[0, 0])
    ax_legend.axis("off")
    # Add legend to the left of the first axis
    ax_legend.legend(
        handles=legend_elements,
        title="nAREA",
        loc="center left",
        bbox_to_anchor=(0.25, 0.5),  # adjust leftward placement as needed
        fontsize=20,
        title_fontsize=22,
        frameon=False,
    )

    return cmap, norm


def plot_dailycycle(groups_dicts, ax, quadrant_linestyles):
    gr_data = []
    for groups_dict in groups_dicts:
        for cluster_id, group_df in groups_dict.items():
            mask = ~np.isnan(group_df["VPD"]) & ~np.isnan(group_df["SF"])
            group = group_df[mask]
            if group.empty:
                continue

            points = np.column_stack((group["VPD"], group["SF"]))

            # Extract quadrant number
            quadrant = ""
            for c in cluster_id[1:]:
                if c.isdigit():
                    quadrant += c
                else:
                    break
            linestyle = quadrant_linestyles[cluster_id]

            # Draw convex hull if enough points
            if len(points) > 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(
                        points[simplex, 0],
                        points[simplex, 1],
                        color="black",
                        linestyle=linestyle,
                        lw=5,
                    )

            vpd = group["VPD"]
            sap = group["SF"]
            # Select points between min VPD and max SF
            start_idx = vpd.idxmin()
            end_idx = sap.idxmax()

            if start_idx <= end_idx:
                # normal case
                mask_slice = (vpd.index >= start_idx) & (vpd.index <= end_idx)
            else:
                # overnight case: select indices >= start OR <= end
                mask_slice = (vpd.index >= start_idx) | (vpd.index <= end_idx)

            vpd_slice = vpd[mask_slice]
            sap_slice = sap[mask_slice]
            reg = single_slope(vpd_slice, sap_slice)

            # extract slope and intercept (first two elements)
            slope, intercept = reg[:2]
            x_vals = np.array([vpd_slice.min(), vpd_slice.max()])
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color="#1f77b4", linestyle=linestyle, lw=5)

            gr_data.append(group)

    return gr_data


def plot_hysteresis_main(mean_cycles, fig=None, outer_spec=None):
    quadrant_linestyles = {
        "hot & wet": "solid",
        "cold & wet": "dashed",
        "cold & dry": "dashdot",
        "hot & dry": "dotted",
    }
    site_initials = sorted(set(k[0] for k in mean_cycles if k != "all"))
    sites = sorted(set(name for name in mean_cycles if name != "all"))

    n_sites = len(site_initials)
    total_cols = max(n_sites, 4)  # force at least 4 columns

    if fig is None or outer_spec is None:
        fig = plt.figure(figsize=(6 * total_cols, 7))
        spec = gridspec.GridSpec(ncols=total_cols, nrows=1)
    else:
        spec = outer_spec.subgridspec(1, total_cols)

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
            
            # Add subplot labels
            subplot_labels = ['d)', 'e)', 'f)']
            if j < len(subplot_labels):
                ax.text(-0.05, 1.1, subplot_labels[j], transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')
        else:
            # First column: legend
            ax = fig.add_subplot(spec[0, 0])
            ax.axis("off")  # Hide axes
            quadrant_labels = ["hot & wet", "hot & dry", "cold & dry", "cold & wet"]

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=quadrant_linestyles[label],
                    lw=3,
                    label=label,
                )
                for label in quadrant_labels
            ]

            ax.legend(
                handles=legend_elements,
                loc="center",
                fontsize=32,
                handlelength=2,
                frameon=False,
                labelspacing=1.5,
            )

        axs.append(ax)

    # Shared labels
    axs[0].set_ylabel(r"Sapflux (SF) in ($\frac{m^{3}}{s}$)", fontsize=26)
    axs[n_sites // 2].set_xlabel(r"VPD in $kPa$", fontsize=26)


def plot_patterns(extreme_anomalies, mean_cycles, code):
    fig = plt.figure(figsize=(30, 15))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Top: cluster centroid plot based on percentiles
    gray_cmap, grayscale_norm = plot_percentiles(
        extreme_anomalies, code, fig=fig, outer_spec=outer[0]
    )

    # Bottom: hysteresis loops
    plot_hysteresis_main(
        mean_cycles, fig=fig, outer_spec=outer[1]
    )  # remove colors and switch to different linestyles

    # Add shared colorbar for slope anomaly
    sm = plt.cm.ScalarMappable(cmap=gray_cmap, norm=grayscale_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.54, 0.015, 0.35])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("sSLOPE", fontsize=26)
    cbar.ax.tick_params(labelsize=22)

    return fig


def plot_rates(site, fig, spec):
    # File paths for environmental and sap flux data
    environment = CSV_ROOT / f"plant/{site}_env_data.csv"
    sapflux = CSV_ROOT / f"plant/{site}_sapf_data.csv"

    # Get subdaily VPD and sap flux data
    subdaily = get_subdaily(sapflux, environment)
    subdaily.index = pd.to_datetime(subdaily.index)

    # RESAMPLINGS for hysteresis and ranges plot
    resamplings = get_resampled(subdaily)

    # Define specific sample rates in order
    sample_rates = ["8TPD", "4TPD", "3TPDnight", "3TPDday"]

    # Map sample rates to their corresponding subplot positions
    positions = [
        spec[0, 0],  # T8PD
        spec[1, 0],  # T3PDday
        spec[1, 1],  # T3PDnight
        spec[0, 1],  # T4PD
    ]

    # Loop through the specified sample rates and their mapped positions
    for i, samplerate in enumerate(sample_rates):
        resampling = resamplings[i]  # Directly access resampling by index

        if resampling is None:
            continue

        concept = (
            resampling[["SF", "VPD"]]
            .groupby(resampling[["SF", "VPD"]].index.hour)
            .mean()
        )

        # Append the first row to the end of the DataFrame to close the cycle
        first_row = concept.iloc[[0]]
        concept = pd.concat([concept, first_row])

        # Create a new subplot for each sample rate at the specified position
        ax_inner = fig.add_subplot(positions[i])  # Use mapped position from the list

        # Plot hourly reference line in grey first (behind)
        hourly_data = subdaily[["SF", "VPD"]].groupby(subdaily.index.hour).mean()
        ax_inner.plot(
            hourly_data["VPD"],
            hourly_data["SF"],
            color="grey",
            linestyle="-",
            lw=2,
            label="Hourly Reference",
        )

        # Calculate min and max values for hourly data
        hourly_vpdmin = hourly_data["VPD"].min()
        hourly_vpdmax = hourly_data["VPD"].max()
        hourly_sapmin = hourly_data["SF"].min()
        hourly_sapmax = hourly_data["SF"].max()

        # Plotting the sample rate in black on top of hourly reference
        ax_inner.scatter(concept["VPD"], concept["SF"], color="k", label=samplerate)
        ax_inner.plot(concept["VPD"], concept["SF"], color="k", lw=3)

        vpd = concept["VPD"]
        sap = concept["SF"]

        xmax, ymax = vpd.max(), sap.max()
        xmin, ymin = vpd.min(), sap.min()

        # Determine subplot row/column position (0 or 1)
        row = 0 if i in [0, 3] else 1  # based on your mapping
        col = 0 if i in [0, 1] else 1

        # Add ticks only on outer edges
        if row == 1:  # bottom row → show x ticks
            ax_inner.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=14)
            ax_inner.set_xlabel("VPD (kPa)", fontsize=22)
        else:  # top row → hide x ticks
            ax_inner.tick_params(axis="x", bottom=False, labelbottom=False)

        if col == 0:  # left column → show y ticks
            ax_inner.tick_params(axis="y", left=True, labelleft=True, labelsize=14)
            ax_inner.set_ylabel(r"Sapflux ($\frac{m^{3}}{s}$)", fontsize=22)
        else:  # right column → hide y ticks
            ax_inner.tick_params(axis="y", left=False, labelleft=False)

        # Draw dashed lines at min and max values for SF and VPD
        ax_inner.axvline(xmin, ls="--", lw=1.5, color="black")
        ax_inner.axvline(xmax, ls="--", lw=1.5, color="black")
        ax_inner.axhline(ymin, ls="--", lw=1.5, color="black")
        ax_inner.axhline(ymax, ls="--", lw=1.5, color="black")

        # Draw dashed lines for hourly min/max values
        ax_inner.axvline(hourly_vpdmin, ls="--", lw=1.5, color="gray")
        ax_inner.axvline(hourly_vpdmax, ls="--", lw=1.5, color="gray")
        ax_inner.axhline(hourly_sapmin, ls="--", lw=1.5, color="gray")
        ax_inner.axhline(hourly_sapmax, ls="--", lw=1.5, color="gray")

        # Add title for each sample rate
        ax_inner.set_title(samplerate, fontsize=22)


def plot_coefficients(slope_longterm, area_longterm, fig, spec):
    ax = fig.add_subplot(spec[:, 2])

    # Prepare long-form data for seaborn
    slope_melted = slope_longterm.melt(var_name="SampleRate", value_name="R2")
    slope_melted["Type"] = "Slope"

    area_melted = area_longterm.melt(var_name="SampleRate", value_name="R2")
    area_melted["Type"] = "Area"

    data = pd.concat([slope_melted, area_melted], ignore_index=True)
    data = data.dropna()

    # --- Print medians and IQR ends (25th and 75th percentiles) ---
    grouped = data.groupby(["SampleRate", "Type"])["R2"]
    stats = grouped.agg(
        median="median", Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75)
    )

    # Plot boxplot
    sns.boxplot(
        data=data,
        x="SampleRate",
        y="R2",
        hue="Type",
        palette={"Slope": "#1f77b4", "Area": "#2c2c2c"},
        dodge=True,
        ax=ax,
    )

    # Aesthetic tweaks
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
    fig = plt.figure(figsize=(18, 10))

    spec = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        left=0.05,
        right=0.95,
        height_ratios=[1, 1],
        width_ratios=[1, 1, 2],
        hspace=0.2,
        wspace=0.2,
    )

    # Draw left and right panels
    plot_rates("FRA_PUE", fig, spec)
    plot_coefficients(slope_longterm, area_longterm, fig, spec)

    # -------------------------------------------------------
    # PANEL LABELS
    # -------------------------------------------------------

    # a) for the left 2×2 block of rates plots
    fig.text(
        0.03,  # left margin
        0.97,  # near top of figure
        "a)",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # b) for the right large boxplot panel
    fig.text(
        0.53,  # shifted into the right panel column
        0.97,
        "b)",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_distributions_focus(extreme_anomalies):
    """Plot distributions of SLOPE (upper row) and AREA (lower row) of percentile clusters per focus site."""
    # Define linestyles for clusters
    quadrant_linestyles = {
        "hot & wet": "solid",
        "cold & wet": "dashed",
        "cold & dry": "dashdot",
        "hot & dry": "dotted",
    }

    # Define colors
    slope_color = "#1f77b4"
    area_color = "#2c2c2c"

    # Get site names
    sites = list(extreme_anomalies.keys())
    n_sites = len(sites)

    # Create figure with 2 rows (SLOPE and AREA) and n_sites columns
    fig, axes = plt.subplots(2, n_sites, figsize=(6 * n_sites, 10))

    # Ensure axes is 2D array even with single site
    if n_sites == 1:
        axes = axes.reshape(2, 1)

    # Plot for each site
    for col_idx, site in enumerate(sites):
        df = extreme_anomalies[site]

        # Get unique clusters (excluding None)
        clusters = df[df["Cluster"].notna()]["Cluster"].unique()

        # --- SLOPE distributions (upper row) ---
        ax_slope = axes[0, col_idx]

        for cluster in clusters:
            cluster_data = df[df["Cluster"] == cluster]["SLOPE"].values

            if len(cluster_data) > 1:
                # Plot KDE for clusters with multiple points
                try:
                    kde = stats.gaussian_kde(cluster_data)
                    x_range = np.linspace(
                        cluster_data.min() - 0.1, cluster_data.max() + 0.1, 200
                    )
                    density = kde(x_range)
                    ax_slope.plot(
                        x_range,
                        density,
                        linestyle=quadrant_linestyles[cluster],
                        color=slope_color,
                        linewidth=4,
                        label=cluster,
                        alpha=0.7,
                        zorder=3,
                    )

                    # Plot points on the distribution curve
                    point_densities = kde(cluster_data)
                    ax_slope.scatter(
                        cluster_data,
                        point_densities,
                        alpha=0.6,
                        s=30,
                        color="crimson",
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=2,
                    )
                except:
                    # If KDE fails, just plot points at y=0
                    ax_slope.scatter(
                        cluster_data,
                        np.zeros_like(cluster_data),
                        alpha=0.6,
                        s=30,
                        color="crimson",
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=2,
                    )
            else:
                # For single points, plot at y=0 with a marker
                ax_slope.scatter(
                    cluster_data,
                    [0],
                    alpha=0.6,
                    s=30,
                    color="crimson",
                    edgecolors="white",
                    linewidth=0.5,
                    label=cluster,
                    zorder=2,
                )

        ax_slope.set_xlabel("sSLOPE", fontsize=13, fontweight="bold")
        ax_slope.set_ylabel("Density", fontsize=13)
        ax_slope.set_title(f"{site}", fontsize=14, fontweight="bold")
        ax_slope.legend(loc="best", fontsize=11)
        ax_slope.grid(True, alpha=0.3, linestyle="--")
        ax_slope.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        
        # Add subplot labels for top row
        subplot_labels_top = ['p', 'q', 'r']
        if col_idx < len(subplot_labels_top):
            ax_slope.text(-0.1, 1.1, subplot_labels_top[col_idx], transform=ax_slope.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

        # --- AREA distributions (lower row) ---
        ax_area = axes[1, col_idx]

        for cluster in clusters:
            cluster_data = df[df["Cluster"] == cluster]["AREA"].values

            if len(cluster_data) > 1:
                # Plot KDE for clusters with multiple points
                try:
                    kde = stats.gaussian_kde(cluster_data)
                    x_range = np.linspace(
                        cluster_data.min() - 0.05, cluster_data.max() + 0.05, 200
                    )
                    density = kde(x_range)
                    ax_area.plot(
                        x_range,
                        density,
                        linestyle=quadrant_linestyles[cluster],
                        color=area_color,
                        linewidth=4,
                        label=cluster,
                        alpha=0.7,
                        zorder=3,
                    )

                    # Plot points on the distribution curve
                    point_densities = kde(cluster_data)
                    ax_area.scatter(
                        cluster_data,
                        point_densities,
                        alpha=0.6,
                        s=30,
                        color="crimson",
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=2,
                    )
                except:
                    # If KDE fails, just plot points at y=0
                    ax_area.scatter(
                        cluster_data,
                        np.zeros_like(cluster_data),
                        alpha=0.6,
                        s=30,
                        color="crimson",
                        edgecolors="white",
                        linewidth=0.5,
                        zorder=2,
                    )
            else:
                # For single points, plot at y=0 with a marker
                ax_area.scatter(
                    cluster_data,
                    [0],
                    alpha=0.6,
                    s=30,
                    color="crimson",
                    edgecolors="white",
                    linewidth=0.5,
                    label=cluster,
                    zorder=2,
                )

        ax_area.set_xlabel("nAREA", fontsize=13, fontweight="bold")
        ax_area.set_ylabel("Density", fontsize=13)
        ax_area.legend(loc="best", fontsize=11)
        ax_area.grid(True, alpha=0.3, linestyle="--")
        ax_area.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        
        # Add subplot labels for bottom row
        subplot_labels_bottom = ['s', 't', 'u']
        if col_idx < len(subplot_labels_bottom):
            ax_area.text(-0.1, 1.1, subplot_labels_bottom[col_idx], transform=ax_area.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

        # For SLOPE legend
        legend_slope = ax_slope.legend(loc="best", fontsize=11)
        for handle in legend_slope.legend_handles:
            handle.set_linewidth(2.5)
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(8)
        # Make legend lines longer
        for line in legend_slope.get_lines():
            line.set_linewidth(2.5)
        ax_slope.legend(loc="best", fontsize=11, handlelength=3.5, handleheight=1.5)

        # For AREA legend
        legend_area = ax_area.legend(loc="best", fontsize=11)
        for handle in legend_area.legend_handles:
            handle.set_linewidth(2.5)
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(8)
        # Make legend lines longer
        for line in legend_area.get_lines():
            line.set_linewidth(2.5)
        ax_area.legend(loc="best", fontsize=11, handlelength=3.5, handleheight=1.5)

    plt.tight_layout()

    return fig


def plot_distribution_TSM_TAir(anomalies_TAir_TSM):
    """
    Plot distributions of TAir and TSM anomalies for each SLOPE-AREA combination.
    """
    # Define colors for each combination
    combination_colors = {
        "high SLOPE & low AREA": "#1f77b4",  # blue
        "low SLOPE & high AREA": "#d62728",  # red
        "high SLOPE & high AREA": "#2c2c2c",  # black
        "low SLOPE & low AREA": "#7f7f7f",  # grey
    }

    # Create figure with 2 panels (TAir and TSM)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique combinations
    combinations = anomalies_TAir_TSM["combination"].unique()

    # --- TAir anomaly distribution (left panel) ---
    ax_tair = axes[0]

    for combination in combinations:
        data = anomalies_TAir_TSM[anomalies_TAir_TSM["combination"] == combination][
            "TAir_anomaly"
        ]

        if len(data) > 1:
            # Plot KDE
            from scipy import stats

            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min() - 1, data.max() + 1, 300)
                density = kde(x_range)

                # Calculate mean
                mean_val = data.mean()

                # SECOND: Plot KDE line (on top) with mean in label
                ax_tair.plot(
                    x_range,
                    density,
                    color=combination_colors[combination],
                    linewidth=2.5,
                    label=f"{combination} (μ={mean_val:.2f}°C)",
                    alpha=0.8,
                    zorder=3,
                )

                # Add a subtle fill under the curve
                ax_tair.fill_between(
                    x_range,
                    density,
                    alpha=0.2,
                    color=combination_colors[combination],
                    zorder=1,
                )

                # Add vertical line at the mean
                ax_tair.axvline(
                    x=mean_val,
                    color=combination_colors[combination],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    zorder=3,
                )
            except:
                pass

    ax_tair.set_xlabel("TAir Anomaly (°C)", fontsize=14, fontweight="bold")
    ax_tair.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax_tair.legend(loc="best", fontsize=11, handlelength=1, handleheight=1.5)
    ax_tair.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_tair.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    
    # Add subplot label
    ax_tair.text(-0.1, 1.05, 'v', transform=ax_tair.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    # --- TSM anomaly distribution (right panel) ---
    ax_tsm = axes[1]

    for combination in combinations:
        data = anomalies_TAir_TSM[anomalies_TAir_TSM["combination"] == combination][
            "TSM_anomaly"
        ]

        if len(data) > 1:
            # Plot KDE
            from scipy import stats

            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min() - 0.01, data.max() + 0.01, 300)
                density = kde(x_range)

                # Calculate mean
                mean_val = data.mean()

                # SECOND: Plot KDE line (on top) with mean in label
                ax_tsm.plot(
                    x_range,
                    density,
                    color=combination_colors[combination],
                    linewidth=2.5,
                    label=f"{combination} (μ={mean_val:.3f})",
                    alpha=0.8,
                    zorder=3,
                )

                # Add a subtle fill under the curve
                ax_tsm.fill_between(
                    x_range,
                    density,
                    alpha=0.2,
                    color=combination_colors[combination],
                    zorder=1,
                )

                # Add vertical line at the mean
                ax_tsm.axvline(
                    x=mean_val,
                    color=combination_colors[combination],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    zorder=3,
                )
            except:
                pass

    ax_tsm.set_xlabel("TSM Anomaly (cm³/cm³)", fontsize=14, fontweight="bold")
    ax_tsm.set_ylabel("Density", fontsize=14, fontweight="bold")
    # ax_tsm.set_title(
    #     "Soil Moisture Anomaly Distribution", fontsize=13, fontweight="bold"
    # )
    ax_tsm.legend(loc="best", fontsize=11, handlelength=1, handleheight=1.5)
    ax_tsm.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_tsm.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    
    # Add subplot label
    ax_tsm.text(-0.1, 1.05, 'w', transform=ax_tsm.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    return fig


def plot_distributions_SLOPE_AREA(slope_area_distributions):
    # Define colors for each combination
    combination_colors = {
        "high TAir & low TSM": "#d62728",  # red
        "low TAir & high TSM": "#1f77b4",  # blue
        "high TAir & high TSM": "#2c2c2c",  # black
        "low TAir & low TSM": "#7f7f7f",  # grey
    }

    # Create figure with 2 panels (SLOPE and AREA)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique combinations
    combinations = slope_area_distributions["combination"].unique()

    # --- SLOPE distribution (left panel) ---
    ax_slope = axes[0]

    for combination in combinations:
        data = slope_area_distributions[
            slope_area_distributions["combination"] == combination
        ]["SLOPE"]

        if len(data) > 1:
            # Plot KDE
            from scipy import stats

            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min() - 0.1, data.max() + 0.1, 300)
                density = kde(x_range)

                # Calculate median
                median_val = data.median()

                # Plot KDE line with mean in label
                ax_slope.plot(
                    x_range,
                    density,
                    color=combination_colors[combination],
                    linewidth=2.5,
                    label=f"{combination}",
                    alpha=0.8,
                    zorder=3,
                )

                # Add a subtle fill under the curve
                ax_slope.fill_between(
                    x_range,
                    density,
                    alpha=0.2,
                    color=combination_colors[combination],
                    zorder=1,
                )

            except:
                pass

    ax_slope.set_xlabel("sSLOPE", fontsize=14, fontweight="bold")
    ax_slope.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax_slope.legend(loc="best", fontsize=11, handlelength=1, handleheight=1.5)
    ax_slope.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_slope.axvline(
        x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1
    )
    
    # Add subplot label
    ax_slope.text(-0.1, 1.05, 'a)', transform=ax_slope.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    # --- AREA distribution (right panel) ---
    ax_area = axes[1]

    for combination in combinations:
        data = slope_area_distributions[
            slope_area_distributions["combination"] == combination
        ]["AREA"]

        if len(data) > 1:
            # Plot KDE
            from scipy import stats

            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min() - 0.01, data.max() + 0.01, 300)
                density = kde(x_range)

                # Calculate mean
                median_val = data.median()

                # Plot KDE line with mean in label
                ax_area.plot(
                    x_range,
                    density,
                    color=combination_colors[combination],
                    linewidth=2.5,
                    label=f"{combination}",
                    alpha=0.8,
                    zorder=3,
                )

                # Add a subtle fill under the curve
                ax_area.fill_between(
                    x_range,
                    density,
                    alpha=0.2,
                    color=combination_colors[combination],
                    zorder=1,
                )

            except:
                pass

    ax_area.set_xlabel("nAREA", fontsize=14, fontweight="bold")
    ax_area.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax_area.legend(loc="best", fontsize=11, handlelength=1, handleheight=1.5)
    ax_area.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_area.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    
    # Add subplot label
    ax_area.text(-0.1, 1.05, 'b)', transform=ax_area.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    return fig
