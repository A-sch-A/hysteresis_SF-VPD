# analyse.py — site-level processing and figure generation.

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from config import CSV_ROOT, FIG_DIR, GROWING_SEASON_DAYLENGTH, GROWING_SEASON_TEMP, get_logger
from util import (
    get_anomalies_TAIR_TSM,
    get_classification,
    get_cluster_cycles,
    get_cluster_dates,
    get_combined_variables,
    get_concept,
    get_correlations_to_hourly,
    get_metadata_per_site,
    get_percentiles,
    get_seasonal,
    get_seasonal_cycle_correlations,
    get_site_data,
    get_standardized_metrics,
    fix_site_and_update_counters,
)
from visualization import (
    plot_classification,
    plot_cycle,
    plot_distributions_SLOPE_AREA,
    plot_heatmap_summary_param,
    plot_patterns,
    plot_srs,
)

log = get_logger(__name__)


@dataclass
class SiteOutputs:
    site: str
    subdaily: pd.DataFrame
    daily: pd.DataFrame
    slopeframe: pd.DataFrame
    areaframe: pd.DataFrame
    dl_frame: pd.DataFrame
    growing_season: pd.DataFrame
    growing_season_complete: pd.DataFrame
    seasonal_corrs: Dict[str, Any]
    corr_slope_to_hourly: pd.Series
    corr_area_to_hourly: pd.Series


def process_all_sites(
    input_data: pd.DataFrame, sites: List[str], focus_sites: List[str]
) -> Dict[str, Any]:
    # iterate over all sites, compute metrics, correlations, and growing-season subsets
    site_outputs: Dict[str, SiteOutputs] = {}
    growing_season_list: List[pd.DataFrame] = []
    growing_season_anom_list = []
    global_subdaily_list: List[pd.DataFrame] = []
    subdaily_focus: Dict[str, pd.DataFrame] = {}
    slope_coefficients_to_hourly = pd.DataFrame()
    area_coefficients_to_hourly = pd.DataFrame()
    all_seasonal_correlations: Dict[str, Any] = {}
    focus_growing_season: Dict[str, pd.DataFrame] = {}
    combined_all: List[pd.DataFrame] = []

    # collect metadata across all sites
    site_metadata = get_metadata_per_site(CSV_ROOT, sites)

    counters = {
        "combined_sites": set(),
        "reduced_sites": set(),
        "combined_sites_unique": set(),
        "reduced_sites_unique": set(),
    }

    for site in sites:
        # load pre-computed site data
        subdaily, daily, slopeframe, areaframe, dl_frame = get_site_data(
            site, input_data
        )
        global_subdaily_list.append(subdaily)

        slopeframe.index = pd.to_datetime(slopeframe.index)
        areaframe.index = pd.to_datetime(areaframe.index)

        # seasonal-cycle correlations with environmental drivers
        seasonal_corrs = get_seasonal_cycle_correlations(site, slopeframe, areaframe)
        all_seasonal_correlations[site] = seasonal_corrs

        # R-squared of resampled metrics against hourly reference
        corr_slope_to_hourly, corr_area_to_hourly = get_correlations_to_hourly(
            slopeframe, areaframe
        )

        # accumulate R-squared values across sites
        slope_df = corr_slope_to_hourly.to_frame().T
        area_df = corr_area_to_hourly.to_frame().T
        slope_coefficients_to_hourly = pd.concat(
            [slope_coefficients_to_hourly, slope_df], ignore_index=True
        )
        area_coefficients_to_hourly = pd.concat(
            [area_coefficients_to_hourly, area_df], ignore_index=True
        )

        # merge SLOPE/AREA with environmental drivers
        # sites drop from ~45 to ~25 when TSM or PPFD is unavailable
        df_combined, df_reduced = get_combined_variables(
            slopeframe=slopeframe,
            areaframe=areaframe,
            dl_frame=dl_frame,
            daily=daily,
            site=site,
        )
        combined_all.append(df_combined)

        df_combined, df_reduced = fix_site_and_update_counters(
            df_combined=df_combined,
            df_reduced=df_reduced,
            site=site,
            counters=counters,
        )

        # growing-season filter: daylength and temperature thresholds
        growing_season = df_reduced.loc[
            (df_reduced.daylength >= GROWING_SEASON_DAYLENGTH)
            & (df_reduced.TAir >= GROWING_SEASON_TEMP)
        ]

        growing_season_standardized = get_standardized_metrics(growing_season)
        growing_season_list.append(growing_season_standardized)

        # anomalies for distribution plots
        growing_season_stand_anom = get_anomalies_TAIR_TSM(growing_season_standardized)
        growing_season_anom_list.append(growing_season_stand_anom)

        # extra handling for focus sites
        if site in focus_sites:
            focus_growing_season[site] = growing_season_standardized
            subdaily_focus[site] = subdaily

        site_outputs[site] = SiteOutputs(
            site=site,
            subdaily=subdaily,
            daily=daily,
            slopeframe=slopeframe,
            areaframe=areaframe,
            dl_frame=dl_frame,
            growing_season=growing_season_standardized,
            growing_season_complete=df_combined,
            seasonal_corrs=seasonal_corrs,
            corr_slope_to_hourly=corr_slope_to_hourly,
            corr_area_to_hourly=corr_area_to_hourly,
        )

    log.info("Combined sites: %d", len(counters["combined_sites"]))
    log.info("Combined unique sites: %d", len(counters["combined_sites_unique"]))
    log.info("Reduced sites: %d", len(counters["reduced_sites"]))
    log.info("Reduced unique sites: %d", len(counters["reduced_sites_unique"]))

    # aggregated outputs consumed by the main analysis pipeline
    return {
        "site_outputs": site_outputs,                           # per-site SiteOutputs dataclass
        "growing_season_list": growing_season_list,             # standardised growing-season frames
        "combined_all": combined_all,                           # full SLOPE/AREA/driver frames
        "subdaily_focus": subdaily_focus,                       # hourly data for focus sites only
        "slope_coefficients_to_hourly": slope_coefficients_to_hourly,  # R-squared vs hourly
        "area_coefficients_to_hourly": area_coefficients_to_hourly,
        "all_seasonal_correlations": all_seasonal_correlations, # seasonal-cycle r/p values
        "focus_growing_season": focus_growing_season,           # growing-season data for focus sites
        "growing_season_stand_anom_list": growing_season_anom_list,  # anomaly frames
        "site_metadata": site_metadata,                         # metadata table for all sites
    }


# ---- helper: save figure as PDF + PNG at 300 DPI ----


def _save_fig(fig, pdf_path):
    # save to PDF via PdfPages, then save a 300 DPI PNG alongside
    with PdfPages(pdf_path) as pp:
        pp.savefig(fig, bbox_inches="tight")
    fig.savefig(
        pdf_path.replace(".pdf", ".png") if isinstance(pdf_path, str)
        else str(pdf_path).replace(".pdf", ".png"),
        bbox_inches="tight", dpi=300,
    )
    plt.close(fig)


# ---- Fig. A1: climate classification ----


def calc_climate_classification(
    combined_all: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, Any]:
    # clean Site column across all site frames, then classify in climate space
    cleaned_frames = []

    for i, df in enumerate(combined_all):
        if df.empty:
            log.warning("Empty dataframe at index %d", i)
            continue

        site_vals = df["Site"].dropna().unique()

        if len(site_vals) == 1:
            site_name = site_vals[0]
            df = df.copy()
            df["Site"] = site_name
            cleaned_frames.append(df)
        elif len(site_vals) == 0:
            log.warning("No Site value found at index %d", i)
            continue
        else:
            log.warning("Multiple Site values found at index %d: %s", i, site_vals)
            site_name = site_vals[0]
            df = df.copy()
            df["Site"] = site_name
            cleaned_frames.append(df)

    all_sites = pd.concat(cleaned_frames, axis=0)

    # classification uses all rows; reduction happens inside get_classification
    df_classification = get_classification(all_sites)
    log.info("%d sites classified", len(df_classification["data"]))
    return all_sites, df_classification  # concatenated frame + classification dict for plotting


def plot_climate_classification(
    df_classification: Any,
    out_pdf: str = str(FIG_DIR / "figA1.pdf"),
):
    fig = plot_classification(df_classification)
    _save_fig(fig, out_pdf)


# ---- Fig. 3: seasonal cycles for focus sites ----


def calc_cycles_all_sites(
    site_outputs: Dict[str, SiteOutputs], sites_info: List[dict]
) -> Dict[str, Any]:
    # prepare per-site objects for the seasonal-cycle figure
    ppfd_min, ppfd_max = float("inf"), float("-inf")
    for si in sites_info:
        df_env = pd.read_csv(si["env_file"], parse_dates=["solar_TIMESTAMP"]).set_index(
            "solar_TIMESTAMP"
        )
        ppfd = (
            df_env["ppfd_in"].resample("D").mean().groupby(lambda x: x.dayofyear).mean()
        )
        ppfd_min = min(ppfd_min, ppfd.min())
        ppfd_max = max(ppfd_max, ppfd.max())

    per_site = {}
    for si in sites_info:
        site = si["site"]
        if site not in site_outputs:
            continue
        out = site_outputs[site]
        concept = get_concept(out.subdaily)
        seasonal = get_seasonal(out.slopeframe, out.areaframe, out.daily)
        per_site[site] = {
            "concept": concept,
            "seasonal": seasonal,
            "subdaily": out.subdaily,
            "dl_frame": out.dl_frame,
            "slopeframe": out.slopeframe,
            "areaframe": out.areaframe,
            "env_file": si["env_file"],
            "years": si.get("years"),
        }

    # pre-computed objects passed to plot_cycles_all_sites
    return {
        "ppfd_min": ppfd_min,       # global PPFD min for shared colorbar
        "ppfd_max": ppfd_max,       # global PPFD max for shared colorbar
        "per_site": per_site,       # {site: concept, seasonal, subdaily, ...}
        "sites_info": sites_info,   # original focus site config entries
    }


def plot_cycles_all_sites(
    calc_obj: Dict[str, Any],
    out_pdf: str = str(FIG_DIR / "fig03.pdf"),
):
    ppfd_min = calc_obj["ppfd_min"]
    ppfd_max = calc_obj["ppfd_max"]
    per_site = calc_obj["per_site"]
    sites_info = calc_obj["sites_info"]

    fig = plt.figure(figsize=(25, 23))
    outer_spec = gridspec.GridSpec(ncols=1, nrows=len(sites_info), figure=fig)
    im = None

    panel_labels = [("a", "b"), ("c", "d"), ("e", "f")]
    for i, si in enumerate(sites_info):
        site = si["site"]
        if site not in per_site:
            continue
        ps = per_site[site]
        inner_spec = gridspec.GridSpecFromSubplotSpec(
            nrows=2, ncols=2,
            subplot_spec=outer_spec[i],
            width_ratios=[0.4, 1],
            height_ratios=[0.5, 1],
            wspace=0.15, hspace=0.05,
        )
        im = plot_cycle(
            ps["concept"], ps["seasonal"], ps["dl_frame"],
            ps["slopeframe"], ps["areaframe"],
            site, ps["env_file"],
            ppfd_min, ppfd_max,
            panel_labels[i], ps["years"],
            fig=fig, spec=inner_spec,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    if im is not None:
        cbar_ax = fig.add_axes([0.4, 0.055, 0.5, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(
            r"Mean Seasonal Cycle of daily PPFD in $\frac{\mathrm{\mu mol}}{\mathrm{m}^2\,\mathrm{s}}$",
            fontsize=30, labelpad=-3,
        )
        cbar.ax.tick_params(labelsize=20)

    _save_fig(fig, out_pdf)


# ---- Fig. 4: correlation heatmap ----


def plot_heatmap_parameters(
    all_correlations,
    site_metadata_file: str,
    out_pdf: str = str(FIG_DIR / "fig04.pdf"),
):
    fig, ax = plot_heatmap_summary_param(all_correlations, site_metadata_file)
    _save_fig(fig, out_pdf)


# ---- Fig. 5: percentile patterns and hysteresis fingerprints ----


def calc_hysteresis_patterns(
    cluster_data: Dict[str, pd.DataFrame], subdaily_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    # compute extreme anomalies and mean diurnal cycles per cluster
    desired_order = ["FRA_PUE", "GUF_GUY_GUY", "RUS_POG_VAR"]
    cluster_data = {k: cluster_data[k] for k in desired_order if k in cluster_data}

    if not cluster_data:
        return {"extreme_anomalies": None, "mean_cycles": None, "all_cycles": None}  # no focus sites

    extreme_anomalies = get_percentiles(cluster_data)

    # combine all subdaily data for the "all" key used by get_cluster_cycles
    df_subdaily = (
        pd.concat(list(subdaily_data.values())) if subdaily_data else pd.DataFrame()
    )
    subdaily_data_local = subdaily_data.copy()
    subdaily_data_local["all"] = df_subdaily

    nested_dates = get_cluster_dates(extreme_anomalies)
    mean_cycles, all_cycles = get_cluster_cycles(nested_dates, subdaily_data_local)

    # inputs for plot_hysteresis_patterns
    return {
        "extreme_anomalies": extreme_anomalies,  # per-site frames with Cluster column
        "mean_cycles": mean_cycles,              # per-site, per-cluster mean diurnal loops
        "all_cycles": all_cycles,                # per-site, per-cluster raw subdaily data
    }


def plot_hysteresis_patterns(
    calc_obj: Dict[str, Any],
    out_pdf: str = str(FIG_DIR / "fig05.pdf"),
):
    if calc_obj["extreme_anomalies"] is None:
        return
    fig = plot_patterns(
        calc_obj["extreme_anomalies"], calc_obj["mean_cycles"], code="supp",
    )
    _save_fig(fig, out_pdf)


# ---- Fig. 6: SLOPE and AREA distributions ----


def calc_distributions_slope_area(growing_season_anom_list):
    # SLOPE and AREA distributions at 20th/80th percentiles of TAir and TSM anomalies
    combined_df = pd.concat(growing_season_anom_list, ignore_index=False)

    tair_20 = combined_df["TAir_anomaly"].quantile(0.20)
    tair_80 = combined_df["TAir_anomaly"].quantile(0.80)
    tsm_20 = combined_df["TSM_anomaly"].quantile(0.20)
    tsm_80 = combined_df["TSM_anomaly"].quantile(0.80)

    # 4 combinations of high/low TAir and TSM anomalies
    high_tair_low_tsm = combined_df[
        (combined_df["TAir_anomaly"] >= tair_80)
        & (combined_df["TSM_anomaly"] <= tsm_20)
    ][["SLOPE", "AREA"]].copy()
    high_tair_low_tsm["combination"] = "high TAir & low TSM"

    low_tair_high_tsm = combined_df[
        (combined_df["TAir_anomaly"] <= tair_20)
        & (combined_df["TSM_anomaly"] >= tsm_80)
    ][["SLOPE", "AREA"]].copy()
    low_tair_high_tsm["combination"] = "low TAir & high TSM"

    high_tair_high_tsm = combined_df[
        (combined_df["TAir_anomaly"] >= tair_80)
        & (combined_df["TSM_anomaly"] >= tsm_80)
    ][["SLOPE", "AREA"]].copy()
    high_tair_high_tsm["combination"] = "high TAir & high TSM"

    low_tair_low_tsm = combined_df[
        (combined_df["TAir_anomaly"] <= tair_20)
        & (combined_df["TSM_anomaly"] <= tsm_20)
    ][["SLOPE", "AREA"]].copy()
    low_tair_low_tsm["combination"] = "low TAir & low TSM"

    return pd.concat(
        [high_tair_low_tsm, low_tair_high_tsm, high_tair_high_tsm, low_tair_low_tsm],
        ignore_index=False,
    )  # SLOPE and AREA values labelled by anomaly combination, for KDE plots


def plot_distributions_slope_area(
    slope_area_distributions,
    out_pdf=str(FIG_DIR / "fig06.pdf"),
):
    fig = plot_distributions_SLOPE_AREA(slope_area_distributions)
    _save_fig(fig, out_pdf)


# ---- Fig. 7: sample-rate comparison ----


def calc_samplerates(
    slope_coefficients: pd.DataFrame, area_coefficients: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    # pass-through packaging for plot_samplerates
    return {
        "slope_coefficients": slope_coefficients,  # R-squared of SLOPE per sample rate
        "area_coefficients": area_coefficients,    # R-squared of AREA per sample rate
    }


def plot_samplerates(
    calc_obj: Dict[str, pd.DataFrame],
    out_pdf: str = str(FIG_DIR / "fig07.pdf"),
):
    plot_srs(
        calc_obj["slope_coefficients"],
        calc_obj["area_coefficients"],
    )
    # plot_srs creates its own figure internally
    fig = plt.gcf()
    _save_fig(fig, out_pdf)
