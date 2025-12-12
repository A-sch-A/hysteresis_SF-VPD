from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from config import FIG_DIR, GROWING_SEASON_DAYLENGTH, GROWING_SEASON_TEMP
from util import (
    get_anomalies_TAIR_TSM,
    get_classification,
    get_cluster_cycles,
    get_cluster_dates,
    get_combined_variables,
    get_concept,
    get_correlations_to_hourly,
    get_percentiles,
    get_seasonal,
    get_seasonal_cycle_correlations,
    get_site_data,
    get_standardized_metrics,
)
from visualization import (
    plot_classification,
    plot_cycle,
    plot_distribution_TSM_TAir,
    plot_distributions_focus,
    plot_heatmap_summary,
    # plot_map_summary,
    plot_patterns,
    plot_srs,
)


@dataclass
class SiteOutputs:
    site: str
    subdaily: pd.DataFrame
    daily: pd.DataFrame
    slopeframe: pd.DataFrame
    areaframe: pd.DataFrame
    dl_frame: pd.DataFrame
    growing_season: pd.DataFrame
    seasonal_corrs: Dict[str, Any]
    corr_slope_to_hourly: pd.Series
    corr_area_to_hourly: pd.Series


def process_all_sites(
    input_data: pd.DataFrame, sites: List[str], focus_sites: List[str]
) -> Dict[str, Any]:
    site_outputs: Dict[str, SiteOutputs] = {}
    growing_season_list: List[pd.DataFrame] = []
    growing_season_anom_list = []
    global_subdaily_list: List[pd.DataFrame] = []
    subdaily_focus: Dict[str, pd.DataFrame] = {}
    slope_coefficients_to_hourly = pd.DataFrame()
    area_coefficients_to_hourly = pd.DataFrame()
    all_seasonal_correlations: Dict[str, Any] = {}
    selected_sites: List[str] = sites
    focus_growing_season: Dict[str, pd.DataFrame] = {}

    for site in sites:
        # load site data
        subdaily, daily, slopeframe, areaframe, dl_frame = get_site_data(
            site, input_data
        )
        global_subdaily_list.append(subdaily)
        # defensive index conversion
        slopeframe.index = pd.to_datetime(slopeframe.index)
        areaframe.index = pd.to_datetime(areaframe.index)

        # -------------- SEASONAL RESPONSE METRICS TO ENV DRIVERS
        # seasonal correlations for maps/heatmaps
        seasonal_corrs = get_seasonal_cycle_correlations(site, slopeframe, areaframe)
        all_seasonal_correlations[site] = seasonal_corrs

        # -------------- FOR SAMPLERATES COMPARISON
        # correlation of metrics from resampled data to hourly reference
        corr_slope_to_hourly, corr_area_to_hourly = get_correlations_to_hourly(
            slopeframe, areaframe
        )

        # accumulate coefficients for storing values for all sites
        slope_df = corr_slope_to_hourly.to_frame().T
        area_df = corr_area_to_hourly.to_frame().T
        slope_coefficients_to_hourly = pd.concat(
            [slope_coefficients_to_hourly, slope_df], ignore_index=True
        )
        area_coefficients_to_hourly = pd.concat(
            [area_coefficients_to_hourly, area_df], ignore_index=True
        )

        # ---------------- PREPARATION FOR ANALYSIS OF HYSTERESIS DURING EXTREMES CONDITIONS
        df_combined = get_combined_variables(
            slopeframe=slopeframe,
            areaframe=areaframe,
            dl_frame=dl_frame,
            daily=daily,
            site=site,
        )

        # growing-season filter
        growing_season = df_combined.loc[
            (df_combined.daylength >= GROWING_SEASON_DAYLENGTH)
            & (df_combined.TAir >= GROWING_SEASON_TEMP)
        ]

        growing_season_standardized = get_standardized_metrics(growing_season)
        growing_season_list.append(growing_season_standardized)
        # print(growing_season_standardized)

        # Get anomalies
        growing_season_stand_anom = get_anomalies_TAIR_TSM(growing_season_standardized)
        growing_season_anom_list.append(growing_season_stand_anom)
        # print(growing_season_stand_anom)

        # ----------------- extra handling of sites in focus:
        if site in focus_sites:
            focus_growing_season[site] = growing_season_standardized
            subdaily_focus[site] = subdaily

        # store site outputs for downstream single-pass usage
        site_outputs[site] = SiteOutputs(
            site=site,
            subdaily=subdaily,
            daily=daily,
            slopeframe=slopeframe,
            areaframe=areaframe,
            dl_frame=dl_frame,
            growing_season=growing_season_standardized,
            seasonal_corrs=seasonal_corrs,
            corr_slope_to_hourly=corr_slope_to_hourly,
            corr_area_to_hourly=corr_area_to_hourly,
        )

    # aggregated return
    return {
        "site_outputs": site_outputs,
        "growing_season_list": growing_season_list,
        "global_subdaily_list": global_subdaily_list,
        "subdaily_focus": subdaily_focus,  # probably not needed?
        "slope_coefficients_to_hourly": slope_coefficients_to_hourly,
        "area_coefficients_to_hourly": area_coefficients_to_hourly,
        "all_seasonal_correlations": all_seasonal_correlations,
        "selected_sites": selected_sites,
        "focus_growing_season": focus_growing_season,
        "growing_season_stand_anom_list": growing_season_anom_list,
    }


# -------------------------
# 1) classification_climate
# -------------------------
def calc_climate_classification(
    growing_season_list: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, Any]:
    """Return combined growing season frames and classification frame of means for diagram"""
    growing_all_sites = pd.concat(growing_season_list)
    df_classification = get_classification(growing_all_sites)
    return growing_all_sites, df_classification


def plot_climate_classification(
    df_classification: Any, out_pdf: str = str(FIG_DIR / "classification_climate.pdf")
):
    with PdfPages(out_pdf) as pp:
        fig = plot_classification(df_classification)
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# -------------------------
# 2) cycle_all_sites
# -------------------------
def calc_cycles_all_sites(
    site_outputs: Dict[str, SiteOutputs], sites_info: List[dict]
) -> Dict[str, Any]:
    """
    Prepare the objects needed by the plotting function (concepts, seasonals, rad_min/rad_max, cached per-site data).
    Returns a dict with keys needed for plotting.
    """
    # compute global rad min/max
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

    # prepare per-site objects using cached site_outputs to avoid re-looping
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

    return {
        "ppfd_min": ppfd_min,
        "ppfd_max": ppfd_max,
        "per_site": per_site,
        "sites_info": sites_info,
    }


def plot_cycles_all_sites(
    calc_obj: Dict[str, Any], out_pdf: str = str(FIG_DIR / "cycle_all_sites.pdf")
):
    ppfd_min = calc_obj["ppfd_min"]
    ppfd_max = calc_obj["ppfd_max"]
    per_site = calc_obj["per_site"]
    sites_info = calc_obj["sites_info"]

    fig = plt.figure(figsize=(25, 23))
    outer_spec = gridspec.GridSpec(ncols=1, nrows=len(sites_info), figure=fig)
    im = None

    for i, si in enumerate(sites_info):
        site = si["site"]
        if site not in per_site:
            continue
        ps = per_site[site]
        inner_spec = gridspec.GridSpecFromSubplotSpec(
            nrows=2,
            ncols=2,
            subplot_spec=outer_spec[i],
            width_ratios=[0.4, 1],
            height_ratios=[0.5, 1],
            wspace=0.15,
            hspace=0.05,
        )
        im = plot_cycle(
            ps["concept"],
            ps["seasonal"],
            ps["dl_frame"],
            ps["slopeframe"],
            ps["areaframe"],
            site,
            ps["env_file"],
            ppfd_min,
            ppfd_max,
            ps["years"],
            fig=fig,
            spec=inner_spec,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    if im is not None:
        cbar_ax = fig.add_axes([0.4, 0.055, 0.5, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(
            r"Mean Seasonal Cycle of daily PPFD in $\frac{\mathrm{\mu mol}}{\mathrm{m}^2\,\mathrm{s}}$",
            fontsize=30,
            labelpad=-3,
        )
        cbar.ax.tick_params(labelsize=20)

    with PdfPages(out_pdf) as pp:
        pp.savefig(fig)
        plt.close(fig)


# -------------------------
# 3) summary_heatmap of seasonal correlations and significance
# -------------------------
# def calc_heatmap(all_correlations: Dict[str, Any]) -> Dict[str, Any]:
#     return {"all_correlations": all_correlations}


def plot_heatmap(
    all_correlations,
    out_pdf: str = str(FIG_DIR / "summary_heatmap_org_avg.pdf"),
):
    all_correlations = all_correlations
    with PdfPages(out_pdf) as pp:
        fig, ax = plot_heatmap_summary(all_correlations)
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# -------------------------
# 4) coefficients_maps
# -------------------------
def calc_maps_of_coefficients(
    all_correlations: Dict[str, Any], input_data: pd.DataFrame
) -> Dict[str, Any]:
    # Build coordinates from input_data and return with correlations

    selected_sites_md = input_data.loc[input_data["site"].isin(all_correlations.keys())]
    coordinates = dict(
        zip(
            selected_sites_md["site"],
            zip(selected_sites_md["longitude"], selected_sites_md["latitude"]),
        )
    )
    variables_to_plot = [
        "SLOPE-TSM",
        "AREA-TSM",
        "SLOPE-PPFD",
        "AREA-PPFD",
        "SLOPE-TAir",
        "AREA-TAir",
    ]
    return {
        "all_correlations": all_correlations,
        "coordinates": coordinates,
        "variables": variables_to_plot,
    }


def plot_maps_of_coefficients(
    calc_obj: Dict[str, Any], out_pdf: str = str(FIG_DIR / "coefficients_maps.pdf")
):
    all_correlations = calc_obj["all_correlations"]
    coordinates = calc_obj["coordinates"]
    variables = calc_obj["variables"]
    with PdfPages(out_pdf) as pp:
        fig, axs = plot_map_summary(
            all_correlations,
            coordinates=coordinates,
            variables=variables,
            title="Seasonal Cycle Correlations Across Sites",
        )
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# -------------------------
# 5) patterns_supplements (percentiles & hysteresis)
# -------------------------
def calc_hysteresis_patterns(
    cluster_data: Dict[str, pd.DataFrame], subdaily_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Compute extreme anomalies and mean cycles needed for patterns supplement plotting."""
    # Keep only desired order if present
    desired_order = ["FRA_PUE", "GUF_GUY_GUY", "RUS_POG_VAR"]
    cluster_data = {k: cluster_data[k] for k in desired_order if k in cluster_data}

    if not cluster_data:
        return {"extreme_anomalies": None, "mean_cycles": None, "all_cycles": None}

    extreme_anomalies = get_percentiles(cluster_data)
    # concat subdaily into 'all' for get_cluster_cycles where needed
    df_subdaily = (
        pd.concat(list(subdaily_data.values())) if subdaily_data else pd.DataFrame()
    )
    subdaily_data_local = subdaily_data.copy()
    subdaily_data_local["all"] = df_subdaily

    nested_dates = get_cluster_dates(extreme_anomalies)
    mean_cycles, all_cycles = get_cluster_cycles(nested_dates, subdaily_data_local)

    return {
        "extreme_anomalies": extreme_anomalies,
        "mean_cycles": mean_cycles,
        "all_cycles": all_cycles,
    }


def plot_hysteresis_patterns(
    calc_obj: Dict[str, Any], out_pdf: str = str(FIG_DIR / "patterns_supplements.pdf")
):
    if calc_obj["extreme_anomalies"] is None:
        # nothing to plot
        return
    fig = plot_patterns(
        calc_obj["extreme_anomalies"], calc_obj["mean_cycles"], code="supp"
    )
    with PdfPages(out_pdf) as pp:
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_distributions_metrics(
    extreme_anomalies, out_pdf=str(FIG_DIR / "distributions_focus.pdf")
):
    fig = plot_distributions_focus(extreme_anomalies)
    # Save to PDF
    with PdfPages(out_pdf) as pp:
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def calc_distributions_anomalies(growing_season_anom_list):
    """
    Calculate anomalies of TSM and TAir at 20% percentiles of combinations of SLOPE and AREA.
    """
    # Combine list of dataframes into one
    combined_df = pd.concat(growing_season_anom_list, ignore_index=False)

    # Calculate 20th and 80th percentiles for SLOPE and AREA
    slope_20 = combined_df["SLOPE"].quantile(0.20)
    slope_80 = combined_df["SLOPE"].quantile(0.80)
    area_20 = combined_df["AREA"].quantile(0.20)
    area_80 = combined_df["AREA"].quantile(0.80)

    # Define the 4 combinations based on percentiles
    # high SLOPE & low AREA
    high_slope_low_area = combined_df[
        (combined_df["SLOPE"] >= slope_80) & (combined_df["AREA"] <= area_20)
    ][["TAir_anomaly", "TSM_anomaly"]].copy()
    high_slope_low_area["combination"] = "high SLOPE & low AREA"

    # low SLOPE & high AREA
    low_slope_high_area = combined_df[
        (combined_df["SLOPE"] <= slope_20) & (combined_df["AREA"] >= area_80)
    ][["TAir_anomaly", "TSM_anomaly"]].copy()
    low_slope_high_area["combination"] = "low SLOPE & high AREA"

    # high SLOPE & high AREA
    high_slope_high_area = combined_df[
        (combined_df["SLOPE"] >= slope_80) & (combined_df["AREA"] >= area_80)
    ][["TAir_anomaly", "TSM_anomaly"]].copy()
    high_slope_high_area["combination"] = "high SLOPE & high AREA"

    # low SLOPE & low AREA
    low_slope_low_area = combined_df[
        (combined_df["SLOPE"] <= slope_20) & (combined_df["AREA"] <= area_20)
    ][["TAir_anomaly", "TSM_anomaly"]].copy()
    low_slope_low_area["combination"] = "low SLOPE & low AREA"

    # Combine all combinations into one dataframe
    anomalies_TAir_TSM = pd.concat(
        [
            high_slope_low_area,
            low_slope_high_area,
            high_slope_high_area,
            low_slope_low_area,
        ],
        ignore_index=False,
    )

    return anomalies_TAir_TSM


def plot_distributions_anomalies(
    anomalies_TAir_TSM, out_pdf=str(FIG_DIR / "distributions_anomalies.pdf")
):
    fig = plot_distribution_TSM_TAir(anomalies_TAir_TSM)
    # Save to PDF
    with PdfPages(out_pdf) as pp:
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# -------------------------
# 6) samplerates
# -------------------------
def calc_samplerates(
    slope_coefficients: pd.DataFrame, area_coefficients: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    return {
        "slope_coefficients": slope_coefficients,
        "area_coefficients": area_coefficients,
    }


def plot_samplerates(
    calc_obj: Dict[str, pd.DataFrame], out_pdf: str = str(FIG_DIR / "samplerates.pdf")
):
    with PdfPages(out_pdf) as pp:
        plot_srs(
            calc_obj["slope_coefficients"],
            calc_obj["area_coefficients"],
        )
        pp.savefig(bbox_inches="tight")
        plt.close()
