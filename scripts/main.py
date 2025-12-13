"""
Main script for the analysis of vegetation health based on sub-daily sap flow variability.

This script uses data from the SAPFLUXNET database.

Instructions:
-------------
1. Set your directories in `config.py`
2. Create your environment using `environment.yml`
3. Read the notes in `README.md`
4. Explore the diurnal cycles of SF and VPD!
5. Have fun :)
"""

import argparse

import pandas as pd

from analyse import (
    calc_climate_classification,
    calc_cycles_all_sites,
    calc_distributions_anomalies,
    calc_distributions_slope_area,
    # calc_heatmap,
    calc_hysteresis_patterns,
    # calc_maps_of_coefficients,
    calc_samplerates,
    plot_climate_classification,
    plot_cycles_all_sites,
    plot_distributions_anomalies,
    plot_distributions_metrics,
    plot_distributions_slope_area,
    plot_heatmap,
    plot_hysteresis_patterns,
    # plot_maps_of_coefficients,
    plot_samplerates,
    process_all_sites,
)
from config import (
    CSV_ROOT,
    FIG_DIR,
    FOCUS_SITES,
    FOCUS_SITES_INFO,
    LEVEL,
    TMP_DIR,
    get_logger,
)
from mapping import plot_map
from select_sites import get_selection
from util import get_metrics, get_resampled, get_subdaily

log = get_logger(__name__)


def prepare():
    """
    Preparation step SAPFLUXNET:
    Evaluates the overlap between hysteresis data (sap flow, VPD)
    and hydrometeorological drivers (TAir, TSM, etc.) for all sites
    in the selected SAPFLUXNET data level(s).
    It then calculates the hysteresis metrics: SLOPE and AREA.
    """

    log.info("Starting preparation step (site selection)...")
    # get_selection(CSV_ROOT, LEVEL, TMP_DIR)
    log.info("Preparation completed. Site CSVs saved to TMP_DIR.")
    # load input data created in select_sites -> selected_sites
    selected_sites = pd.read_csv(TMP_DIR / "plant_sites.csv")["site"]  # only TSM+TAir!
    sites_to_prepare = selected_sites
    log.info("Plotting the map ...")
    plot_map(
        site_csv=TMP_DIR / "plant_sites.csv",
        output_path=FIG_DIR / "map.pdf",
        projection=None,
        figsize=(16, 8),
    )
    exit()
    for site in sites_to_prepare:
        environment = CSV_ROOT / f"plant/{site}_env_data.csv"
        sapflux = CSV_ROOT / f"plant/{site}_sapf_data.csv"

        # Retrieve sub-daily vapor pressure deficit (VPD) and sap flux (SF) data
        log.info(f"Get subdaily data for {site}")
        subdaily = get_subdaily(sapflux, environment)

        # Perform resampling of sub-daily data for assessment of hysteresis from space
        resamplings = get_resampled(subdaily)

        log.info(f"Preapare csv files for hysteresis metrics SLOPE and AREA for {site}")
        get_metrics(resamplings, site)
        # Additional processing can be added here as needed

    log.info("Data preparation complete. Proceeding to main processing...")
    main()  # Call the main function after preparation


def main():
    """
    Main analysis pipeline entry point.
    """
    # Load input data containing site information
    input_data = pd.read_csv(TMP_DIR / "plant_sites.csv")
    input_data = input_data[input_data["code"] != "TAir-only"]

    sites = input_data["site"].to_list()

    # ---------- process through sites ----------
    agg = process_all_sites(input_data=input_data, sites=sites, focus_sites=FOCUS_SITES)

    # unpack aggregated outputs
    site_outputs = agg["site_outputs"]
    growing_season_list = agg["growing_season_list"]
    # subdaily_list = agg["subdaily_list"]
    subdaily_focus = agg["subdaily_focus"]
    slope_coefficients_to_hourly = agg["slope_coefficients_to_hourly"]
    area_coefficients_to_hourly = agg["area_coefficients_to_hourly"]
    all_seasonal_correlations = agg["all_seasonal_correlations"]
    growing_season_anom_list = agg["growing_season_stand_anom_list"]
    site_metadata = agg["site_metadata"]
    print(site_metadata)
    exit()
    # # ---------- 1. classification_climate for real observations ----------
    df_clusters = calc_climate_classification(growing_season_list)
    plot_climate_classification(df_clusters)

    # # ---------- 2. cycle_all_sites ----------
    cycles_all_sites = calc_cycles_all_sites(site_outputs, FOCUS_SITES_INFO)
    plot_cycles_all_sites(cycles_all_sites)

    # # ---------- 3. summary_heatmap_org_avg ----------
    # heatmap_values = calc_heatmap(all_seasonal_correlations)
    plot_heatmap(all_seasonal_correlations)

    # # ---------- 4. coefficients_maps
    # maps_coefficients = calc_maps_of_coefficients(all_seasonal_correlations, input_data) # i we include it again, we need to care include significance, which is part of correlations now
    # plot_maps_of_coefficients(maps_coefficients)

    # # ---------- 5. patterns_supplements ----------
    hysteresis_patterns = calc_hysteresis_patterns(
        agg["focus_growing_season"], subdaily_focus
    )
    plot_hysteresis_patterns(hysteresis_patterns)
    plot_distributions_metrics(hysteresis_patterns["extreme_anomalies"])

    anomalies_TAir_TSM = calc_distributions_anomalies(growing_season_anom_list)
    plot_distributions_anomalies(anomalies_TAir_TSM)

    distributions_SLOPE_AREA = calc_distributions_slope_area(growing_season_anom_list)
    plot_distributions_slope_area(distributions_SLOPE_AREA)

    # # ---------- 6. samplerates
    samplerates = calc_samplerates(
        slope_coefficients_to_hourly, area_coefficients_to_hourly
    )
    plot_samplerates(samplerates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process or prepare sapfluxnet data.")
    parser.add_argument(
        "--mode",
        choices=["prepare", "main"],
        required=True,
        help="Specify 'prepare' to run the preparation function or 'main' to run the main processing function.",
    )
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    elif args.mode == "main":
        main()
