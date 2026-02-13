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
    calc_distributions_slope_area,
    calc_hysteresis_patterns,
    calc_samplerates,
    plot_climate_classification,
    plot_cycles_all_sites,
    plot_distributions_slope_area,
    plot_heatmap_parameters,
    plot_hysteresis_patterns,
    plot_samplerates,
    process_all_sites,
)
from config import (
    CSV_ROOT,
    FIG_DIR,
    FOCUS_SITES,
    FOCUS_SITES_INFO,
    TMP_DIR,
    get_logger,
)
from mapping import plot_map
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
    log.info("Preparation completed. Site CSVs saved to TMP_DIR.")
    # load input data created in select_sites -> selected_sites
    input_data = pd.read_csv(TMP_DIR / "plant_sites.csv")

    input_data = input_data[input_data["code"] != "TAir-only"]
    print(input_data)

     # for testing, limit to first 5 sites
    sites_to_prepare = input_data["site"].to_list()
    print(sites_to_prepare)

    log.info("Plotting the map ...")
    plot_map(
        site_csv=TMP_DIR / "plant_sites.csv",
        output_path=FIG_DIR / "map.pdf",
        projection=None,
        figsize=(16, 8),
    )

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
    combined_all = agg["combined_all"]
    subdaily_focus = agg["subdaily_focus"]
    slope_coefficients_to_hourly = agg["slope_coefficients_to_hourly"]
    area_coefficients_to_hourly = agg["area_coefficients_to_hourly"]
    all_seasonal_correlations = agg["all_seasonal_correlations"]
    growing_season_anom_list = agg["growing_season_stand_anom_list"]
    site_metadata = agg["site_metadata"]
    # export site metadata
    site_metadata_df = pd.DataFrame(site_metadata)
    site_metadata_df.to_csv(TMP_DIR / "site_metadata.csv", index=False)

    # # ---------- 1. classification_climate for real observations ----------
    df_clusters = calc_climate_classification(combined_all)
    plot_climate_classification(df_clusters) # 45 sites, including plots

    # # ---------- 2. cycle_all_sites ----------
    cycles_all_sites = calc_cycles_all_sites(site_outputs, FOCUS_SITES_INFO)
    plot_cycles_all_sites(cycles_all_sites)

    # # ---------- 3. summary_heatmap_parameters ----------
    plot_heatmap_parameters(all_seasonal_correlations, TMP_DIR/"site_metadata.csv")

    # # ---------- 5. patterns_supplements ----------
    hysteresis_patterns = calc_hysteresis_patterns(
        agg["focus_growing_season"], subdaily_focus
    )
    plot_hysteresis_patterns(hysteresis_patterns)


    distributions_SLOPE_AREA = calc_distributions_slope_area(growing_season_anom_list)
    plot_distributions_slope_area(distributions_SLOPE_AREA)

    # # ---------- 6. samplerates
    print(len(slope_coefficients_to_hourly), "sites for evaluation of samplerates")
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
