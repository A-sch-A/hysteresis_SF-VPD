# main.py — entry point for the sap flow hysteresis analysis.
#
# Usage:
#   python main.py --mode prepare   # compute metrics from raw SAPFLUXNET data
#   python main.py --mode main      # run the analysis and generate figures
#
# Prerequisites:
#   1. Set your directories in config.py
#   2. Run select_sites.py first to produce the site selection CSV
#   3. Create your environment using environment.yml

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
    # compute SLOPE and AREA metrics at multiple sample rates for each selected site.
    # reads the site selection CSV produced by select_sites.py.

    log.info("Starting preparation step ...")
    input_data = pd.read_csv(TMP_DIR / "plant_sites.csv")
    input_data = input_data[input_data["code"] != "TAir-only"]

    sites_to_prepare = input_data["site"].to_list()
    log.info("%d sites to prepare", len(sites_to_prepare))

    log.info("Plotting the map ...")
    plot_map(
        site_csv=TMP_DIR / "plant_sites.csv",
        output_path=FIG_DIR / "fig02.pdf",
        projection=None,
        figsize=(16, 8),
    )

    for site in sites_to_prepare:
        environment = CSV_ROOT / f"plant/{site}_env_data.csv"
        sapflux = CSV_ROOT / f"plant/{site}_sapf_data.csv"

        log.info("Get subdaily data for %s", site)
        subdaily = get_subdaily(sapflux, environment)

        resamplings = get_resampled(subdaily)

        log.info("Compute SLOPE and AREA for %s", site)
        get_metrics(resamplings, site)

    log.info("Preparation complete. Proceeding to main analysis ...")
    main()


def main():
    # run the full analysis pipeline: process sites, then generate all figures.

    input_data = pd.read_csv(TMP_DIR / "plant_sites.csv")
    input_data = input_data[input_data["code"] != "TAir-only"]

    sites = input_data["site"].to_list()

    # process all sites
    agg = process_all_sites(input_data=input_data, sites=sites, focus_sites=FOCUS_SITES)

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

    # Fig. A1: climate classification
    df_clusters = calc_climate_classification(combined_all)
    plot_climate_classification(df_clusters)

    # Fig. 3: seasonal cycles for focus sites
    cycles_all_sites = calc_cycles_all_sites(site_outputs, FOCUS_SITES_INFO)
    plot_cycles_all_sites(cycles_all_sites)

    # Fig. 4: correlation heatmap
    plot_heatmap_parameters(all_seasonal_correlations, TMP_DIR / "site_metadata.csv")

    # Fig. 5: percentile patterns and hysteresis fingerprints
    hysteresis_patterns = calc_hysteresis_patterns(
        agg["focus_growing_season"], subdaily_focus
    )
    plot_hysteresis_patterns(hysteresis_patterns)

    # Fig. 6: SLOPE and AREA distributions
    distributions_SLOPE_AREA = calc_distributions_slope_area(growing_season_anom_list)
    plot_distributions_slope_area(distributions_SLOPE_AREA)

    # Fig. 7: sample-rate comparison
    log.info("%d sites for sample-rate evaluation", len(slope_coefficients_to_hourly))
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
        help="'prepare' to compute metrics, 'main' to run the analysis.",
    )
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    elif args.mode == "main":
        main()
