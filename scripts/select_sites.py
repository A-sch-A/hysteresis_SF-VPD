# select_sites.py
from pathlib import Path

import numpy as np
import pandas as pd

import util
from config import (
    CSV_ROOT,
    CSV_WRITE_KWARGS,
    GROWING_SEASON_DAYLENGTH,
    GROWING_SEASON_TEMP,
    LEVEL,
    OVERLAP_THRESHOLD,
    TMP_DIR,
    NO_TREATMENT_VALUES,
    get_logger,
)
from daylength import daylength

log = get_logger(__name__)


#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import logging

log = logging.getLogger(__name__)


def has_no_treatment(site_md: Path) -> bool:
    """
    Check if a site has no treatment applied by examining the stand metadata.
    
    Parameters:
    -----------
    site_md : Path
        Path to the site metadata CSV file (typically *_stand_md.csv)
    
    Returns:
    --------
    bool
        True if site has no treatment (or only pre-treatment measurements),
        False otherwise
    """
    try:
        df = pd.read_csv(site_md)
    except Exception as e:
        log.warning(f"Could not read {site_md.name}: {e}")
        return False
    
    # Check if the treatment column exists
    if 'st_treatment' not in df.columns:
        log.warning(f"{site_md.name}: column 'st_treatment' missing - assuming no treatment")
        return True
    
    # Get all treatment values for this site (handle NaN and strip whitespace)
    treatments = (
        df['st_treatment']
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
    
    # If no treatments listed, assume no treatment
    if len(treatments) == 0:
        return True
    
    # Check if all treatments are in the "no treatment" set
    return all(t in NO_TREATMENT_VALUES for t in treatments)


def get_selection(csv_root: Path = CSV_ROOT, level=LEVEL, tmp_dir: Path = TMP_DIR):
    """
    This function checks the overlap of hysteresis (SF & VPD) data with hydrometeorological drivers
    at each site in the specified level of data from the SAPFLUXNET database.
    
    Only sites with no treatment applied are included in the selection.

    It uses the following modules:
        - util.py (utilities for calculating subdaily and daily time series)
        - daylength.py (script for the calculation of the seasonal cycle of the daylength at a specific location)

    and these functions (defined below):
        - has_no_treatment:
            - check if the site has no treatment applied based on stand metadata
        - max_overlapping_days_list:
            - find the maximum number of overlapping days with hysteresis data as well as temperature and top soil moisture per year
            - count this year as valid if number of overlap >= 80% of the defined growing season (here only days with a daylength >= 12 hours)
            - declare the site as valid if at least 2 years match this criterion
        - save_selection: create a dataframe used for plotting the map later

    Output:
    The results are saved as CSV files named '_sites.csv' in the TMP_DIR.
    Next to this also the '_daylength.csv' is saved.
    """

    par_dir = csv_root
    d = {}

    # Loop through each data level directory (e.g. plant, leaf, sapwood)
    lev_dir = par_dir / level
    d_level = {}

    # Iterate over all available files in the directory
    for filename in lev_dir.glob("*sapf_data.csv"):
        sapfile = filename
        site = filename.stem.replace("_sapf_data", "")
        site_md = lev_dir / f"{site}_site_md.csv"
        stand_md = lev_dir / f"{site}_stand_md.csv"
        envfile = lev_dir / f"{site}_env_data.csv"

        log.info("--------------------------------------------------")
        log.info(f"Processing site: {site}")

        # Check if site has no treatment applied - skip if treatment detected
        if not has_no_treatment(stand_md):
            log.info(f"Skipping {site}: treatment detected")
            log.info("--------------------------------------------------")
            continue

        log.info(f"{site}: No treatment - proceeding with analysis")

        # get sub daily time series of hysteresis data (SF and VPD)
        subdaily = util.get_subdaily(sapfile, envfile)
        # get daily time series of environmental data (TAir, TSM, PRECIP, PPFD)
        daily = util.get_daily(envfile)

        # get the seasonal cycle of the daylength at the site coordinates
        daylength(
            site_md,
            site,
            subdaily.iloc[0].name.year,
            subdaily.iloc[-1].name.year,
            tmp_dir,
        )

        dl_frame = pd.read_csv(tmp_dir / f"{site}_daylength.csv")
        dl_frame.index = dl_frame["date"]
        dl_frame.drop(columns=["date"], inplace=True)

        # check the days with overlapping hysteresis and (SWC/TAir) data
        ls = max_overlapping_days_list(site_md, subdaily, daily, dl_frame)

        if ls != []:
            ls = [site] + ls
            d_level[str(site)] = ls

        log.info("--------------------------------------------------")

    d[level] = d_level

    save_selection(d, tmp_dir)

# def get_selection(csv_root: Path = CSV_ROOT, level=LEVEL, tmp_dir: Path = TMP_DIR):
#     """
#     This function checks the overlap of hysteresis (SF & VPD) data with hydrometeorological drivers
#     at each site in the specified level of data from the SAPFLUXNET database.

#     It uses the following modules:
#         - util.py (utilities for calculating subdaily and daily time series)
#         - daylength.py (script for the calculation of the seasonal cycle of the daylength at a specific location)

#     and these functions (defined below):
#         - max_overlapping_days_list:
#             - find the maximum number of overlapping days with hysteresis data as well as temperature and top soil moisture per year
#             - count this year as valid if number of overlap >= 80% of the defined growing season (here only days with a daylength >= 12 hours)
#             - declare the site as valid if at least 2 years match this criterion
#         - save_selection: create a dataframe used for plotting the map later

#     Output:
#     The results are saved as CSV files named '_sites.csv' in the TMP_DIR.
#     Next to this also the '_daylength.csv' is saved.
#     """

#     par_dir = csv_root
#     d = {}

#     # Loop through each data level directory (e.g. plant, leaf, sapwood)
#     lev_dir = par_dir / level
#     d_level = {}

#     # Iterate over all available files in the directory
#     for filename in lev_dir.glob("*sapf_data.csv"):
#         sapfile = filename
#         site = filename.stem.replace("_sapf_data", "")
#         site_md = lev_dir / f"{site}_site_md.csv"
#         envfile = lev_dir / f"{site}_env_data.csv"

#         # get sub daily time series of hysteresis data (SF and VPD)
#         subdaily = util.get_subdaily(sapfile, envfile)
#         # get daily time series of environmental data (TAir, TSM, PRECIP, PPFD)
#         daily = util.get_daily(envfile)

#         # get the seasonal cycle of the daylength at the site coordinates
#         daylength(
#             site_md,
#             site,
#             subdaily.iloc[0].name.year,
#             subdaily.iloc[-1].name.year,
#             tmp_dir,
#         )

#         dl_frame = pd.read_csv(tmp_dir / f"{site}_daylength.csv")
#         dl_frame.index = dl_frame["date"]
#         dl_frame.drop(columns=["date"], inplace=True)

#         log.info("--------------------------------------------------")
#         log.info(site)

#         # check the days with overlapping hysteresis and (SWC/TAir) data
#         ls = max_overlapping_days_list(site_md, subdaily, daily, dl_frame)

#         if ls != []:
#             ls = [site] + ls
#             d_level[str(site)] = ls

#         log.info("--------------------------------------------------")

#     d[level] = d_level

#     save_selection(d, tmp_dir)

# TODO: define the growing season also with temperature > GROWING_SEASON_TAIR (defined in config)!!!!
# def max_overlapping_days_list(
#     site_md: str, subdaily: pd.DataFrame, daily: pd.DataFrame, dl_frame: pd.DataFrame
# ):
#     # --- Metadata ---
#     site = pd.read_csv(site_md)
#     latitude = site["si_lat"][0]
#     longitude = site["si_long"][0]
#     site_str = f"Site(lat={latitude:.3f}, lon={longitude:.3f})"

#     log.info("Starting overlap evaluation for %s", site_str)

#     # --- Initialization ---
#     dates = np.array([], dtype="datetime64")
#     max_season, tot_T, tot_TS = [], [], []
#     n_years = 0

#     for year in np.unique(daily.index.year):
#         log.debug("%s — YEAR %s", site_str, year)

#         # Get valid daily data
#         annual = daily.loc[daily.index.year == year]
#         if annual.empty:
#             log.debug("%s — %s: no daily data, skipping", site_str, year)
#             continue

#         first_doy = annual.index[0].timetuple().tm_yday
#         last_doy = annual.index[-1].timetuple().tm_yday

#         # Get annual daylength data
#         dl_annual = dl_frame.loc[pd.to_datetime(dl_frame.index).year == year]
#         dl_annual = dl_annual.iloc[first_doy - 1 : last_doy]
#         dl_annual.index = annual.index  # Align indices

#         # Combine valid data with daylength
#         annual = pd.concat([annual, dl_annual], axis=1)

#         max_length = len(
#             annual["daylength"].loc[annual["daylength"] >= GROWING_SEASON_DAYLENGTH]
#         )
#         log.debug(
#             "%s — %s: %d growing-season days (≥ 12.0h)", site_str, year, max_length
#         )
#         if max_length == 0:
#             log.debug("%s — %s: no growing-season days, skipping", site_str, year)
#             continue

#         season_dates = (
#             annual["daylength"]
#             .loc[annual["daylength"] >= GROWING_SEASON_DAYLENGTH]
#             .index
#         )

#         # --- Subdaily VPD check ---
#         if "VPD" not in subdaily.columns:
#             log.debug("%s — %s: VPD not available, skipping year", site_str, year)
#             continue

#         vpd_inseason = (
#             subdaily["VPD"]
#             .loc[subdaily["VPD"].index.isin(pd.to_datetime(season_dates))]
#             .dropna()
#         )
#         vpd_inseason_daily = vpd_inseason.groupby(
#             [vpd_inseason.index.year, vpd_inseason.index.month, vpd_inseason.index.day]
#         ).mean()
#         daily_index = set(vpd_inseason.index.date)
#         vpd_inseason_daily.index = daily_index

#         sap_inseason = (
#             subdaily["SF"]
#             .loc[subdaily["SF"].index.isin(pd.to_datetime(season_dates))]
#             .dropna()
#         )
#         sap_inseason_daily = sap_inseason.groupby(
#             [sap_inseason.index.year, sap_inseason.index.month, sap_inseason.index.day]
#         ).mean()
#         daily_index = set(sap_inseason.index.date)
#         sap_inseason_daily.index = daily_index

#         hysteresis_dates = vpd_inseason_daily.index.intersection(
#             sap_inseason_daily.index
#         )
#         log.debug(
#             "%s — %s: hysteresis data on %d days", site_str, year, len(hysteresis_dates)
#         )
#         # --- Overlap evaluation ---
#         if (
#             "TSM" in annual.columns
#             and "TAir" in annual.columns
#             and annual["TSM"].notna().all()
#         ):
#             T = (
#                 annual["TAir"]
#                 .loc[
#                     pd.to_datetime(annual["TAir"].index.date).isin(
#                         pd.to_datetime(hysteresis_dates)
#                     )
#                 ]
#                 .dropna()
#             )
#             S = (
#                 annual["TSM"]
#                 .loc[
#                     pd.to_datetime(annual["TSM"].index.date).isin(
#                         pd.to_datetime(hysteresis_dates)
#                     )
#                 ]
#                 .dropna()
#             )
#             TS = S.loc[S.index.isin(T.index)].dropna()
#             n_TS = len(TS)
#             tot_TS.append(n_TS)
#             ratio = n_TS / max_length if max_length != 0 else np.nan
#             log.debug(
#                 "%s — %s: overlap TAir+TSM %d days, ratio=%.2f",
#                 site_str,
#                 year,
#                 n_TS,
#                 ratio,
#             )
#             if max_length != 0 and ratio >= OVERLAP_THRESHOLD:
#                 n_years += 1

#         elif "TAir" in annual.columns and annual["TAir"].notna().all():
#             T = (
#                 annual["TAir"]
#                 .loc[
#                     pd.to_datetime(annual["TAir"].index.date).isin(
#                         pd.to_datetime(hysteresis_dates)
#                     )
#                 ]
#                 .dropna()
#             )
#             n_T = len(T)
#             tot_T.append(n_T)
#             ratio = n_T / max_length if max_length != 0 else np.nan
#             log.debug(
#                 "%s — %s: overlap TAir %d days, ratio=%.2f", site_str, year, n_T, ratio
#             )
#             if max_length != 0 and ratio >= 0.8:
#                 n_years += 1

#         max_season.append(max_length)

#     # --- Final site evaluation ---
#     if tot_TS and n_years >= 2:
#         ratio_final = sum(tot_TS) / sum(max_season) if sum(max_season) != 0 else np.nan
#         level_ls = [
#             longitude,
#             latitude,
#             "TSM+TAir",
#             ratio_final,
#             sum(tot_TS),
#             n_years,
#             sum(max_season),
#         ]
#         log.info(
#             "%s — PASSED: %d valid years, ratio=%.2f, type=TSM+TAir",
#             site_str,
#             n_years,
#             ratio_final,
#         )

#     elif tot_T and n_years >= 2:
#         ratio_final = sum(tot_T) / sum(max_season) if sum(max_season) != 0 else np.nan
#         level_ls = [
#             longitude,
#             latitude,
#             "TAir-only",
#             ratio_final,
#             sum(tot_T),
#             n_years,
#             sum(max_season),
#         ]
#         log.info(
#             "%s — PASSED: %d valid years, ratio=%.2f, type=TAir",
#             site_str,
#             n_years,
#             ratio_final,
#         )

#     else:
#         level_ls = []
#         log.warning("%s — FAILED: insufficient overlap or valid years", site_str)

#     return level_ls

# def max_overlapping_days_list(
#     site_md: str, subdaily: pd.DataFrame, daily: pd.DataFrame, dl_frame: pd.DataFrame
# ):
#     # --- Metadata ---
#     site = pd.read_csv(site_md)
#     latitude = site["si_lat"][0]
#     longitude = site["si_long"][0]
#     site_str = f"Site(lat={latitude:.3f}, lon={longitude:.3f})"

#     log.info("Starting overlap evaluation for %s", site_str)

#     # --- Initialization ---
#     dates = np.array([], dtype="datetime64")
#     max_season, tot_T, tot_TS = [], [], []
#     n_years = 0

#     for year in np.unique(daily.index.year):
#         log.debug("%s — YEAR %s", site_str, year)

#         annual = daily.loc[daily.index.year == year]
#         if annual.empty:
#             continue

#         first_doy = annual.index[0].timetuple().tm_yday
#         last_doy = annual.index[-1].timetuple().tm_yday

#         dl_annual = dl_frame.loc[pd.to_datetime(dl_frame.index).year == year]
#         dl_annual = dl_annual.iloc[first_doy - 1 : last_doy]
#         dl_annual.index = annual.index

#         annual = pd.concat([annual, dl_annual], axis=1)

#         # --------------------------------------------------
#         # Growing season definition
#         # --------------------------------------------------
#         if "TAir" not in annual.columns:
#             log.debug(
#                 "%s — %s: TAir missing, cannot define growing season, skipping year",
#                 site_str,
#                 year,
#             )
#             continue

#         gs_mask = (
#             (annual["daylength"] >= GROWING_SEASON_DAYLENGTH)
#             & (annual["TAir"] > GROWING_SEASON_TEMP)
#         )

#         season_dates = annual.index[gs_mask]
#         max_length = len(season_dates)

#         log.debug(
#             "%s — %s: growing season days = %d (daylength ≥ %.1fh AND TAir > %.1f°C)",
#             site_str,
#             year,
#             max_length,
#             GROWING_SEASON_DAYLENGTH,
#             GROWING_SEASON_TAIR,
#         )

#         if max_length == 0:
#             log.debug("%s — %s: no growing-season days, skipping", site_str, year)
#             continue

#         # --- Subdaily VPD check ---
#         if "VPD" not in subdaily.columns:
#             continue

#         vpd_inseason = (
#             subdaily["VPD"]
#             .loc[subdaily.index.normalize().isin(season_dates.normalize())]
#             .dropna()
#         )
#         vpd_inseason_daily = vpd_inseason.groupby(vpd_inseason.index.date).mean()

#         sap_inseason = (
#             subdaily["SF"]
#             .loc[subdaily.index.normalize().isin(season_dates.normalize())]
#             .dropna()
#         )
#         sap_inseason_daily = sap_inseason.groupby(sap_inseason.index.date).mean()

#         hysteresis_dates = vpd_inseason_daily.index.intersection(
#             sap_inseason_daily.index
#         )

#         # --- Overlap evaluation ---
#         if (
#             "TSM" in annual.columns
#             and "TAir" in annual.columns
#             and annual["TSM"].notna().all()
#         ):
#             T = annual.loc[annual.index.date.astype("O").isin(hysteresis_dates), "TAir"]
#             S = annual.loc[annual.index.date.astype("O").isin(hysteresis_dates), "TSM"]
#             TS = S.loc[S.index.isin(T.index)].dropna()

#             n_TS = len(TS)
#             tot_TS.append(n_TS)
#             ratio = n_TS / max_length if max_length else np.nan

#             if ratio >= OVERLAP_THRESHOLD:
#                 n_years += 1

#         elif "TAir" in annual.columns and annual["TAir"].notna().all():
#             T = annual.loc[annual.index.date.astype("O").isin(hysteresis_dates), "TAir"]
#             n_T = len(T)
#             tot_T.append(n_T)
#             ratio = n_T / max_length if max_length else np.nan

#             if ratio >= 0.8:
#                 n_years += 1

#         max_season.append(max_length)

#     # --- Final evaluation ---
#     if tot_TS and n_years >= 2:
#         ratio_final = sum(tot_TS) / sum(max_season)
#         level_ls = [
#             longitude,
#             latitude,
#             "TSM+TAir",
#             ratio_final,
#             sum(tot_TS),
#             n_years,
#             sum(max_season),
#         ]

#     elif tot_T and n_years >= 2:
#         ratio_final = sum(tot_T) / sum(max_season)
#         level_ls = [
#             longitude,
#             latitude,
#             "TAir-only",
#             ratio_final,
#             sum(tot_T),
#             n_years,
#             sum(max_season),
#         ]

#     else:
#         level_ls = []

#     return level_ls

def save_selection(data_dict, tmp_dir: Path = TMP_DIR, level=LEVEL):
    """
    This functions saves the dataframe to a csv for later plotting the map
    """
    data = pd.DataFrame.from_dict(data_dict[LEVEL]).T

    for i, df in enumerate([data]):
        df.columns = [
            "site",
            "longitude",  # needed for the location on the map
            "latitude",
            "code",  # needed for identification of existing TAir and TSM data
            "ratio",  # ratio of overlap
            "n_hyst",  # number of hysteresis days with data of hydrometeorological drivers
            "n_years",  # number of years with valid data
            "tot_dl_days",  # maximum number of days in the growing season
        ]

        file_name = LEVEL + "_sites.csv"
        output_file = tmp_dir / file_name
        df.to_csv(output_file, **CSV_WRITE_KWARGS)
        log.info(f"Saved site selection to {output_file}")
