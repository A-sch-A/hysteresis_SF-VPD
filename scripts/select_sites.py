# select_sites.py — site selection based on data overlap with hydrometeorological drivers.
#
# Standalone pre-processing script. Run this before main.py to produce the
# site selection CSV (plant_sites.csv) consumed by the analysis pipeline.

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
    NO_TREATMENT_VALUES,
    OVERLAP_THRESHOLD,
    TMP_DIR,
    get_logger,
)
from daylength import daylength

log = get_logger(__name__)


def has_no_treatment(site_md: Path) -> bool:
    # check if a site has no experimental treatment based on the stand metadata
    try:
        df = pd.read_csv(site_md)
    except Exception as e:
        log.warning(f"Could not read {site_md.name}: {e}")
        return False

    if "st_treatment" not in df.columns:
        log.warning(f"{site_md.name}: column 'st_treatment' missing - assuming no treatment")
        return True

    treatments = (
        df["st_treatment"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(treatments) == 0:
        return True  # no treatment entries means untreated

    return all(t in NO_TREATMENT_VALUES for t in treatments)


def max_overlapping_days_list(site_md, subdaily, daily, dl_frame):
    # evaluate per-year overlap of hysteresis data (SF + VPD) with TAir and TSM,
    # filtered to the growing season (daylength >= 12 h AND TAir > 5 C).
    # a site passes if at least 2 years have overlap >= OVERLAP_THRESHOLD.

    site = pd.read_csv(site_md)
    latitude = site["si_lat"][0]
    longitude = site["si_long"][0]
    site_str = f"Site(lat={latitude:.3f}, lon={longitude:.3f})"

    log.info("Starting overlap evaluation for %s", site_str)

    max_season, tot_T, tot_TS = [], [], []
    n_years = 0

    for year in np.unique(daily.index.year):
        log.debug("%s — YEAR %s", site_str, year)

        annual = daily.loc[daily.index.year == year]
        if annual.empty:
            continue

        first_doy = annual.index[0].timetuple().tm_yday
        last_doy = annual.index[-1].timetuple().tm_yday

        dl_annual = dl_frame.loc[pd.to_datetime(dl_frame.index).year == year]
        dl_annual = dl_annual.iloc[first_doy - 1 : last_doy]
        dl_annual.index = annual.index

        annual = pd.concat([annual, dl_annual], axis=1)

        # growing season: daylength and temperature thresholds
        if "TAir" not in annual.columns:
            log.debug(
                "%s — %s: TAir missing, cannot define growing season, skipping year",
                site_str, year,
            )
            continue

        gs_mask = (
            (annual["daylength"] >= GROWING_SEASON_DAYLENGTH)
            & (annual["TAir"] > GROWING_SEASON_TEMP)
        )

        season_dates = annual.index[gs_mask]
        max_length = len(season_dates)

        log.debug(
            "%s — %s: growing season days = %d (daylength >= %.1fh AND TAir > %.1f C)",
            site_str, year, max_length,
            GROWING_SEASON_DAYLENGTH, GROWING_SEASON_TEMP,
        )

        if max_length == 0:
            log.debug("%s — %s: no growing-season days, skipping", site_str, year)
            continue

        # subdaily VPD check
        if "VPD" not in subdaily.columns:
            continue

        vpd_inseason = (
            subdaily["VPD"]
            .loc[subdaily.index.normalize().isin(season_dates.normalize())]
            .dropna()
        )
        vpd_inseason_daily = vpd_inseason.groupby(vpd_inseason.index.date).mean()

        sap_inseason = (
            subdaily["SF"]
            .loc[subdaily.index.normalize().isin(season_dates.normalize())]
            .dropna()
        )
        sap_inseason_daily = sap_inseason.groupby(sap_inseason.index.date).mean()

        hysteresis_dates = vpd_inseason_daily.index.intersection(
            sap_inseason_daily.index
        )

        # overlap evaluation: prefer TSM+TAir, fall back to TAir-only
        if (
            "TSM" in annual.columns
            and "TAir" in annual.columns
            and annual["TSM"].notna().all()
        ):
            T = annual.loc[annual.index.date.astype("O").isin(hysteresis_dates), "TAir"]
            S = annual.loc[annual.index.date.astype("O").isin(hysteresis_dates), "TSM"]
            TS = S.loc[S.index.isin(T.index)].dropna()

            n_TS = len(TS)
            tot_TS.append(n_TS)
            ratio = n_TS / max_length if max_length else np.nan

            if ratio >= OVERLAP_THRESHOLD:
                n_years += 1

        elif "TAir" in annual.columns and annual["TAir"].notna().all():
            T = annual.loc[annual.index.date.astype("O").isin(hysteresis_dates), "TAir"]
            n_T = len(T)
            tot_T.append(n_T)
            ratio = n_T / max_length if max_length else np.nan

            if ratio >= 0.8:
                n_years += 1

        max_season.append(max_length)

    # final site evaluation: need at least 2 valid years
    if tot_TS and n_years >= 2:
        ratio_final = sum(tot_TS) / sum(max_season)
        level_ls = [
            longitude, latitude, "TSM+TAir",
            ratio_final, sum(tot_TS), n_years, sum(max_season),
        ]

    elif tot_T and n_years >= 2:
        ratio_final = sum(tot_T) / sum(max_season)
        level_ls = [
            longitude, latitude, "TAir-only",
            ratio_final, sum(tot_T), n_years, sum(max_season),
        ]

    else:
        level_ls = []

    return level_ls  # [] if site fails, otherwise [lon, lat, code, ratio, n_overlap, n_years, n_season]


def get_selection(csv_root: Path = CSV_ROOT, level=LEVEL, tmp_dir: Path = TMP_DIR):
    # iterate over all sites in the specified data level, compute daylength,
    # check treatment status, evaluate overlap, and save the selection CSV.

    par_dir = csv_root
    d = {}

    lev_dir = par_dir / level
    d_level = {}

    for filename in lev_dir.glob("*sapf_data.csv"):
        sapfile = filename
        site = filename.stem.replace("_sapf_data", "")
        site_md = lev_dir / f"{site}_site_md.csv"
        stand_md = lev_dir / f"{site}_stand_md.csv"
        envfile = lev_dir / f"{site}_env_data.csv"

        log.info("--------------------------------------------------")
        log.info(f"Processing site: {site}")

        # skip sites with experimental treatments
        if not has_no_treatment(stand_md):
            log.info(f"Skipping {site}: treatment detected")
            log.info("--------------------------------------------------")
            continue

        log.info(f"{site}: No treatment - proceeding with analysis")

        # get subdaily and daily time series
        subdaily = util.get_subdaily(sapfile, envfile)
        daily = util.get_daily(envfile)

        # compute daylength for the site
        daylength(
            site_md, site,
            subdaily.iloc[0].name.year,
            subdaily.iloc[-1].name.year,
            tmp_dir,
        )

        dl_frame = pd.read_csv(tmp_dir / f"{site}_daylength.csv")
        dl_frame.index = dl_frame["date"]
        dl_frame.drop(columns=["date"], inplace=True)

        # evaluate overlap
        ls = max_overlapping_days_list(site_md, subdaily, daily, dl_frame)

        if ls != []:
            ls = [site] + ls
            d_level[str(site)] = ls

        log.info("--------------------------------------------------")

    d[level] = d_level

    save_selection(d, tmp_dir)


def save_selection(data_dict, tmp_dir: Path = TMP_DIR, level=LEVEL):
    # save the site selection as a CSV for the map and downstream pipeline
    data = pd.DataFrame.from_dict(data_dict[LEVEL]).T

    for i, df in enumerate([data]):
        df.columns = [
            "site",
            "longitude",       # map x-coordinate
            "latitude",        # map y-coordinate
            "code",            # "TSM+TAir" or "TAir-only"
            "ratio",           # overlap ratio
            "n_hyst",          # number of days with hysteresis + driver data
            "n_years",         # number of valid years
            "tot_dl_days",     # total growing-season days across years
        ]

        file_name = LEVEL + "_sites.csv"
        output_file = tmp_dir / file_name
        df.to_csv(output_file, **CSV_WRITE_KWARGS)
        log.info(f"Saved site selection to {output_file}")
