# util.py — data loading, metric computation, and processing utilities.

import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
from shapely.geometry import Polygon

from config import CSV_ROOT, GYMNOSPERM_GENERA, TMP_DIR


# ---- preparation ----


def get_subdaily(sap, env):
    # load sap flow data
    df_sap = pd.read_csv(sap)

    # average across all tree sensors, convert to litres (SAPFLUXNET convention)
    trees = df_sap.columns[2:]
    df_sap["SF"] = df_sap[trees].mean(axis=1) / 1000

    # load environmental data
    df_env = pd.read_csv(env)

    # pick the solar timestamp column
    ts_candidates = ["TIMESTAMP_solar", "solar_TIMESTAMP", "SOLAR_TIMESTAMP"]
    ts_col = next((c for c in ts_candidates if c in df_sap.columns), None)

    # combine sap flow and VPD into a single frame
    df = df_sap[["SF", ts_col]].rename(columns={ts_col: "TIMESTAMP"})
    if "vpd" in df_env.columns:
        df["VPD"] = df_env["vpd"]

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.replace(-9999999, np.nan, inplace=True)
    df.set_index("TIMESTAMP", inplace=True)
    df = df.where(df >= 0)

    return df.resample("h").mean()  # hourly SF and VPD for hysteresis analysis


def get_daily(env):
    # load environmental data
    df = pd.read_csv(env)

    # pick the solar timestamp column
    ts_candidates = ["TIMESTAMP_solar", "solar_TIMESTAMP", "SOLAR_TIMESTAMP"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        ts_col = "TIMESTAMP"

    df["TIMESTAMP"] = pd.to_datetime(df[ts_col])

    # standardise column names for downstream use
    valid_columns = {
        "ta": "TAir",
        "swc_shallow": "TSM",
        "precip": "PRECIP",
        "ppfd_in": "PPFD",
    }
    for col, new_name in valid_columns.items():
        df[new_name] = df[col] if col in df.columns else np.nan

    relevant_columns = ["TIMESTAMP"] + list(valid_columns.values())
    df = df[relevant_columns]

    df.replace(-9999999, np.nan, inplace=True)
    df.set_index("TIMESTAMP", inplace=True)

    return df.resample("D").mean()  # daily means of TAir, TSM, PRECIP, PPFD


def get_resampled(sd_data):
    # create sub-sampled versions of hourly data at different temporal resolutions
    T8PD = sd_data.iloc[sd_data.index.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
    T4PD = sd_data.iloc[sd_data.index.hour.isin([0, 6, 12, 18])]
    T3PD_night = sd_data.iloc[sd_data.index.hour.isin([0, 6, 18])]
    T3PD_day = sd_data.iloc[sd_data.index.hour.isin([6, 12, 18])]

    return [T8PD, T4PD, T3PD_night, T3PD_day, sd_data]  # last element is hourly reference


def get_regression(ref, sampled):
    ref.index = sampled.index
    mask = ~np.isnan(ref) & ~np.isnan(sampled)

    if len(ref[mask]) > 1 and len(sampled[mask]) > 1 and len(set(ref[mask])) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            ref[mask], sampled[mask]
        )
        return slope, intercept, r_value, p_value, std_err  # linear regression stats

    return np.nan, np.nan, np.nan, np.nan, np.nan  # insufficient data for regression


def single_slope(vpd, sap):
    # compute the morning-branch regression slope of the SF-VPD hysteresis loop
    if not vpd.isna().all() and not sap.isna().all():
        idxv_min = vpd.idxmin()
        idxs_max = sap.idxmax()
        vpd.index = pd.to_datetime(vpd.index)
        sap.index = pd.to_datetime(sap.index)
        if not isinstance(idxv_min, str):
            morning_data = pd.concat([vpd, sap], axis=1).between_time(
                idxv_min.strftime("%H:%M"), idxs_max.strftime("%H:%M")
            )
        else:
            morning_data = pd.concat([vpd, sap], axis=1).between_time(
                idxv_min, idxs_max
            )
        x_morning = morning_data["VPD"]
        y_morning = morning_data["SF"]
        if x_morning.dropna().nunique() <= 1 or y_morning.dropna().nunique() <= 1:
            return np.nan, np.nan, np.nan, np.nan, np.nan  # too few unique values

        return get_regression(x_morning, y_morning)  # (slope, intercept, r, p, se) of morning branch

    return np.nan, np.nan, np.nan, np.nan, np.nan  # all-NaN input


def get_slope(subdaily):
    # daily SLOPE: regression slope of morning SF-VPD relationship
    slopes = []
    for date in np.unique(subdaily.index.date):
        x = subdaily["VPD"].loc[str(date)]
        y = subdaily["SF"].loc[str(date)]
        reg = single_slope(x, y)[0]
        slopes.append(reg if isinstance(reg, float) else np.nan)

    slope_df = pd.DataFrame(
        {"date": pd.to_datetime(np.unique(subdaily.index.date)), "SLOPE": slopes}
    )
    return slope_df.set_index("date")  # daily time series of SLOPE


def single_area(vpd, sap):
    # enclosed area of the daily SF-VPD hysteresis loop
    mask = ~np.isnan(vpd) & ~np.isnan(sap)

    if len(vpd) == 2:
        area = 0.0
    elif len(vpd[mask]) >= 3 and len(sap[mask]) >= 3:
        vpd_list = vpd.dropna().values.tolist()
        sap_list = sap.dropna().values.tolist()
        polygon = Polygon(zip(vpd_list, sap_list))
        area = gpd.GeoSeries([polygon]).area.item()
    else:
        area = np.nan

    return area  # scalar: polygon area in VPD-SF space, or NaN if insufficient data


def get_area(subdaily):
    # daily AREA: enclosed polygon area of the SF-VPD hysteresis loop
    vpd = subdaily["VPD"]
    sap = subdaily["SF"]
    areas = []

    for date in np.unique(sap.index.date):
        x = vpd.loc[str(date)]
        y = sap.loc[str(date)]
        areas.append(single_area(x, y))

    area_df = pd.DataFrame(
        {"date": pd.to_datetime(np.unique(sap.index.date)), "AREA": areas}
    )
    return area_df.set_index("date")  # daily time series of AREA


def get_metrics(resamplings, site):
    # compute SLOPE and AREA for each resampling and save to CSV
    slopeseries, areaseries = [], []
    for resampling in resamplings:
        slopeseries.append(get_slope(resampling))
        areaseries.append(get_area(resampling))

    slopeframe = pd.concat(slopeseries, axis=1)
    areaframe = pd.concat(areaseries, axis=1)

    slopeframe.columns = ["8TPD", "4TPD", "3TPDnight", "3TPDday", "hourly"]
    areaframe.columns = ["8TPD", "4TPD", "3TPDnight", "3TPDday", "hourly"]

    slopeframe.to_csv(TMP_DIR / f"{site}_slopes.csv")
    areaframe.to_csv(TMP_DIR / f"{site}_areas.csv")


# ---- processing ----


def get_site_data(site, input_data):
    # load pre-computed subdaily, daily, slope, area, and daylength data for one site
    env_file = CSV_ROOT / f"plant/{site}_env_data.csv"
    sf_file = CSV_ROOT / f"plant/{site}_sapf_data.csv"

    subdaily = get_subdaily(sf_file, env_file)
    subdaily.index = pd.to_datetime(subdaily.index)
    daily = get_daily(env_file)

    slopeframe = pd.read_csv(TMP_DIR / f"{site}_slopes.csv", index_col="date")
    slopeframe.index = pd.to_datetime(slopeframe.index)

    areaframe = pd.read_csv(TMP_DIR / f"{site}_areas.csv", index_col="date")
    areaframe.index = pd.to_datetime(areaframe.index)

    dl_frame = pd.read_csv(TMP_DIR / f"{site}_daylength.csv")

    return subdaily, daily, slopeframe, areaframe, dl_frame  # all input frames needed per site


def corr_and_p(x, y):
    # Pearson correlation with p-value, guarded for short series
    if x is None or y is None:
        return (np.nan, np.nan)  # no data
    if len(x) < 3:
        return (np.nan, np.nan)  # too few points
    r, p = pearsonr(x, y)
    return (r, p)  # (correlation coefficient, two-sided p-value)


def pairwise_corr(x, y):
    # pairwise correlation on overlapping DOYs only
    if x is None or y is None:
        return None  # missing input
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return None  # too few overlapping points
    return corr_and_p(xy.iloc[:, 0], xy.iloc[:, 1])  # (r, p) on shared DOYs


def get_seasonal_cycle_correlations(site, slopeframe, areaframe):
    # seasonal-cycle correlations between SLOPE/AREA and environmental drivers
    env_file = CSV_ROOT / f"plant/{site}_env_data.csv"
    df = pd.read_csv(env_file)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)

    tair_col = "ta"
    tsm_col = "swc_shallow"
    ppfd_col = "ppfd_in"

    # daily means
    tair_daily = df[tair_col].resample("D").mean() if tair_col in df.columns else None
    tsm_daily = df[tsm_col].resample("D").mean() if tsm_col in df.columns else None
    ppfd_daily = df[ppfd_col].resample("D").mean() if ppfd_col in df.columns else None

    # mean seasonal cycles (group by DOY)
    tair_cycle = (
        tair_daily.groupby(tair_daily.index.dayofyear).mean()
        if tair_daily is not None else None
    )
    tsm_cycle = (
        tsm_daily.groupby(tsm_daily.index.dayofyear).mean()
        if tsm_daily is not None else None
    )
    ppfd_cycle = (
        ppfd_daily.groupby(ppfd_daily.index.dayofyear).mean()
        if ppfd_daily is not None else None
    )

    slope_cycle = slopeframe["hourly"].groupby(slopeframe.index.dayofyear).median()
    area_cycle = areaframe["hourly"].groupby(areaframe.index.dayofyear).median()

    correlations = {
        "SLOPE-TSM": pairwise_corr(slope_cycle, tsm_cycle),
        "SLOPE-TAir": pairwise_corr(slope_cycle, tair_cycle),
        "SLOPE-PPFD": pairwise_corr(slope_cycle, ppfd_cycle),
        "AREA-TSM": pairwise_corr(area_cycle, tsm_cycle),
        "AREA-TAir": pairwise_corr(area_cycle, tair_cycle),
        "AREA-PPFD": pairwise_corr(area_cycle, ppfd_cycle),
    }

    return correlations  # dict of {metric-driver: (r, p)} for one site


def get_correlations_to_hourly(slopeframe, areaframe):
    # R-squared of each resampled metric against the hourly reference
    slope_coeffs = {}
    area_coeffs = {}

    for proposal in slopeframe.columns[:-1]:  # exclude "hourly" itself
        coeff_slope = get_regression(slopeframe["hourly"], slopeframe[proposal])[2] ** 2
        slope_coeffs[proposal] = coeff_slope

        coeff_area = get_regression(areaframe["hourly"], areaframe[proposal])[2] ** 2
        area_coeffs[proposal] = coeff_area

    longterm_sdf = pd.Series(slope_coeffs)
    longterm_adf = pd.Series(area_coeffs)

    return longterm_sdf, longterm_adf  # R-squared per sample rate, for SLOPE and AREA


def get_combined_variables(slopeframe, areaframe, dl_frame, daily, site):
    # merge SLOPE, AREA, daylength, and environmental drivers into a single frame
    slopeframe = slopeframe.copy()
    areaframe = areaframe.copy()
    slopeframe["SLOPE"] = slopeframe["hourly"]
    areaframe["AREA"] = areaframe["hourly"]

    dl_frame = dl_frame.copy()
    dl_frame["date"] = pd.to_datetime(dl_frame["date"], errors="coerce")
    dl_frame = dl_frame.set_index("date")

    daily = daily.copy()
    daily.index = slopeframe.index

    # cap slopes at the 95th percentile and keep only positive values
    slope_capped = slopeframe["SLOPE"].clip(upper=slopeframe["SLOPE"].quantile(0.95))
    slope_filtered = slope_capped[slope_capped > 0]

    tsm = daily["TSM"]
    tair = daily["TAir"]
    precip = daily["PRECIP"]

    df_cluster = pd.concat(
        [
            tsm,
            tair,
            slope_filtered,
            areaframe["AREA"],
            dl_frame,
            pd.Series(site, index=tsm.index, name="Site"),
            precip,
        ],
        axis=1,
    )

    return df_cluster, df_cluster.dropna()  # full and NaN-dropped versions


def scale_by_95th(series):
    # scale a series by its range (max - min)
    series = series.copy()
    return series / (series.max() - series.min())  # range-normalised series


def get_standardization_by95(daily):
    # centre on median, then scale by range
    standardized = daily - daily.median()
    return scale_by_95th(standardized)  # median-centred, range-normalised series


def get_standardized_metrics(growing_cluster):
    # standardise SLOPE and AREA within the growing-season subset
    gc = growing_cluster.copy()

    # keep original values for reference
    gc["SLOPE_org"] = gc["SLOPE"]
    gc["AREA_org"] = gc["AREA"]

    gc["SLOPE"] = get_standardization_by95(gc["SLOPE"])
    gc["AREA"] = scale_by_95th(gc["AREA"])

    return gc  # growing-season frame with standardised SLOPE/AREA


def get_classification(growing_all_sites):
    # per-site median SLOPE/AREA and climate coordinates for the classification plot
    results = []
    for site in growing_all_sites["Site"].unique():
        md_path = CSV_ROOT / f"plant/{site}_site_md.csv"
        site_md = pd.read_csv(md_path)

        si_mat = site_md["si_mat"].mean()    # mean annual temperature
        si_map = site_md["si_map"].mean()    # mean annual precipitation

        site_df = growing_all_sites[growing_all_sites["Site"] == site]
        mean_area = site_df["AREA"].median()
        mean_slope = site_df["SLOPE"].median()

        results.append({
            "Site": site,
            "TAir": si_mat,
            "PRECIP": si_map,
            "AREA": mean_area,
            "SLOPE": mean_slope,
        })

    df_classification = pd.DataFrame(results).set_index("Site")

    return {
        "slope": df_classification.groupby("Site")["SLOPE"].mean(),
        "area": df_classification.groupby("Site")["AREA"].mean(),
        "data": df_classification,
    }  # dict consumed by plot_classification


def _fix_site(df, site_name):
    # ensure the Site column contains a single consistent value
    if df.empty:
        return df  # pass through empty frames unchanged
    df = df.copy()
    site_vals = df["Site"].dropna().unique()
    if len(site_vals) == 1:
        df["Site"] = site_vals[0]
    else:
        df["Site"] = site_name
    return df  # frame with a uniform Site column


def fix_site_and_update_counters(df_combined, df_reduced, site, counters):
    # fix Site column and update per-site tracking counters
    df_combined_fixed = _fix_site(df_combined, site)
    df_reduced_fixed = _fix_site(df_reduced, site)

    for s in df_combined_fixed["Site"].dropna().unique():
        counters["combined_sites"].add(s)
        counters["combined_sites_unique"].add(s[:7])

    for s in df_reduced_fixed["Site"].dropna().unique():
        counters["reduced_sites"].add(s)
        counters["reduced_sites_unique"].add(s[:7])

    return df_combined_fixed, df_reduced_fixed  # site-fixed frames, counters mutated in place


def get_concept(subdaily):
    # mean diurnal cycle of SF and VPD across all days at a site
    concept = (
        subdaily[["SF", "VPD"]]
        .groupby([subdaily.index.hour, subdaily.index.minute])
        .mean()
        .reset_index(drop=True)
    )

    start = subdaily.resample("d").mean().index[0]
    end = start + datetime.timedelta(hours=23)
    times = pd.date_range(freq="1h", start=start, end=end)
    concept.index = times

    return concept  # hourly mean diurnal cycle, used for the conceptual hysteresis panel


def get_seasonal(slopeframe, areaframe, daily):
    # annual seasonal cycles of SLOPE and AREA, aligned to a common 365-day calendar
    slope_cycles, area_cycles = {}, {}

    for year in np.unique(daily.index.year):
        start = pd.to_datetime(f"{year}-01-01")
        end = pd.to_datetime(f"{year}-12-31")

        empty = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="1d"))

        annual_slope = slopeframe["hourly"].loc[slopeframe["hourly"].index.year == year]
        annual_area = areaframe["hourly"].loc[areaframe["hourly"].index.year == year]

        # remove Feb 29 so all years have 365 rows
        annual_slope = annual_slope[
            ~(annual_slope.index.month == 2) | (annual_slope.index.day != 29)
        ]
        annual_area = annual_area[
            ~(annual_area.index.month == 2) | (annual_area.index.day != 29)
        ]
        empty = empty[~(empty.index.month == 2) | (empty.index.day != 29)]

        empty["SLOPE"] = annual_slope
        empty["AREA"] = annual_area

        slope_cycles[str(year)] = empty["SLOPE"].reset_index(drop=True)
        area_cycles[str(year)] = empty["AREA"].reset_index(drop=True)

    df_slope = pd.DataFrame.from_dict(slope_cycles)
    df_area = pd.DataFrame.from_dict(area_cycles)

    index = empty.index.strftime("%m-%d")
    df_slope.index = index
    df_area.index = index

    return df_slope, df_area  # year-columns x DOY-rows, for the seasonal-cycle plot


def get_percentiles(cluster_data):
    # classify growing-season days into 4 quadrants based on TAir/TSM percentiles
    extreme_anomalies = {}

    for site, df in cluster_data.items():
        tsm_low, tsm_high = df["TSM"].quantile([0.2, 0.8])
        tair_low, tair_high = df["TAir"].quantile([0.2, 0.8])

        def classify(row):
            high_tsm = row["TSM"] >= tsm_high
            low_tsm = row["TSM"] <= tsm_low
            high_tair = row["TAir"] >= tair_high
            low_tair = row["TAir"] <= tair_low

            if high_tair and high_tsm:
                return "hot & wet"
            elif high_tair and low_tsm:
                return "hot & dry"
            elif low_tair and low_tsm:
                return "cold & dry"
            elif low_tair and high_tsm:
                return "cold & wet"
            else:
                return None

        df["Cluster"] = df.apply(classify, axis=1)
        extreme_anomalies[site] = df

    return extreme_anomalies  # per-site frames with a Cluster column


def get_cluster_dates(anomalies_dict):
    # extract date lists per cluster per site
    cluster_dates = {}
    for site, df in anomalies_dict.items():
        cluster_dict = {}
        for cluster_id, group in df.groupby("Cluster"):
            cluster_dict[cluster_id] = list(group.index)
        cluster_dates[site] = cluster_dict
    return cluster_dates  # {site: {cluster_label: [dates]}}


def get_cluster_cycles(nested_dates, subdaily_data):
    # mean diurnal cycles of SF and VPD per cluster per site
    mean_cycles, all_cycles = {}, {}

    for site in nested_dates:
        site_dates = nested_dates[site]
        subdaily = subdaily_data[site]
        mean_cycles[site], all_cycles[site] = {}, {}

        for cluster_id in site_dates:
            target_dates = [pd.to_datetime(d).date() for d in site_dates[cluster_id]]
            mask = pd.Series(subdaily.index.date).isin(target_dates)
            cluster_data = subdaily[mask.values]
            if cluster_data.empty:
                continue

            # group by time-of-day and close the loop
            grouped = cluster_data.groupby(cluster_data.index.time).mean().sort_index()
            closed_cycle = pd.concat([grouped, grouped.iloc[[0]]])
            str_index = [t.strftime("%H:%M") for t in closed_cycle.index]
            closed_cycle.index = str_index

            mean_cycles[site][cluster_id] = closed_cycle
            all_cycles[site][cluster_id] = cluster_data

    return mean_cycles, all_cycles  # per-site, per-cluster mean diurnal loops


def get_anomalies_TAIR_TSM(growing_season_standardized):
    # compute TAir and TSM anomalies (deviation from mean) for distribution plots
    selected_columns = ["TSM", "TAir", "SLOPE", "AREA"]
    df_subset = growing_season_standardized[selected_columns].copy()

    df_subset["TAir_anomaly"] = df_subset["TAir"] - df_subset["TAir"].mean()
    df_subset["TSM_anomaly"] = df_subset["TSM"] - df_subset["TSM"].mean()

    return df_subset  # growing-season frame with anomaly columns appended


def get_plant_group(species_list):
    # classify a list of species names as angiosperm, gymnosperm, or mixed
    if not species_list or (isinstance(species_list, float) and np.isnan(species_list)):
        return np.nan  # no species info available

    has_gymnosperm = False
    has_angiosperm = False

    for species in species_list:
        genus = species.split()[0]
        if genus in GYMNOSPERM_GENERA:
            has_gymnosperm = True
        else:
            has_angiosperm = True

    if has_gymnosperm and has_angiosperm:
        return "mixed"        # both groups present at site
    elif has_gymnosperm:
        return "gymnosperm"   # conifers only
    else:
        return "angiosperm"   # broadleaf only


def get_metadata_per_site(csv_root, sites):
    # extract site-level metadata (DBH, LAI, soil, species, biome, etc.) for all sites
    csv_root = Path(csv_root)
    plant_dir = csv_root / "plant"

    if not plant_dir.exists():
        raise FileNotFoundError(f"Plant directory not found: {plant_dir}")

    metadata_list = []

    for site_id in sites:
        try:
            # stand metadata
            stand_file = plant_dir / f"{site_id}_stand_md.csv"
            if not stand_file.exists():
                metadata_list.append(_empty_metadata(site_id))
                continue

            stand_md = pd.read_csv(stand_file)
            if len(stand_md) > 0:
                lai = stand_md["st_lai"].iloc[0] if "st_lai" in stand_md.columns else np.nan
                soil_texture = (
                    stand_md["st_USDA_soil_texture"].iloc[0]
                    if "st_USDA_soil_texture" in stand_md.columns else np.nan
                )
                treatment = (
                    stand_md["st_growth_condition"].iloc[0]
                    if "st_growth_condition" in stand_md.columns else np.nan
                )
            else:
                lai = soil_texture = treatment = np.nan

            # plant metadata
            plant_file = plant_dir / f"{site_id}_plant_md.csv"
            species = []
            if plant_file.exists():
                plant_md = pd.read_csv(plant_file)
                dbh_mean = (
                    plant_md["pl_dbh"].mean()
                    if "pl_dbh" in plant_md.columns and len(plant_md) > 0
                    else np.nan
                )
                if "pl_species" in plant_md.columns:
                    species = plant_md["pl_species"].unique().tolist()
            else:
                dbh_mean = np.nan

            # site metadata
            site_file = plant_dir / f"{site_id}_site_md.csv"
            biome = np.nan
            paper = np.nan
            precip = np.nan
            if site_file.exists():
                site_md = pd.read_csv(site_file)
                if len(site_md) > 0:
                    if "si_biome" in site_md.columns:
                        biome = site_md["si_biome"].iloc[0]
                    if "si_paper" in site_md.columns:
                        paper = site_md["si_paper"].iloc[0]
                    if "si_map" in site_md.columns:
                        precip = site_md["si_map"].iloc[0]

            plant_group = get_plant_group(species)

            metadata_list.append({
                "Site": site_id,
                "DBH_mean": dbh_mean,
                "LAI": lai,
                "soil_texture": soil_texture,
                "species": species,
                "biome": biome,
                "plant_group": plant_group,
                "treatment": treatment,
                "paper": paper,
                "precip": precip,
            })

        except Exception as e:
            metadata_list.append(_empty_metadata(site_id))

    return pd.DataFrame(metadata_list)  # one row per site


def _empty_metadata(site_id):
    # fallback row when metadata files are missing or unreadable
    return {
        "Site": site_id,
        "DBH_mean": np.nan,
        "LAI": np.nan,
        "soil_texture": np.nan,
        "species": np.nan,
        "biome": np.nan,
        "plant_group": np.nan,
        "treatment": np.nan,
        "paper": np.nan,
        "precip": np.nan,
    }
