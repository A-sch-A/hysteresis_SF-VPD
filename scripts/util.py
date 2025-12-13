import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
from shapely.geometry import Polygon

from config import CSV_ROOT, TMP_DIR

TMP = TMP_DIR

# ---------------------------
# PREPARATION
# ---------------------------


def get_subdaily(sap, env):
    # Load sap flow data
    df_sap = pd.read_csv(sap)

    trees = df_sap.columns[2:]
    # Assuming first two columns are not tree measurements, the SAPFLUXNET data contains TIMESTAMPs there
    df_sap["SF"] = df_sap[trees].mean(axis=1) / 1000
    # Convert to liters, only for SAPFLUXNET!

    # Load environmental data
    df_env = pd.read_csv(env)

    # Decide which timestamp column to use
    ts_candidates = ["TIMESTAMP_solar", "solar_TIMESTAMP", "SOLAR_TIMESTAMP"]
    ts_col = next((c for c in ts_candidates if c in df_sap.columns), None)

    # Combine sap flow and environmental data
    df = df_sap[["SF", ts_col]].rename(columns={ts_col: "TIMESTAMP"})

    # Add VPD if available
    if "vpd" in df_env.columns:
        df["VPD"] = df_env["vpd"]

    # Process timestamps and handle missing values
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.replace(-9999999, np.nan, inplace=True)

    # Set index and resample to hourly averages, since some sites contains only hourly data
    df.set_index("TIMESTAMP", inplace=True)
    df = df.where(df >= 0)

    return df.resample("h").mean()


def get_daily(env):
    # Load environmental data
    df = pd.read_csv(env)

    # Decide which timestamp column to use
    ts_candidates = ["TIMESTAMP_solar", "solar_TIMESTAMP", "SOLAR_TIMESTAMP"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        ts_col = "TIMESTAMP"  # fallback if no solar timestamp available

    # Convert to datetime
    df["TIMESTAMP"] = pd.to_datetime(df[ts_col])

    # Mapping: original column name -> standardized name
    valid_columns = {
        "ta": "TAir",
        "swc_shallow": "TSM",
        "precip": "PRECIP",
        "ppfd_in": "PPFD",
    }

    # Initialize standardized columns
    for col, new_name in valid_columns.items():
        df[new_name] = df[col] if col in df.columns else np.nan

    # Keep only the selected TIMESTAMP + standardized valid columns (in fixed order)
    relevant_columns = ["TIMESTAMP"] + list(valid_columns.values())
    df = df[relevant_columns]

    # Replace invalid values with NaN and set index
    df.replace(-9999999, np.nan, inplace=True)
    df.set_index("TIMESTAMP", inplace=True)

    # Calculate daily mean
    daily_mean = df.resample("D").mean()
    return daily_mean


def get_resampled(sd_data):
    # Original hourly data
    hourly = sd_data

    # Resampling for different time intervals
    T8PD = sd_data.iloc[sd_data.index.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]

    T4PD = sd_data.iloc[sd_data.index.hour.isin([0, 6, 12, 18])]

    T3PD_night = sd_data.iloc[sd_data.index.hour.isin([0, 6, 18])]
    T3PD_day = sd_data.iloc[sd_data.index.hour.isin([6, 12, 18])]

    return [T8PD, T4PD, T3PD_night, T3PD_day, hourly]


def get_regression(ref, sampled):
    ref.index = sampled.index

    mask = ~np.isnan(ref) & ~np.isnan(sampled)

    if len(ref[mask]) > 1 and len(sampled[mask]) > 1 and len(set(ref[mask])) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            ref[mask], sampled[mask]
        )
        return slope, intercept, r_value, p_value, std_err

    return np.nan, np.nan, np.nan, np.nan, np.nan


def single_slope(vpd, sap):
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
            return np.nan, np.nan, np.nan, np.nan, np.nan

        #       return get_robust_regression(x_morning, y_morning)
        return get_regression(x_morning, y_morning)

    return np.nan, np.nan, np.nan, np.nan, np.nan


def get_slope(subdaily):
    slopes = []

    for date in np.unique(subdaily.index.date):
        x = subdaily["VPD"].loc[str(date)]
        y = subdaily["SF"].loc[str(date)]

        reg = single_slope(x, y)[0]
        slopes.append(reg if isinstance(reg, float) else np.nan)

    slope_df = pd.DataFrame(
        {"date": pd.to_datetime(np.unique(subdaily.index.date)), "SLOPE": slopes}
    )

    return slope_df.set_index("date")


def single_area(vpd, sap):
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

    return area


def get_area(subdaily):
    #   subdaily = scale_by_95th(subdaily) # ensure metrics are comparable
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

    return area_df.set_index("date")


def get_metrics(resamplings, site):
    slopeseries, areaseries = [], []

    for resampling in resamplings:
        slopeseries.append(get_slope(resampling))
        areaseries.append(get_area(resampling))

    slopeframe = pd.concat(slopeseries, axis=1)
    areaframe = pd.concat(areaseries, axis=1)

    slopeframe.columns = ["8TPD", "4TPD", "3TPDnight", "3TPDday", "hourly"]

    areaframe.columns = ["8TPD", "4TPD", "3TPDnight", "3TPDday", "hourly"]

    # Save the slope and area DataFrames to CSV files
    slopeframe.to_csv(TMP / f"{site}_slopes.csv")
    areaframe.to_csv(TMP / f"{site}_areas.csv")


# ---------------------------
# PROCESSING
# ---------------------------


def get_site_data(site, input_data):
    ENV_FILE = CSV_ROOT / f"plant/{site}_env_data.csv"
    SF_FILE = CSV_ROOT / f"plant/{site}_sapf_data.csv"

    environment_path = ENV_FILE
    sapflux_path = SF_FILE

    # Get sub-daily VPD and SF, and daily TAir (and SWC if needed)
    subdaily = get_subdaily(sapflux_path, environment_path)
    subdaily.index = pd.to_datetime(
        subdaily.index
    )  # Ensure the index is in datetime format
    daily = get_daily(environment_path)  # Get daily metrics

    # Load slope and area data from CSV files
    slopeframe = pd.read_csv(TMP / f"{site}_slopes.csv", index_col="date")
    slopeframe.index = pd.to_datetime(slopeframe.index)  # Convert index to datetime

    areaframe = pd.read_csv(TMP / f"{site}_areas.csv", index_col="date")
    areaframe.index = pd.to_datetime(areaframe.index)  # Convert index to datetime

    dl_frame = pd.read_csv(TMP / f"{site}_daylength.csv")  # Load day length data

    return subdaily, daily, slopeframe, areaframe, dl_frame


def corr_and_p(x, y):
    if x is None or y is None:
        return (np.nan, np.nan)
    if len(x) < 3:
        return (np.nan, np.nan)
    r, p = pearsonr(x, y)
    return (r, p)


def get_seasonal_cycle_correlations(site, slopeframe, areaframe):
    ENV_FILE = CSV_ROOT / f"plant/{site}_env_data.csv"

    env_file = ENV_FILE
    # Load and preprocess
    df = pd.read_csv(env_file)
    TS = "TIMESTAMP"
    df[TS] = pd.to_datetime(df[TS])
    df.set_index(TS, inplace=True)

    # Columns of interest
    tair_col = "ta"
    tsm_col = "swc_shallow"
    ppfd_col = "ppfd_in"

    # Resample subdaily env data to daily means
    tair_daily = df[tair_col].resample("D").mean()
    tsm_daily = df[tsm_col].resample("D").mean()
    ppfd_daily = df[ppfd_col].resample("D").mean() if ppfd_col in df.columns else None

    # Create seasonal cycles (mean per day of year)
    tair_cycle = tair_daily.groupby(tair_daily.index.dayofyear).mean()
    tsm_cycle = tsm_daily.groupby(tsm_daily.index.dayofyear).mean()
    ppfd_cycle = (
        ppfd_daily.groupby(ppfd_daily.index.dayofyear).mean()
        if ppfd_daily is not None
        else None
    )

    slope_cycle = slopeframe["hourly"].groupby(slopeframe.index.dayofyear).median()
    area_cycle = areaframe["hourly"].groupby(areaframe.index.dayofyear).median()

    # Align all series on day of year
    data = pd.DataFrame(
        {
            "TSM": tsm_cycle,
            "TAir": tair_cycle,
            "PPFD": ppfd_cycle if ppfd_cycle is not None else pd.NA,
            "SLOPE": slope_cycle,
            "AREA": area_cycle,
        }
    )

    # Drop rows with any missing values
    data = data.dropna()

    correlations = {
        "SLOPE-TSM": corr_and_p(data["SLOPE"], data["TSM"]),
        "SLOPE-TAir": corr_and_p(data["SLOPE"], data["TAir"]),
        "SLOPE-PPFD": corr_and_p(data["SLOPE"], data.get("PPFD")),
        "AREA-TSM": corr_and_p(data["AREA"], data["TSM"]),
        "AREA-TAir": corr_and_p(data["AREA"], data["TAir"]),
        "AREA-PPFD": corr_and_p(data["AREA"], data.get("PPFD")),
    }

    return correlations


def get_correlations_to_hourly(slopeframe, areaframe):
    slope_coeffs = {}
    area_coeffs = {}

    # Loop over proposals (columns)
    for proposal in slopeframe.columns[:-1]:  # exclude last column if irrelevant
        # Slope
        coeff_slope = get_regression(slopeframe["hourly"], slopeframe[proposal])[2] ** 2
        slope_coeffs[proposal] = coeff_slope

        # Area
        coeff_area = get_regression(areaframe["hourly"], areaframe[proposal])[2] ** 2
        area_coeffs[proposal] = coeff_area

    # Convert to Series for easy use
    longterm_sdf = pd.Series(slope_coeffs)
    longterm_adf = pd.Series(area_coeffs)

    return longterm_sdf, longterm_adf


def get_combined_variables(slopeframe, areaframe, dl_frame, daily, site):
    # --- Copy and rename columns to standard representation ---
    slopeframe = slopeframe.copy()
    areaframe = areaframe.copy()
    slopeframe["SLOPE"] = slopeframe["hourly"]
    areaframe["AREA"] = areaframe["hourly"]

    # --- Normalize dl_frame index ---
    dl_frame = dl_frame.copy()
    dl_frame["date"] = pd.to_datetime(dl_frame["date"], errors="coerce")
    dl_frame = dl_frame.set_index("date")

    # --- Align daily index with subdaily-derived frames ---
    daily = daily.copy()
    daily.index = slopeframe.index

    # --- Cap slopes at 95th percentile + filter negatives ---
    slope_capped = slopeframe["SLOPE"].clip(upper=slopeframe["SLOPE"].quantile(0.95))
    slope_filtered = slope_capped[slope_capped > 0]

    # --- Extract drivers ---
    tsm = daily["TSM"]
    tair = daily["TAir"]
    precip = daily["PRECIP"]

    # --- Assemble cluster dataframe ---
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
    ).dropna()

    return df_cluster


def scale_by_95th(series):
    series = series.copy()
    scaled = pd.DataFrame(index=series.index)
    max = series.max()
    min = series.min()
    scaled = series / (max - min)
    return scaled


def get_standardization_by95(daily):
    standardized = daily - daily.median()
    standardized = scale_by_95th(standardized)
    return standardized


def get_standardized_metrics(growing_cluster):
    gc = growing_cluster.copy()

    # store originals
    gc["SLOPE_org"] = gc["SLOPE"]
    gc["AREA_org"] = gc["AREA"]

    # apply scaling
    gc["SLOPE"] = get_standardization_by95(gc["SLOPE"])
    gc["AREA"] = scale_by_95th(gc["AREA"])

    return gc


def get_classification(growing_all_sites):
    results = []

    for site in growing_all_sites["Site"].unique():
        # load site metadata
        md_path = CSV_ROOT / f"plant/{site}_site_md.csv"
        site_md = pd.read_csv(md_path)

        # extract correct mean values from metadata
        si_mat = site_md["si_mat"].mean()  # TAir
        si_map = site_md["si_map"].mean()  # PRECIP

        # compute AREA and SLOPE from df_all
        site_df = growing_all_sites[growing_all_sites["Site"] == site]
        mean_area = site_df["AREA"].median()
        mean_slope = site_df["SLOPE"].median()

        results.append(
            {
                "Site": site,
                "TAir": si_mat,
                "PRECIP": si_map,
                "AREA": mean_area,
                "SLOPE": mean_slope,
            }
        )

    df_classification = pd.DataFrame(results)

    df_classification = pd.DataFrame(results).set_index("Site")
    # mimic old return format
    return {
        "slope": df_classification.groupby("Site")["SLOPE"].mean(),
        "area": df_classification.groupby("Site")["AREA"].mean(),
        "data": df_classification,
    }


def get_concept(subdaily):
    # Group by hour and minute, calculating the mean for each combination
    concept = (
        subdaily[["SF", "VPD"]]
        .groupby([subdaily.index.hour, subdaily.index.minute])
        .mean()
        .reset_index(drop=True)
    )

    # Get the start date for resampling
    start = subdaily.resample("d").mean().index[0]

    # Define the end time as 23:00 of the same day
    end = start + datetime.timedelta(hours=23)

    # Create a range of hourly timestamps from start to end
    times = pd.date_range(freq="1h", start=start, end=end)

    # Set the newly created hourly timestamps as the index of the concept DataFrame
    concept.index = times

    return concept  # Return the resulting DataFrame with mean values


def get_seasonal(slopeframe, areaframe, daily):
    slope_cycles, area_cycles = (
        {},
        {},
    )  # Initialize dictionaries to store annual slope and area data

    # Loop through each unique year in the daily DataFrame
    for year in np.unique(daily.index.year):
        start = pd.to_datetime(f"{year}-01-01")  # Start of the year
        end = pd.to_datetime(f"{year}-12-31")  # End of the year

        empty = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="1d"))

        annual_slope = slopeframe["hourly"].loc[slopeframe["hourly"].index.year == year]
        annual_area = areaframe["hourly"].loc[areaframe["hourly"].index.year == year]

        # Remove February 29th if it exists
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

    return df_slope, df_area  # Return the seasonal slope and area DataFrames


def get_percentiles(cluster_data):
    extreme_anomalies = {}

    for site, df in cluster_data.items():
        # Compute 20th and 80th percentiles
        tsm_low, tsm_high = df["TSM"].quantile([0.2, 0.8])
        tair_low, tair_high = df["TAir"].quantile([0.2, 0.8])

        def classify(row):
            high_tsm = row["TSM"] >= tsm_high
            low_tsm = row["TSM"] <= tsm_low
            high_tair = row["TAir"] >= tair_high
            low_tair = row["TAir"] <= tair_low

            if high_tair and high_tsm:
                return "hot & wet"  # High TAir, High TSM
            elif high_tair and low_tsm:
                return "hot & dry"  # High TAir, Low TSM
            elif low_tair and low_tsm:
                return "cold & dry"  # Low TAir, Low TSM
            elif low_tair and high_tsm:
                return "cold & wet"  # Low TAir, High TSM
            else:
                return None

        # Apply classification and store result
        df["Cluster"] = df.apply(classify, axis=1)
        extreme_anomalies[site] = df

    return extreme_anomalies


def get_cluster_dates(anomalies_dict):
    cluster_dates = {}
    for site, df in anomalies_dict.items():
        cluster_dict = {}
        for cluster_id, group in df.groupby("Cluster"):
            cluster_dict[cluster_id] = list(group.index)
        cluster_dates[site] = cluster_dict
    return cluster_dates


def get_cluster_cycles(nested_dates, subdaily_data):
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

            # Group by time (hour:minute)
            grouped = cluster_data.groupby(cluster_data.index.time).mean().sort_index()
            # Close the cycle by repeating the first row
            closed_cycle = pd.concat([grouped, grouped.iloc[[0]]])
            # Create string time index using .strftime('%H:%M')
            str_index = [t.strftime("%H:%M") for t in closed_cycle.index]
            closed_cycle.index = str_index
            # Store using the cluster ID
            mean_cycles[site][cluster_id] = closed_cycle
            all_cycles[site][cluster_id] = cluster_data

    return mean_cycles, all_cycles


def get_anomalies_TAIR_TSM(growing_season_standardized):
    """
    Extract TSM, TAir, SLOPE, AREA columns and calculate anomalies for TAir and TSM for distributions plot.
    """
    # Select only the required columns
    selected_columns = ["TSM", "TAir", "SLOPE", "AREA"]
    df_subset = growing_season_standardized[selected_columns].copy()

    # Calculate anomalies (deviation from mean)
    df_subset["TAir_anomaly"] = df_subset["TAir"] - df_subset["TAir"].mean()
    df_subset["TSM_anomaly"] = df_subset["TSM"] - df_subset["TSM"].mean()

    return df_subset


def get_metadata_per_site(CSV_ROOT, sites):
    """
    Extract site metadata (DBH, LAI, soil texture) for selected sites.
    """
    from pathlib import Path

    CSV_ROOT = Path(CSV_ROOT)
    plant_dir = CSV_ROOT / "plant"

    if not plant_dir.exists():
        raise FileNotFoundError(f"Plant directory not found: {plant_dir}")

    metadata_list = []

    # Only process sites in the provided list
    for site_id in sites:
        try:
            # Read stand metadata for LAI and soil texture
            stand_file = plant_dir / f"{site_id}_stand_md.csv"

            if not stand_file.exists():
                print(f"Warning: Stand metadata file not found for {site_id}")
                metadata_list.append(
                    {
                        "Site": site_id,
                        "DBH_mean": np.nan,
                        "LAI": np.nan,
                        "soil_texture": np.nan,
                    }
                )
                continue

            stand_md = pd.read_csv(stand_file)

            if len(stand_md) > 0:
                lai = (
                    stand_md["st_lai"].iloc[0]
                    if "st_lai" in stand_md.columns
                    else np.nan
                )
                soil_texture = (
                    stand_md["st_USDA_soil_texture"].iloc[0]
                    if "st_USDA_soil_texture" in stand_md.columns
                    else np.nan
                )
            else:
                lai = np.nan
                soil_texture = np.nan

            # Read plant metadata for DBH
            plant_file = plant_dir / f"{site_id}_plant_md.csv"

            if plant_file.exists():
                plant_md = pd.read_csv(plant_file)

                # Calculate mean DBH across all plants at the site
                if "pl_dbh" in plant_md.columns and len(plant_md) > 0:
                    dbh_mean = plant_md["pl_dbh"].mean()
                else:
                    dbh_mean = np.nan
            else:
                dbh_mean = np.nan
                print(f"Warning: Plant metadata file not found for {site_id}")

            # Store metadata for this site
            metadata_list.append(
                {
                    "Site": site_id,
                    "DBH_mean": dbh_mean,
                    "LAI": lai,
                    "soil_texture": soil_texture,
                }
            )

        except Exception as e:
            print(f"Error processing {site_id}: {e}")
            metadata_list.append(
                {
                    "Site": site_id,
                    "DBH_mean": np.nan,
                    "LAI": np.nan,
                    "soil_texture": np.nan,
                }
            )

    # Create DataFrame with sites as index
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df = metadata_df.set_index("Site")

    return metadata_df
