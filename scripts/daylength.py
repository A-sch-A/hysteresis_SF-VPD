from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timezonefinder
from suntimes import SunTimes


def daylength(
    md_file: Path, si: str, first_year: int, last_year: int, tmp_dir: Path
) -> None:
    """
    Compute and save daily daylength (sunlight duration) for a given site.

    This function calculates daily sunrise and sunset times based on site metadata
    (latitude and longitude), then derives the daylength (in hours) for every day
    between `first_year` and `last_year`. The results are stored as a CSV file in
    the specified temporary directory.

    Parameters
    ----------
    par_dir : Path
        Base directory containing all SAPFLUXNET data (e.g. CSV_ROOT).
    level : str
        Data level (e.g., "leaf", "plant", or "sapwood").
    si : str
        Site identifier used to locate the metadata file (e.g., "DE-Tha").
    first_year : int
        Start year for daylength computation.
    last_year : int
        End year for daylength computation (inclusive).
    tmp_dir : Path
        Output directory for saving the daylength results.

    Output
    ------
    Creates a CSV file named ``<site>_daylength.csv`` in ``tmp_dir`` containing:
        - sunrise_hour
        - sunset_hour
        - daylength (sunset - sunrise, in hours)

    Notes
    -----
    - The function infers the local timezone automatically using `timezonefinder`.
    - If sunrise or sunset times cannot be computed for a date, NaN values are assigned.
    - Altitude is assumed to be 0 unless extended metadata are available.
    """

    # ------------------------------------------------------------------
    # Load site metadata
    # ------------------------------------------------------------------
    md = pd.read_csv(md_file)
    latitude = md["si_lat"][0]
    longitude = md["si_long"][0]

    # ------------------------------------------------------------------
    # Determine timezone
    # ------------------------------------------------------------------
    tf = timezonefinder.TimezoneFinder()
    timezone_str = tf.certain_timezone_at(lat=latitude, lng=longitude)
    if timezone_str is None:
        raise ValueError(
            f"Could not determine timezone for site '{si}' (lat={latitude}, lon={longitude})."
        )

    # ------------------------------------------------------------------
    # Initialize SunTimes and date range
    # ------------------------------------------------------------------
    sun = SunTimes(latitude=latitude, longitude=longitude, altitude=0)
    dates = pd.date_range(start=f"1/1/{first_year}", end=f"31/12/{last_year}")

    # ------------------------------------------------------------------
    # Compute sunrise/sunset and daylength
    # ------------------------------------------------------------------
    records = []
    for date in dates:
        try:
            sunrise = sun.risewhere(date, timezone_str)
            sunset = sun.setwhere(date, timezone_str)
            if isinstance(sunrise, datetime) and isinstance(sunset, datetime):
                records.append(
                    (date, sunrise.hour, sunset.hour, sunset.hour - sunrise.hour)
                )
            else:
                records.append((date, np.nan, np.nan, np.nan))
        except Exception:
            records.append((date, np.nan, np.nan, np.nan))

    df = pd.DataFrame(
        records, columns=["date", "sunrise_hour", "sunset_hour", "daylength"]
    )
    df.set_index("date", inplace=True)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_file = tmp_dir / f"{si}_daylength.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file)
