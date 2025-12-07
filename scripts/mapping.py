import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)


def plot_map(site_csv: Path, output_path: Path, projection=None, figsize=(12, 6)):
    if projection is None:
        projection = ccrs.PlateCarree()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(site_csv)
    if "site" not in df.columns:
        raise ValueError("CSV must contain a 'site' column.")

    df = df.set_index("site")

    # Filter TSM+TAir sites
    tsm_sites = df[df["code"] == "TSM+TAir"]

    # Identify reference sites
    site_most_years = tsm_sites.loc[tsm_sites["n_years"].idxmax()]
    site_north = tsm_sites.loc[tsm_sites["latitude"].idxmax()]
    # Site closest to equator (latitude closest to 0)
    site_south = tsm_sites.loc[(tsm_sites["latitude"].abs()).idxmin()]

    log.info("Reference sites (TSM+TAir):")
    log.info(
        "  Most valid years: %s (%d years)",
        site_most_years.name,
        site_most_years.n_years,
    )
    log.info("  Northernmost: %s (lat=%.2f)", site_north.name, site_north.latitude)
    log.info("  Southernmost: %s (lat=%.2f)", site_south.name, site_south.latitude)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
    ax.set_global()

    # Add land background
    ax.add_feature(cfeature.LAND, color="lightgrey", zorder=0)

    # Plot sites
    for _, row in df.iterrows():
        size = row["n_years"] * 50
        color = "crimson" if row["code"] == "TSM+TAir" else "black"
        ax.scatter(
            row["longitude"],
            row["latitude"],
            s=size,
            color=color,
            marker="H",
            alpha=0.5,
            edgecolor="white",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    # Annotate reference sites
    _annotate_sites(ax, df, site_north.name, site_south.name, site_most_years.name)

    # Remove axes
    ax.axis("off")

    # Add legend
    _add_legend(ax, df)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Map saved to %s", output_path)

    return fig, ax


def _add_legend(ax, df):
    """Add custom legend, including automatic size categories."""

    # Site type legend
    type_handles = [
        Line2D(
            [],
            [],
            color="k",
            lw=0,
            marker="H",
            markeredgecolor="white",
            markersize=20,
            alpha=0.5,
            label="TAir",
        ),
        Line2D(
            [],
            [],
            color="crimson",
            lw=0,
            marker="H",
            markeredgecolor="white",
            markersize=20,
            alpha=0.5,
            label="TSM+TAir",
        ),
    ]

    # Choose meaningful breakpoints from the data
    year_values = sorted(df["n_years"].unique())
    year_values = [year_values[0], year_values[-1]]

    size_handles = [
        Line2D(
            [],
            [],
            color="gray",
            lw=0,
            marker="H",
            markeredgecolor="white",
            alpha=0.5,
            markersize=(years * 50) ** 0.5,
            label=f"{years} years",
        )
        for years in year_values
    ]

    ax.legend(
        handles=type_handles + size_handles,
        bbox_to_anchor=(0.25, 0.5),
        fontsize=14,
        facecolor="white",
        frameon=False,
        ncols=1,
        title="Legend",
    )


def _annotate_sites(ax, data, north_name, south_name, most_years_name):
    """Annotate representative sites."""
    for site_name, label_offset in zip(
        [north_name, south_name, most_years_name],
        [(20, 3), (20, 3), (3, -20)],
    ):
        ax.annotate(
            site_name,
            xy=[data.loc[site_name, "longitude"], data.loc[site_name, "latitude"]],
            xytext=label_offset,
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="k"),
            fontsize=14,
        )
