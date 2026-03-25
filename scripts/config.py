# config.py — project-wide settings and directory layout.
#
# Set the paths below to match your local setup before running any script.

import logging
import warnings
from pathlib import Path

import cartopy.crs as ccrs

# ---- paths and directories (edit these) ----

# focus sites for detailed cycle and pattern analysis
FOCUS_SITES = ["FRA_PUE", "GUF_GUY_GUY", "RUS_POG_VAR"]

# root directory containing SAPFLUXNET data
DATA_ROOT = Path("")

# versioned subfolder of the SAPFLUXNET database
VERSION = "0.1.5"

# CSV-level folder structure
CSV_ROOT = DATA_ROOT / VERSION / "csv"

CODE_DIR = Path("")

# temporary output directory for intermediate files
TMP_DIR = CODE_DIR / "output/tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# figure output directory
FIG_DIR = CODE_DIR / "output/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- data levels and thresholds ----

LEVEL = "plant"
OVERLAP_THRESHOLD = 0.8
GROWING_SEASON_DAYLENGTH = 12   # hours
GROWING_SEASON_TEMP = 5         # degree Celsius

# per-site info used for the seasonal-cycle figure (Fig. 3)
# env_file paths are derived from CSV_ROOT and LEVEL defined above
FOCUS_SITES_INFO = [
    {
        "site": site,
        "env_file": CSV_ROOT / LEVEL / f"{site}_env_data.csv",
        "years": list(range(2003, 2016)) if site == "FRA_PUE" else None,
    }
    for site in FOCUS_SITES
]

GYMNOSPERM_GENERA = {
    "Pinus", "Picea", "Larix", "Juniperus", "Agathis",
    "Abies", "Cedrus", "Pseudotsuga", "Tsuga", "Thuja",
}

# treatment values that represent natural/untreated conditions
NO_TREATMENT_VALUES = {
    "natural conditions",
    "Control - Unthinned, unpruned, unfertilised",
    "Control",
    "control",
    "Reference",
    "Ambient Control",
    "non_thinned",
    "Before Thinning",
    "Before thinning",
    "Pre-thinning",
}

# ---- CSV I/O conventions ----

CSV_READ_KWARGS = dict(index_col=0)
CSV_WRITE_KWARGS = dict(index=False)

# ---- plotting and map settings ----

DEFAULT_PROJECTION = ccrs.Mollweide()
FIGSIZE_LARGE = (25, 23)

# ---- warning control ----

warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

# ---- logging ----

LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
