"""
Configuration file for the SLAINTE project.

Set the following paths to match your local setup before running any script.
"""

import logging
import warnings
from pathlib import Path

import cartopy.crs as ccrs

# ---------------------------------------------------------------------
# Paths and directories  (EDIT THESE TO MATCH YOUR SYSTEM)
# ---------------------------------------------------------------------

FOCUS_SITES = ["FRA_PUE", "GUF_GUY_GUY", "RUS_POG_VAR"]

# Root directory containing SAPFLUXNET data
DATA_ROOT = Path("/home/tigris/DATA/SLAINTE")

# Versioned subfolder of the SAPFLUXNET database
VERSION = "0.1.5"

# CSV-level folder structure
CSV_ROOT = DATA_ROOT / VERSION / "csv"

CODE_DIR = Path("/home/tigris/Dokumente/SLAINTE/code_pub")

# Temporary output directory for intermediate files
TMP_DIR = CODE_DIR / "output/tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Figure output directory
FIG_DIR = CODE_DIR / "output/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FOCUS_SITES_INFO = [
    {
        "site": "FRA_PUE",
        "env_file": "/home/tigris/DATA/SLAINTE/0.1.5/csv/plant/FRA_PUE_env_data.csv",
        "years": list(range(2003, 2016)),
    },
    {
        "site": "GUF_GUY_GUY",
        "env_file": "/home/tigris/DATA/SLAINTE/0.1.5/csv/plant/GUF_GUY_GUY_env_data.csv",
        "years": None,
    },
    {
        "site": "RUS_POG_VAR",
        "env_file": "/home/tigris/DATA/SLAINTE/0.1.5/csv/plant/RUS_POG_VAR_env_data.csv",
        "years": None,
    },
]

# ---------------------------------------------------------------------
# Data levels and thresholds
# ---------------------------------------------------------------------
LEVEL = "plant"
OVERLAP_THRESHOLD = 0.8
GROWING_SEASON_DAYLENGTH = 12  # hours
GROWING_SEASON_TEMP = 5  # degree celsius
GYMNOSPERM_GENERA = {
    'Pinus', 'Picea', 'Larix', 'Juniperus', 'Agathis',
    'Abies', 'Cedrus', 'Pseudotsuga', 'Tsuga', 'Thuja'
}

# Define treatment values that represent "no treatment" or natural conditions
NO_TREATMENT_VALUES = {
    # Clear "no treatment" values
    'natural conditions',
    'Control - Unthinned, unpruned, unfertilised',
    'Control',
    'control',
    'Reference',
    'Ambient Control',
    # Borderline cases (minimal/no intervention)
    'non_thinned',
    'Before Thinning',
    'Before thinning',
    'Pre-thinning',
}

# ---------------------------------------------------------------------
# CSV I/O conventions
# ---------------------------------------------------------------------
CSV_READ_KWARGS = dict(index_col=0)
CSV_WRITE_KWARGS = dict(index=False)

# ---------------------------------------------------------------------
# Plotting and map settings
# ---------------------------------------------------------------------
DEFAULT_PROJECTION = ccrs.Mollweide()
FIGSIZE_LARGE = (25, 23)

# ---------------------------------------------------------------------
# Warning control
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
