# Detection of tree stress from sub-daily sap flow variability

Code accompanying the manuscript:

> Schackow, A. T., Steele-Dunne, S. C., Milodowski, D. T., Limousin, J.-M., and Bastos, A.:
> Detection of tree stress from sub-daily sap flow variability,
> *Biogeosciences*, submitted, 2026.


## Overview

This repository contains the analysis code for quantifying vegetation stress
from sub-daily sap flow (SF) and vapour pressure deficit (VPD) hysteresis
dynamics, using data from the SAPFLUXNET database (Poyatos et al., 2021).

Two diagnostic metrics are derived from diurnal SF–VPD curves at each site:

- **SLOPE** — the regression slope of the morning branch, reflecting plant
  hydraulic sensitivity to atmospheric demand.
- **AREA** — the enclosed area of the hysteresis loop, capturing the overall
  magnitude of the diurnal SF–VPD decoupling.

The code selects sites with sufficient data overlap, analyses the seasonal
co-variation of these metrics with environmental drivers (air temperature, soil
moisture, radiation), classifies sites in climate space, and assesses the
temporal sampling resolution required to capture hysteresis signatures from
potential future satellite observations.


## Repository structure

```
.
├── config.py            # paths, thresholds, and project-wide settings
├── main.py              # entry point (--mode prepare | --mode main)
├── select_sites.py      # site selection based on data overlap criteria
├── daylength.py         # compute daily sunlight duration per site
├── util.py              # data loading, metric computation, processing
├── analyse.py           # site-level processing and figure generation
├── visualization.py     # all plotting functions
├── mapping.py           # world map of selected sites
├── check_treatment.py   # standalone utility: list treatment values
├── environment.yml      # conda environment specification
├── main.tex             # manuscript source (Copernicus LaTeX template)
└── output/              # created at runtime, not tracked in git
    ├── tmp/             # intermediate CSV files
    └── figures/         # PDF and PNG figures (fig01–fig07, figA1)
```


## Data

This analysis uses sap flow and environmental data from the SAPFLUXNET
database (version 0.1.5):

> Poyatos, R., et al.: SAPFLUXNET: A global database of sap flow measurements,
> Zenodo, https://doi.org/10.5281/zenodo.2530798, 2020.

The data is not included in this repository. To reproduce the analysis:

1. Download the SAPFLUXNET data from the Zenodo link above.
2. Set `DATA_ROOT` in `config.py` to point to the directory containing the
   downloaded data (the code expects `DATA_ROOT/0.1.5/csv/plant/` with the
   site-level CSV files).


## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate treepulse
```

Then edit `config.py`:

- Set `DATA_ROOT` to your local SAPFLUXNET data directory.
- Set `CODE_DIR` to the path of this repository on your system.


## Usage

The analysis runs in two stages:

**1. Site selection** (run once):

```bash
python select_sites.py
```

This evaluates all SAPFLUXNET sites for sufficient data overlap between
hysteresis data (SF + VPD) and hydrometeorological drivers (air temperature,
soil moisture), filtered by treatment status and growing-season criteria. The
output is `output/tmp/plant_sites.csv`.

**2. Metric computation and analysis:**

```bash
python main.py --mode prepare
```

This computes SLOPE and AREA at multiple temporal resolutions for each selected
site, generates the site map (fig02), and then automatically proceeds to the
main analysis, which produces all remaining figures (fig03–fig07, figA1).

To re-run only the analysis (after metrics have been computed):

```bash
python main.py --mode main
```


## Figures

All figures are saved as both PDF and PNG (300 DPI) in `output/figures/`:

| Output file | Manuscript figure | Description |
|-------------|-------------------|-------------|
| `fig01`     | Fig. 1            | Conceptual diagram (not code-generated) |
| `fig02`     | Fig. 2            | Map of selected SAPFLUXNET sites |
| `fig03`     | Fig. 3            | Seasonal cycles at three focus sites |
| `fig04`     | Fig. 4            | Correlation heatmap of metrics vs. drivers |
| `fig05`     | Fig. 5            | Percentile-based hysteresis fingerprints |
| `fig06`     | Fig. 6            | SLOPE and AREA distributions under extremes |
| `fig07`     | Fig. 7            | Sample-rate sensitivity analysis |
| `figA1`     | Fig. A1           | Climate classification of all sites |


## Contact

Anna T. Schackow — corresponding author
