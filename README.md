# SHAP-Based Interpretation of Heatwave Reconstructions with Autoencoders

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Companion code for:**  


---

## Overview

This repository contains the code used to compute and visualise **SHAP (SHapley Additive exPlanations)** values for a pre-trained Autoencoder (AE) applied to European heatwave reconstruction. The workflow:

1. A `DeepExplainer` is built using the post-industrial training period as background data.
2. SHAP values are computed for **5 HW** and **5 NO-HW** case studies across four input variables (Z500, PEva, MSL, SM).
3. Per-variable normalised image plots and mean-sum `.npy` arrays are saved.
4. Publication-ready comparative figures are generated from the saved arrays.

---

## Repository Structure

```
.
├── config/
│   ├── model.json          # Model, domain, preprocessing, and clustering config
│   └── case_studies.json   # HW and NO-HW case study periods
├── scripts/
│   ├── shap_regression.py  # Batch SHAP computation (main pipeline)
│   ├── shap_figures.py     # Comparative/panel figures from .npy arrays
│   └── shap_composite.py   # Composite anomaly + SHAP overlay figures
├── notebooks/
│   ├── 01_shap_regression.ipynb  # Interactive SHAP exploration
│   ├── 02_shap_figures.ipynb     # Interactive figure tuning
│   └── 03_shap_composite.ipynb   # Interactive composite + SHAP overlay
├── shap/
│   ├── mod512/             # SHAP .npy and .png outputs (per case study)
│   ├── figures_paper/      # Comparative panel figures
│   ├── comp_shap/          # Composite anomaly + SHAP overlay figures
│   └── clusters/           # KMeans cluster labels and visualisations
├── requirements.txt
└── README.md
```

> **Note:** The `models/` and `data/` directories (containing the pre-trained `.h5` AE weights and the ERA5-based NetCDF files) are **not included** in this repository due to size constraints. See [Data & Models](#data--models) below.

---

## Case Studies

| # | Type  | Period                  | Description              |
|---|-------|-------------------------|--------------------------|
| 1 | HW    | 01 Aug – 10 Aug 2003    | European Heatwave 2003   |
| 2 | HW    | 19 Jul – 28 Jul 2014    | Heatwave 2014            |
| 3 | HW    | 25 Jul – 03 Aug 2018    | Heatwave 2018            |
| 4 | HW    | 20 Jul – 29 Jul 2019    | Heatwave 2019            |
| 5 | HW    | 10 Aug – 19 Aug 2022    | Heatwave 2022            |
| 1 | NO-HW | 01 Jul – 10 Jul 2004    | Control period 2004      |
| 2 | NO-HW | 15 Jul – 24 Jul 2012    | Control period 2012      |
| 3 | NO-HW | 10 Jul – 19 Jul 2013    | Control period 2013      |
| 4 | NO-HW | 01 Aug – 10 Aug 2019    | Control period 2019      |
| 5 | NO-HW | 13 Jun – 22 Jun 2021    | Control period 2021      |

Periods are defined in [`config/case_studies.json`](config/case_studies.json) and can be freely modified without touching any Python code.

---

## Input Variables

| Index | Name  | Description                          |
|-------|-------|--------------------------------------|
| 0     | Z500  | Geopotential height at 500 hPa       |
| 1     | PEva  | Potential evapotranspiration         |
| 2     | MSL   | Mean sea-level pressure              |
| 3     | SM    | Soil moisture                        |

---

## Installation

```bash
git clone https://github.com/your-org/shap-heatwave.git
cd shap-heatwave
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The `src/va_am` package (autoencoder utilities and preprocessing) must also be installed or placed on the Python path. Follow that package's own installation instructions.

---

## Configuration

All run-time parameters live in **two JSON files** — no hardcoded values in the Python scripts.

### `config/model.json`

Controls the domain, datasets, model paths, preprocessing, and clustering:

```jsonc
{
  "name": "all512",
  "domain": { "latitude_min": 32, ... },
  "datasets": { "prs_dataset": "~/data/...", ... },
  "model":    { "file_AE_post": "./models/MvAE_fr_mv512.h5", ... },
  "clustering": { "perform_cluster": true, "n_clusters": 8, ... }
}
```

### `config/case_studies.json`

Lists the HW and NO-HW event windows. Add, remove, or edit entries here to change the case studies — the scripts adapt automatically:

```jsonc
{
  "hw":    [ { "id": 1, "label": "...", "start": "2003-08-01", "end": "2003-08-10" }, ... ],
  "no_hw": [ { "id": 1, "label": "...", "start": "2004-07-01", "end": "2004-07-10" }, ... ]
}
```

---

## Usage

### 1. Compute SHAP values

```bash
python scripts/shap_regression.py \
    --config config/model.json \
    --case-studies config/case_studies.json \
    --output-dir ./shap/mod512 \
    --meansum \
    --verbose
```

Outputs are written to `shap/mod512/cs_hw{N}/` and `shap/mod512/cs_nohw{N}/` for each case study pair.

| Flag | Description |
|------|-------------|
| `--config` | Model & domain config (default: `config/model.json`) |
| `--case-studies` | Case study periods (default: `config/case_studies.json`) |
| `--output-dir` | Root output directory |
| `--meansum` | Save mean-of-sum arrays (default: plain sum) |
| `--telegram` | Send Telegram notifications on completion / error |
| `-v / --verbose` | Enable DEBUG logging |

### 2. Generate figures

```bash
# For each variable: z500, peva, msl, sm
python scripts/shap_figures.py \
    --config config/model.json \
    --case-studies config/case_studies.json \
    --var msl \
    --shap-dir ./shap/mod512 \
    --output-dir ./shap/figures_paper \
    --verbose

# Optional flags
#   --threshold 0.05      zero-out values below this normalised magnitude
#   --percentile 99       use 99th percentile as normalisation maximum
#   --meandiff            only produce the HW-mean minus NO-HW-mean map
#   --nine                also produce 3×3 diagnostic grids per case study
```

### 3. Generate composite anomaly + SHAP overlay figures

Overlays the normalised SHAP map on the climatological anomaly field for each
variable and case study:

```bash
python scripts/shap_composite.py \
    --config config/model.json \
    --case-studies config/case_studies.json \
    --vars z500 msl peva sm \
    --verbose
```

Outputs are saved under `shap/comp_shap/{var}/`.

| Flag | Description |
|------|-------------|
| `--vars` | One or more of `z500 msl peva sm` (default: all four) |
| `--no-individual` | Save only the 5-case mean figure, skip per-case figures |
| `--dataset` | Override the NetCDF path from config |

**Variable-specific colour settings applied automatically:**

| Variable | Composite cmap | SHAP threshold | Composite limits |
|----------|---------------|----------------|------------------|
| z500 | BrBG → PuOr blend | 0.30 | data-driven |
| msl  | BrBG → PuOr blend | 0.60 | data-driven |
| peva | RdBu_r (white centre, cf11) | — | ±1e-4 |
| sm   | RdBu_r (white centre, cf7)  | — | ±0.20 |

### 3. Interactive exploration (notebooks)

```bash
jupyter notebook notebooks/01_shap_regression.ipynb
jupyter notebook notebooks/02_shap_figures.ipynb
```

---

## Output Structure

After running both scripts, the `shap/` directory will contain:

```
shap/
├── clusters/
│   ├── clusterall512.npy          # KMeans labels
│   └── clusterall512.png          # Cluster visualisation
├── mod512/
│   ├── cs_hw1/
│   │   ├── shap_hw_zpms_all512.png          # All-variable SHAP image plot
│   │   ├── shap_hw_meansum_msl_all512.png   # MSL mean-sum figure
│   │   └── shap_hw_meansum_msl_all512.npy   # MSL mean-sum array
│   ├── cs_nohw1/ ...
│   └── cs_hw2/  ... cs_nohw5/ ...
└── figures_paper/
    ├── hw/
    │   ├── comparative_meansum_hw_mslall512.png
    │   ├── comparative_meansum_hw_mslall512_contour.png
    │   └── subfig/
    │       ├── mean_hw_mslall512.png
    │       └── event1_hw_mslall512.png  ... event5_...
    ├── nohw/ ...
    └── diff/
        └── meandiff_mslall512.png
```

---

## Data & Models

The following files are required but **not included** in this repository:

| File | Description |
|------|-------------|
| `models/MvAE_fr_mv512.h5` | Pre-trained multi-variable AE (post-industrial) |
| `models/MvAE_atrib_fr_512_pre.h5` | Pre-trained AE (pre-industrial) |
| `data/data_dailyMean_zpms_1940-2022.nc` | ERA5 Z500 / PEva / MSL / SM daily means |
| `data/data_dailyMax_t2m_1940-2022.nc` | ERA5 daily maximum 2 m temperature |

The NetCDF files are derived from **ERA5 reanalysis** (Hersbach et al., 2020) and can be obtained from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

---

## Telegram Notifications (optional)

Create a `secret.txt` file (excluded from version control by `.gitignore`) with three lines:

```
<bot_token>
<chat_id>
<your_username>
```

Then pass `--telegram` to either script.

---

## Citation

If you use this code in your research, please cite:



---

## License

This project is licensed under the EUPL 1.2 License. See [LICENSE](LICENSE) for details.
