## [1.1.0] – 2025

### Added
- `scripts/shap_composite.py` – overlays normalised SHAP values on the
  climatological composite anomaly (field − day-of-year climatology, 1980–2010
  baseline) for each variable and case study. Produces individual per-case and
  5-case-mean figures for both HW and NO-HW event types.
- `notebooks/03_shap_composite.ipynb` – interactive companion for the
  composite script, with step-by-step cell-by-cell exploration.
- `shap/comp_shap/` output directory stubs added to the repo structure.

### Details
- **Z500 / MSL composite cmap** – BrBG (cool tail) blended with PuOr (warm
  tail), with a semi-transparent black band at the zero crossing
  (colors[7] = [0,0,0,0.3]), as specified.
- **PEva / SM composite cmap** – RdBu_r with three central levels set to
  white to highlight the zero band.
- **SHAP thresholds** – z500: 0.30; msl: 0.60; peva: cf11; sm: cf7.
- **Normalisation** – z500/msl use the per-event-type global maximum;
  peva/sm use the combined (HW+NO-HW) maximum so both event types share a
  consistent colour scale.

---

# Changelog

All notable changes to this project are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.0.0] – 2025

### Added
- Initial public release.
- `scripts/shap_regression.py` – batch SHAP computation for all HW / NO-HW
  case study pairs with a single invocation; loops over all 5 pairs, shares
  background data and the loaded model across pairs for efficiency.
- `scripts/shap_figures.py` – generates comparative 3×4 panels, standalone
  single-panel maps, optional 3×3 diagnostic grids, and HW-mean minus
  NO-HW-mean difference maps.
- `config/model.json` – nested, self-documenting configuration for domain,
  datasets, model paths, preprocessing, and clustering.  Replaces the flat,
  monolithic original JSON.
- `config/case_studies.json` – all HW and NO-HW periods defined in one place
  with human-readable labels; no hardcoded dates in any Python file.
- `notebooks/01_shap_regression.ipynb` – interactive companion for exploring
  SHAP values for individual case studies.
- `notebooks/02_shap_figures.ipynb` – interactive companion for tuning
  normalisation, thresholds, and layouts before batch generation.

### Changed
- Case studies updated to 5 HW + 5 NO-HW periods (2003, 2014, 2018, 2019,
  2022 / 2004, 2012, 2013, 2019, 2021).
- Output directories restructured to `shap/mod512/cs_hw{N}/` and
  `shap/mod512/cs_nohw{N}/` for clear per-case-study isolation.
- All subplot titles now include the full human-readable date range label.
- Colourbar logic refactored into reusable helpers.

### Fixed
- **`norm_msl` / `norm_sm` index swap** – the original code derived
  `norm_msl` from channel index 3 (SM) and `norm_sm` from index 2 (MSL),
  causing each variable to be normalised by the other's dynamic range.
  Both are now correctly derived from their own channels.
- `where_save_nohw` string-slice derivation replaced with explicit path
  construction (`os.path.join`).
- Cluster `KMeans` call now uses `n_init="auto"` to suppress the
  scikit-learn deprecation warning.
