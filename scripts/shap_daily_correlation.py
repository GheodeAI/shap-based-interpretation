"""
shap_daily_correlation.py
==========================
Day-resolved pixel-wise correlation between normalised SHAP values and
anomaly fields — same statistical machinery as ``shap_spatial_correlation.py``
(``pixelwise_pearson`` / ``summary_statistics`` / ``_correlation_figure`` are
literally copy-identical: neither function cares what axis 0 represents),
but axis 0 here is **days within a case-study window**, not case studies.

This requires the per-day ("sum") SHAP ``.npy`` files, NOT the ``meansum``
ones used by ``shap_composite.py`` / ``shap_figures.py`` /
``shap_spatial_correlation.py``. Those are produced by running
``shap_regression.py`` WITHOUT ``--meansum`` (see that script's
``_save_shap_figure``: ``result = arr_sum.mean(axis=0) if meansum else
arr_sum`` — the day axis is only collapsed when ``meansum=True``).

Three complementary analyses are produced per (var, event_type)
--------------------------------------------------------------
1. **Per case study** (N = n_days, e.g. 10): for each of the 5 case
   studies independently, a pixel-wise correlation map and summary stats
   computed across that case's own days. Much more statistical power per
   map than the original 5-case analysis (df = n_days−2 instead of 3).

2. **Pooled** (N = n_cases × n_days, e.g. 50): all case studies'
   day-resolved arrays concatenated into one stack and treated as N
   exchangeable samples per pixel — ignores which case each day came
   from. This is the "all together" / "10 days × 5 case studies" view.

3. **Averaged-across-cases**: the 5 per-case r_maps (from #1) averaged
   pixel-wise, plus simple arithmetic means of the 5 cases' scalar stats.
   This is descriptive only — p-values are NOT combined into a formal
   joint significance test (that would need e.g. Fisher's method, which
   isn't implemented here) — it keeps case identity separate, unlike #2
   which mixes all days into one flat sample set.

Outputs are saved under ``{output_dir}/{var}/daily/``.

Usage
-----
    python scripts/shap_daily_correlation.py

    python scripts/shap_daily_correlation.py \\
        --config        config/model.json \\
        --case-studies  config/case_studies.json \\
        --dataset       ~/data/data_dailyMean_zpms_1940-2022.nc \\
        --shap-dir      ./shap/mod512 \\
        --output-dir    ./shap/comp_shap \\
        --vars z500 msl peva sm \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import global_land_mask as glm
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from mpl_toolkits.basemap import Basemap
from scipy import stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — kept identical to shap_spatial_correlation.py
# ---------------------------------------------------------------------------
CLIM_START = "1980-01-01"
CLIM_END = "2010-12-31"

SHAP_LAT, SHAP_LON = 20, 30
N_COLORS = 21

# Significance level for stippling
ALPHA = 0.05

VAR_META: dict[str, dict] = {
    "z500": dict(field="z",    threshold=0.30, n_levels=13, vmin=-1200,  vmax=1200),
    "msl":  dict(field="msl",  threshold=0.20, n_levels=13, vmin=-1000,  vmax=1000),
    "peva": dict(field="peva", threshold=None, n_levels=13, vmin=-1e-4,  vmax=1e-4),
    "sm":   dict(field="sm",   threshold=None, n_levels=13, vmin=-0.20,  vmax=0.20),
}

# Tolerance for the "is this value ≈ 0?" test in _nonzero_figure /
# strict_nonzero. Tune based on the magnitude of the sentinel value you
# find in your own unique-value inspection — they differ a lot between
# variables (see shap_spatial_correlation.py conversation history).
_NONZERO_ATOL = {
    "sm":   1e-5,
    "peva": 1e-6,
    "z500": 1e-2,
    "msl":  1e-2,
}


# ---------------------------------------------------------------------------
# Colormaps (unused by _correlation_figure directly — kept for parity with
# shap_spatial_correlation.py / shap_composite.py's cmap_comp parameter,
# which is currently inert there too since the isobar overlay is commented
# out; kept so a future re-enable doesn't need new plumbing).
# ---------------------------------------------------------------------------
def _cmap_isobar() -> ListedColormap:
    c1 = plt.get_cmap("BrBG")(np.linspace(0, 1, N_COLORS))[:-6]
    c2 = plt.get_cmap("PuOr")(np.linspace(0, 1, N_COLORS))
    c1[7] = [0.0, 0.0, 0.0, 0.3]
    c1[8:] = c2[14:]
    return ListedColormap(c1)


def _cmap_white_centre() -> ListedColormap:
    c1 = plt.get_cmap("BrBG")(np.linspace(0, 1, N_COLORS))[:-6]
    c2 = plt.get_cmap("PuOr")(np.linspace(0, 1, N_COLORS))
    c1[6] = [1, 1, 1, 1]
    c1[7] = [1, 1, 1, 1]
    c1[8] = [1, 1, 1, 1]
    c1[9:] = c2[15:]
    return ListedColormap(c1)


# ---------------------------------------------------------------------------
# Config / dataset (identical to shap_spatial_correlation.py)
# ---------------------------------------------------------------------------
def _load_configs(model_path: str, cs_path: str):
    with open(model_path) as f:
        cfg = json.load(f)
    with open(cs_path) as f:
        cs = json.load(f)
    return cfg["domain"], cfg["name"], cs["hw"], cs["no_hw"]


def _open_dataset(path: str, domain: dict) -> xr.Dataset:
    log.info("Opening dataset  %s", path)
    ds = xr.open_dataset(path)
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    return (
        ds
        .sortby("longitude")
        .sortby("latitude")
        .sel(
            latitude=slice(domain["latitude_min"],  domain["latitude_max"]),
            longitude=slice(domain["longitude_min"], domain["longitude_max"]),
        )
    )


def _compute_climatology(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(time=slice(CLIM_START, CLIM_END)).groupby("time.dayofyear").mean("time")


def _event_anomaly_daily(ds, start, end, climatology, field) -> np.ndarray:
    """
    Day-resolved anomaly for *field* over the event window [start, end] —
    the day-axis equivalent of ``_event_anomaly_mean`` (which collapses it
    via ``.mean(dim="time")``). Returns shape ``(N_DAYS, LAT, LON)``.
    """
    event = ds.sel(time=slice(start, end))
    anom = event.groupby("time.dayofyear") - climatology
    return anom[field].values   # (N_DAYS, LAT, LON) — no time-averaging


# ---------------------------------------------------------------------------
# SHAP helpers — per-day ("sum") variant
# ---------------------------------------------------------------------------
def _shap_path_daily(shap_dir, event_type, cs_id, var, name) -> str:
    """
    Path to the per-day SHAP array — saved by shap_regression.py when run
    WITHOUT --meansum (``ms_tag = "sum"``). Shape on disk is
    ``(N_DAYS, H, W, 1)`` (per-variable slice keeps the trailing
    singleton channel dim).
    """
    return os.path.join(
        shap_dir,
        f"cs_{event_type}{cs_id}",
        f"shap_{event_type}_sum_{var}{name}.npy",
    )


def _load_shap_daily(path: str) -> np.ndarray:
    """
    Load a per-day SHAP ``.npy`` and return ``(N_DAYS, SHAP_LAT, SHAP_LON)``,
    in the same lat-ascending orientation as the dataset (flip applied on
    the LAT axis, which is axis 1 here since axis 0 is now days — in the
    case-level loader axis 0 IS the lat axis, so it flips axis 0 there).
    """
    arr = np.load(path)
    if arr.ndim == 4:
        arr = np.squeeze(arr, axis=-1)          # (N_DAYS, H, W) — explicit
                                                 # axis avoids accidentally
                                                 # squeezing N_DAYS==1 away
    elif arr.ndim != 3:
        raise ValueError(
            f"Unexpected shape {arr.shape} for daily SHAP file {path!r}; "
            "expected (N_DAYS, H, W) or (N_DAYS, H, W, 1). Did you run "
            "shap_regression.py WITHOUT --meansum?"
        )
    arr = arr.reshape(arr.shape[0], SHAP_LAT, SHAP_LON)
    return np.flip(arr, axis=1)


def _global_max_daily(shap_dir, event_type, cs_ids, var, name) -> float:
    paths = [_shap_path_daily(shap_dir, event_type, cs_id, var, name) for cs_id in cs_ids]
    maxima = [
        float(np.max(np.abs(_load_shap_daily(p))))
        for p in paths
        if os.path.isfile(p)
    ]
    if not maxima:
        raise FileNotFoundError(
            f"No per-day SHAP .npy found for var={var!r}, event={event_type!r} "
            f"(looked for shap_{event_type}_sum_{var}*.npy under {shap_dir!r}). "
            "Re-run shap_regression.py WITHOUT --meansum to generate these."
        )
    return float(np.max(maxima))


def _combined_max_daily(shap_dir, cs_ids, var, name) -> float:
    return max(
        _global_max_daily(shap_dir, "hw",   cs_ids, var, name),
        _global_max_daily(shap_dir, "nohw", cs_ids, var, name),
    )


def _normalise(arr, norm, threshold):
    out = arr / norm
    if threshold is not None:
        out[np.abs(out) < threshold] = 0.0
    return out


# ---------------------------------------------------------------------------
# Basemap (identical to shap_spatial_correlation.py — half-resolution
# buffer so Basemap.imshow's enforced extent matches the actual pixel grid;
# see that script's docstring for why this matters).
# ---------------------------------------------------------------------------
def _make_basemap(ax: plt.Axes, domain: dict) -> Basemap:
    half_res = domain["resolution"] / 2
    m = Basemap(
        ax=ax, resolution="l",
        llcrnrlon=domain["longitude_min"]-half_res, llcrnrlat=domain["latitude_min"]-half_res,
        urcrnrlon=domain["longitude_max"]+half_res, urcrnrlat=domain["latitude_max"]+half_res,
    )
    m.drawcountries(color="#303338")
    m.drawcoastlines(color="#000000")
    m.drawmeridians(range(0, 360, 10), color="k", labels=[0, 0, 0, 1])
    m.drawparallels(range(-90, 100, 10), color="k", labels=[1, 0, 0, 0])
    return m


# Minimum standard deviation required in both the field and SHAP vectors
# before calling pearsonr at a pixel.  Pixels below this threshold have
# effectively zero variance — correlation is undefined and pearsonr would
# return NaN silently.  The main cause: _normalise's threshold parameter
# sets low-importance SHAP values to exactly 0.0 (not NaN), so a pixel
# where ALL N days have sub-threshold SHAP produces a constant-zero vector.
_ZV_THRESHOLD = 1e-12

# ---------------------------------------------------------------------------
# Core: pixel-wise Pearson r — IDENTICAL to shap_spatial_correlation.py.
# Axis 0 of the inputs can be cases, days, or pooled day×case samples —
# this function only ever sees "N samples per pixel".
# ---------------------------------------------------------------------------
def pixelwise_pearson(
    composites: np.ndarray,   # shape (N, LAT, LON)
    shap_norm:  np.ndarray,   # shape (N, LAT, LON)
    valid_mask: Optional[np.ndarray] = None,   # shape (LAT, LON)
) -> tuple[np.ndarray, np.ndarray]:
    """
    At every grid point compute the Pearson r and two-sided p-value across
    the N samples (cases, days, or pooled day×case — caller's choice).

    valid_mask : optional (LAT, LON) boolean array (e.g. land mask). Points
                 where this is False are skipped entirely and left as NaN.
    """
    N, nlat, nlon = composites.shape
    r_map = np.full((nlat, nlon), np.nan)
    p_map = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            if valid_mask is not None and not valid_mask[i, j]:
                continue
            x = composites[:, i, j]
            y = shap_norm[:, i, j]
            # Drop NaN pairs (e.g. from land mask applied day-by-day)
            finite = ~np.isnan(x) & ~np.isnan(y)
            if finite.sum() < 3:
                continue   # need at least 3 points for df=1 in t-test
            xf, yf = x[finite], y[finite]
            # Skip zero-variance arrays — pearsonr would return NaN with a
            # RuntimeWarning.  This happens for z500/msl when the threshold
            # in _normalise zeroes out ALL days' SHAP at this pixel: the
            # resulting constant vector has undefined correlation.  Leaving
            # r_map as NaN here (explicit skip) rather than relying on
            # pearsonr's silent NaN keeps the behaviour documented and
            # ensures valid_mf_map (which is built from mean arrays, not
            # r_map) still sees these as "non-NaN but zero-variance" —
            # the _ZV_THRESHOLD guard below makes both views consistent.
            if xf.std() < _ZV_THRESHOLD or yf.std() < _ZV_THRESHOLD:
                continue
            r, p = stats.pearsonr(xf, yf)
            r_map[i, j] = r
            p_map[i, j] = p

    return r_map, p_map


# ---------------------------------------------------------------------------
# Summary statistics — IDENTICAL to shap_spatial_correlation.py.
# ---------------------------------------------------------------------------
def summary_statistics(
    composites: np.ndarray,   # (N, LAT, LON)
    shap_norm:  np.ndarray,   # (N, LAT, LON)
    r_map:      np.ndarray,   # (LAT, LON)
    p_map:      np.ndarray,   # (LAT, LON)
    event_type: str,
    var: str,
    *,
    strict_nonzero: bool = False,
    atol: float = 1e-6,
    log_label: str = "",
) -> dict:
    """
    Compute and log scalar correlation measures. See
    shap_spatial_correlation.py's version for the full statistical
    rationale (mean-field / point-cloud / pixel-wise-derived stats,
    NaN-poisoning of pearsonr/spearmanr, strict_nonzero semantics) — the
    logic here is unchanged; only ``log_label`` is new, to disambiguate
    log lines when this is called many times per (var, event_type) (once
    per case, once pooled, etc).
    """
    mean_comp_2d = composites.mean(axis=0)
    mean_shap_2d = shap_norm.mean(axis=0)
    # Base validity: non-NaN in both mean maps AND non-zero-variance at this
    # pixel (pixels skipped by pixelwise_pearson due to zero variance — e.g.
    # all-zero SHAP from the threshold in _normalise — are NaN in r_map; we
    # propagate that here so valid_mf_map matches what actually got computed).
    valid_mf_map = (
        ~np.isnan(mean_comp_2d) &
        ~np.isnan(mean_shap_2d) &
        ~np.isnan(r_map)          # excludes zero-variance pixels
    )

    if strict_nonzero:
        nonzero_mf_map = (
            ~np.isclose(mean_comp_2d, 0.0, atol=atol) &
            ~np.isclose(mean_shap_2d, 0.0, atol=atol)
        )
        valid_mf_map = valid_mf_map & nonzero_mf_map

    mean_comp = mean_comp_2d.ravel()
    mean_shap = mean_shap_2d.ravel()
    valid_mf = valid_mf_map.ravel()

    mfp_r, mfp_p = stats.pearsonr(mean_comp[valid_mf], mean_shap[valid_mf])

    flat_comp = composites.ravel()
    flat_shap = shap_norm.ravel()
    valid_pc = ~np.isnan(flat_comp) & ~np.isnan(flat_shap)
    if strict_nonzero:
        valid_pc = valid_pc & ~np.isclose(flat_comp, 0.0, atol=atol) \
                             & ~np.isclose(flat_shap, 0.0, atol=atol)
    pc_r, pc_p = stats.pearsonr(flat_comp[valid_pc], flat_shap[valid_pc])
    sp_r, sp_p = stats.spearmanr(flat_comp[valid_pc], flat_shap[valid_pc])

    valid = ~np.isnan(r_map)
    if strict_nonzero:
        valid = valid & nonzero_mf_map
    mean_r = float(np.nanmean(np.abs(r_map[valid]))) if valid.any() else np.nan

    # Pixel-wise Spearman across the N samples (same valid mask as r_map).
    nlat, nlon = r_map.shape
    spearman_map = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            if not valid[i, j]:
                continue
            x = composites[:, i, j]
            y = shap_norm[:, i, j]
            finite = ~np.isnan(x) & ~np.isnan(y)
            if finite.sum() < 3:
                continue
            xf, yf = x[finite], y[finite]
            if xf.std() < _ZV_THRESHOLD or yf.std() < _ZV_THRESHOLD:
                continue
            spearman_map[i, j], _ = stats.spearmanr(xf, yf)

    mean_spearman = float(np.nanmean(np.abs(spearman_map[valid]))) if valid.any() else np.nan

    frac_sig = (
        float(np.sum(p_map[valid] < ALPHA) / np.sum(valid))
        if valid.any() else np.nan
    )
    sig_mask = valid & (p_map < ALPHA)
    n_significant = int(sig_mask.sum())
    mean_r_significant = float(np.mean(r_map[sig_mask])) if sig_mask.any() else np.nan

    n_valid_mf,  n_total_mf  = int(np.sum(valid_mf)),  int(valid_mf.size)
    n_valid_pc,  n_total_pc  = int(np.sum(valid_pc)),  int(valid_pc.size)
    n_valid_map, n_total_map = int(np.sum(valid)),     int(valid.size)

    log.info(
        "\n  ──── Valid-pixel counts  [%s / %s%s]  (strict_nonzero=%s%s) ────\n"
        "  Mean-field  (map)         : %d / %d valid  (%.1f %%)\n"
        "  Point-cloud (map × N)     : %d / %d valid  (%.1f %%)\n"
        "  r_map / p_map (map)       : %d / %d valid  (%.1f %%)",
        var, event_type, log_label, strict_nonzero,
        f", atol={atol:g}" if strict_nonzero else "",
        n_valid_mf,  n_total_mf,  100 * n_valid_mf  / n_total_mf,
        n_valid_pc,  n_total_pc,  100 * n_valid_pc  / n_total_pc,
        n_valid_map, n_total_map, 100 * n_valid_map / n_total_map,
    )

    log.info(
        "\n  ──── Spatial correlation  [%s / %s%s] ────\n"
        "  Mean-field Pearson r          : %+.4f   (p = %.4f)\n"
        "  Point-cloud Pearson r         : %+.4f   (p = %.4f)\n"
        "  Point-cloud Spearman ρ        : %+.4f   (p = %.4f)\n"
        "  Mean |pixel-wise r|           : %.4f\n"
        "  Mean |pixel-wise Spearman ρ|  : %.4f\n"
        "  Fraction significant          : %.1f %%\n"
        "  Mean r (significant)          : %+.4f   (n = %d)",
        var, event_type, log_label,
        mfp_r, mfp_p,
        pc_r,  pc_p,
        sp_r,  sp_p,
        mean_r,
        mean_spearman,
        frac_sig * 100,
        mean_r_significant, n_significant,
    )

    return dict(
        mean_field_pearson_r=mfp_r,   mean_field_pearson_p=mfp_p,
        pointcloud_pearson_r=pc_r,    pointcloud_pearson_p=pc_p,
        pointcloud_spearman_r=sp_r,   pointcloud_spearman_p=sp_p,
        mean_pixelwise_r=mean_r,
        mean_pixelwise_spearman=mean_spearman,
        spearman_map=spearman_map,
        fraction_significant=frac_sig,
        mean_r_significant=mean_r_significant,
        n_significant=n_significant,
        valid_mf_map=valid_mf_map,
    )


# ---------------------------------------------------------------------------
# Figure — same layout as shap_spatial_correlation.py's _correlation_figure,
# with the colorbar/title sample-size label made explicit since it's no
# longer always "5 cases".
# ---------------------------------------------------------------------------
def _correlation_figure(
    r_map:        np.ndarray,
    p_map:        np.ndarray,
    mean_comp:    np.ndarray,
    domain:       dict,
    X:            np.ndarray,
    Y:            np.ndarray,
    cmap_comp:    ListedColormap,
    var:          str,
    event_type:   str,
    stats_dict:   dict,
    out_path:     str,
    *,
    sample_label: str = "N samples",
    title_suffix: str = "",
    fixed_vmin: Optional[float] = None,
    fixed_vmax: Optional[float] = None,
    n_levels:   int = 13,
    dpi:        int = 200,
) -> None:
    vmin_c = fixed_vmin if fixed_vmin is not None else float(mean_comp.min())
    vmax_c = fixed_vmax if fixed_vmax is not None else float(mean_comp.max())
    _ = np.linspace(vmin_c, vmax_c, n_levels)   # kept for parity / future isobar re-enable

    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    norm_r = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    img_r = m.imshow(r_map, origin='lower', cmap="RdBu_r", norm=norm_r)

    sig_mask = p_map < ALPHA
    if sig_mask.any():
        lats_sig = Y[sig_mask]
        lons_sig = X[sig_mask]
        xs, ys = m(lons_sig, lats_sig)
        ax.scatter(
            xs, ys, marker=".", s=50, color="k", alpha=0.55, zorder=10,
            label=f"p < {ALPHA}",
        )

    cbar = fig.colorbar(img_r, ax=ax, extend="neither", fraction=0.025,
                        pad=0.02, shrink=0.85)
    cbar.set_label(f"Pixel-wise Pearson r  (across {sample_label})", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=10)

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    txt = (
        f"{ev_label}  ─  {var.upper()}{title_suffix}\n"
        f"Mean |pixel-wise r|  = {stats_dict['mean_pixelwise_r']:.3f}\n"
        f"Mean |pixel-wise ρ|  = {stats_dict['mean_pixelwise_spearman']:.3f}"
    )
    ax.text(
        0.01, 0.98, txt, transform=ax.transAxes, fontsize=20,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.80, ec="grey"),
        family="monospace", zorder=20,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)


# ---------------------------------------------------------------------------
# Figure: averaged-across-cases r_map (Variant B — descriptive only)
# ---------------------------------------------------------------------------
def _avg_rmap_figure(
    r_map_avg: np.ndarray,   # (LAT, LON) — nanmean of the 5 per-case r_maps
    domain:    dict,
    var:       str,
    event_type: str,
    avg_stats: dict,         # arithmetic means of the 5 per-case scalar stats
    out_path:  str,
    *,
    dpi: int = 200,
) -> None:
    """
    Pixel-wise average of the 5 per-case r_maps, kept separate (NOT mixed
    with the pooled N=50 analysis). This is explicitly descriptive: no new
    p-value is computed here — averaging p-values across 5 independent
    tests is not a valid combined-significance procedure (Fisher's method
    would be the correct tool if you want that; not implemented here).
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    norm_r = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    img_r = m.imshow(r_map_avg, origin="lower", cmap="RdBu_r", norm=norm_r)

    cbar = fig.colorbar(img_r, ax=ax, extend="neither", fraction=0.025,
                        pad=0.02, shrink=0.85)
    cbar.set_label("Pixel-wise Pearson r — mean of the 5 per-case r_maps", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=10)

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    txt = (
        f"{ev_label}  ─  {var.upper()}  (mean of 5 per-case results)\n"
        f"Mean |pixel-wise r|  (avg) = {avg_stats['mean_pixelwise_r']:.3f}\n"
        f"Mean |pixel-wise ρ|  (avg) = {avg_stats['mean_pixelwise_spearman']:.3f}"
    )
    ax.text(
        0.01, 0.98, txt, transform=ax.transAxes, fontsize=20,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.80, ec="grey"),
        family="monospace", zorder=20,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)


# ---------------------------------------------------------------------------
# Figure: ranked-pixel absolute correlation map (same as in
# shap_spatial_correlation.py — see that script for full docstring)
# ---------------------------------------------------------------------------
_RANK_EDGES  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_RANK_COLORS = ["#f0f0f0", "#fdd49e", "#fdbb84", "#fc8d59", "#b30000"]


def _ranked_correlation_figure(
    r_map:      np.ndarray,
    domain:     dict,
    var:        str,
    event_type: str,
    out_path:   str,
    *,
    edges:        list[float] = _RANK_EDGES,
    colors:       list[str]   = _RANK_COLORS,
    title_suffix: str         = "",
    dpi: int = 200,
) -> None:
    abs_r = np.abs(r_map)
    n_bins = len(edges) - 1
    cat = np.full(r_map.shape, np.nan)
    valid = ~np.isnan(abs_r)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = valid & (abs_r >= lo) & (abs_r < hi if k < n_bins - 1 else abs_r <= hi)
        cat[mask] = k

    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)
    cmap = ListedColormap(colors)
    m.imshow(np.ma.masked_invalid(cat), origin="lower", cmap=cmap,
             vmin=-0.5, vmax=n_bins - 0.5)

    n_valid = int(valid.sum())
    counts = [int(np.sum(cat == k)) for k in range(n_bins)]
    handles = [
        Patch(
            facecolor=colors[k], edgecolor="k",
            label=(
                f"|r| ∈ [{edges[k]:.1f}, {edges[k+1]:.1f}"
                f"{')'if k<n_bins-1 else ']'}"
                f"  —  {counts[k]} px  ({100*counts[k]/n_valid:.1f}%)"
            ),
        )
        for k in range(n_bins)
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=8,
              title=f"Absolute pixel-wise |r|  (N valid = {n_valid})")

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    ax.set_title(f"Ranked |r| map  —  {var.upper()} / {ev_label}{title_suffix}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)


# ---------------------------------------------------------------------------
# Figure: valid_mf mask
# ---------------------------------------------------------------------------
def _valid_mask_figure(
    valid_mask: np.ndarray,
    domain:     dict,
    var:        str,
    event_type: str,
    out_path:   str,
    *,
    title_suffix: str = "",
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    cmap = ListedColormap(["#d9d9d9", "#2b8cbe"])
    m.imshow(valid_mask.astype(float), origin="lower", cmap=cmap, vmin=0, vmax=1)

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    n_valid, n_total = int(valid_mask.sum()), int(valid_mask.size)
    ax.set_title(
        f"Valid mean-field pixels  —  {var.upper()} / {ev_label}{title_suffix}\n"
        f"{n_valid} / {n_total}  ({100 * n_valid / n_total:.1f} %) valid"
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)


# ---------------------------------------------------------------------------
# Per-variable, per-event pipeline
# ---------------------------------------------------------------------------
def _process_variable_daily(
    var:         str,
    event_type:  str,
    ds:          xr.Dataset,
    climatology: xr.Dataset,
    hw_cases:    list[dict],
    nohw_cases:  list[dict],
    domain:      dict,
    X:           np.ndarray,
    Y:           np.ndarray,
    shap_dir:    str,
    output_dir:  str,
    model_name:  str,
    *,
    strict_nonzero: bool = False,
) -> dict[str, dict]:
    """
    Day-resolved version of shap_spatial_correlation.py's
    ``_process_variable``. Returns a dict of stats keyed by
    ``"case{id}"`` (×5), ``"pooled"``, and ``"avgcases"``.
    """
    meta = VAR_META[var]
    threshold = meta["threshold"]
    n_levels = meta["n_levels"]
    fixed_vmin = meta["vmin"]
    fixed_vmax = meta["vmax"]
    atol = _NONZERO_ATOL.get(var, 1e-6)

    cmap_comp = _cmap_isobar() if var in ("z500", "msl") else _cmap_white_centre()

    cases = hw_cases if event_type == "hw" else nohw_cases
    cs_ids = [c["id"] for c in cases]

    land_mask = None
    if var in ("sm", "peva"):
        land_mask = glm.is_land(Y, X)
        log.info("  [%s / %s]  Land mask derived: %.1f%% land points",
                 var, event_type, 100 * np.mean(land_mask))

    if var in ("z500", "msl"):
        norm = _global_max_daily(shap_dir, event_type, cs_ids, var, model_name)
    else:
        norm = _combined_max_daily(shap_dir, cs_ids, var, model_name)
    log.info("  [%s / %s]  norm (daily) = %.6g", var, event_type, norm)

    out_dir = os.path.join(output_dir, var, "daily")

    field_per_case: list[np.ndarray] = []
    shap_per_case:  list[np.ndarray] = []
    results: dict[str, dict] = {}

    # ── 1. Per-case-study, day-resolved analysis ────────────────────────────
    for case in tqdm(cases, desc=f"Loading {var}/{event_type} (daily)", leave=False):
        cs_id = case["id"]

        field_daily = _event_anomaly_daily(
            ds, case["start"], case["end"], climatology, meta["field"]
        )   # (N_DAYS, LAT, LON)

        shap_raw_daily = _load_shap_daily(
            _shap_path_daily(shap_dir, event_type, cs_id, var, model_name)
        )   # (N_DAYS, SHAP_LAT, SHAP_LON)
        shap_daily = _normalise(shap_raw_daily, norm, threshold)

        if field_daily.shape[0] != shap_daily.shape[0]:
            raise ValueError(
                f"Day-count mismatch for case {cs_id} ({var}/{event_type}): "
                f"field has {field_daily.shape[0]} days, SHAP has "
                f"{shap_daily.shape[0]} days. Check that the case-study "
                f"window in case_studies.json matches what was used when "
                f"shap_regression.py generated this .npy."
            )

        if land_mask is not None:
            field_daily[:, ~land_mask] = np.nan
            shap_daily[:,  ~land_mask] = np.nan

        n_days = field_daily.shape[0]

        r_map_c, p_map_c = pixelwise_pearson(field_daily, shap_daily, valid_mask=land_mask)
        stats_c = summary_statistics(
            field_daily, shap_daily, r_map_c, p_map_c, event_type, var,
            strict_nonzero=strict_nonzero, atol=atol,
            log_label=f", case {cs_id}, {n_days} days",
        )
        stats_c["r_map"] = r_map_c
        stats_c["n_days"] = n_days

        mean_field_c = field_daily.mean(axis=0)
        mean_shap_c  = shap_daily.mean(axis=0)
        if land_mask is not None:
            mean_field_c[~land_mask] = np.nan
            mean_shap_c[~land_mask]  = np.nan

        case_out_path = os.path.join(
            out_dir, f"daily_corr_{event_type}_{var}_case{cs_id}{model_name}.png"
        )
        _correlation_figure(
            r_map_c, p_map_c, mean_field_c,
            domain, X, Y, cmap_comp, var, event_type, stats_c,
            case_out_path,
            sample_label=f"{n_days} days, case {cs_id}",
            title_suffix=f"  (case {cs_id}, {n_days} days)",
            fixed_vmin=fixed_vmin, fixed_vmax=fixed_vmax, n_levels=n_levels,
        )

        _ranked_correlation_figure(
            r_map_c, domain, var, event_type,
            os.path.join(out_dir, f"ranked_corr_{event_type}_{var}_case{cs_id}{model_name}.png"),
            title_suffix=f"  (case {cs_id}, {n_days} days)",
        )

        results[f"case{cs_id}"] = stats_c
        field_per_case.append(field_daily)
        shap_per_case.append(shap_daily)

    # ── 2. Pooled (all cases' days mixed into one N=Σdays sample set) ──────
    field_pooled = np.concatenate(field_per_case, axis=0)   # (Σ days, LAT, LON)
    shap_pooled  = np.concatenate(shap_per_case,  axis=0)
    n_pooled = field_pooled.shape[0]

    r_map_p, p_map_p = pixelwise_pearson(field_pooled, shap_pooled, valid_mask=land_mask)
    stats_p = summary_statistics(
        field_pooled, shap_pooled, r_map_p, p_map_p, event_type, var,
        strict_nonzero=strict_nonzero, atol=atol,
        log_label=f", pooled, {n_pooled} day×case samples",
    )

    mean_field_p = field_pooled.mean(axis=0)
    mean_shap_p  = shap_pooled.mean(axis=0)
    if land_mask is not None:
        mean_field_p[~land_mask] = np.nan
        mean_shap_p[~land_mask]  = np.nan

    pooled_out_path = os.path.join(
        out_dir, f"daily_corr_pooled_{event_type}_{var}{model_name}.png"
    )
    _correlation_figure(
        r_map_p, p_map_p, mean_field_p,
        domain, X, Y, cmap_comp, var, event_type, stats_p,
        pooled_out_path,
        sample_label=f"{n_pooled} pooled day×case samples",
        title_suffix=f"  (pooled, {n_pooled} samples)",
        fixed_vmin=fixed_vmin, fixed_vmax=fixed_vmax, n_levels=n_levels,
    )

    _ranked_correlation_figure(
        r_map_p, domain, var, event_type,
        os.path.join(out_dir, f"ranked_corr_pooled_{event_type}_{var}{model_name}.png"),
        title_suffix=f"  (pooled, {n_pooled} samples)",
    )

    _valid_mask_figure(
        stats_p["valid_mf_map"], domain, var, event_type,
        os.path.join(out_dir, f"valid_mask_pooled_{event_type}_{var}{model_name}.png"),
        title_suffix=f"  (pooled, {n_pooled} samples)",
    )

    results["pooled"] = stats_p

    # ── 3. Averaged-across-cases (descriptive; case identity preserved) ────
    r_maps_stack = np.stack([results[f"case{cid}"]["r_map"] for cid in cs_ids], axis=0)
    r_map_avg = np.nanmean(r_maps_stack, axis=0)

    scalar_keys = [
        "mean_field_pearson_r", "pointcloud_pearson_r", "pointcloud_spearman_r",
        "mean_pixelwise_r", "mean_pixelwise_spearman", "fraction_significant",
        "mean_r_significant",
    ]
    avg_stats = {
        k: float(np.nanmean([results[f"case{cid}"][k] for cid in cs_ids]))
        for k in scalar_keys
    }
    avg_stats["n_significant"] = int(round(
        float(np.mean([results[f"case{cid}"]["n_significant"] for cid in cs_ids]))
    ))
    avg_stats["mean_field_pearson_p"] = float("nan")
    avg_stats["pointcloud_pearson_p"] = float("nan")
    avg_stats["pointcloud_spearman_p"] = float("nan")

    log.info(
        "\n  ──── Averaged across %d cases  [%s / %s] (descriptive only) ────\n"
        "  Mean |pixel-wise r| (avg) : %.4f\n"
        "  Mean |pixel-wise ρ| (avg) : %.4f",
        len(cs_ids), var, event_type,
        avg_stats["mean_pixelwise_r"],
        avg_stats["mean_pixelwise_spearman"],
    )

    avgcases_out_path = os.path.join(
        out_dir, f"daily_corr_avgcases_{event_type}_{var}{model_name}.png"
    )
    _avg_rmap_figure(
        r_map_avg, domain, var, event_type, avg_stats, avgcases_out_path,
    )

    _ranked_correlation_figure(
        r_map_avg, domain, var, event_type,
        os.path.join(out_dir, f"ranked_corr_avgcases_{event_type}_{var}{model_name}.png"),
        title_suffix=f"  (avg of {len(cs_ids)} per-case r_maps)",
    )

    results["avgcases"] = avg_stats

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Day-resolved pixel-wise correlation between normalised SHAP "
            "values and anomaly fields — per case study, pooled across all "
            "cases, and averaged across cases."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="./config/model.json")
    p.add_argument("--case-studies", default="./config/case_studies.json",
                   dest="case_studies")
    p.add_argument("--dataset",      default=None)
    p.add_argument("--shap-dir",     default="./shap/mod512", dest="shap_dir")
    p.add_argument("--output-dir",   default="./shap/comp_shap", dest="output_dir")
    p.add_argument("--vars", nargs="+",
                   default=["z500", "msl", "peva", "sm"],
                   choices=["z500", "msl", "peva", "sm"])
    p.add_argument(
        "--strict-nonzero", action="store_true", dest="strict_nonzero",
        help=(
            "Same semantics as in shap_spatial_correlation.py: valid_mf/"
            "valid_pc/valid additionally require both field and SHAP "
            "non-zero (within a variable-specific atol), not just non-NaN."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = _build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    domain, model_name, hw_cases, nohw_cases = _load_configs(
        args.config, args.case_studies
    )
    with open(args.config) as f:
        cfg = json.load(f)

    dataset_path = args.dataset or cfg["datasets"]["pred_dataset"]

    lon_range = np.arange(
        domain["longitude_min"],
        domain["longitude_max"] + domain["resolution"],
        domain["resolution"],
    )
    lat_range = np.arange(
        domain["latitude_min"],
        domain["latitude_max"] + domain["resolution"],
        domain["resolution"],
    )
    X, Y = np.meshgrid(lon_range, lat_range)

    ds = _open_dataset(dataset_path, domain)
    climatology = _compute_climatology(ds)
    log.info("Climatology computed  (%s – %s)", CLIM_START, CLIM_END)
    log.info(
        "strict_nonzero = %s", args.strict_nonzero,
    )

    plt.rcParams.update({"font.size": 13})

    all_stats: dict[str, dict] = {}

    combos = [(var, evt) for var in args.vars for evt in ("hw", "nohw")]
    for var, event_type in tqdm(combos, desc="Processing"):
        log.info("Processing  var=%-4s  event=%s  (daily)", var, event_type)
        try:
            res = _process_variable_daily(
                var=var,
                event_type=event_type,
                ds=ds,
                climatology=climatology,
                hw_cases=hw_cases,
                nohw_cases=nohw_cases,
                domain=domain,
                X=X, Y=Y,
                shap_dir=args.shap_dir,
                output_dir=args.output_dir,
                model_name=model_name,
                strict_nonzero=args.strict_nonzero,
            )
            for sub_key, s in res.items():
                all_stats[f"{var}_{event_type}_{sub_key}"] = s
        except FileNotFoundError as exc:
            log.error("  Skipping %s / %s – %s", var, event_type, exc)

    # ── Final summary table ───────────────────────────────────────────────
    log.info("\n\n  ═══════════════════════════════════════════════════")
    log.info("                 SUMMARY TABLE  (daily)")
    log.info("  ═══════════════════════════════════════════════════")
    header = f"  {'Key':<26} {'Mean|r|':>10} {'Mean|ρ|':>10}"
    log.info(header)
    log.info("  " + "─" * 50)
    for key, s in all_stats.items():
        log.info(
            "  %-26s  %.4f      %.4f",
            key,
            s["mean_pixelwise_r"],
            s["mean_pixelwise_spearman"],
        )
    log.info("  ═══════════════════════════════════════════════════")
    log.info("Done.  Figures saved to  %s", args.output_dir)


if __name__ == "__main__":
    main()