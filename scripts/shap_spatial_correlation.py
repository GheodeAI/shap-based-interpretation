"""
shap_spatial_correlation.py
============================
Pixel-wise spatial correlation between normalised SHAP values and composite
anomaly fields across the 5 HW and 5 NO-HW case studies.

For each variable two outputs are produced:

  1. **Pixel-wise Pearson-r map** – at every grid point (lat, lon) the
     Pearson r is computed across the N=5 cases (anomaly vs SHAP).
     Gives a (LAT, LON) map of *local* agreement.

  2. **Summary statistics (printed / logged)**:
       • Mean-field Pearson r  – correlation between the mean composite and
         mean SHAP maps (scalar, one per event type).
       • Point-cloud Pearson r – all (lat×lon×N_cases) pairs flattened.
       • Point-cloud Spearman ρ – non-parametric version of the above.

Two figures are saved per variable (HW + NO-HW), styled the same way as
the existing ``shap_composite.py`` output, showing:

  * Background  : pixel-wise Pearson-r map (filled colour, RdBu_r).
  * Contour lines: mean anomaly isobars for spatial orientation.
  * Stippling    : grid points where the correlation is significant
                   at the p < 0.05 level (two-sided t-test with N−2 d.f.).

Outputs are saved under ``{output_dir}/{var}/``.

Usage
-----
    python scripts/shap_spatial_correlation.py

    python scripts/shap_spatial_correlation.py \\
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
# Constants – kept identical to shap_composite.py
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


# ---------------------------------------------------------------------------
# Colormaps  (re-used from shap_composite.py)
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
# Config / dataset  (mirror of shap_composite.py helpers)
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


def _event_anomaly_mean(ds, start, end, climatology, field) -> np.ndarray:
    event = ds.sel(time=slice(start, end))
    anom = event.groupby("time.dayofyear") - climatology
    return anom.mean(dim="time")[field].values


# ---------------------------------------------------------------------------
# SHAP helpers  (mirror of shap_composite.py)
# ---------------------------------------------------------------------------
def _shap_path(shap_dir, event_type, cs_id, var, name):
    return os.path.join(
        shap_dir,
        f"cs_{event_type}{cs_id}",
        f"shap_{event_type}_meansum_{var}{name}.npy",
    )


def _load_shap(path: str) -> np.ndarray:
    arr = np.load(path)
    return np.flip(arr.squeeze().reshape(SHAP_LAT, SHAP_LON), axis=0)


def _global_max(shap_dir, event_type, cs_ids, var, name) -> float:
    maxima = [
        float(np.max(np.abs(_load_shap(_shap_path(shap_dir, event_type, cs_id, var, name)))))
        for cs_id in cs_ids
        if os.path.isfile(_shap_path(shap_dir, event_type, cs_id, var, name))
    ]
    if not maxima:
        raise FileNotFoundError(f"No SHAP .npy for var={var!r}, event={event_type!r}")
    return float(np.max(maxima))


def _combined_max(shap_dir, cs_ids, var, name) -> float:
    return max(
        _global_max(shap_dir, "hw",   cs_ids, var, name),
        _global_max(shap_dir, "nohw", cs_ids, var, name),
    )


def _normalise(arr, norm, threshold):
    out = arr / norm
    if threshold is not None:
        out[np.abs(out) < threshold] = 0.0
    return out


# ---------------------------------------------------------------------------
# Basemap
# ---------------------------------------------------------------------------
def _make_basemap(ax: plt.Axes, domain: dict) -> Basemap:
    """
    NOTE: Basemap.imshow() silently OVERWRITES any ``extent=`` kwarg with
    (self.llcrnrx, self.urcrnrx, self.llcrnry, self.urcrnry) — i.e. the
    corners passed to this Basemap constructor. To keep the pixel-wise
    r_map imshow in ``_correlation_figure`` aligned with the actual grid
    (pixel centres at the lat/lon grid points), the corners here must use
    a half-resolution buffer, matching the ``extent`` computed there —
    NOT a full-resolution buffer (which shifts the image by half a grid
    cell relative to the true coastline / grid points).
    """
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


# Minimum standard deviation required before calling pearsonr at a pixel.
# Pixels below this have zero variance — the threshold in _normalise sets
# low-importance SHAP to exactly 0.0 (not NaN), so if ALL N cases at a
# pixel are sub-threshold, pearsonr would return NaN silently. We skip
# explicitly and propagate to valid_mf_map via summary_statistics.
_ZV_THRESHOLD = 1e-12

# ---------------------------------------------------------------------------
# Core: pixel-wise Pearson r
# ---------------------------------------------------------------------------
def pixelwise_pearson(
    composites: np.ndarray,   # shape (N, LAT, LON)
    shap_norm:  np.ndarray,   # shape (N, LAT, LON)
    valid_mask: Optional[np.ndarray] = None,   # shape (LAT, LON)
) -> tuple[np.ndarray, np.ndarray]:
    """
    At every grid point compute the Pearson r and two-sided p-value across
    the N case studies.

    Parameters
    ----------
    composites : (N, LAT, LON) anomaly stack.
    shap_norm  : (N, LAT, LON) normalised SHAP stack.
    valid_mask : optional (LAT, LON) boolean array (e.g. land mask). Points
                 where this is False are skipped entirely and left as NaN
                 in the outputs — avoids pearsonr being called on all-NaN
                 columns (sea points for SM/PEva) and the resulting flood
                 of RuntimeWarnings.

    Returns
    -------
    r_map : (LAT, LON) Pearson correlation coefficient.
    p_map : (LAT, LON) two-sided p-value (t-test, df = N−2).
    """
    N, nlat, nlon = composites.shape
    r_map = np.full((nlat, nlon), np.nan)
    p_map = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            if valid_mask is not None and not valid_mask[i, j]:
                continue  # not land (or otherwise masked) → leave as NaN
            x = composites[:, i, j]
            y = shap_norm[:, i, j]
            finite = ~np.isnan(x) & ~np.isnan(y)
            if finite.sum() < 3:
                continue
            xf, yf = x[finite], y[finite]
            # Skip zero-variance: _normalise's threshold sets sub-threshold
            # SHAP to 0.0 (not NaN), so if ALL N cases at a pixel are below
            # threshold, the SHAP vector is constant-zero → r undefined.
            if xf.std() < _ZV_THRESHOLD or yf.std() < _ZV_THRESHOLD:
                continue
            r, p = stats.pearsonr(xf, yf)
            r_map[i, j] = r
            p_map[i, j] = p

    return r_map, p_map


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary_statistics(
    composites: np.ndarray,   # (N, LAT, LON)
    shap_norm:  np.ndarray,   # (N, LAT, LON)
    r_map:      np.ndarray,   # (LAT, LON)
    p_map:      np.ndarray,   # (LAT, LON)
    event_type: str,
    var: str,
) -> dict:
    """
    Compute and log scalar correlation measures for one variable/event type.

    Three complementary statistics are reported:

    mean_field_pearson
        Pearson r between the N-case *mean* composite map and the N-case
        *mean* SHAP map (both flattened to 1-D).  Answers: "Do the spatial
        patterns of the typical event agree?"

    pointcloud_pearson / pointcloud_spearman
        Computed over all (N × LAT × LON) paired values.  Answers: "Across
        all cases and all locations, do high-anomaly regions tend to have
        high SHAP importance?"

    mean_r / fraction_significant
        Derived from the pixel-wise r_map / p_map.  Answers: "Where do
        case-to-case variations in the anomaly co-vary with case-to-case
        SHAP variations?"

    Notes
    -----
    Sea/non-land points are NaN in ``composites`` / ``shap_norm`` (and
    consequently in ``r_map`` / ``p_map``) whenever a land mask was applied
    upstream (SM, PEva). ``scipy.stats.pearsonr``/``spearmanr`` do **not**
    ignore NaNs — a single NaN anywhere in the input poisons the *entire*
    result to NaN, not just that one point. So we explicitly drop NaN
    pairs before correlating.
    """
    mean_comp_2d = composites.mean(axis=0)   # (LAT, LON)
    mean_shap_2d = shap_norm.mean(axis=0)    # (LAT, LON)
    # Include ~np.isnan(r_map) so that pixels skipped by pixelwise_pearson
    # due to zero variance (all-zero SHAP from _normalise's threshold) are
    # also excluded here — keeps valid_mf_map consistent with the figure.
    valid_mf_map = (
        ~np.isnan(mean_comp_2d) &
        ~np.isnan(mean_shap_2d) &
        ~np.isnan(r_map)
    )

    mean_comp = mean_comp_2d.ravel()
    mean_shap = mean_shap_2d.ravel()
    valid_mf = valid_mf_map.ravel()

    mfp_r, mfp_p = stats.pearsonr(mean_comp[valid_mf], mean_shap[valid_mf])

    flat_comp = composites.ravel()
    flat_shap = shap_norm.ravel()
    valid_pc = ~np.isnan(flat_comp) & ~np.isnan(flat_shap)
    pc_r, pc_p = stats.pearsonr(flat_comp[valid_pc], flat_shap[valid_pc])
    sp_r, sp_p = stats.spearmanr(flat_comp[valid_pc], flat_shap[valid_pc])

    valid = ~np.isnan(r_map)
    mean_r = float(np.nanmean(np.abs(r_map[valid]))) if valid.any() else np.nan

    # Pixel-wise Spearman: same valid mask, computed per-pixel across N samples.
    # We reuse the flat arrays but compute row-by-row across axis=0.
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
        "\n  ──── Valid-pixel counts  [%s / %s] ────\n"
        "  Mean-field  (map)         : %d / %d valid  (%.1f %%)\n"
        "  Point-cloud (map × cases) : %d / %d valid  (%.1f %%)\n"
        "  r_map / p_map (map)       : %d / %d valid  (%.1f %%)",
        var, event_type,
        n_valid_mf,  n_total_mf,  100 * n_valid_mf  / n_total_mf,
        n_valid_pc,  n_total_pc,  100 * n_valid_pc  / n_total_pc,
        n_valid_map, n_total_map, 100 * n_valid_map / n_total_map,
    )

    log.info(
        "\n  ──── Spatial correlation  [%s / %s] ────\n"
        "  Mean-field Pearson r          : %+.4f   (p = %.4f)\n"
        "  Point-cloud Pearson r         : %+.4f   (p = %.4f)\n"
        "  Point-cloud Spearman ρ        : %+.4f   (p = %.4f)\n"
        "  Mean |pixel-wise r|           : %.4f\n"
        "  Mean |pixel-wise Spearman ρ|  : %.4f\n"
        "  Fraction significant          : %.1f %%\n"
        "  Mean r (significant)          : %+.4f   (n = %d)",
        var, event_type,
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
# Figure
# ---------------------------------------------------------------------------
def _correlation_figure(
    r_map:      np.ndarray,   # (LAT, LON)  pixel-wise Pearson r
    p_map:      np.ndarray,   # (LAT, LON)  two-sided p-value
    mean_comp:  np.ndarray,   # (LAT, LON)  mean anomaly (for isobar overlay)
    domain:     dict,
    X:          np.ndarray,
    Y:          np.ndarray,
    cmap_comp:  ListedColormap,
    var:        str,
    event_type: str,
    stats_dict: dict,
    out_path:   str,
    *,
    fixed_vmin: Optional[float] = None,
    fixed_vmax: Optional[float] = None,
    n_levels:   int = 13,
    dpi:        int = 200,
) -> None:
    """
    Plot the pixel-wise Pearson-r correlation map.

    Figure layers (bottom → top)
    ----------------------------
    1. ``imshow``   – pixel-wise Pearson r (RdBu_r, −1 … +1 centred on 0).
    2. ``contourf`` / ``contour`` – mean anomaly isobars for orientation.
    3. ``scatter``  – stippling at grid points where p < ALPHA.
    4. Two colourbars + summary-stats text box.
    """
    vmin_c = fixed_vmin if fixed_vmin is not None else float(mean_comp.min())
    vmax_c = fixed_vmax if fixed_vmax is not None else float(mean_comp.max())
    comp_levels = np.linspace(vmin_c, vmax_c, n_levels)

    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    # ── 1. Pixel-wise r background ────────────────────────────────────────
    # Diverging, symmetric around 0, always in [−1, +1]
    norm_r = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # NOTE: Basemap.imshow() ignores any extent= kwarg and always uses the
    # Basemap's own corners (llcrnrx/urcrnrx/llcrnry/urcrnry, set in
    # _make_basemap with a half-resolution buffer) — so we don't pass
    # extent here; passing one would be silently discarded and could give
    # the false impression that it controls the alignment.
    img_r = m.imshow(
        r_map,
        origin='lower',
        cmap="RdBu_r",
        norm=norm_r
    )

    # ── 2. Mean-anomaly isobar overlay ────────────────────────────────────
    #if var in ("z500", "msl"):
    #    ax.contour(
    #        X, Y, mean_comp,
    #        levels=comp_levels,
    #        linewidths=2.5,
    #        cmap=cmap_comp,
    #        alpha=0.40,
    #    )
    #else:
    #    ax.contourf(
    #        X, Y, mean_comp,
    #        levels=comp_levels,
    #        cmap=cmap_comp,
    #        alpha=0.35,
    #        zorder=3,
    #        vmin=vmin_c, vmax=vmax_c,
    #    )

    # ── 3. Stippling: significant pixels (p < ALPHA) ──────────────────────
    sig_mask = p_map < ALPHA
    if sig_mask.any():
        # Convert grid indices to map coords for scatter
        lats_sig = Y[sig_mask]
        lons_sig = X[sig_mask]
        xs, ys = m(lons_sig, lats_sig)
        ax.scatter(
            xs, ys,
            marker=".", s=50, color="k", alpha=0.55, zorder=10,
            label=f"p < {ALPHA}",
        )

    # ── 4. Colourbar: pixel-wise r ────────────────────────────────────────
    cbar = fig.colorbar(img_r, ax=ax, extend="neither", fraction=0.025,
                        pad=0.02, shrink=0.85)
    cbar.set_label("Pixel-wise Pearson r", fontsize=15)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=15)

    # ── 5. Stats text box ─────────────────────────────────────────────────
    ev_label = "HW" if event_type == "hw" else "NO-HW"
    txt = (
        f"{ev_label}  ─  {var.upper()}\n"
        f"Mean |pixel-wise r|  = {stats_dict['mean_pixelwise_r']:.3f}\n"
        f"Mean |pixel-wise ρ|  = {stats_dict['mean_pixelwise_spearman']:.3f}"
    )
    ax.text(
        0.01, 0.98, txt,
        transform=ax.transAxes, fontsize=20,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.80, ec="grey"),
        family="monospace",
        zorder=20,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)


# ---------------------------------------------------------------------------
# Figure: valid_mf mask (sanity check against real coastlines)
# ---------------------------------------------------------------------------
def _valid_mask_figure(
    valid_mask: np.ndarray,   # (LAT, LON) bool — True = used in the stat
    domain:     dict,
    var:        str,
    event_type: str,
    out_path:   str,
    *,
    dpi: int = 200,
) -> None:
    """
    Plot which grid points are flagged valid (e.g. ``valid_mf`` from
    ``summary_statistics``) on top of the same basemap (real coastlines)
    used elsewhere — a quick visual sanity check that the mask lines up
    with actual land, independent of whatever numbers get logged.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    cmap = ListedColormap(["#d9d9d9", "#2b8cbe"])  # grey = excluded, blue = valid
    m.imshow(
        valid_mask.astype(float),
        origin="lower",
        cmap=cmap,
        vmin=0, vmax=1,
    )

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    n_valid, n_total = int(valid_mask.sum()), int(valid_mask.size)
    ax.set_title(
        f"Valid mean-field pixels  —  {var.upper()} / {ev_label}\n"
        f"{n_valid} / {n_total}  ({100 * n_valid / n_total:.1f} %) valid"
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)



# ---------------------------------------------------------------------------
# Figure: ranked-pixel absolute correlation map
# ---------------------------------------------------------------------------
# Bin edges for |r| — half-open intervals [edges[i], edges[i+1]).
# The last bin is closed at 1.0. Tune if you want finer/coarser granularity.
_RANK_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Colours for each bin — from light grey (low |r|) to deep red (high |r|).
# Must have len(_RANK_EDGES) - 1 entries.
_RANK_COLORS = ["#f0f0f0", "#fdd49e", "#fdbb84", "#fc8d59", "#b30000"]


def _ranked_correlation_figure(
    r_map:      np.ndarray,    # (LAT, LON) — pixel-wise Pearson r; NaN = invalid
    domain:     dict,
    var:        str,
    event_type: str,
    out_path:   str,
    *,
    edges:  list[float] = _RANK_EDGES,
    colors: list[str]   = _RANK_COLORS,
    title_suffix: str   = "",
    dpi: int = 200,
) -> None:
    """
    Rank pixels by the absolute value of their pixel-wise Pearson r into
    ``len(edges)-1`` bins defined by ``edges`` and plot them on the basemap.

    Each bin covers the half-open interval [edges[i], edges[i+1]), except
    the last which is closed at 1.0.  Invalid pixels (NaN in r_map — ocean,
    zero-variance, or land-masked) are shown in the background basemap colour
    and are not assigned to any bin.

    The intention is to give a spatial intuition of *where* strong
    correlations are concentrated, independent of sign (which the main
    _correlation_figure already shows) — e.g. "are high-|r| pixels clustered
    over the Iberian Peninsula or scattered uniformly?".

    Parameters
    ----------
    r_map : (LAT, LON)
        Pixel-wise Pearson r — the same array plotted in _correlation_figure.
    edges : list of float
        Bin boundaries for |r|, e.g. [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
    colors : list of str
        One colour per bin (len = len(edges)-1). Default: light-to-dark
        orange-red sequential palette.
    """
    assert len(colors) == len(edges) - 1, \
        f"Need {len(edges)-1} colours for {len(edges)-1} bins, got {len(colors)}"

    abs_r = np.abs(r_map)   # NaN preserved
    n_bins = len(edges) - 1

    # Build integer category array: 0 = first bin, ..., n_bins-1 = last bin,
    # NaN stays NaN (we'll use a masked array for imshow).
    cat = np.full(r_map.shape, np.nan)
    valid = ~np.isnan(abs_r)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k < n_bins - 1:
            mask = valid & (abs_r >= lo) & (abs_r < hi)
        else:
            mask = valid & (abs_r >= lo) & (abs_r <= hi)
        cat[mask] = k

    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    cmap = ListedColormap(colors)
    m.imshow(
        np.ma.masked_invalid(cat),
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=n_bins - 0.5,
    )

    # Legend
    counts = [
        int(np.sum(cat == k)) for k in range(n_bins)
    ]
    n_valid = int(valid.sum())
    handles = [
        Patch(
            facecolor=colors[k], edgecolor="k",
            label=(
                f"|r| ∈ [{edges[k]:.1f}, {edges[k+1]:.1f}{')'if k<n_bins-1 else ']'}"
                f"  —  {counts[k]} px  ({100*counts[k]/n_valid:.1f}%)"
            ),
        )
        for k in range(n_bins)
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=14,
          title=f"Absolute pixel-wise |r|", title_fontsize=16)

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    # ax.set_title(
    #     f"Ranked |r| map  —  {var.upper()} / {ev_label}{title_suffix}"
    # )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)


# ---------------------------------------------------------------------------
# Figure: exact-zero vs non-zero map
# Tolerance for the "is this value ≈ 0?" test in _nonzero_figure. Tune
# these based on the magnitude of the sentinel value you find in your own
# unique-value inspection — they differ a lot between variables.
_NONZERO_ATOL = {
    "sm":   1e-5,
    "peva": 1e-6,
    "z500": 1e-2,
    "msl": 1e-2,
}


def _nonzero_figure(
    field:      np.ndarray,   # (LAT, LON) — may or may not contain NaN
    domain:     dict,
    var:        str,
    event_type: str,
    label:      str,          # e.g. "raw field (climatology, day 0)" or "mean_comp"
    out_path:   str,
    *,
    atol: float = 1e-6,
    dpi: int = 200,
) -> None:
    """
    Categorical map over the same basemap: grey = NaN (already masked
    upstream), yellow = "zero" within ``atol``, blue = non-zero. Lets you
    compare, pixel by pixel, where a naive "value ≈ 0 ⇒ ocean" rule would
    and would not have matched the real land mask plotted by
    ``_valid_mask_figure`` — for both the raw field (before any masking)
    and a processed field such as ``mean_comp`` (after masking).

    atol : float
        Absolute tolerance passed to ``np.isclose(field, 0.0, atol=atol)``.
        Use this instead of a bit-exact ``== 0`` test: fill/sentinel
        values often survive scale_factor/add_offset decoding as a tiny
        non-zero float (e.g. ~1.9e-6 for SM, ~5e-10 for PEva in this
        dataset) rather than a clean zero. NOTE the two variables differ
        by several orders of magnitude here — a single default won't
        suit both equally well, so tune ``atol`` per call if needed.
    """
    is_nan     = np.isnan(field)
    is_zero    = (~is_nan) & np.isclose(field, 0.0, atol=atol)
    is_nonzero = (~is_nan) & ~np.isclose(field, 0.0, atol=atol)

    cat = np.zeros(field.shape, dtype=float)   # 0 = NaN/masked
    cat[is_zero]    = 1.0
    cat[is_nonzero] = 2.0

    fig, ax = plt.subplots(figsize=(12, 7))
    m = _make_basemap(ax, domain)

    cmap = ListedColormap(["#d9d9d9", "#f0e442", "#2b8cbe"])
    m.imshow(cat, origin="lower", cmap=cmap, vmin=0, vmax=2)

    ev_label = "HW" if event_type == "hw" else "NO-HW"
    n_total = int(field.size)
    n_nan, n_zero, n_nonzero = int(is_nan.sum()), int(is_zero.sum()), int(is_nonzero.sum())
    ax.set_title(
        f"Zero / non-zero map — {label}  (atol={atol:g})\n"
        f"{var.upper()} / {ev_label}    "
        f"NaN: {n_nan} ({100*n_nan/n_total:.1f}%)   "
        f"zero: {n_zero} ({100*n_zero/n_total:.1f}%)   "
        f"non-zero: {n_nonzero} ({100*n_nonzero/n_total:.1f}%)"
    )

    handles = [
        Patch(facecolor="#d9d9d9", edgecolor="k", label="NaN / masked"),
        Patch(facecolor="#f0e442", edgecolor="k", label=f"zero (atol={atol:g})"),
        Patch(facecolor="#2b8cbe", edgecolor="k", label="non-zero"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved  %s", out_path)
# ---------------------------------------------------------------------------
def _process_variable(
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
) -> dict:
    """
    Build stacks of composites and SHAP maps, compute pixel-wise Pearson r,
    log summary statistics, and generate the correlation figure.

    Returns the stats dict for downstream use (e.g. CSV export).
    """
    meta = VAR_META[var]
    threshold = meta["threshold"]
    n_levels = meta["n_levels"]
    fixed_vmin = meta["vmin"]
    fixed_vmax = meta["vmax"]

    cmap_comp = _cmap_isobar() if var in ("z500", "msl") else _cmap_white_centre()

    cases = hw_cases if event_type == "hw" else nohw_cases
    cs_ids = [c["id"] for c in cases]

    # Land mask for surface variables (SM and PEva) using global_land_mask
    land_mask = None
    if var in ("sm", "peva"):
        land_mask = glm.is_land(Y, X)   # True = land, False = ocean
        log.info("  [%s / %s]  Land mask derived: %.1f%% land points",
                 var, event_type, 100 * np.mean(land_mask))

    # Normalisation constant (same logic as shap_composite.py)
    if var in ("z500", "msl"):
        norm = _global_max(shap_dir, event_type, cs_ids, var, model_name)
    else:
        norm = _combined_max(shap_dir, cs_ids, var, model_name)

    log.info("  [%s / %s]  norm = %.6g", var, event_type, norm)

    # ── Collect arrays ────────────────────────────────────────────────────
    composites_list: list[np.ndarray] = []
    shap_list:       list[np.ndarray] = []

    for case in tqdm(cases, desc=f"Loading {var}/{event_type}", leave=False):
        cs_id = case["id"]
        comp = _event_anomaly_mean(
            ds, case["start"], case["end"], climatology, meta["field"]
        )
        shap_raw = _load_shap(_shap_path(shap_dir, event_type, cs_id, var, model_name))
        shap_norm = _normalise(shap_raw, norm, threshold)

        if land_mask is not None:
            comp[~land_mask] = np.nan
            shap_norm[~land_mask] = np.nan

        composites_list.append(comp)
        shap_list.append(shap_norm)

    composites = np.stack(composites_list, axis=0)   # (N, LAT, LON)
    shap_stack = np.stack(shap_list,       axis=0)   # (N, LAT, LON)

    # ── Pixel-wise Pearson r ──────────────────────────────────────────────
    log.info("  [%s / %s]  computing pixel-wise Pearson r …", var, event_type)
    r_map, p_map = pixelwise_pearson(composites, shap_stack, valid_mask=land_mask)

    # ── Summary statistics ────────────────────────────────────────────────
    stats_dict = summary_statistics(composites, shap_stack, r_map, p_map, event_type, var)

    # ── Figure ────────────────────────────────────────────────────────────
    mean_comp = composites.mean(axis=0)
    if land_mask is not None:
        mean_comp[~land_mask] = np.nan
    
    out_dir = os.path.join(output_dir, var)
    out_path = os.path.join(
        out_dir,
        f"spatial_corr_{event_type}_{var}{model_name}.png",
    )

    _correlation_figure(
        r_map, p_map, mean_comp,
        domain, X, Y,
        cmap_comp, var, event_type, stats_dict,
        out_path,
        fixed_vmin=fixed_vmin, fixed_vmax=fixed_vmax,
        n_levels=n_levels,
    )

    # ── Ranked |r| figure ─────────────────────────────────────────────────
    _ranked_correlation_figure(
        r_map, domain, var, event_type,
        os.path.join(out_dir, f"ranked_corr_{event_type}_{var}{model_name}.png"),
    )

    # ── Valid-mask sanity-check figure ──────────────────────────────────────
    valid_mask_path = os.path.join(
        out_dir,
        f"valid_mask_{event_type}_{var}{model_name}.png",
    )
    _valid_mask_figure(
        stats_dict["valid_mf_map"],
        domain, var, event_type,
        valid_mask_path,
    )

    # ── Non-zero sanity-check figures (raw field & mean_comp) ───────────────
    # Lets you compare, side by side with valid_mask_*.png, where a naive
    # "value ≈ 0 ⇒ ocean" rule would and would not have matched the
    # land_mask actually used (glm.is_land). atol is variable-specific: the
    # sentinel magnitude found for SM (~1.9e-6) and PEva (~5e-10) differ by
    # several orders of magnitude — tune _NONZERO_ATOL below if needed.
    if var in ("sm", "peva", "z500", "msl"):
        atol = _NONZERO_ATOL.get(var, 1e-6)
        raw_field = climatology[meta["field"]].isel(dayofyear=0).values
        _nonzero_figure(
            raw_field, domain, var, event_type,
            "raw field (climatology, day 0)",
            os.path.join(out_dir, f"nonzero_raw_{event_type}_{var}{model_name}.png"),
            atol=atol,
        )
        _nonzero_figure(
            mean_comp, domain, var, event_type,
            "mean_comp (already land-masked)",
            os.path.join(out_dir, f"nonzero_meancomp_{event_type}_{var}{model_name}.png"),
            atol=atol,
        )

    return stats_dict


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pixel-wise spatial correlation between normalised SHAP values "
            "and composite anomaly fields."
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

    # Geographic mesh
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

    # Dataset & climatology
    ds = _open_dataset(dataset_path, domain)
    climatology = _compute_climatology(ds)
    log.info("Climatology computed  (%s – %s)", CLIM_START, CLIM_END)

    plt.rcParams.update({"font.size": 13})

    # Collect all stats for a final summary table
    all_stats: dict[str, dict] = {}

    combos = [(var, evt) for var in args.vars for evt in ("hw", "nohw")]
    for var, event_type in tqdm(combos, desc="Processing"):
        log.info("Processing  var=%-4s  event=%s", var, event_type)
        try:
            s = _process_variable(
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
            )
            all_stats[f"{var}_{event_type}"] = s
        except FileNotFoundError as exc:
            log.error("  Skipping %s / %s – %s", var, event_type, exc)

    # ── Final summary table ───────────────────────────────────────────────
    log.info("\n\n  ═══════════════════════════════════════════════════")
    log.info("                 SUMMARY TABLE")
    log.info("  ═══════════════════════════════════════════════════")
    header = f"  {'Key':<20} {'Mean|r|':>10} {'Mean|ρ|':>10}"
    log.info(header)
    log.info("  " + "─" * 44)
    for key, s in all_stats.items():
        log.info(
            "  %-20s  %.4f      %.4f",
            key,
            s["mean_pixelwise_r"],
            s["mean_pixelwise_spearman"],
        )
    log.info("  ═══════════════════════════════════════════════════")
    log.info("Done.  Figures saved to  %s", args.output_dir)


if __name__ == "__main__":
    main()