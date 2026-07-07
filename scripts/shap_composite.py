"""
shap_composite.py
=================
Generate composite analysis figures that overlay SHAP values on top of
the climatological anomaly (field − daily climatology mean) for each
variable and case study.

For each variable the **composite** is:
    anomaly = event_mean_field − day-of-year climatology (1980–2010 baseline)

The SHAP mean-sum arrays (produced by ``shap_regression.py --meansum``) are
normalised and thresholded, then overlaid as a semi-transparent ``contourf``
on top of the composite field (``imshow`` background + ``contour`` isobars).

Two output kinds are produced for every variable and event type:

  * **Individual** – one figure per case study (5 HW + 5 NO-HW).
  * **Mean**       – one figure showing the 5-case mean composite and mean SHAP.

Outputs are saved under ``shap/comp_shap/{var}/``.

Variable-specific settings
--------------------------
+------+------------------------------+---------------+------------------+
| Var  | Composite cmap               | SHAP threshold| Composite limits |
+======+==============================+===============+==================+
| z500 | BrBG→PuOr blend (isobars)   | 0.30          | data-driven      |
| msl  | BrBG→PuOr blend (isobars)   | 0.30          | data-driven      |
| peva | RdBu_r (white centre)        | cf11 band     | ±1e-4            |
| sm   | RdBu_r (white centre)        | cf7  band     | ±0.20            |
+------+------------------------------+---------------+------------------+

Usage
-----
    python scripts/shap_composite.py
    python scripts/shap_composite.py \\
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
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
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
# Variable metadata
# ---------------------------------------------------------------------------
# field     : variable name inside the NetCDF dataset
# threshold : normalised SHAP values below this magnitude are zeroed out.
#             For z500/msl a hard threshold is applied before plotting.
#             For peva/sm no threshold is used – the white centre of the
#             colormap naturally hides near-zero values.
# n_levels  : number of contourf levels for the SHAP overlay.
#             z500/msl use the full 21 levels; peva uses 11 (coarser, less
#             noisy); sm uses 7 (coarsest – high small-scale variability).
# vmin/vmax : fixed composite colour limits (None = data-driven per case)
VAR_META: dict[str, dict] = {
    "z500": dict(field="z",    threshold=0.30, n_levels=13, vmin=-1200,  vmax=1200),
    "msl":  dict(field="msl",  threshold=0.20, n_levels=13, vmin=-1000,  vmax=1000),
    "peva": dict(field="peva", threshold=None, n_levels=13, vmin=-1e-4, vmax=1e-4),
    "sm":   dict(field="sm",   threshold=None, n_levels=13,  vmin=-0.20, vmax=0.20),
}

# Reference period for the climatology
CLIM_START = "1980-01-01"
CLIM_END   = "2010-12-31"

# Spatial dimensions of the SHAP .npy files
SHAP_LAT, SHAP_LON = 20, 30

# Number of colour levels used throughout
N_COLORS = 21


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def _cmap_isobar() -> ListedColormap:
    """
    BrBG blended with PuOr for Z500 / MSL composite background + isobars.

    Levels 0-6  : BrBG cool tail  (brown, negative anomalies)
    Level  7    : semi-transparent black  (near-zero band)
    Levels 8-14 : PuOr warm tail  (positive anomalies, purple/orange)

    Matches the specification:
        colors[7]  = [0, 0, 0, 0.3]
        colors[8:] = colors2[14:]
    """
    c1 = plt.get_cmap("BrBG")(np.linspace(0, 1, N_COLORS))[:-6]   # 15 entries
    c2 = plt.get_cmap("PuOr")(np.linspace(0, 1, N_COLORS))         # 21 entries
    c1[7]  = [0.0, 0.0, 0.0, 0.3]
    c1[8:] = c2[14:]
    #c1 = c1[4:-4]
    return ListedColormap(c1)


def _cmap_white_centre() -> ListedColormap:
    """
    RdBu_r with three central cells set to white.
    Used for PEva and SM composite backgrounds (anomaly fields that can
    be positive or negative around zero).
    """
    c1 = plt.get_cmap("BrBG")(np.linspace(0, 1, N_COLORS))[:-6]   # 15 entries
    c2 = plt.get_cmap("PuOr")(np.linspace(0, 1, N_COLORS))         # 21 entries
    c1[6]  = [1, 1, 1, 1]
    c1[7]  = [1, 1, 1, 1]
    c1[8]  = [1, 1, 1, 1]
    c1[9:] = c2[15:]
    return ListedColormap(c1)


def _cmap_shap() -> ListedColormap:
    """Plain RdBu_r for Z500 / MSL SHAP overlay (threshold already masks zeros)."""
    return ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0, 1, N_COLORS)))


def _cmap_shap_white_centre() -> ListedColormap:
    """
    RdBu_r with a wider white band in the centre for PEva / SM SHAP overlay.
    The white band covers the near-zero region naturally via the colour mapping.
    """
    colors = plt.get_cmap("RdBu_r")(np.linspace(0, 1, N_COLORS))
    colors[8]  = [1, 1, 1, 1]
    colors[9]  = [1, 1, 1, 1]
    colors[10] = [1, 1, 1, 1]
    colors[11] = [1, 1, 1, 1]
    colors[12] = [1, 1, 1, 1]
    return ListedColormap(colors)


def _build_cmaps(var: str) -> tuple[ListedColormap, ListedColormap]:
    """Return ``(composite_cmap, shap_cmap)`` for *var*."""
    if var in ("z500", "msl"):
        return _cmap_isobar(), _cmap_shap_white_centre()
    else:  # peva, sm
        return _cmap_white_centre(), _cmap_shap_white_centre()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_configs(
    model_path: str, cs_path: str
) -> tuple[dict, str, list[dict], list[dict]]:
    """
    Return ``(domain, model_name, hw_cases, nohw_cases)``.
    """
    with open(model_path) as f:
        cfg = json.load(f)
    with open(cs_path) as f:
        cs = json.load(f)
    return cfg["domain"], cfg["name"], cs["hw"], cs["no_hw"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _open_dataset(path: str, domain: dict) -> xr.Dataset:
    """
    Open the NetCDF dataset, normalise longitudes to −180/180, sort, and
    crop to the analysis domain.
    """
    log.info("Opening dataset  %s", path)
    ds = xr.open_dataset(path)

    # Remap longitudes if they are in 0–360 convention
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

    ds = (
        ds
        .sortby("longitude")
        .sortby("latitude")
        .sel(
            latitude=slice(domain["latitude_min"],  domain["latitude_max"]),
            longitude=slice(domain["longitude_min"], domain["longitude_max"]),
        )
    )
    return ds


def _compute_climatology(ds: xr.Dataset) -> xr.Dataset:
    """Day-of-year climatology over the 1980–2010 reference period."""
    ref = ds.sel(time=slice(CLIM_START, CLIM_END))
    return ref.groupby("time.dayofyear").mean(dim="time")


def _event_anomaly_mean(
    ds: xr.Dataset,
    start: str,
    end: str,
    climatology: xr.Dataset,
    field: str,
) -> np.ndarray:
    """
    Return a 2-D ``(lat, lon)`` array of the time-mean anomaly for *field*
    over the event window [*start*, *end*].
    """
    event = ds.sel(time=slice(start, end))
    anom  = event.groupby("time.dayofyear") - climatology
    return anom.mean(dim="time")[field].values


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------
def _shap_path(
    shap_dir: str, event_type: str, cs_id: int, var: str, name: str
) -> str:
    return os.path.join(
        shap_dir,
        f"cs_{event_type}{cs_id}",
        f"shap_{event_type}_meansum_{var}{name}.npy",
    )


def _load_shap(path: str) -> np.ndarray:
    """
    Load a SHAP ``.npy`` and return a ``(SHAP_LAT, SHAP_LON)`` array.
    Handles both ``(LAT, LON)`` and ``(1, LAT, LON)`` saved shapes.
    """
    arr = np.load(path)
    return np.flip(arr.squeeze().reshape(SHAP_LAT, SHAP_LON), axis=0)


def _global_max(
    shap_dir: str, event_type: str, cs_ids: list[int], var: str, name: str
) -> float:
    """
    Maximum |SHAP| across all case studies for one event type.
    Used as the normalisation constant for Z500 and MSL.
    """
    maxima = []
    for cs_id in cs_ids:
        p = _shap_path(shap_dir, event_type, cs_id, var, name)
        if os.path.isfile(p):
            maxima.append(float(np.max(np.abs(_load_shap(p)))))
    if not maxima:
        raise FileNotFoundError(
            f"No SHAP .npy files found for var='{var}', "
            f"event='{event_type}' in {shap_dir!r}. "
            "Run shap_regression.py first."
        )
    return float(np.max(maxima))


def _combined_max(
    shap_dir: str, cs_ids: list[int], var: str, name: str
) -> float:
    """
    Maximum |SHAP| across **both** HW and NO-HW case studies.
    Used for PEva and SM so both event types share the same colour scale.
    """
    return max(
        _global_max(shap_dir, "hw",   cs_ids, var, name),
        _global_max(shap_dir, "nohw", cs_ids, var, name),
    )


def _normalise(
    arr: np.ndarray, norm: float, threshold: Optional[float]
) -> np.ndarray:
    """Divide by *norm* and zero-out values below *threshold*."""
    out = arr / norm
    if threshold is not None:
        out[np.abs(out) < threshold] = 0.0
    return out


# ---------------------------------------------------------------------------
# Basemap
# ---------------------------------------------------------------------------
def _make_basemap(ax: plt.Axes, domain: dict) -> Basemap:
    m = Basemap(
        ax=ax, resolution="l",
        llcrnrlon=domain["longitude_min"], llcrnrlat=domain["latitude_min"],
        urcrnrlon=domain["longitude_max"], urcrnrlat=domain["latitude_max"],
    )
    m.drawcountries(color="#303338")
    m.drawcoastlines(color="#000000")
    m.drawmeridians(range(0, 360, 10), color="k", labels=[0, 0, 0, 1])
    m.drawparallels(range(-90, 100, 10), color="k", labels=[1, 0, 0, 0])
    return m


# ---------------------------------------------------------------------------
# Core figure
# ---------------------------------------------------------------------------
def _composite_figure(
    composite: np.ndarray,
    shap_norm: np.ndarray,
    domain: dict,
    X: np.ndarray,
    Y: np.ndarray,
    lon_range: np.ndarray,
    lat_range: np.ndarray,
    cmap_comp: ListedColormap,
    cmap_shap: ListedColormap,
    var: str,
    title: str,
    out_path: str,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    n_levels: int = 21,
    dpi: int = 200,
) -> None:
    """
    Plot and save one composite-anomaly + SHAP overlay figure.

    Figure layers (bottom to top)
    ------------------------------
    1. ``imshow``   – composite anomaly as a filled colour background.
    2. ``contourf`` – normalised SHAP values as a semi-transparent overlay.
       The number of contourf levels is controlled by *n_levels*:
       - z500 / msl : 21 levels (fine); weak values zeroed by hard threshold.
       - peva       : 11 levels (medium); zero region hidden by white-centre cmap.
       - sm         :  7 levels (coarse); zero region hidden by white-centre cmap.
    3. ``contour``  – composite anomaly isobar lines for orientation.
    4. Two colourbars: left = composite anomaly, right = normalised SHAP.

    Parameters
    ----------
    vmin, vmax :
        Colour limits for the composite imshow. Inferred from data when ``None``.
    n_levels :
        Number of ``contourf`` levels for the SHAP overlay (default 21).
        Fewer levels produce coarser, less noisy fills – appropriate for
        variables with high small-scale variability (peva=11, sm=7).
    """
    _vmin = float(composite.min()) if vmin is None else vmin
    _vmax = float(composite.max()) if vmax is None else vmax

    comp_levels = np.linspace(_vmin, _vmax, 13)#N_COLORS)
    shap_levels = np.linspace(-1.0,  1.0,  n_levels)

    fig, ax = plt.subplots(figsize=(12, 7))
    #fig, ax = plt.subplots(figsize=(24, 14))
    m = _make_basemap(ax, domain)

    # ── 1. Composite background ───────────────────────────────────────────
    img_comp_no = m.imshow(composite, cmap=cmap_comp, vmin=_vmin, vmax=_vmax)

    # ── 2. SHAP contourf overlay ──────────────────────────────────────────
    # For peva/sm the white-centre colormap naturally hides near-zero values;
    # no level-skipping or masking is needed.
    img_shap = ax.contourf(
        X, Y, np.clip(shap_norm, -1.0, 1.0),
        levels=shap_levels,
        cmap=cmap_shap,
        vmin=-1.0, vmax=1.0,
        #alpha=0.85,
        #extend="both",
    )

    # ── 3. Composite isobar lines ─────────────────────────────────────────
    if var in ["z500", "msl"]:
        img_comp = ax.contour(
            X, Y, composite,
            levels=comp_levels,
            linewidths=5,
            cmap=cmap_comp,
            alpha=0.3,
        )
    else:
        img_comp = ax.contourf(
            X, Y, composite,
            levels=comp_levels,
            cmap=cmap_comp,
            alpha=0.5,
            zorder=3,
            vmin=_vmin,
            vmax=_vmax,
        )

    # ── 4. Titles & colourbars ────────────────────────────────────────────
    #ax.set_title(f"({var.upper()})", loc="left",  fontsize=14, fontweight="bold")
    #ax.set_title(title,              loc="right", fontsize=12)

    if var in ["z500", "msl"]:
        cbar_comp = fig.colorbar(img_comp_no, ax=ax, extend="both",)
                                  #fraction=0.025, pad=0.01, shrink=0.85)
    else:
        cbar_comp = fig.colorbar(img_comp_no, ax=ax,)

    if var in ["peva"]:
        cbar_comp.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar_comp.formatter.set_powerlimits((0,0))
        cbar_comp.update_ticks()
        

    cbar_comp.set_label(f"Anomaly [{_units(var)}]", fontsize=26)
    cbar_comp.ax.tick_params(labelsize=24)

    cbar_shap = fig.colorbar(img_shap, ax=ax,)# extend="neither",)
                              #fraction=0.025, pad=0.05, shrink=0.85)
    cbar_shap.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    cbar_shap.set_label("Norm. SHAP", fontsize=26)
    cbar_shap.ax.tick_params(labelsize=24)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.debug("    Saved  %s", out_path)


def _units(var: str) -> str:
    return {"z500": "m²/s²", "msl": "Pa", "peva": "m/day", "sm": "m³/m³"}.get(var, "")


# ---------------------------------------------------------------------------
# Per-variable pipeline
# ---------------------------------------------------------------------------
def _process_variable(
    var: str,
    event_type: str,
    ds: xr.Dataset,
    climatology: xr.Dataset,
    hw_cases: list[dict],
    nohw_cases: list[dict],
    domain: dict,
    X: np.ndarray,
    Y: np.ndarray,
    lon_range: np.ndarray,
    lat_range: np.ndarray,
    shap_dir: str,
    output_dir: str,
    model_name: str,
    save_individual: bool,
) -> None:
    """
    Individual + mean composite+SHAP figures for one variable / event type.
    """
    meta      = VAR_META[var]
    threshold = meta["threshold"]
    n_levels  = meta["n_levels"]
    fixed_vmin, fixed_vmax = meta["vmin"], meta["vmax"]

    cmap_comp, cmap_shap = _build_cmaps(var)
    cases  = hw_cases if event_type == "hw" else nohw_cases
    cs_ids = [c["id"] for c in cases]
    ev_tag = "HW" if event_type == "hw" else "NO-HW"

    out_dir = os.path.join(output_dir, var)
    os.makedirs(out_dir, exist_ok=True)

    # Normalisation constant
    if var in ("z500", "msl"):
        norm = _global_max(shap_dir, event_type, cs_ids, var, model_name)
    else:
        norm = _combined_max(shap_dir, cs_ids, var, model_name)
    log.info("  [%s / %s]  norm = %.6g", var, event_type, norm)

    # Collect arrays for the mean figure
    all_composites: list[np.ndarray] = []
    all_shap:       list[np.ndarray] = []

    for case in tqdm(cases, desc=f"{var}/{event_type}", leave=False):
        cs_id = case["id"]
        label = case["label"]

        # Composite anomaly
        composite = _event_anomaly_mean(
            ds, case["start"], case["end"], climatology, meta["field"]
        )

        # SHAP
        shap_raw  = _load_shap(
            _shap_path(shap_dir, event_type, cs_id, var, model_name)
        )
        shap_norm = _normalise(shap_raw, norm, threshold)

        all_composites.append(composite)
        all_shap.append(shap_norm)

        if save_individual:
            vmin = fixed_vmin if fixed_vmin is not None else float(composite.min())
            vmax = fixed_vmax if fixed_vmax is not None else float(composite.max())
            _composite_figure(
                composite, shap_norm,
                domain, X, Y, lon_range, lat_range,
                cmap_comp, cmap_shap,
                var=var,
                title=f"{ev_tag} {cs_id} – {label}",
                out_path=os.path.join(
                    out_dir,
                    f"comp_shap_{event_type}{cs_id}_{var}{model_name}.png",
                ),
                vmin=vmin, vmax=vmax,
                n_levels=n_levels,
            )

    # ── Mean figure ───────────────────────────────────────────────────────
    mean_comp = np.mean(all_composites, axis=0)
    mean_shap = np.mean(all_shap,       axis=0)

    # Re-normalise the averaged SHAP so its scale stays in [−1, 1]
    ms_max = float(np.max(np.abs(mean_shap)))
    if ms_max > 0:
        mean_shap = mean_shap / ms_max

    if threshold is not None:
        mean_shap[np.abs(mean_shap) < threshold] = 0.0

    vmin_m = fixed_vmin if fixed_vmin is not None else float(mean_comp.min())
    vmax_m = fixed_vmax if fixed_vmax is not None else float(mean_comp.max())

    _composite_figure(
        mean_comp, mean_shap,
        domain, X, Y, lon_range, lat_range,
        cmap_comp, cmap_shap,
        var=var,
        title=f"{ev_tag} Mean ({len(cases)} cases)",
        out_path=os.path.join(
            out_dir,
            f"comp_shap_{event_type}_mean_{var}{model_name}.png",
        ),
        vmin=vmin_m, vmax=vmax_m,
        n_levels=n_levels,
    )
    log.info("  [%s / %s]  done", var, event_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate composite anomaly + SHAP overlay figures for each "
            "variable and case study."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",        default="./config/model.json",
                   help="Model & domain config JSON.")
    p.add_argument("--case-studies",  default="./config/case_studies.json",
                   dest="case_studies",
                   help="Case-study periods JSON.")
    p.add_argument("--dataset",       default=None,
                   help="Path to the NetCDF dataset (overrides config value).")
    p.add_argument("--shap-dir",      default="./shap/mod512",
                   dest="shap_dir",
                   help="Root directory containing the cs_*/ SHAP folders.")
    p.add_argument("--output-dir",    default="./shap/comp_shap",
                   dest="output_dir",
                   help="Root directory for saved figures.")
    p.add_argument("--vars", nargs="+",
                   default=["z500", "msl", "peva", "sm"],
                   choices=["z500", "msl", "peva", "sm"],
                   help="Variables to process.")
    p.add_argument("--no-individual", action="store_true", dest="no_individual",
                   help="Skip individual case-study figures; save mean only.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = _build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Config ────────────────────────────────────────────────────────────
    domain, model_name, hw_cases, nohw_cases = _load_configs(
        args.config, args.case_studies
    )
    with open(args.config) as f:
        cfg = json.load(f)

    dataset_path = args.dataset or cfg["datasets"]["pred_dataset"]

    # ── Geographic mesh ───────────────────────────────────────────────────
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

    # ── Dataset & climatology ─────────────────────────────────────────────
    ds          = _open_dataset(dataset_path, domain)
    climatology = _compute_climatology(ds)
    log.info("Climatology computed  (%s – %s)", CLIM_START, CLIM_END)

    # ── Main loop ─────────────────────────────────────────────────────────
    #plt.rcParams.update({"font.size": 13})
    plt.rcParams.update({"font.size": 26})
 
    # Register and apply the CMU Bold font
    _CMU_FONT_PATH = "/usr/share/fonts/truetype/cmu/cmunbx.ttf"
    if os.path.isfile(_CMU_FONT_PATH):
        font_manager.fontManager.addfont(_CMU_FONT_PATH)
        _cmu_name = font_manager.FontProperties(fname=_CMU_FONT_PATH).get_name()
        plt.rcParams["font.family"] = _cmu_name
        log.info("Using font: %s (%s)", _cmu_name, _CMU_FONT_PATH)
    else:
        log.warning("CMU font not found at %s – using default font", _CMU_FONT_PATH)


    combos = [
        (var, evt)
        for var in args.vars
        for evt in ("hw", "nohw")
    ]

    for var, event_type in tqdm(combos, desc="Processing"):
        log.info("Processing  var=%-4s  event=%s", var, event_type)
        try:
            _process_variable(
                var=var,
                event_type=event_type,
                ds=ds,
                climatology=climatology,
                hw_cases=hw_cases,
                nohw_cases=nohw_cases,
                domain=domain,
                X=X, Y=Y,
                lon_range=lon_range,
                lat_range=lat_range,
                shap_dir=args.shap_dir,
                output_dir=args.output_dir,
                model_name=model_name,
                save_individual=not args.no_individual,
            )
        except FileNotFoundError as exc:
            log.error("  Skipping %s / %s – %s", var, event_type, exc)

    log.info("Done.  Figures saved to  %s", args.output_dir)


if __name__ == "__main__":
    main()
