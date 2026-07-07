"""
shap_figures.py
===============
Generate publication-ready figures from the SHAP ``.npy`` arrays produced by
``shap_regression.py``.

For each combination of time period (all / post / pre) and event type
(HW / NO-HW) the script produces:

  * A 3×4 comparative panel (mean + 5 individual cases + their deviations).
  * Same panel with contour fill overlaid.
  * Standalone single-panel maps for the mean and each individual case study.
  * (Optional) 3×3 diagnostic grids for each individual case study (``--nine``).
  * (Optional) A single HW-mean minus NO-HW-mean difference map (``--meandiff``).

Usage
-----
    python scripts/shap_figures.py
    python scripts/shap_figures.py \\
        --config config/model.json \\
        --case-studies config/case_studies.json \\
        --var msl \\
        --shap-dir ./shap/mod512 \\
        --output-dir ./shap/figures_paper \\
        --percentile 99 --threshold 0.05 \\
        --verbose
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

# Subplot letter sequence used in panel titles
_PANEL_LETTERS = list("abcdefghijk")

# Number of discrete colours for the colormaps
_N_COLORS = 21


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_configs(model_path: str, cs_path: str) -> tuple[dict, list[dict], list[dict]]:
    """
    Load domain parameters, HW labels, and NO-HW labels from config files.

    Returns
    -------
    domain : dict
        Geographic bounds and resolution required for the Basemap and meshgrid.
    hw_cases : list of dict
        HW case studies with ``id``, ``label``, ``start``, ``end``.
    nohw_cases : list of dict
        NO-HW (control) case studies in the same format.
    """
    with open(model_path) as f:
        cfg = json.load(f)
    with open(cs_path) as f:
        cs = json.load(f)

    domain = cfg["domain"]  # latitude_min/max, longitude_min/max, resolution
    return domain, cs["hw"], cs["no_hw"]


# ---------------------------------------------------------------------------
# Basemap helper
# ---------------------------------------------------------------------------
def _make_basemap(ax: plt.Axes, domain: dict) -> Basemap:
    """Create a Basemap on *ax* with coastlines, countries, and graticules."""
    m = Basemap(
        ax=ax,
        resolution="l",
        llcrnrlon=domain["longitude_min"],
        llcrnrlat=domain["latitude_min"],
        urcrnrlon=domain["longitude_max"],
        urcrnrlat=domain["latitude_max"],
    )
    m.drawcountries(color="#303338")
    m.drawcoastlines(color="#000000")
    m.drawmeridians(range(0, 360, 10), color="k", labels=[0, 0, 0, 1])
    m.drawparallels(range(-90, 100, 10), color="k", labels=[1, 0, 0, 0])
    return m


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def _build_cmaps() -> tuple[ListedColormap, ListedColormap, ListedColormap]:
    """Return (seismic, pink-green, transparent) discrete colormaps."""
    discrete_seismic = ListedColormap(
        plt.get_cmap("RdBu_r")(np.linspace(0, 1, _N_COLORS))
    )
    discrete_pg = ListedColormap(
        plt.get_cmap("PiYG")(np.linspace(0, 1, _N_COLORS))
    )
    transparent = ListedColormap(["none", "none"])
    return discrete_seismic, discrete_pg, transparent


# ---------------------------------------------------------------------------
# SHAP file discovery
# ---------------------------------------------------------------------------
def _discover_files(shap_dir: str, var: str, is_atribution: bool = False) -> tuple[list, list, list]:
    """
    Glob and sort the per-variable mean-sum ``.npy`` files, then split by
    time-period suffix (all / post / pre).

    Each case study contributes exactly 3 files (one per period) if the 
    flag is_atribution is True, so the sorted list interleaves them:
        cs_hw1/…all…,  cs_hw1/…post…,  cs_hw1/…pre…,
        cs_hw2/…all…,  …,  cs_nohw1/…, …
    If not, only all is expected
    """
    pattern = os.path.join(shap_dir, "cs_*", f"shap_*_meansum_{var}*.npy")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No SHAP .npy files found for variable '{var}'.\n"
            f"Pattern searched: {pattern}\n"
            "Run shap_regression.py first."
        )
    if is_atribution:
        if len(files) % 3 != 0:
            warnings.warn(
                f"Expected a multiple of 3 files (all/post/pre × case studies) "
                f"but found {len(files)}. Results may be incorrect.",
                stacklevel=2,
            )
        return files[0::3], files[1::3], files[2::3]
    else:
        return files, [], []
    


# ---------------------------------------------------------------------------
# Single-panel helpers
# ---------------------------------------------------------------------------
def _save_single(
    data: np.ndarray,
    cmap: ListedColormap,
    domain: dict,
    path_stem: str,
    *,
    vmax: float = 1.0,
    mask: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    transparent_cmap: Optional[ListedColormap] = None,
    lon_range: Optional[np.ndarray] = None,
    lat_range: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """
    Save a standalone map without colourbar, then re-save with colourbar.

    Both versions are written: ``{path_stem}.png`` and
    ``{path_stem}_cbar.png``.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    m   = _make_basemap(ax, domain)
    img = m.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)

    if mask is not None and X is not None and Y is not None:
        ax.pcolor(lon_range, lat_range, mask, hatch="//",
                  cmap=transparent_cmap, zorder=1.1)
        ax.contour(X, Y, mask.mask.astype(int),
                   levels=[0.5], linestyles="dotted", colors="white")

    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=dpi, bbox_inches="tight")

    m.colorbar(img, extend="both", location="bottom", pad=0.3)
    fig.tight_layout()
    fig.savefig(f"{path_stem}_cbar.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3×4 Comparative panel
# ---------------------------------------------------------------------------
def _comparative_panel(
    ms_shap: np.ndarray,
    mean_ms_shap: np.ndarray,
    diff_shap: np.ndarray,
    diff_mask: np.ndarray,
    event_type: str,
    case_labels: list[str],
    domain: dict,
    lon_range: np.ndarray,
    lat_range: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cmaps: tuple,
    path_stem: str,
    *,
    contour: bool = False,
    vmax: float = 1.0,
    dpi: int = 200,
) -> None:
    """
    Produce and save the main 3×4 comparative panel.

    Layout:
        Col 1        Col 2             Col 3             Col 4
        [Mean]  [colourbar col]  [Case Study i]   [CS i − Mean]
         …
        (3 rows × 4 columns = 12 subplot positions, col 2 holds colourbars)

    Parameters
    ----------
    ms_shap : shape ``(n_cs, lat, lon)``
    mean_ms_shap : shape ``(lat, lon)``
    diff_shap : shape ``(n_cs, lat, lon)``
    diff_mask : masked array, shape ``(n_cs, lat, lon)``
    event_type : ``"hw"`` or ``"nohw"``
    case_labels : human-readable label for each case study
    contour : overlay contourf on top of imshow
    """
    discrete_seismic, discrete_pg, cw = cmaps
    n_cs    = ms_shap.shape[0]
    ev_tag  = "HW" if event_type == "hw" else "NO HW"

    fig = plt.figure(figsize=(20, 10))

    # ── Position 1: Mean map ──────────────────────────────────────────────
    ax = plt.subplot(3, 4, 1)
    m  = _make_basemap(ax, domain)
    ax.set_title("(a)", loc="left")
    ax.set_title("Mean", loc="right")
    img = m.imshow(mean_ms_shap, cmap=discrete_seismic, vmin=-vmax, vmax=vmax)
    if contour:
        lvl = np.linspace(mean_ms_shap.min(), mean_ms_shap.max(), _N_COLORS)
        plt.contourf(X, Y, mean_ms_shap, levels=lvl,
                     cmap=discrete_seismic, vmin=-vmax, vmax=vmax)

    # ── Position 2: Colourbar placeholder column ──────────────────────────
    axcb = plt.subplot(3, 4, 2)
    axcb.axis("off")
    axcb.set_visible(False)

    cax_top = inset_axes(
        axcb, width="80%", height="5%", loc="upper center",
        bbox_to_anchor=(0, -0.3, 1, 1), bbox_transform=axcb.transAxes,
        borderpad=0,
    )
    cbar_top = plt.colorbar(
        img, extend="both", cax=cax_top, orientation="horizontal",
        fraction=0.046, pad=0.04,
    )
    cbar_top.ax.xaxis.set_ticks_position("bottom")
    cbar_top.ax.xaxis.set_label_position("bottom")

    # ── Positions 3–12: Individual case studies ───────────────────────────
    # Subplot positions for case studies and their diffs
    sp_pairs = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    first_diff_img = None

    for cs_i, (sp_cs, sp_diff) in enumerate(sp_pairs[:n_cs]):
        letter_cs   = _PANEL_LETTERS[1 + cs_i * 2]
        letter_diff = _PANEL_LETTERS[2 + cs_i * 2]

        # Case-study map
        ax = plt.subplot(3, 4, sp_cs)
        m  = _make_basemap(ax, domain)
        ax.set_title(f"({letter_cs})", loc="left")
        ax.set_title(f"{ev_tag} {cs_i + 1} – {case_labels[cs_i]}", loc="right",
                     fontsize=7)
        img_cs = m.imshow(ms_shap[cs_i], cmap=discrete_seismic,
                          vmin=-vmax, vmax=vmax)
        if contour:
            lvl = np.linspace(ms_shap[cs_i].min(), ms_shap[cs_i].max(),
                              _N_COLORS)
            plt.contourf(X, Y, ms_shap[cs_i], levels=lvl,
                         cmap=discrete_seismic, vmin=-vmax, vmax=vmax)

        # Deviation map (case study − mean)
        ax = plt.subplot(3, 4, sp_diff)
        m  = _make_basemap(ax, domain)
        ax.set_title(f"({letter_diff})", loc="left")
        ax.set_title(f"{ev_tag} {cs_i + 1} $-$ Mean", loc="right", fontsize=7)
        img_diff = m.imshow(diff_shap[cs_i], cmap=discrete_pg,
                            vmin=-vmax, vmax=vmax)
        if contour:
            lvl = np.linspace(diff_shap[cs_i].min(), diff_shap[cs_i].max(),
                              _N_COLORS)
            plt.contourf(X, Y, diff_shap[cs_i], levels=lvl,
                         cmap=discrete_pg, vmin=-vmax, vmax=vmax)
        ax.pcolor(lon_range, lat_range, diff_mask[cs_i],
                  hatch="//", cmap=cw, zorder=1.1)
        ax.contour(X, Y, diff_mask[cs_i].mask.astype(int),
                   levels=[0.5], linestyles="dotted", colors="white")

        if cs_i == 0:
            first_diff_img = img_diff

    # Colourbar for deviation maps
    if first_diff_img is not None:
        cax_bot = inset_axes(
            axcb, width="80%", height="5%", loc="lower center",
            bbox_to_anchor=(0, 0.3, 1, 1), bbox_transform=axcb.transAxes,
            borderpad=0,
        )
        cbar_bot = plt.colorbar(
            first_diff_img, extend="both", cax=cax_bot,
            orientation="horizontal", fraction=0.046,
        )
        cbar_bot.ax.xaxis.set_ticks_position("top")
        cbar_bot.ax.xaxis.set_label_position("top")

    fig.tight_layout()
    ct = "_contour" if contour else ""
    fig.savefig(f"{path_stem}{ct}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3×3 Diagnostic panel (optional)
# ---------------------------------------------------------------------------
def _nine_panel(
    ms_shap: np.ndarray,
    mean_ms_shap: np.ndarray,
    diff_shap: np.ndarray,
    diff_mask: np.ndarray,
    event_type: str,
    cs_idx: int,
    case_label: str,
    domain: dict,
    lon_range: np.ndarray,
    lat_range: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cmaps: tuple,
    path_stem: str,
    dpi: int = 200,
) -> None:
    """3×3 grid: imshow / imshow+contourf / contourf  ×  [CS, mean, CS−mean]."""
    discrete_seismic, discrete_pg, cw = cmaps
    ev_tag = "HW" if event_type == "hw" else "NO HW"
    label  = f"{ev_tag} {cs_idx + 1} – {case_label}"
    vmax   = 1.0

    rows = [
        (ms_shap,       discrete_seismic, label,        False),
        (mean_ms_shap,  discrete_seismic, "Mean",       False),
        (diff_shap,     discrete_pg,      f"{label} $-$ Mean", True),
    ]

    fig = plt.figure(figsize=(20, 20))
    letter_map = {
        (0, 0): "a", (0, 1): "d", (0, 2): "g",
        (1, 0): "b", (1, 1): "e", (1, 2): "h",
        (2, 0): "c", (2, 1): "f", (2, 2): "i",
    }

    for r, (data, cmap, title_r, is_diff) in enumerate(rows):
        for c, (mode, title_c) in enumerate(
            [("imshow", ""), ("imshow+contour", " contour"), ("contour", " def")]
        ):
            ax = plt.subplot(3, 3, r * 3 + c + 1)
            m  = _make_basemap(ax, domain)
            ax.set_title(f"({letter_map[(r, c)]})", loc="left")
            ax.set_title(f"{title_r}{title_c}", loc="right", fontsize=8)
            img = m.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)

            if "contour" in mode:
                lvl = np.linspace(data.min(), data.max(), _N_COLORS)
                plt.contourf(X, Y, data, levels=lvl,
                             cmap=cmap, vmin=-vmax, vmax=vmax)

            m.colorbar(img, extend="both", location="bottom", pad=0.3)

            if is_diff:
                ax.pcolor(lon_range, lat_range, diff_mask,
                          hatch="//", cmap=cw, zorder=1.1)
                ax.contour(X, Y, diff_mask.mask.astype(int),
                           levels=[0.5], linestyles="dotted", colors="white")

    fig.tight_layout()
    fig.savefig(f"{path_stem}_9x9.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mean-diff map
# ---------------------------------------------------------------------------
def _meandiff_map(
    mean_hw: np.ndarray,
    mean_nohw: np.ndarray,
    domain: dict,
    lon_range: np.ndarray,
    lat_range: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cmaps: tuple,
    out_dir: str,
    period: str,
    var: str,
    tag: str,
    dpi: int = 300,
) -> None:
    """Save HW-mean minus NO-HW-mean difference maps (with and without contour)."""
    _, discrete_pg, cw = cmaps
    diff        = mean_hw - mean_nohw
    diff_max    = 0.5
    diff_std    = float(np.std(diff))
    diff_mask   = np.ma.masked_where(np.abs(diff) >= diff_std,
                                      np.abs(diff) < diff_std)

    for contour in (False, True):
        fig, ax = plt.subplots(figsize=(10, 8))
        m  = _make_basemap(ax, domain)
        img = m.imshow(diff, cmap=discrete_pg, vmin=-diff_max, vmax=diff_max)
        ax.pcolor(lon_range, lat_range, diff_mask,
                  hatch="//", cmap=cw, zorder=1.1)
        ax.contour(X, Y, diff_mask.mask.astype(int),
                   levels=[0.5], linestyles="dotted", colors="white")
        if contour:
            lvl = np.linspace(diff.min(), diff.max(), _N_COLORS)
            plt.contourf(X, Y, diff, levels=lvl,
                         cmap=discrete_pg, vmin=-diff_max, vmax=diff_max)

        ct = "_contour" if contour else ""
        stem = os.path.join(out_dir, "diff", f"meandiff_{tag}{ct}_{var}{period}512")
        fig.tight_layout()
        fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
        m.colorbar(img, extend="both", location="bottom", pad=0.3)
        fig.tight_layout()
        fig.savefig(f"{stem}_cbar.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _process_period(
    period_name: str,
    file_paths: list[str],
    hw_cases: list[dict],
    nohw_cases: list[dict],
    domain: dict,
    cmaps: tuple,
    lon_range: np.ndarray,
    lat_range: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    out_dir: str,
    var: str,
    tag: str,
    threshold: Optional[float],
    do_meandiff: bool,
    do_nine: bool,
) -> None:
    """Run the complete figure pipeline for one time period."""
    n_files = len(file_paths)
    n_cs    = n_files // 2          # half HW, half NO-HW
    log.info("Period: %-6s  |  %d files  |  %d case studies each",
             period_name, n_files, n_cs)

    # ── Load and normalise ────────────────────────────────────────────────
    log.info("file_paths: %s", file_paths)
    for p in file_paths:
        log.info("p shape: %s",np.shape(np.load(p)))
    raw = np.array(
        [np.flip(np.load(p).reshape(20, 30), axis=0) for p in file_paths]
        #[np.flip(np.squeeze(np.load(p))) for p in file_paths]
    )
    ms_max = float(np.max(np.abs(raw)))
    if ms_max == 0:
        log.warning("  All SHAP values are zero for period '%s' – skipping.",
                    period_name)
        return
    ms_shap = raw / ms_max
    #log.info("ms_shap: %s",ms_shap)
    log.info("ms_shap shape: %s",ms_shap.shape)
    log.info("raw shape: %s",raw.shape)
    log.info("ms_max shape: %s",np.shape(ms_max))
    log.info("ms_max: %s",ms_max)

    if threshold is not None:
        log.info("  Applying threshold %.4f", threshold)
        ms_shap[np.abs(ms_shap) < threshold] = 0.0

    # Reshape → (2, n_cs, lat, lon): dim 0 = [HW, NO-HW]
    ms_shap      = ms_shap.reshape(2, n_cs, *ms_shap.shape[1:])
    mean_ms_shap = ms_shap.mean(axis=1)               # (2, lat, lon)
    diff_shap    = ms_shap - mean_ms_shap[:, np.newaxis]

    diff_std  = float(np.std(diff_shap))
    diff_mask = np.ma.masked_where(
        np.abs(diff_shap) >= diff_std, np.abs(diff_shap) < diff_std
    )
    vmax = 1.0

    # ── Mean-diff mode ────────────────────────────────────────────────────
    if do_meandiff:
        os.makedirs(os.path.join(out_dir, "diff"), exist_ok=True)
        _meandiff_map(
            mean_ms_shap[0], mean_ms_shap[1],
            domain, lon_range, lat_range, X, Y,
            cmaps, out_dir, period_name, var, tag,
        )
        return

    # ── Comparative panels + single-panel subfigures ──────────────────────
    for i, event_type in enumerate(("hw", "nohw")):
        is_hw      = (event_type == "hw")
        cases_meta = hw_cases if is_hw else nohw_cases
        case_labels = [c["label"] for c in cases_meta[:n_cs]]
        ev_dir      = os.path.join(out_dir, event_type)
        sub_dir     = os.path.join(ev_dir, "subfig")
        os.makedirs(ev_dir,  exist_ok=True)
        os.makedirs(sub_dir, exist_ok=True)

        # Comparative 3×4 panel (imshow + imshow+contour)
        panel_stem = os.path.join(
            ev_dir, f"comparative_meansum_{event_type}_{tag}{var}{period_name}512"
        )
        for contour_mode in (False, True):
            _comparative_panel(
                ms_shap[i], mean_ms_shap[i], diff_shap[i], diff_mask[i],
                event_type=event_type,
                case_labels=case_labels,
                domain=domain,
                lon_range=lon_range, lat_range=lat_range,
                X=X, Y=Y,
                cmaps=cmaps,
                path_stem=panel_stem,
                contour=contour_mode,
            )

        # Standalone mean map
        _save_single(
            mean_ms_shap[i],
            cmaps[0],  # seismic
            domain,
            os.path.join(sub_dir, f"mean_{event_type}_{tag}{var}{period_name}512"),
        )

        # Per-case-study maps and deviation maps
        for j in tqdm(range(n_cs), desc=f"{period_name}/{event_type} subfigs",
                      leave=False):
            cs_label_short = cases_meta[j]["label"]

            _save_single(
                ms_shap[i, j], cmaps[0], domain,
                os.path.join(
                    sub_dir,
                    f"event{j+1}_{event_type}_{tag}{var}{period_name}512",
                ),
            )
            _save_single(
                diff_shap[i, j], cmaps[1], domain,
                os.path.join(
                    sub_dir,
                    f"diff{j+1}_{event_type}_{tag}{var}{period_name}512",
                ),
                mask=diff_mask[i, j],
                X=X, Y=Y,
                transparent_cmap=cmaps[2],
                lon_range=lon_range, lat_range=lat_range,
            )

            # Optional 3×3 diagnostic grid
            if do_nine:
                nine_stem = os.path.join(
                    ev_dir,
                    f"meansum_{event_type}_{tag}{var}{period_name}512_cs{j+1}",
                )
                _nine_panel(
                    ms_shap[i, j], mean_ms_shap[i], diff_shap[i, j],
                    diff_mask[i, j],
                    event_type=event_type,
                    cs_idx=j,
                    case_label=cs_label_short,
                    domain=domain,
                    lon_range=lon_range, lat_range=lat_range,
                    X=X, Y=Y,
                    cmaps=cmaps,
                    path_stem=nine_stem,
                )

    log.info("  Period '%s' done", period_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate publication figures from SHAP .npy arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", dest="config",
        default="./config/model.json",
        help="Model & domain config JSON.",
    )
    p.add_argument(
        "--case-studies", dest="case_studies",
        default="./config/case_studies.json",
        help="Case-study periods JSON.",
    )
    p.add_argument(
        "--var", default="msl",
        choices=["z500", "peva", "msl", "sm"],
        help="Variable to generate figures for.",
    )
    p.add_argument(
        "--shap-dir", dest="shap_dir",
        default="./shap/mod512",
        help="Directory containing the cs_*/ SHAP output folders.",
    )
    p.add_argument(
        "--output-dir", dest="output_dir",
        default="./shap/figures_paper",
        help="Root directory for saved figures.",
    )
    p.add_argument(
        "--threshold", dest="threshold", type=float, default=None,
        help="Zero-out normalised SHAP values below this absolute threshold.",
    )
    p.add_argument(
        "--percentile", dest="percentile", type=float, default=None,
        help="Use this percentile of |SHAP| as the normalisation maximum "
             "instead of the actual maximum.",
    )
    p.add_argument(
        "--meandiff", action="store_true",
        help="Only generate the HW-mean minus NO-HW-mean difference map.",
    )
    p.add_argument(
        "--nine", action="store_true",
        help="Also generate 3×3 diagnostic grids for each case study.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--is-atribution", dest="is_atribution", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = _build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Config ────────────────────────────────────────────────────────────
    domain, hw_cases, nohw_cases = _load_configs(args.config, args.case_studies)

    # ── Build colormaps and mesh ──────────────────────────────────────────
    cmaps = _build_cmaps()
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

    # ── File discovery ────────────────────────────────────────────────────
    paths_all, paths_post, paths_pre = _discover_files(args.shap_dir, args.var, args.is_atribution)

    log.info("Variable      : %s", args.var)
    log.info("SHAP dir      : %s", args.shap_dir)
    log.info("Output dir    : %s", args.output_dir)
    log.info("Files found   : %d  (all=%d, post=%d, pre=%d)",
             len(paths_all) + len(paths_post) + len(paths_pre),
             len(paths_all), len(paths_post), len(paths_pre))

    # ── Filter tag for file names ─────────────────────────────────────────
    tag = ""
    if args.threshold  is not None:
        tag += f"th{args.threshold}"
    if args.percentile is not None:
        tag += f"p{int(args.percentile)}"

    # ── Process each period ───────────────────────────────────────────────
    log.info("Paths all   : %s", paths_all)
    log.info("Paths post   : %s", paths_post)
    log.info("Paths pre   : %s", paths_pre)
    os.makedirs(args.output_dir, exist_ok=True)
    for period_name, file_paths in tqdm(
        [("all", paths_all), ("post", paths_post), ("pre", paths_pre)] if args.is_atribution else [("all", paths_all)],
        desc="Periods",
    ):
        log.info("Files paths   : %s", file_paths)
        _process_period(
            period_name=period_name,
            file_paths=file_paths,
            hw_cases=hw_cases,
            nohw_cases=nohw_cases,
            domain=domain,
            cmaps=cmaps,
            lon_range=lon_range,
            lat_range=lat_range,
            X=X,
            Y=Y,
            out_dir=args.output_dir,
            var=args.var,
            tag=tag,
            threshold=args.threshold,
            do_meandiff=args.meandiff,
            do_nine=args.nine,
        )

    log.info("All periods done.")


if __name__ == "__main__":
    main()
