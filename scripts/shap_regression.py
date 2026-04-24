"""
shap_regression.py
==================
Compute SHAP (SHapley Additive exPlanations) values for every HW / NO-HW
case-study pair and save the results as ``.png`` figures and ``.npy`` arrays
for downstream analysis with ``shap_figures.py``.

The script loads a pre-trained Autoencoder (AE), builds a DeepExplainer
using the post-industrial training period as background data, and then for
each case-study pair it:

  1. Loads the event-period pressure/temperature fields.
  2. Computes SHAP values for both the HW and NO-HW windows.
  3. Normalises values per variable using the joint (HW ∪ NO-HW) percentile.
  4. Optionally applies KMeans clustering on the latent space.
  5. Saves per-variable SHAP image plots and mean-sum ``.npy`` arrays.

Variable index mapping (last axis of the 4-D input arrays)
----------------------------------------------------------
    0 → Z500   (geopotential height at 500 hPa)
    1 → PEva   (potential evapotranspiration)
    2 → MSL    (mean sea-level pressure)
    3 → SM     (soil moisture)

Usage
-----
    python scripts/shap_regression.py
    python scripts/shap_regression.py \\
        --config config/model.json \\
        --case-studies config/case_studies.json \\
        --output-dir ./shap/mod512 \\
        --meansum --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import shap
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow import keras

from va_am.utils import AutoEncoders
from va_am.va_am import perform_preprocess, square_dims

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
# Variable configuration: (name, channel_slice, joint_percentile)
# ---------------------------------------------------------------------------
VARIABLES: list[tuple[str, slice, int]] = [
    ("z500", slice(0, 1), 99),
    ("peva", slice(1, 2), 100),
    ("msl",  slice(2, 3), 99),
    ("sm",   slice(3, 4), 100),
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_config(model_path: str, cs_path: str) -> tuple[dict, list[dict], list[dict]]:
    """
    Load and validate both configuration files.

    Returns
    -------
    params : dict
        Flat parameter dict expected by ``perform_preprocess``.
    hw_cases : list of dict
        Heatwave case studies, each with ``id``, ``label``, ``start``, ``end``.
    nohw_cases : list of dict
        Control (NO-HW) case studies in the same format.
    """
    with open(model_path) as f:
        cfg = json.load(f)
    with open(cs_path) as f:
        cs = json.load(f)

    hw_cases   = cs["hw"]
    nohw_cases = cs["no_hw"]
    if len(hw_cases) != len(nohw_cases):
        raise ValueError(
            f"config/case_studies.json: 'hw' and 'no_hw' must have the same "
            f"number of entries ({len(hw_cases)} vs {len(nohw_cases)})."
        )

    # Flatten nested config into the flat dict that perform_preprocess expects
    params: dict = {}
    params["name"]   = cfg["name"]
    params["season"] = cfg["season"]

    domain = cfg["domain"]
    params.update(domain)  # latitude_min/max, longitude_min/max, resolution

    tp = cfg["time_periods"]
    params.update(tp)  # pre_init, pre_end, post_init, post_end

    ds = cfg["datasets"]
    params.update(ds)  # prs_dataset, temp_dataset, ident_dataset, temp_var_name

    model = cfg["model"]
    params["file_AE_post"] = model["file_AE_post"]
    params["file_AE_pre"]  = model["file_AE_pre"]
    params["latent_dim"]   = model["latent_dim"]
    params["arch"]         = model["arch"]
    params["period"]       = model["period"]

    prep = cfg["preprocessing"]
    params.update(prep)  # interest_region, per_what, remove_year, …

    clust = cfg["clustering"]
    params["perform_cluster"] = clust["perform_cluster"]
    params["n_clusters"]      = clust.get("n_clusters", 8)
    if "path_cluster" in clust and clust["path_cluster"]:
        params["path_cluster"] = clust["path_cluster"]

    params["verbose"] = cfg.get("verbose", False)

    # Fields set at runtime – not stored in config
    params["teleg"]       = False
    params["secret_file"] = ""
    params["out_preprocess"] = [
        "params", "img_size", "data_prs", "data_temp",
        "time_indust_prs", "data_of_interest_prs", "data_of_interest_temp",
        "x_train_ind_prs", "x_test_ind_prs", "indust_prs", "indust_temp",
    ]

    return params, hw_cases, nohw_cases


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _joint_percentile(
    shap_nohw: np.ndarray,
    shap_hw: np.ndarray,
    var_slice: slice,
    pct: int,
) -> float:
    """Return *pct*-th percentile of |shap_nohw[…,sl]| ∪ |shap_hw[…,sl]|."""
    combined = np.concatenate(
        (np.abs(np.array(shap_nohw)[..., var_slice]),
         np.abs(np.array(shap_hw)[..., var_slice]))
    )
    return float(np.nanpercentile(combined, pct))


def compute_normalisations(
    shap_nohw: np.ndarray,
    shap_hw: np.ndarray,
) -> dict[str, float]:
    """
    Compute per-variable SHAP normalisation constants.

    Each constant is derived from the joint distribution of |NO-HW| and |HW|
    SHAP values for that variable, ensuring a consistent colour scale across
    both event types.

    Returns a dict with keys ``"all"``, ``"z500"``, ``"peva"``, ``"msl"``, ``"sm"``.
    """
    norms: dict[str, float] = {
        "all": float(
            np.nanpercentile(
                np.concatenate((np.abs(shap_nohw), np.abs(shap_hw))), 99
            )
        )
    }
    for name, sl, pct in VARIABLES:
        norms[name] = _joint_percentile(shap_nohw, shap_hw, sl, pct)
    return norms


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _cluster_mean(
    shap_arr: np.ndarray,
    labels: np.ndarray,
    n_clust: int,
) -> np.ndarray:
    """Average SHAP values within each cluster. Returns shape ``(n_clust, …)``."""
    out = np.zeros((n_clust,) + shap_arr.shape[1:])
    for c in range(n_clust):
        mask = labels == c
        if mask.any():
            out[c] = shap_arr[mask].mean(axis=0)
    return out


def _apply_clustering(
    shap_nohw: np.ndarray,
    shap_hw: np.ndarray,
    params: dict,
    enc: tf.keras.Model,
    x_train_clust: np.ndarray,
    output_dir: str,
    save_cluster_files: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply KMeans clustering in the latent space and return cluster-averaged
    SHAP arrays together with the cluster weights.

    Parameters
    ----------
    save_cluster_files : bool
        If ``True`` save the cluster label ``.npy`` and visualisation ``.png``
        (only done once, for the first case study).
    """
    n_clust = params.get("n_clusters", 8)

    if "path_cluster" in params and os.path.isfile(params["path_cluster"]):
        labels = np.load(params["path_cluster"])
        log.info("  Cluster labels loaded from  %s", params["path_cluster"])
    else:
        log.info("  Fitting KMeans (k=%d) on latent space …", n_clust)
        km = KMeans(n_clusters=n_clust, random_state=0, n_init="auto")
        km.fit(enc.predict(x_train_clust).T)
        labels = km.labels_

        if save_cluster_files:
            cluster_dir = os.path.join(output_dir, "..", "clusters")
            os.makedirs(cluster_dir, exist_ok=True)
            label_path = os.path.join(cluster_dir, f'cluster{params["name"]}.npy')
            np.save(label_path, labels)
            log.info("  Cluster labels saved to  %s", label_path)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(labels.reshape(square_dims(len(labels))), cmap="Set1")
            fig.colorbar(im, ax=ax, label="Cluster")
            ax.set_title(f'Latent-space clusters – {params["name"]}')
            fig.tight_layout()
            fig.savefig(label_path.replace(".npy", ".png"), dpi=150)
            plt.close(fig)

    weights = np.array([np.sum(labels == c) for c in range(n_clust)], dtype=float)
    shap_nohw_c = _cluster_mean(shap_nohw, labels, n_clust)
    shap_hw_c   = _cluster_mean(shap_hw,   labels, n_clust)
    return shap_nohw_c, shap_hw_c, weights


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
def _save_shap_figure(
    shap_arr: np.ndarray,
    data_arr: np.ndarray,
    base_path: str,
    meansum: bool,
    is_cluster: bool,
    weights: Optional[np.ndarray] = None,
) -> None:
    """
    Save a SHAP mean-sum (or plain-sum) figure as ``.png`` and ``.npy``.

    When *is_cluster* is ``True``, two variants are saved:
      * unweighted cluster-sum   → ``base_path`` with ``_sum`` → ``_sumclust``
      * weighted cluster-sum     → ``base_path``  (primary output used by figures script)

    Parameters
    ----------
    shap_arr :
        SHAP values, shape ``(n_samples, H, W, C)`` or ``(n_clusters, H, W, C)``.
    data_arr :
        Corresponding input fields used as background for the image plot.
    base_path :
        Output path **without** extension; ``.png`` and ``.npy`` are appended.
    meansum :
        Average over samples (``True``) or keep full batch (``False``).
    is_cluster :
        Whether clustering was applied.
    weights :
        Per-cluster sample counts; required when *is_cluster* is ``True``.
    """
    arr     = np.array(shap_arr)
    arr_sum = arr.sum(axis=0)

    def _write(plot_shap: np.ndarray, plot_data: np.ndarray, path: str) -> None:
        fig = plt.figure()
        shap.image_plot(list([plot_shap]), plot_data, show=False)
        plt.savefig(f"{path}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.save(f"{path}.npy", plot_shap)

    if is_cluster and weights is not None:
        # Unweighted cluster-sum (kept for reference)
        tag = "meansum" if meansum else "sum"
        clust_path = base_path.replace(f"_{tag}_", f"_{tag}clust_")
        unweighted = arr_sum.mean(axis=0) if meansum else arr_sum
        _write(
            unweighted,
            data_arr.mean(axis=0) if meansum else data_arr,
            clust_path,
        )
        # Weighted cluster-sum (primary output)
        w_sum = (arr.T * weights).T.sum(axis=0)
        weighted = w_sum.mean(axis=0) if meansum else w_sum
        _write(
            weighted,
            data_arr.mean(axis=0) if meansum else data_arr,
            base_path,
        )
    else:
        result = arr_sum.mean(axis=0) if meansum else arr_sum
        _write(
            result,
            data_arr.mean(axis=0) if meansum else data_arr,
            base_path,
        )


# ---------------------------------------------------------------------------
# Case-study processing
# ---------------------------------------------------------------------------
def process_case_study(
    cs_idx: int,
    hw: dict,
    nohw: dict,
    params: dict,
    enc: tf.keras.Model,
    x_train_bg: np.ndarray,
    x_train_clust: np.ndarray,
    meansum: bool,
    output_dir: str,
) -> None:
    """
    Run SHAP analysis for one HW / NO-HW pair and save all outputs.

    Parameters
    ----------
    cs_idx :
        0-based index in the case-study list.
    hw :
        HW case-study dict with keys ``id``, ``label``, ``start``, ``end``.
    nohw :
        Corresponding NO-HW case-study dict.
    params :
        Flat configuration dict (date fields are mutated internally).
    enc :
        Pre-trained Keras autoencoder.
    x_train_bg :
        Background data for the SHAP DeepExplainer (subset of the
        post-industrial training set).
    x_train_clust :
        Full post-industrial set used for clustering.
    meansum :
        Whether to save the mean-of-sum or the raw sum.
    output_dir :
        Root directory under which ``cs_hw{N}/`` and ``cs_nohw{N}/``
        subdirectories are created.
    """
    cs_num = hw["id"]
    log.info("─" * 60)
    log.info("Case study %d  |  HW: %s  |  NO-HW: %s",
             cs_num, hw["label"], nohw["label"])
    log.info("  HW    : %s → %s", hw["start"], hw["end"])
    log.info("  NO-HW : %s → %s", nohw["start"], nohw["end"])

    dir_hw   = os.path.join(output_dir, f"cs_hw{cs_num}")
    dir_nohw = os.path.join(output_dir, f"cs_nohw{cs_num}")
    os.makedirs(dir_hw,   exist_ok=True)
    os.makedirs(dir_nohw, exist_ok=True)

    # ── Load event data ───────────────────────────────────────────────────
    params["data_of_interest_init"], params["data_of_interest_end"] = (
        hw["start"], hw["end"]
    )
    params["out_preprocess"] = ["data_of_interest_pred"]
    data_hw_prs = perform_preprocess(params)[0]
    data_hw_prs = np.flip(data_hw_prs, axis=1)

    params["data_of_interest_init"], params["data_of_interest_end"] = (
        nohw["start"], nohw["end"]
    )
    data_nohw_prs = perform_preprocess(params)[0]
    data_nohw_prs = np.flip(data_nohw_prs, axis=1)

    log.info("  Data loaded  →  HW %s   NO-HW %s",
             data_hw_prs.shape, data_nohw_prs.shape)

    # ── SHAP explainer ────────────────────────────────────────────────────
    explainer = shap.DeepExplainer(model=enc, data=x_train_bg)

    sv_nohw   = explainer.shap_values(data_nohw_prs)
    shap_nohw = np.array(sv_nohw).transpose(4, 0, 1, 2, 3)

    sv_hw   = explainer.shap_values(data_hw_prs)
    shap_hw = np.array(sv_hw).transpose(4, 0, 1, 2, 3)

    log.info("  SHAP values computed  →  NO-HW %s   HW %s",
             shap_nohw.shape, shap_hw.shape)

    # ── Normalisation ─────────────────────────────────────────────────────
    norms = compute_normalisations(shap_nohw, shap_hw)
    log.info(
        "  Norms  all=%.4f  z500=%.4f  peva=%.4f  msl=%.4f  sm=%.4f",
        norms["all"], norms["z500"], norms["peva"], norms["msl"], norms["sm"],
    )

    # ── Optional clustering ───────────────────────────────────────────────
    is_cluster = params.get("perform_cluster", False)
    weights: Optional[np.ndarray] = None

    if is_cluster:
        shap_nohw, shap_hw, weights = _apply_clustering(
            shap_nohw, shap_hw, params, enc, x_train_clust,
            output_dir=output_dir,
            save_cluster_files=(cs_idx == 0),
        )
        log.info("  Clustering applied  (k=%d)", params.get("n_clusters", 8))

    # ── Save figures ──────────────────────────────────────────────────────
    name   = params["name"]
    ms_tag = "meansum" if meansum else "sum"

    for event, shap_arr, data_arr, out_dir in (
        ("hw",   shap_hw,   data_hw_prs,   dir_hw),
        ("nohw", shap_nohw, data_nohw_prs, dir_nohw),
    ):
        # All-variables raw image plot (normalised, not mean-sum)
        fig = plt.figure()
        shap.image_plot(
            list(shap_arr / norms["all"]),
            data_arr, show=False, vmax=1.0,
        )
        plt.savefig(
            os.path.join(out_dir, f"shap_{event}_zpms{name}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        # All-variables mean-sum
        _save_shap_figure(
            shap_arr, data_arr,
            base_path=os.path.join(out_dir, f"shap_{event}_{ms_tag}_zpms{name}"),
            meansum=meansum, is_cluster=is_cluster, weights=weights,
        )

        # Per-variable
        for var_name, var_sl, _ in VARIABLES:
            norm     = norms[var_name]
            v_shap   = shap_arr[..., var_sl]
            v_data   = data_arr[..., var_sl]

            # Raw normalised image plot
            fig = plt.figure()
            shap.image_plot(
                list(v_shap / norm), v_data, show=False, vmax=1.0
            )
            plt.savefig(
                os.path.join(out_dir, f"shap_{event}_{var_name}{name}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

            # Mean-sum
            _save_shap_figure(
                v_shap, v_data,
                base_path=os.path.join(
                    out_dir, f"shap_{event}_{ms_tag}_{var_name}{name}"
                ),
                meansum=meansum, is_cluster=is_cluster, weights=weights,
            )

    log.info("  All figures saved for case study %d", cs_num)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute SHAP values for HW / NO-HW case studies using a "
            "pre-trained Autoencoder and save results for figure generation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", dest="config",
        default="./config/model.json",
        help="Path to the model & preprocessing JSON config.",
    )
    p.add_argument(
        "--case-studies", dest="case_studies",
        default="./config/case_studies.json",
        help="Path to the case-study periods JSON.",
    )
    p.add_argument(
        "--output-dir", dest="output_dir",
        default=None,
        help="Root directory for output files (overrides config value).",
    )
    p.add_argument(
        "--meansum", action="store_true",
        help="Save the mean-of-sum SHAP arrays (default: plain sum).",
    )
    p.add_argument(
        "--secret-file", dest="secret",
        default="secret.txt",
        help="TXT file with Telegram bot credentials (token/chat_id/username).",
    )
    p.add_argument(
        "--telegram", action="store_true",
        help="Send exceptions and completion notices to a Telegram bot.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Telegram setup ────────────────────────────────────────────────────
    token = chat_id = user_name = None
    if args.telegram:
        with open(args.secret) as f:
            token     = f.readline().strip()
            chat_id   = f.readline().strip()
            user_name = f.readline().strip()

    try:
        # ── Load config ───────────────────────────────────────────────────
        params, hw_cases, nohw_cases = _load_config(
            args.config, args.case_studies
        )
        output_dir = (
            args.output_dir
            or _nested_get(
                json.load(open(args.config)), ["output", "output_dir"],
                default="./shap/mod512",
            )
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info("Config loaded  –  %d case study pair(s)", len(hw_cases))
        log.info("Output dir     –  %s", output_dir)

        # ── Load model ────────────────────────────────────────────────────
        enc = tf.keras.models.load_model(
            params["file_AE_post"],
            custom_objects={
                "keras": keras,
                "AutoEncoders": AutoEncoders,
                "ReLU": keras.layers.ReLU,
            },
        )
        log.info("Model loaded   –  %s", params["file_AE_post"])
        if args.verbose:
            enc.summary(print_fn=log.debug)

        # ── Background data (loaded once for all case studies) ────────────
        # Temporarily set the date range to the first HW event so that
        # perform_preprocess returns x_train_ind_prs (post-industrial data).
        params["data_of_interest_init"] = hw_cases[0]["start"]
        params["data_of_interest_end"]  = hw_cases[0]["end"]
        params["out_preprocess"] = ["x_train_ind_pred"]
        x_train_full = perform_preprocess(params)[0]

        x_train_clust = x_train_full.copy()
        # Limit background samples for DeepExplainer (memory / speed trade-off)
        n_bg = 100 if params["latent_dim"] > 900 else 1000
        x_train_bg = np.flip(x_train_full[-n_bg:], axis=1)
        log.info("Background data ready  –  %d samples", n_bg)

        # ── Case-study loop ───────────────────────────────────────────────
        for idx, (hw, nohw) in enumerate(zip(hw_cases, nohw_cases)):
            process_case_study(
                cs_idx=idx,
                hw=hw,
                nohw=nohw,
                params=params,
                enc=enc,
                x_train_bg=x_train_bg,
                x_train_clust=x_train_clust,
                meansum=args.meansum,
                output_dir=output_dir,
            )

        log.info("=" * 60)
        log.info("All %d case studies completed successfully.", len(hw_cases))

    except Exception as ex:
        log.exception("Fatal error")
        if args.telegram and token:
            msg = traceback.format_exc().replace("<", "").replace(">", "")
            _send_telegram(
                token, chat_id,
                f"[<b>{type(ex).__name__}</b>] {user_name}: {msg}",
            )
        raise

    if args.telegram and token:
        _send_telegram(
            token, chat_id,
            f"[DONE] shap_regression.py – {params['name']} finished OK.",
        )


def _nested_get(d: dict, keys: list[str], default=None):
    """Safely retrieve a value from a nested dict using a list of keys."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _send_telegram(token: str, chat_id: str, text: str) -> None:
    """Send a message to a Telegram bot (best-effort; errors are suppressed)."""
    try:
        url = (
            f"https://api.telegram.org/bot{token}/sendMessage"
            f"?chat_id={chat_id}&parse_mode=HTML&text={text}"
        )
        requests.get(url, timeout=10).json()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
