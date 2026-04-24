#!/usr/bin/env python3
"""
Compute detector-wise expected GBM signal from a sky map and compare it with
observed GBM count rates for a GRB trigger.

What this script does
---------------------
1. Reads the trigger geometry from `gll_pt_trigger_frame.py` or a saved JSON.
2. Reads a sky-map histogram (NPZ with `hist`, `abins`, `rbins`) produced by
   the skymap workflow.
3. Converts every sky-map pixel center into RA/Dec and computes the pixel solid
   angle.
4. For every GBM detector, evaluates the effective area over the sky using the
   leaf-response database (`GBMDRMdb002`) and integrates

       source_map * effective_area * dOmega

   over the sky to form an expected detector signal.
5. Reads GBM detector data products from the trigger directory, estimates a
   background-subtracted count rate from trigger time to T90, and compares those
   count rates with the expected detector signals.

Notes
-----
- The expected detector signal is only as absolute as the input source map. If
  your source map is in arbitrary units, the comparison is a relative one.
- By default the observed count-rate estimate prefers CTIME files because they
  are binned light curves and are easy to background-subtract robustly.
- Background is estimated from two side windows around the source interval and
  modeled with a linear interpolation between the pre- and post-burst means.
- The source interval defaults to [TRIGTIME, TRIGTIME + T90] in keeping with
  your request. If a T90 start time is available and you choose to use it later,
  the function can be extended easily.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from astropy.io import fits


# -----------------------------------------------------------------------------
# Dynamic imports
# -----------------------------------------------------------------------------

def _load_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_frame_payload(args: argparse.Namespace) -> Tuple[Dict[str, object], str]:
    if args.frame_json is not None:
        with open(Path(args.frame_json).expanduser().resolve(), "r", encoding="utf-8") as f:
            payload = json.load(f)
        trigger_id = str(payload.get("trigger_id", "unknown"))
        return payload, trigger_id

    frame_mod = _load_module_from_path(Path(args.frame_script).expanduser().resolve(), "gll_pt_trigger_frame_dyn_compare")
    payload = frame_mod.build_payload(
        trigger_id=str(args.trigger_id),
        base_dir=Path('./GRBData/'+args.base_dir).expanduser().resolve(),
        ft2_path=None if args.ft2 is None else Path(args.ft2).expanduser().resolve(),
        tcat_path=None if args.tcat is None else Path(args.tcat).expanduser().resolve(),
        time_override=args.time,
        grb_ra_override=args.grb_ra,
        grb_dec_override=args.grb_dec,
    )
    return payload, str(args.trigger_id)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _radec_to_unit(ra_deg, dec_deg) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


def _unit_to_radec(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec = vec / np.where(norm == 0.0, 1.0, norm)
    x = vec[..., 0]
    y = vec[..., 1]
    z = np.clip(vec[..., 2], -1.0, 1.0)
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arcsin(z))
    return ra, dec


def pixel_centers_from_bins(abins: np.ndarray, rbins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    phi_centers = 0.5 * (abins[:-1] + abins[1:])
    theta_centers_deg = 0.5 * (rbins[:-1] + rbins[1:])
    return phi_centers, theta_centers_deg


def local_pixels_to_radec(payload: Dict[str, object], abins: np.ndarray, rbins: np.ndarray):
    local_frame = payload["local_frame"]
    east = local_frame["east_radec_deg"]
    north = local_frame["north_radec_deg"]
    zenith = local_frame["zenith_radec_deg"]

    east_hat = _radec_to_unit(float(east["ra"]), float(east["dec"]))
    north_hat = _radec_to_unit(float(north["ra"]), float(north["dec"]))
    zen_hat = _radec_to_unit(float(zenith["ra"]), float(zenith["dec"]))

    phi_centers, theta_centers_deg = pixel_centers_from_bins(abins, rbins)
    PHI, THETA_DEG = np.meshgrid(phi_centers, theta_centers_deg, indexing="ij")
    theta = np.deg2rad(THETA_DEG)

    local_x = np.sin(theta) * np.cos(PHI)
    local_y = np.sin(theta) * np.sin(PHI)
    local_z = np.cos(theta)

    vec = (
        local_x[..., None] * east_hat[None, None, :] +
        local_y[..., None] * north_hat[None, None, :] +
        local_z[..., None] * zen_hat[None, None, :]
    )
    ra, dec = _unit_to_radec(vec)
    return {
        "phi_center_rad": PHI,
        "theta_center_deg": THETA_DEG,
        "ra_deg": ra,
        "dec_deg": dec,
    }


def pixel_solid_angle_map(abins: np.ndarray, rbins: np.ndarray) -> np.ndarray:
    """Solid angle per sky-map pixel in sr for hist shape (n_phi, n_theta)."""
    dphi = np.diff(abins)[:, None]  # radians
    theta_lo = np.deg2rad(rbins[:-1])[None, :]
    theta_hi = np.deg2rad(rbins[1:])[None, :]
    return dphi * (np.cos(theta_lo) - np.cos(theta_hi))


# -----------------------------------------------------------------------------
# Sky map and energy grid
# -----------------------------------------------------------------------------

def load_sky_histogram(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = {"hist", "abins", "rbins"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Sky-map NPZ is missing keys: {sorted(missing)}")
    hist = np.asarray(data["hist"], dtype=float)
    abins = np.asarray(data["abins"], dtype=float)
    rbins = np.asarray(data["rbins"], dtype=float)
    if hist.shape != (len(abins) - 1, len(rbins) - 1):
        raise ValueError(
            f"Unexpected histogram shape {hist.shape}; expected ({len(abins)-1}, {len(rbins)-1})"
        )
    return {"hist": hist, "abins": abins, "rbins": rbins}


def parse_energy_grid(args: argparse.Namespace) -> np.ndarray:
    if args.energies is not None:
        return np.array([float(x) for x in args.energies.split(",") if x.strip()], dtype=float)
    if args.energy_min is None or args.energy_max is None:
        raise ValueError("Provide either --energies or both --energy-min and --energy-max")
    if args.n_energy <= 1:
        return np.array([float(args.energy_min)], dtype=float)
    if args.energy_spacing == "log":
        return np.geomspace(float(args.energy_min), float(args.energy_max), int(args.n_energy))
    return np.linspace(float(args.energy_min), float(args.energy_max), int(args.n_energy))


def reduce_energy_axis(aeff_cube: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.mean(aeff_cube, axis=-1)
    if mode == "sum":
        return np.sum(aeff_cube, axis=-1)
    if mode == "max":
        return np.max(aeff_cube, axis=-1)
    raise ValueError(f"Unsupported band reduction mode: {mode}")


# -----------------------------------------------------------------------------
# Trigger data discovery and metadata
# -----------------------------------------------------------------------------

def find_trigger_folder(base_dir: Path, trigger_id: str) -> Path:
    tid = str(trigger_id)
    candidates = [base_dir / f"bn{tid}", base_dir / tid, Path("./GRBData") / f"bn{tid}"]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"Could not find trigger folder for {tid}")


def _find_file(folder: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]
    return None


def read_duration_keywords(trigger_folder: Path, trigger_id: str) -> Dict[str, Optional[float]]:
    candidates = [
        f"glg_bcat_all_bn{trigger_id}_v*.fit*",
        f"glg_bcat_all_{trigger_id}_v*.fit*",
        f"glg_tcat_all_bn{trigger_id}_v*.fit*",
        f"glg_tcat_all_{trigger_id}_v*.fit*",
        "glg_bcat*.fit*",
        "glg_tcat*.fit*",
    ]
    path = _find_file(trigger_folder, candidates)
    out = {"trigtime": None, "t90": None, "t90start": None, "detmask": None, "source_file": None}
    if path is None:
        return out

    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        out["source_file"] = str(path)
        for key in ("TRIGTIME", "TRIGTM", "TRIG_MET", "TIME", "TSTART"):
            if key in header:
                try:
                    out["trigtime"] = float(header[key])
                    break
                except Exception:
                    pass
        for key in ("T90", "DURATION", "T90_DUR"):
            if key in header:
                try:
                    out["t90"] = float(header[key])
                    break
                except Exception:
                    pass
        for key in ("T90START", "T90_ST", "T90START1"):
            if key in header:
                try:
                    out["t90start"] = float(header[key])
                    break
                except Exception:
                    pass
        for key in ("DET_MASK", "DETMASK"):
            if key in header:
                out["detmask"] = str(header[key])
                break
    return out


# -----------------------------------------------------------------------------
# GBM time series reading
# -----------------------------------------------------------------------------

def _find_time_energy_hdu(hdul: fits.HDUList):
    for name in ("SPECTRUM", "COUNTS", "RATE", "BATSE BURST SPECTRA"):
        if name in hdul:
            return hdul[name]
    for hdu in hdul:
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and hdu.data is not None:
            cols = {c.upper() for c in hdu.columns.names or []}
            if ("TIME" in cols or "START" in cols) and ("COUNTS" in cols or "RATE" in cols):
                return hdu
    raise KeyError("Could not locate time series extension in FITS file")


def _col(data, cols: Sequence[str], *candidates: str):
    cmap = {str(c).upper(): str(c) for c in cols}
    for c in candidates:
        if c.upper() in cmap:
            return data[cmap[c.upper()]]
    return None


def read_gbm_binned_timeseries(path: Path) -> Dict[str, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        hdu = _find_time_energy_hdu(hdul)
        data = hdu.data
        cols = list(hdu.columns.names or [])
        time = _col(data, cols, "TIME", "START")
        endtime = _col(data, cols, "ENDTIME", "STOP")
        exposure = _col(data, cols, "EXPOSURE")
        counts = _col(data, cols, "COUNTS", "PHA", "RATE")
        if time is None or counts is None:
            raise KeyError(f"Could not read TIME/COUNTS from {path}")
        t0 = np.asarray(time, dtype=float)
        if endtime is not None:
            t1 = np.asarray(endtime, dtype=float)
        elif exposure is not None:
            t1 = t0 + np.asarray(exposure, dtype=float)
        else:
            # Infer from adjacent bin widths.
            dt = np.median(np.diff(t0)) if len(t0) > 1 else 0.064
            t1 = t0 + dt
        exp = np.asarray(exposure, dtype=float) if exposure is not None else np.asarray(t1 - t0, dtype=float)
        cnt = np.asarray(counts, dtype=float)
        if cnt.ndim == 1:
            cnt = cnt[:, None]

        # Energy bounds if available.
        emin = emax = None
        if "EBOUNDS" in hdul:
            eb = hdul["EBOUNDS"].data
            eb_cols = list(hdul["EBOUNDS"].columns.names or [])
            emin_arr = _col(eb, eb_cols, "E_MIN", "ELOW")
            emax_arr = _col(eb, eb_cols, "E_MAX", "EHIGH")
            if emin_arr is not None and emax_arr is not None:
                emin = np.asarray(emin_arr, dtype=float)
                emax = np.asarray(emax_arr, dtype=float)

        return {
            "tstart": t0,
            "tstop": t1,
            "exposure": exp,
            "counts": cnt,
            "emin": emin,
            "emax": emax,
            "file": str(path),
        }


def choose_detector_data_file(trigger_folder: Path, trigger_id: str, det: str) -> Optional[Path]:
    det = det.lower()
    # GBM burst products commonly store the time series in .pha files, while
    # .rsp/.rsp2 are response files and should not be used here. Prefer CTIME
    # .pha, then CSPEC .pha, then FIT/FITS equivalents if present.
    patterns = [
        f"glg_ctime_{det}_bn{trigger_id}_v*.pha",
        f"glg_ctime_{det}_{trigger_id}_v*.pha",
        f"glg_ctime_{det}_*.pha",
        f"glg_cspec_{det}_bn{trigger_id}_v*.pha",
        f"glg_cspec_{det}_{trigger_id}_v*.pha",
        f"glg_cspec_{det}_*.pha",
        f"glg_ctime_{det}_bn{trigger_id}_v*.fit*",
        f"glg_ctime_{det}_{trigger_id}_v*.fit*",
        f"glg_ctime_{det}_*.fit*",
        f"glg_cspec_{det}_bn{trigger_id}_v*.fit*",
        f"glg_cspec_{det}_{trigger_id}_v*.fit*",
        f"glg_cspec_{det}_*.fit*",
    ]
    path = _find_file(trigger_folder, patterns)
    if path is None:
        return None
    # Guard against accidentally selecting response files.
    if path.suffix.lower() in {".rsp", ".rsp2"}:
        return None
    return path


def select_energy_channels(ts: Dict[str, np.ndarray], emin_keV: Optional[float], emax_keV: Optional[float]) -> np.ndarray:
    counts = ts["counts"]
    nchan = counts.shape[1]
    if emin_keV is None and emax_keV is None:
        return np.ones(nchan, dtype=bool)
    emin = ts.get("emin")
    emax = ts.get("emax")
    if emin is None or emax is None:
        return np.ones(nchan, dtype=bool)
    mask = np.ones_like(emin, dtype=bool)
    if emin_keV is not None:
        mask &= emax >= float(emin_keV)
    if emax_keV is not None:
        mask &= emin <= float(emax_keV)
    if not np.any(mask):
        return np.ones(nchan, dtype=bool)
    return mask


def estimate_background_subtracted_rate(
    ts: Dict[str, np.ndarray],
    trigtime: float,
    source_duration: float,
    emin_keV: Optional[float],
    emax_keV: Optional[float],
    bg_window: float,
    gap: float,
) -> Dict[str, float | str | np.ndarray]:
    """Estimate background-subtracted average count rate over [trigtime, trigtime + source_duration]."""
    tstart = np.asarray(ts["tstart"], dtype=float)
    tstop = np.asarray(ts["tstop"], dtype=float)
    exposure = np.asarray(ts["exposure"], dtype=float)
    chan_mask = select_energy_channels(ts, emin_keV, emax_keV)
    counts = np.asarray(ts["counts"], dtype=float)[:, chan_mask].sum(axis=1)
    rates = counts / np.maximum(exposure, 1e-12)
    tmid = 0.5 * (tstart + tstop)

    src_t0 = float(trigtime)
    src_t1 = float(trigtime + source_duration)

    src_mask = (tstop > src_t0) & (tstart < src_t1)
    pre_mask = (tmid >= src_t0 - gap - bg_window) & (tmid < src_t0 - gap)
    post_mask = (tmid > src_t1 + gap) & (tmid <= src_t1 + gap + bg_window)

    if np.sum(src_mask) == 0:
        raise ValueError("No source bins overlap the requested source interval")
    if np.sum(pre_mask) == 0 or np.sum(post_mask) == 0:
        # Fall back to global median background if side windows are unavailable.
        bg_rate = float(np.nanmedian(rates))
        src_counts = float(np.sum(counts[src_mask]))
        src_exp = float(np.sum(exposure[src_mask]))
        net_rate = src_counts / max(src_exp, 1e-12) - bg_rate
        return {
            "file": ts["file"],
            "src_t0": src_t0,
            "src_t1": src_t1,
            "background_model": "global_median",
            "background_rate_cps": bg_rate,
            "net_rate_cps": net_rate,
            "gross_rate_cps": src_counts / max(src_exp, 1e-12),
            "n_source_bins": int(np.sum(src_mask)),
        }

    pre_rate = float(np.sum(counts[pre_mask]) / max(np.sum(exposure[pre_mask]), 1e-12))
    post_rate = float(np.sum(counts[post_mask]) / max(np.sum(exposure[post_mask]), 1e-12))
    pre_time = float(np.mean(tmid[pre_mask]))
    post_time = float(np.mean(tmid[post_mask]))

    src_mid = tmid[src_mask]
    if abs(post_time - pre_time) < 1e-12:
        bg_src = np.full_like(src_mid, 0.5 * (pre_rate + post_rate), dtype=float)
    else:
        bg_src = pre_rate + (post_rate - pre_rate) * (src_mid - pre_time) / (post_time - pre_time)

    gross_counts = float(np.sum(counts[src_mask]))
    bg_counts = float(np.sum(bg_src * exposure[src_mask]))
    src_exp = float(np.sum(exposure[src_mask]))
    gross_rate = gross_counts / max(src_exp, 1e-12)
    net_rate = (gross_counts - bg_counts) / max(src_exp, 1e-12)

    return {
        "file": ts["file"],
        "src_t0": src_t0,
        "src_t1": src_t1,
        "background_model": "linear_sidebands",
        "pre_rate_cps": pre_rate,
        "post_rate_cps": post_rate,
        "gross_rate_cps": gross_rate,
        "net_rate_cps": net_rate,
        "n_source_bins": int(np.sum(src_mask)),
    }


# -----------------------------------------------------------------------------
# Family normalization helpers
# -----------------------------------------------------------------------------

def detector_family(det: str, leaf_mod) -> str:
    meta = leaf_mod.DETECTORS.get(str(det).lower(), {})
    typ = str(meta.get("type", "")).lower()
    return "BGO" if typ == "bgo" else "NaI"


def add_family_normalized_columns(merged_rows: List[Dict[str, object]], leaf_mod) -> List[Dict[str, object]]:
    family_stats: Dict[str, Dict[str, float]] = {}
    for family in ("NaI", "BGO"):
        exp_vals = []
        obs_vals = []
        for row in merged_rows:
            if detector_family(str(row.get("detector", "")), leaf_mod) != family:
                continue
            expv = float(row.get("expected_signal", float("nan")))
            obsv = float(row.get("net_rate_cps", float("nan")))
            if np.isfinite(expv):
                exp_vals.append(expv)
            if np.isfinite(obsv):
                obs_vals.append(obsv)
        family_stats[family] = {
            "mean_expected_signal": float(np.nanmean(exp_vals)) if len(exp_vals) else float("nan"),
            "mean_net_rate_cps": float(np.nanmean(obs_vals)) if len(obs_vals) else float("nan"),
        }

    out: List[Dict[str, object]] = []
    for row in merged_rows:
        det = str(row.get("detector", "")).lower()
        family = detector_family(det, leaf_mod)
        expv = float(row.get("expected_signal", float("nan")))
        obsv = float(row.get("net_rate_cps", float("nan")))
        mean_exp = family_stats[family]["mean_expected_signal"]
        mean_obs = family_stats[family]["mean_net_rate_cps"]
        exp_norm = expv / mean_exp if np.isfinite(expv) and np.isfinite(mean_exp) and mean_exp != 0.0 else float("nan")
        obs_norm = obsv / mean_obs if np.isfinite(obsv) and np.isfinite(mean_obs) and mean_obs != 0.0 else float("nan")
        ratio_norm = obs_norm / exp_norm if np.isfinite(obs_norm) and np.isfinite(exp_norm) and exp_norm != 0.0 else float("nan")
        log10_ratio_norm = float(np.log10(ratio_norm)) if np.isfinite(ratio_norm) and ratio_norm > 0.0 else float("nan")
        scaled_expected_rate = expv * (mean_obs / mean_exp) if np.isfinite(expv) and np.isfinite(mean_obs) and np.isfinite(mean_exp) and mean_exp != 0.0 else float("nan")
        row2 = dict(row)
        row2.update({
            "detector_family": family,
            "family_mean_expected_signal": mean_exp,
            "family_mean_net_rate_cps": mean_obs,
            "expected_signal_norm_family": exp_norm,
            "net_rate_cps_norm_family": obs_norm,
            "expected_signal_scaled_to_rate": scaled_expected_rate,
            "ratio_norm_family": ratio_norm,
            "log10_ratio_norm_family": log10_ratio_norm,
        })
        out.append(row2)
    return out

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_expected_vs_observed(detectors: Sequence[str], expected: np.ndarray, observed: np.ndarray,
                              out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(expected, observed)
    for d, x, y in zip(detectors, expected, observed):
        ax.annotate(d, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=9)
    ax.set_xlabel("Expected signal (sky map × effective area integrated over sky)")
    ax.set_ylabel("Observed background-subtracted count rate (counts/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# -----------------------------------------------------------------------------
# Local-frame projection helpers
# -----------------------------------------------------------------------------

def radec_to_local_angles(payload: Dict[str, object], ra_deg: float, dec_deg: float) -> Dict[str, float]:
    local_frame = payload["local_frame"]
    east = local_frame["east_radec_deg"]
    north = local_frame["north_radec_deg"]
    zenith = local_frame["zenith_radec_deg"]

    src = _radec_to_unit(float(ra_deg), float(dec_deg))
    east_hat = _radec_to_unit(float(east["ra"]), float(east["dec"]))
    north_hat = _radec_to_unit(float(north["ra"]), float(north["dec"]))
    zen_hat = _radec_to_unit(float(zenith["ra"]), float(zenith["dec"]))

    x = float(np.dot(src, east_hat))
    y = float(np.dot(src, north_hat))
    z = float(np.dot(src, zen_hat))
    theta_deg = float(np.degrees(np.arccos(np.clip(z, -1.0, 1.0))))
    phi_deg = float(np.degrees(np.arctan2(y, x)) % 360.0)
    return {
        "x_east": x,
        "y_north": y,
        "z_zenith": z,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
    }


def spacecraft_direction_to_radec(leaf_mod, att: Dict[str, float], az_deg: float, zen_deg: float) -> Tuple[float, float]:
    xhat, yhat, zhat = leaf_mod.orthonormal_spacecraft_axes(
        scz_ra_deg=float(att["ra_scz_deg"]),
        scz_dec_deg=float(att["dec_scz_deg"]),
        scx_ra_deg=float(att["ra_scx_deg"]),
        scx_dec_deg=float(att["dec_scx_deg"]),
    )
    vsc = np.asarray(leaf_mod.azzen_to_unit(float(az_deg), float(zen_deg)), dtype=float)
    if vsc.ndim > 1:
        vsc = vsc.reshape(-1)
    vcel = vsc[0] * xhat + vsc[1] * yhat + vsc[2] * zhat
    ra, dec = _unit_to_radec(vcel)
    return float(ra), float(dec)


def earth_centered_projection_xy(theta_local_deg: float, phi_local_deg: float) -> Tuple[float, float, float]:
    r = 180.0 - float(theta_local_deg)
    phi = np.deg2rad(float(phi_local_deg))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return float(x), float(y), float(r)


def plot_local_earth_projection(payload: Dict[str, object], leaf_mod, merged_rows: Sequence[Dict[str, object]],
                                out_png: Path, title: str, earth_radius_deg: float = 67.0) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 8.8))

    max_r = 180.0
    earth_patch = Circle((0.0, 0.0), radius=float(earth_radius_deg), facecolor='0.8', edgecolor='0.4',
                         alpha=0.8, lw=1.2, zorder=0)
    ax.add_patch(earth_patch)
    ax.text(0.0, 0.0, 'Earth', ha='center', va='center', fontsize=11, zorder=1)

    for rr in (30.0, 60.0, 90.0, 120.0, 150.0, 180.0):
        circ = Circle((0.0, 0.0), radius=rr, facecolor='none', edgecolor='0.85' if rr < 180.0 else '0.6',
                      ls='--', lw=0.8 if rr < 180.0 else 1.2, zorder=0)
        ax.add_patch(circ)
        if rr < 180.0:
            ax.text(rr, 0.0, f'{rr:.0f}°', fontsize=8, color='0.4', va='bottom', ha='left')

    ax.axhline(0.0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0.0, color='0.85', lw=0.8, zorder=0)
    ax.text(max_r, 0.0, 'E', ha='left', va='center', fontsize=11)
    ax.text(-max_r, 0.0, 'W', ha='right', va='center', fontsize=11)
    ax.text(0.0, max_r, 'N', ha='center', va='bottom', fontsize=11)
    ax.text(0.0, -max_r, 'S', ha='center', va='top', fontsize=11)

    grb_local = payload.get("grb_local_coordinates")
    if isinstance(grb_local, dict):
        gx, gy, _ = earth_centered_projection_xy(float(grb_local["theta_deg"]), float(grb_local["phi_deg"]))
        ax.plot([gx], [gy], marker='*', markersize=16, markerfacecolor='gold', markeredgecolor='black',
                markeredgewidth=1.0, linestyle='None', zorder=6)
        ax.annotate('GRB', (gx, gy), textcoords='offset points', xytext=(8, 8), fontsize=10, weight='bold')

    att = payload["attitude"]
    xs = []
    ys = []
    color_metric = []
    markers = []
    labels = []
    valid_flags = []

    for row in merged_rows:
        det = str(row["detector"]).lower()
        if det not in leaf_mod.DETECTORS:
            continue
        meta = leaf_mod.DETECTORS[det]
        dra, ddec = spacecraft_direction_to_radec(leaf_mod, att, float(meta['az_deg']), float(meta['zen_deg']))
        dlocal = radec_to_local_angles(payload, dra, ddec)
        x, y, _ = earth_centered_projection_xy(dlocal['theta_deg'], dlocal['phi_deg'])
        logratio = float(row.get('log10_ratio_norm_family', float('nan')))

        xs.append(x)
        ys.append(y)
        color_metric.append(logratio)
        markers.append('s' if str(meta.get('type', '')).lower() == 'bgo' else 'o')
        labels.append(det)
        valid_flags.append(np.isfinite(logratio))

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    color_metric = np.asarray(color_metric, dtype=float)
    valid_flags = np.asarray(valid_flags, dtype=bool)
    markers_arr = np.asarray(markers, dtype=object)

    mappable = None
    if np.any(valid_flags):
        vmax = float(np.nanpercentile(np.abs(color_metric[valid_flags]), 95.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(np.abs(color_metric[valid_flags])))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 0.5
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = 'coolwarm'

        for marker_shape in ('o', 's'):
            sel = valid_flags & (markers_arr == marker_shape)
            if np.any(sel):
                sc = ax.scatter(xs[sel], ys[sel], c=color_metric[sel], s=90, marker=marker_shape,
                                cmap=cmap, norm=norm, edgecolors='black', linewidths=0.8, zorder=4)
                if mappable is None:
                    mappable = sc

    invalid = ~valid_flags
    if np.any(invalid):
        for marker_shape in ('o', 's'):
            sel = invalid & (markers_arr == marker_shape)
            if np.any(sel):
                ax.scatter(xs[sel], ys[sel], s=90, marker=marker_shape, color='0.6',
                           edgecolors='black', linewidths=0.8, zorder=4)

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='0.8', alpha=0.8), zorder=6)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'log$_{10}$[(count rate / <count rate>) / (expected / <expected>)]')

    nai_handle = ax.scatter([], [], s=90, marker='o', color='white', edgecolors='black', linewidths=0.8, label='NaI')
    bgo_handle = ax.scatter([], [], s=90, marker='s', color='white', edgecolors='black', linewidths=0.8, label='BGO')
    invalid_handle = ax.scatter([], [], s=90, marker='o', color='0.6', edgecolors='black', linewidths=0.8, label='No ratio')
    ax.legend(handles=[nai_handle, bgo_handle, invalid_handle], loc='lower left', fontsize=9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-max_r - 8.0, max_r + 8.0)
    ax.set_ylim(-max_r - 8.0, max_r + 8.0)
    ax.set_xlabel('Local east-west projection axis (deg from Earth center)')
    ax.set_ylabel('Local north-south projection axis (deg from Earth center)')
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame-json", type=str, help="JSON file written by gll_pt_trigger_frame.py --json")
    src.add_argument("--trigger-id", type=str, help="GBM trigger ID; imports gll_pt_trigger_frame.py directly")

    p.add_argument("--frame-script", type=str, default="./gll_pt_trigger_frame.py")
    p.add_argument("--leaf-tool", type=str, default="./gbm_leaf_aeff_tool.py")
    p.add_argument("--base-dir", type=str, default=".")
    p.add_argument("--ft2", type=str, default=None)
    p.add_argument("--tcat", type=str, default=None)
    p.add_argument("--time", type=float, default=None)
    p.add_argument("--grb-ra", type=float, default=None)
    p.add_argument("--grb-dec", type=float, default=None)

    p.add_argument("--sky-hist", type=str, required=True, help="NPZ histogram produced by skymap_from_trigger.py")
    p.add_argument("--db-root", type=str, required=True, help="Path to GBMDRMdb002")

    p.add_argument("--energies", type=str, default=None, help="Explicit comma-separated energies in keV")
    p.add_argument("--energy-min", type=float, default=None)
    p.add_argument("--energy-max", type=float, default=None)
    p.add_argument("--n-energy", type=int, default=16)
    p.add_argument("--energy-spacing", choices=["log", "linear"], default="log")
    p.add_argument("--energy-reduce", choices=["mean", "sum", "max"], default="mean")
    p.add_argument("--k-neighbors", type=int, default=3)
    p.add_argument("--distance-power", type=float, default=2.0)

    p.add_argument("--count-emin", type=float, default=None, help="Observed-count energy min in keV")
    p.add_argument("--count-emax", type=float, default=None, help="Observed-count energy max in keV")
    p.add_argument("--bg-window", type=float, default=20.0, help="Background side-window length in seconds")
    p.add_argument("--bg-gap", type=float, default=2.0, help="Gap between source and background windows in seconds")
    p.add_argument("--earth-radius-deg", type=float, default=67.0, help="Angular radius of the Earth disk in the Earth-centered local projection")

    p.add_argument("--detectors", type=str, default=None,
                   help="Optional comma-separated detector list; default all detectors in leaf tool")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--print-json", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    payload, trigger_id = load_frame_payload(args)
    trigger_folder = find_trigger_folder(Path('./GRBData/'+args.base_dir).expanduser().resolve(), trigger_id)
    print(trigger_folder)
    duration_info = read_duration_keywords(trigger_folder, trigger_id)

    trigtime = duration_info["trigtime"]
    if trigtime is None:
        trigtime = payload.get("selected_met")
    if trigtime is None:
        raise ValueError("Could not determine TRIGTIME from metadata or frame payload")
    trigtime = float(trigtime)

    t90 = duration_info["t90"]
    if t90 is None:
        raise ValueError("Could not determine T90 from trigger metadata (bcat/tcat)")
    t90 = float(t90)

    leaf_mod = _load_module_from_path(Path(args.leaf_tool).expanduser().resolve(), "gbm_leaf_tool_dyn_compare")
    lib = leaf_mod.GBMLeafAeffLibrary(Path(args.db_root).expanduser().resolve())
    all_detectors = list(leaf_mod.DETECTORS.keys())
    detectors = all_detectors if args.detectors is None else [d.strip().lower() for d in args.detectors.split(",") if d.strip()]

    sky = load_sky_histogram(Path(args.sky_hist).expanduser().resolve())
    coords = local_pixels_to_radec(payload, sky["abins"], sky["rbins"])
    dOmega = pixel_solid_angle_map(sky["abins"], sky["rbins"])
    src_ra = coords["ra_deg"].ravel()
    src_dec = coords["dec_deg"].ravel()
    source_map = np.asarray(sky["hist"], dtype=float)

    energies = parse_energy_grid(args)
    att = payload["attitude"]

    expected_rows: List[Dict[str, object]] = []
    for det in detectors:
        res = lib.compute_detector_from_radec(
            detector=det,
            scz_ra_deg=float(att["ra_scz_deg"]),
            scz_dec_deg=float(att["dec_scz_deg"]),
            scx_ra_deg=float(att["ra_scx_deg"]),
            scx_dec_deg=float(att["dec_scx_deg"]),
            src_ra_deg=src_ra,
            src_dec_deg=src_dec,
            energies_kev=energies,
            k_neighbors=int(args.k_neighbors),
            distance_power=float(args.distance_power),
        )
        nphi, ntheta = source_map.shape
        aeff_cube = np.asarray(res["effective_area_cm2"], dtype=float).reshape(nphi, ntheta, len(energies))
        aeff_band = reduce_energy_axis(aeff_cube, args.energy_reduce)
        expected_signal = float(np.nansum(source_map * aeff_band * dOmega))
        expected_rows.append({
            "detector": det,
            "expected_signal": expected_signal,
            "aeff_band_min_cm2": float(np.nanmin(aeff_band)),
            "aeff_band_max_cm2": float(np.nanmax(aeff_band)),
            "incident_angle_mean_deg": float(np.nanmean(np.asarray(res["incident_angle_deg"]))),
        })

    observed_rows: List[Dict[str, object]] = []
    for row in expected_rows:
        det = str(row["detector"])
        det_file = choose_detector_data_file(trigger_folder, trigger_id, det)
        obs = {
            "detector": det,
            "data_file": None if det_file is None else str(det_file),
            "gross_rate_cps": float("nan"),
            "net_rate_cps": float("nan"),
            "background_model": None,
            "n_source_bins": 0,
        }
        if det_file is not None:
            try:
                ts = read_gbm_binned_timeseries(det_file)
                rate_info = estimate_background_subtracted_rate(
                    ts=ts,
                    trigtime=trigtime,
                    source_duration=t90,
                    emin_keV=args.count_emin,
                    emax_keV=args.count_emax,
                    bg_window=float(args.bg_window),
                    gap=float(args.bg_gap),
                )
                obs.update(rate_info)
            except Exception as exc:
                obs["error"] = str(exc)
        else:
            obs["error"] = "No CTIME/CSPEC file found"
        observed_rows.append(obs)

    merged_raw: List[Dict[str, object]] = []
    for erow, orow in zip(expected_rows, observed_rows):
        merged_raw.append({**erow, **orow})

    merged = add_family_normalized_columns(merged_raw, leaf_mod)

    valid = [r for r in merged if np.isfinite(r["expected_signal"]) and np.isfinite(r["net_rate_cps"])]
    exp_arr = np.array([r["expected_signal"] for r in valid], dtype=float)
    obs_arr = np.array([r["net_rate_cps"] for r in valid], dtype=float)
    det_valid = [str(r["detector"]) for r in valid]

    valid_norm = [r for r in merged if np.isfinite(r["expected_signal_norm_family"]) and np.isfinite(r["net_rate_cps_norm_family"])]
    exp_norm_arr = np.array([r["expected_signal_norm_family"] for r in valid_norm], dtype=float)
    obs_norm_arr = np.array([r["net_rate_cps_norm_family"] for r in valid_norm], dtype=float)
    det_valid_norm = [str(r["detector"]) for r in valid_norm]

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"bn{trigger_id}_all_detectors"

    plot_path = outdir / f"expected_vs_counts_{tag}.png"
    plot_expected_vs_observed(
        det_valid,
        exp_arr,
        obs_arr,
        plot_path,
        title=(
            f"Expected signal vs observed GBM count rate | Trigger {trigger_id}\n"
            f"Observed interval: TRIGTIME to TRIGTIME+T90 ({t90:.3f} s)"
        ),
    )

    plot_norm_path = outdir / f"expected_vs_counts_normalized_{tag}.png"
    if len(valid_norm) > 0:
        plot_expected_vs_observed(
            det_valid_norm,
            exp_norm_arr,
            obs_norm_arr,
            plot_norm_path,
            title=(
                f"Family-normalized expected vs observed GBM count rate | Trigger {trigger_id}\n"
                f"NaI and BGO normalized separately by their family averages"
            ),
        )
    else:
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.text(
            0.5, 0.5,
            "No valid family-normalized detector pairs were available.",
            ha="center", va="center", fontsize=12
        )
        ax.set_axis_off()
        ax.set_title(f"Family-normalized expected vs observed GBM count rate | Trigger {trigger_id}")
        fig.tight_layout()
        fig.savefig(plot_norm_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    proj_path = outdir / f"local_projection_counts_over_expected_{tag}.png"
    plot_local_earth_projection(
        payload=payload,
        leaf_mod=leaf_mod,
        merged_rows=merged,
        out_png=proj_path,
        title=(
            f"GBM detectors in GRB local frame | Trigger {trigger_id}\n"
            f"Colors show log10[(count/<count>_family)/(expected/<expected>_family)]"
        ),
        earth_radius_deg=float(args.earth_radius_deg),
    )

    csv_path = outdir / f"expected_vs_counts_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    summary = {
        "trigger_id": trigger_id,
        "trigger_folder": str(trigger_folder),
        "sky_hist": str(Path(args.sky_hist).expanduser().resolve()),
        "db_root": str(Path(args.db_root).expanduser().resolve()),
        "trigtime": trigtime,
        "t90": t90,
        "duration_metadata": duration_info,
        "energy_grid_kev": energies.tolist(),
        "energy_reduce": args.energy_reduce,
        "count_energy_range_kev": {
            "emin": args.count_emin,
            "emax": args.count_emax,
        },
        "background": {
            "window_sec": float(args.bg_window),
            "gap_sec": float(args.bg_gap),
            "source_interval": [trigtime, trigtime + t90],
        },
        "n_detectors_requested": len(detectors),
        "n_detectors_compared": len(valid),
        "pearson_r": _corrcoef_safe(exp_arr, obs_arr),
        "outputs": {
            "plot_png": str(plot_path),
            "normalized_plot_png": str(plot_norm_path) if len(valid_norm) > 0 else None,
            "local_projection_png": str(proj_path),
            "table_csv": str(csv_path),
        },
        "family_normalization": {
            "n_valid_normalized": len(valid_norm),
            "pearson_r_normalized": _corrcoef_safe(exp_norm_arr, obs_norm_arr) if len(valid_norm) > 0 else float("nan"),
        },
        "rows": merged,
    }

    summary_path = outdir / f"summary_expected_vs_counts_{tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Trigger folder: {trigger_folder}")
    print(f"TRIGTIME: {trigtime:.6f} | T90: {t90:.6f} s")
    print(f"Detectors requested: {len(detectors)} | compared successfully: {len(valid)}")
    print(f"Saved plot:   {plot_path}")
    if len(valid_norm) > 0:
        print(f"Saved norm:   {plot_norm_path}")
    print(f"Saved local:  {proj_path}")
    print(f"Saved table:  {csv_path}")
    print(f"Saved summary:{summary_path}")

    if args.print_json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
