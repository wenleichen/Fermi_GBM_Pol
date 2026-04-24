#!/usr/bin/env python3
"""
Generate a sky map from a simulation output directory by selecting the correct
Pol_<theta>_<phi> folder from the GRB local coordinates produced by
`gll_pt_trigger_frame.py`.

This script supports two ways to obtain the GRB local coordinates:

1) Read a JSON payload written by `gll_pt_trigger_frame.py --json`.
2) Import `gll_pt_trigger_frame.py` directly and compute the payload from a
   trigger ID / trigger folder.

The simulation grid is assumed to be organized in folders like

    SimData/Pol_0_0/
    SimData/Pol_0_15/
    SimData/Pol_15_0/
    ...

with a step size of 15 deg in both theta and phi. By default, a GRB with local
coordinates (theta_deg, phi_deg) is mapped to the folder using the lower-edge
bin values:

    theta_bin = floor(theta_deg / 15) * 15
    phi_bin   = floor(phi_deg   / 15) * 15

This version adds a position angle (PA) correction:

    phi_search_deg = phi_deg - PA_deg

and uses phi_search_deg when selecting the Pol_<theta>_<phi> folder.

After the sky map is built, it is rotated by +PA_deg along the phi direction
before being saved and plotted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


# -----------------------------------------------------------------------------
# Reading GRB local coordinates from gll_pt_trigger_frame.py
# -----------------------------------------------------------------------------

def _load_module_from_path(module_path: Path, module_name: str = "gll_pt_trigger_frame_dyn"):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_payload_from_json(json_path: Path) -> Dict[str, object]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_payload_from_trigger(trigger_id: str, base_dir: Path, frame_script: Path,
                               ft2: Optional[Path] = None, tcat: Optional[Path] = None,
                               time_override: Optional[float] = None,
                               grb_ra_override: Optional[float] = None,
                               grb_dec_override: Optional[float] = None) -> Dict[str, object]:
    mod = _load_module_from_path(frame_script)
    payload = mod.build_payload(
        trigger_id=str(trigger_id),
        base_dir=Path(base_dir),
        ft2_path=None if ft2 is None else Path(ft2),
        tcat_path=None if tcat is None else Path(tcat),
        time_override=time_override,
        grb_ra_override=grb_ra_override,
        grb_dec_override=grb_dec_override,
    )
    return payload


def extract_grb_local_coords(payload: Dict[str, object]) -> Tuple[float, float]:
    loc = payload.get("grb_local_coordinates")
    if not isinstance(loc, dict):
        raise ValueError(
            "Payload does not contain 'grb_local_coordinates'. Ensure the GRB RA/Dec "
            "were available to gll_pt_trigger_frame.py."
        )
    theta_deg = float(loc["theta_deg"])
    phi_deg = float(loc["phi_deg"])
    return theta_deg, phi_deg


# -----------------------------------------------------------------------------
# Selecting the simulation folder
# -----------------------------------------------------------------------------

def bin_angle_to_lower_edge(angle_deg: float, step_deg: float, *, wrap_360: bool = False,
                            max_angle_deg: Optional[float] = None) -> int:
    ang = float(angle_deg)
    if wrap_360:
        ang = ang % 360.0
        if math.isclose(ang, 360.0):
            ang = 0.0
    else:
        if max_angle_deg is not None:
            upper = float(max_angle_deg)
            ang = min(max(ang, 0.0), np.nextafter(upper, -np.inf))
        else:
            ang = max(ang, 0.0)
    return int(math.floor(ang / float(step_deg)) * float(step_deg))


def select_pol_folder(sim_root: Path, theta_deg: float, phi_deg: float,
                      step_deg: float = 15.0, pol_prefix: str = "Pol") -> Tuple[Path, int, int]:
    theta_bin = bin_angle_to_lower_edge(theta_deg, step_deg, wrap_360=False, max_angle_deg=180.0)
    phi_bin = bin_angle_to_lower_edge(phi_deg, step_deg, wrap_360=True)

    folder = sim_root / f"{pol_prefix}_{theta_bin}_{phi_bin}"
    if folder.exists() and folder.is_dir():
        return folder, theta_bin, phi_bin

    candidates: List[Tuple[float, Path, int, int]] = []
    for p in sim_root.glob(f"{pol_prefix}_*_*"):
        if not p.is_dir():
            continue
        parts = p.name.split("_")
        if len(parts) < 3:
            continue
        try:
            t = int(parts[-2])
            ph = int(parts[-1])
        except ValueError:
            continue
        dphi = abs((phi_bin - ph + 180) % 360 - 180)
        dist2 = (theta_bin - t) ** 2 + dphi ** 2
        candidates.append((dist2, p, t, ph))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find simulation folder {folder} and found no {pol_prefix}_*_* folders under {sim_root}"
        )

    candidates.sort(key=lambda x: x[0])
    _, best_folder, best_theta_bin, best_phi_bin = candidates[0]
    return best_folder, best_theta_bin, best_phi_bin


# -----------------------------------------------------------------------------
# Reading simulation data and building the sky map
# -----------------------------------------------------------------------------

def _iter_primary_rows(file_path: Path) -> Iterable[Sequence[str]]:
    data_tmp = np.loadtxt(str(file_path), dtype=str)
    arr = np.asarray(data_tmp)

    if arr.shape == (13,):
        row = data_tmp
        if row[12] == "Primary":
            yield row
        return

    for row in data_tmp:
        if row[12] == "Primary":
            yield row


def load_primary_events_from_folder(pol_folder: Path, filename: str = "DTRout_tmp.dat") -> List[Sequence[str]]:
    files = sorted(pol_folder.glob(filename))
    if not files:
        raise FileNotFoundError(f"No {filename} found under {pol_folder}")

    rows: List[Sequence[str]] = []
    for f in files:
        rows.extend(list(_iter_primary_rows(f)))
    if not rows:
        raise ValueError(f"No rows tagged 'Primary' found in {filename} under {pol_folder}")
    return rows


def build_event_arrays(rows: Sequence[Sequence[str]]) -> Dict[str, np.ndarray]:
    n_gamma = len(rows)
    eng = np.zeros(n_gamma, dtype=float)
    theta = np.zeros(n_gamma, dtype=float)
    phi = np.zeros(n_gamma, dtype=float)

    for k, row in enumerate(rows):
        eng[k] = float(row[4])

        dx = float(row[5]) - float(row[8])
        dy = float(row[6]) - float(row[9])
        dz = float(row[7]) - float(row[10])

        phi_k = math.atan2(dy, dx)
        if phi_k < 0.0:
            phi_k += 2.0 * math.pi

        theta_k = math.atan2(math.sqrt(dx * dx + dy * dy), abs(dz))
        if float(row[10]) - float(row[7]) > 0.0:
            theta_k = math.pi - theta_k

        phi[k] = phi_k
        theta[k] = theta_k

    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    return {
        "energy": eng,
        "theta_rad": theta,
        "phi_rad": phi,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
    }


def make_sky_histogram(theta_deg: np.ndarray, phi_rad: np.ndarray,
                       nr: int = 90, na: int = 90) -> Dict[str, np.ndarray]:
    rbins = np.linspace(0.0, 180.0, nr)
    dr = rbins[1] - rbins[0]
    abins = np.linspace(0.0, 2.0 * np.pi, na)
    da_rad = abins[1] - abins[0]

    hist_gamma_bg, _, _ = np.histogram2d(phi_rad, theta_deg, bins=(abins, rbins))

    theta_centers_rad = np.radians(0.5 * (rbins[1:] + rbins[:-1]))
    solid_angle_per_bin = np.sin(theta_centers_rad) * (dr * np.pi / 180.0) * da_rad
    solid_angle_per_bin = np.where(solid_angle_per_bin == 0.0, np.nan, solid_angle_per_bin)

    hist_gamma_bg = hist_gamma_bg / solid_angle_per_bin[np.newaxis, :]

    A, R = np.meshgrid(abins, rbins)
    return {
        "hist": hist_gamma_bg,
        "abins": abins,
        "rbins": rbins,
        "A": A,
        "R": R,
    }


def rotate_histogram_phi(hist: np.ndarray, abins: np.ndarray, pa_deg: float) -> np.ndarray:
    """Rotate a periodic phi histogram by +pa_deg using linear interpolation."""
    if abs(float(pa_deg)) < 1.0e-12:
        return np.array(hist, copy=True)

    phi_centers_deg = np.degrees(0.5 * (abins[:-1] + abins[1:]))
    hist = np.asarray(hist, dtype=float)
    out = np.empty_like(hist)

    x = phi_centers_deg
    x_ext = np.concatenate([x - 360.0, x, x + 360.0])
    for j in range(hist.shape[1]):
        y = hist[:, j]
        y_ext = np.concatenate([y, y, y])
        query = (x - float(pa_deg)) % 360.0
        out[:, j] = np.interp(query, x_ext, y_ext)
    return out


def save_event_array(events: Dict[str, np.ndarray], out_prefix: Path) -> Path:
    out_arr = np.vstack([events["theta_rad"], events["phi_rad"], events["energy"]])
    out_path = out_prefix.with_suffix(".npy")
    np.save(out_path, out_arr)
    return out_path


def save_histogram_npz(hist: Dict[str, np.ndarray], out_prefix: Path) -> Path:
    out_path = out_prefix.with_suffix(".npz")
    np.savez_compressed(
        out_path,
        hist=hist["hist"],
        abins=hist["abins"],
        rbins=hist["rbins"],
    )
    return out_path


def _robust_lognorm(plot_values: np.ndarray) -> Optional[LogNorm]:
    positive = plot_values[np.isfinite(plot_values) & (plot_values > 0.0)]
    if positive.size == 0:
        return None

    vmin = np.percentile(positive, 15.0)
    vmax = np.percentile(positive, 99.5)

    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = np.min(positive)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = np.max(positive)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0

    return LogNorm(vmin=vmin, vmax=vmax)


def plot_sky_map(hist: Dict[str, np.ndarray], out_png: Path, *, title: str,
                 grb_theta_deg: Optional[float] = None, grb_phi_deg: Optional[float] = None,
                 cmap: str = "hot", show: bool = False) -> Path:
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    ax.grid(False)

    plot_values = np.array(hist["hist"], copy=True)
    plot_values[~np.isfinite(plot_values)] = np.nan
    norm = _robust_lognorm(plot_values)
    if norm is not None:
        plot_values = np.where(plot_values > 0.0, plot_values, np.nan)

    pc = ax.pcolormesh(hist["A"], hist["R"], plot_values.T, cmap=cmap, shading="auto", norm=norm)

    if grb_theta_deg is not None and grb_phi_deg is not None:
        grb_phi_rad = np.deg2rad(float(grb_phi_deg) % 360.0)
        grb_theta_plot = float(grb_theta_deg)
        ax.plot(
            [grb_phi_rad],
            [grb_theta_plot],
            marker="*",
            markersize=14,
            markerfacecolor="cyan",
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            label="GRB direction",
            zorder=10,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10))

    ax.set_title(title)
    cbar = fig.colorbar(pc)
    if norm is not None:
        cbar.set_label(r"Counts $sr^{-1}s^{-1}m^{-2}$ (log scale)")
    else:
        cbar.set_label(r"Counts $sr^{-1}s^{-1}m^{-2}$")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame-json", type=str,
                     help="JSON file produced by gll_pt_trigger_frame.py --json")
    src.add_argument("--trigger-id", type=str,
                     help="GBM trigger ID. If used, this script imports gll_pt_trigger_frame.py and computes the payload directly.")

    p.add_argument("--frame-script", type=str, default="/mnt/data/gll_pt_trigger_frame.py",
                   help="Path to gll_pt_trigger_frame.py when using --trigger-id")
    p.add_argument("--base-dir", type=str, default=".",
                   help="Base directory containing bn<TRIGGER_ID>/ when using --trigger-id")
    p.add_argument("--ft2", type=str, default=None, help="Optional explicit gll_pt_*.fits path")
    p.add_argument("--tcat", type=str, default=None, help="Optional explicit glg_tcat*.fit path")
    p.add_argument("--time", type=float, default=None, help="Optional MET override when using --trigger-id")
    p.add_argument("--grb-ra", type=float, default=None, help="Optional GRB RA override in deg")
    p.add_argument("--grb-dec", type=float, default=None, help="Optional GRB Dec override in deg")

    p.add_argument("--sim-root", type=str, default="./SimData",
                   help="Root directory containing Pol_*_* folders")
    p.add_argument("--pol-prefix", type=str, default="Pol", help="Folder prefix, default: Pol")
    p.add_argument("--step-deg", type=float, default=15.0, help="Simulation angular step size in deg")
    p.add_argument("--input-name", type=str, default="DTRout_tmp.dat",
                   help="Simulation file name inside each Pol folder")
    p.add_argument("--pa-deg", type=float, default=0.0,
                   help="Position angle in degrees. Folder selection uses phi_deg - PA, and the final sky map is rotated by +PA.")

    p.add_argument("--outdir", type=str, default=".", help="Output directory")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag for output filenames. Default uses trigger ID or selected Pol folder")
    p.add_argument("--nr", type=int, default=90, help="Number of radial bin edges for the sky map")
    p.add_argument("--na", type=int, default=90, help="Number of azimuth bin edges for the sky map")
    p.add_argument("--show", action="store_true", help="Display the plot interactively")
    p.add_argument("--print-json", action="store_true", help="Print a summary JSON to stdout")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.frame_json is not None:
        payload = read_payload_from_json(Path(args.frame_json).expanduser().resolve())
        trigger_id = str(payload.get("trigger_id", "unknown"))
    else:
        payload = build_payload_from_trigger(
            trigger_id=args.trigger_id,
            base_dir=Path(args.base_dir).expanduser().resolve(),
            frame_script=Path(args.frame_script).expanduser().resolve(),
            ft2=None if args.ft2 is None else Path(args.ft2).expanduser().resolve(),
            tcat=None if args.tcat is None else Path(args.tcat).expanduser().resolve(),
            time_override=args.time,
            grb_ra_override=args.grb_ra,
            grb_dec_override=args.grb_dec,
        )
        trigger_id = str(args.trigger_id)

    theta_deg, phi_deg = extract_grb_local_coords(payload)
    phi_search_deg = (phi_deg - float(args.pa_deg)) % 360.0

    sim_root = Path(args.sim_root).expanduser().resolve()
    pol_folder, theta_bin, phi_bin = select_pol_folder(
        sim_root=sim_root,
        theta_deg=theta_deg,
        phi_deg=phi_search_deg,
        step_deg=args.step_deg,
        pol_prefix=args.pol_prefix,
    )

    rows = load_primary_events_from_folder(pol_folder, filename=args.input_name)
    events = build_event_arrays(rows)
    hist = make_sky_histogram(events["theta_deg"], events["phi_rad"], nr=args.nr, na=args.na)

    if abs(float(args.pa_deg)) > 1.0e-12:
        hist["hist"] = rotate_histogram_phi(hist["hist"], hist["abins"], float(args.pa_deg))

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    default_tag = args.tag or f"bn{trigger_id}_{pol_folder.name}_pa{args.pa_deg:g}"

    out_arr_path = save_event_array(events, outdir / f"out_arr_{default_tag}")
    hist_path = save_histogram_npz(hist, outdir / f"hist_{default_tag}")
    png_path = plot_sky_map(
        hist,
        out_png=(outdir / f"skymap_{default_tag}.png"),
        title=(
            f"Gamma-Ray Sky Map\n"
            f"Trigger {trigger_id} | GRB local (theta, phi)=({theta_deg:.3f}, {phi_deg:.3f}) deg\n"
            f"PA={args.pa_deg:.3f} deg | Folder search phi=phi-PA={phi_search_deg:.3f} deg\n"
            f"Selected folder: {pol_folder.name}"
        ),
        grb_theta_deg=theta_deg,
        grb_phi_deg=phi_deg,
        show=args.show,
    )

    summary = {
        "trigger_id": trigger_id,
        "grb_local_theta_deg": theta_deg,
        "grb_local_phi_deg": phi_deg,
        "position_angle_deg": float(args.pa_deg),
        "folder_search_phi_deg": phi_search_deg,
        "selected_pol_folder": str(pol_folder),
        "selected_theta_bin_deg": theta_bin,
        "selected_phi_bin_deg": phi_bin,
        "n_primary_events": int(len(rows)),
        "grb_marker_on_final_map": {"theta_deg": theta_deg, "phi_deg": phi_deg},
        "outputs": {
            "event_array_npy": str(out_arr_path),
            "rotated_histogram_npz": str(hist_path),
            "sky_map_png": str(png_path),
        },
    }

    print(f"GRB local coordinates: theta={theta_deg:.6f} deg, phi={phi_deg:.6f} deg")
    print(f"Position angle: {float(args.pa_deg):.6f} deg")
    print(f"Folder-search phi = phi - PA = {phi_search_deg:.6f} deg")
    print(f"Selected simulation folder: {pol_folder}")
    print(f"Saved event array: {out_arr_path}")
    print(f"Saved rotated histogram: {hist_path}")
    print(f"Saved sky map: {png_path}")

    summary_json_path = outdir / f"summary_{default_tag}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_json_path}")

    if args.print_json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
