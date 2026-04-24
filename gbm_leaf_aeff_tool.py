#!/usr/bin/env python3
"""
GBM leaf-response effective-area tool
=====================================

This module reads the GBM Response Generator leaf response files under
`data/GBMDRMdb002` and computes detector effective area curves for one or more
source sky positions.

Inputs
------
- spacecraft Z-axis pointing (RA, Dec)
- spacecraft X-axis pointing (RA, Dec)
- one or more source positions (RA, Dec)
- one or more photon energies (keV)

Method
------
1. Convert each source sky position into spacecraft-frame azimuth / zenith.
2. For each detector, find the nearest leaf response files in the database.
3. Read each leaf `.rsp` FITS file and extract its effective-area curve as the
   DRM summed over detector channels for each photon-energy bin.
4. Interpolate in sky angle using inverse-distance weights over the nearest
   leaf responses.
5. Interpolate in energy onto the user-requested energy grid.

Notes
-----
- This is a geometry-driven, leaf-library lookup tool. It does not reproduce
  every detail of the full GBM response generator pipeline.
- By default it uses the 3 nearest leaf responses per detector, which is often
  smoother than a single nearest-neighbor lookup while remaining fast.
- Requires: numpy, astropy

Example
-------
    from gbm_leaf_aeff_tool import GBMLeafAeffLibrary
    import numpy as np

    lib = GBMLeafAeffLibrary("/path/to/gbmrsp-2.0/data/GBMDRMdb002")

    ra = np.linspace(0.0, 350.0, 36)
    dec = np.zeros_like(ra)
    energies = np.geomspace(8.0, 1000.0, 64)

    result = lib.compute_all_detectors_from_radec(
        scz_ra_deg=120.0,
        scz_dec_deg=-10.0,
        scx_ra_deg=30.0,
        scx_dec_deg=5.0,
        src_ra_deg=ra,
        src_dec_deg=dec,
        energies_kev=energies,
        k_neighbors=3,
    )

    aeff_n0 = result["aeff_cm2"]["n0"]   # shape (n_sources, n_energies)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from astropy.io import fits
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This tool requires astropy. Install it with `pip install astropy`."
    ) from exc


DETECTORS: Dict[str, Dict[str, float | str]] = {
    "n0": {"az_deg": 45.89,  "zen_deg": 20.58, "type": "NaI"},
    "n1": {"az_deg": 45.11,  "zen_deg": 45.31, "type": "NaI"},
    "n2": {"az_deg": 58.44,  "zen_deg": 90.21, "type": "NaI"},
    "n3": {"az_deg": 314.87, "zen_deg": 45.24, "type": "NaI"},
    "n4": {"az_deg": 303.15, "zen_deg": 90.27, "type": "NaI"},
    "n5": {"az_deg": 3.35,   "zen_deg": 89.79, "type": "NaI"},
    "n6": {"az_deg": 224.93, "zen_deg": 20.43, "type": "NaI"},
    "n7": {"az_deg": 224.62, "zen_deg": 46.18, "type": "NaI"},
    "n8": {"az_deg": 236.61, "zen_deg": 89.97, "type": "NaI"},
    "n9": {"az_deg": 135.19, "zen_deg": 45.55, "type": "NaI"},
    "na": {"az_deg": 123.73, "zen_deg": 90.42, "type": "NaI"},
    "nb": {"az_deg": 183.74, "zen_deg": 90.32, "type": "NaI"},
    "b0": {"az_deg": 0.00,   "zen_deg": 90.00, "type": "BGO"},
    "b1": {"az_deg": 180.00, "zen_deg": 90.00, "type": "BGO"},
}

DETECTOR_DIRS: Dict[str, str] = {
    "n0": "NAI_00",
    "n1": "NAI_01",
    "n2": "NAI_02",
    "n3": "NAI_03",
    "n4": "NAI_04",
    "n5": "NAI_05",
    "n6": "NAI_06",
    "n7": "NAI_07",
    "n8": "NAI_08",
    "n9": "NAI_09",
    "na": "NAI_10",
    "nb": "NAI_11",
    "b0": "BGO_00",
    "b1": "BGO_01",
}

LEAF_RE = re.compile(r"glg_leaf_([nb][0-9ab])_z(\d{6})_az(\d{6})_v\d+\.rsp$", re.IGNORECASE)


@dataclass(frozen=True)
class LeafPoint:
    detector: str
    az_deg: float
    zen_deg: float
    path: Path
    unit_vec: np.ndarray


@dataclass
class EffectiveAreaCurve:
    energy_lo_kev: np.ndarray
    energy_hi_kev: np.ndarray
    energy_mid_kev: np.ndarray
    effective_area_cm2: np.ndarray


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def _normalize(v: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    if np.any(n == 0.0):
        raise ValueError("Zero-length vector encountered during normalization")
    return v / n


def radec_to_unit(ra_deg: np.ndarray | float, dec_deg: np.ndarray | float) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def unit_to_radec(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vec = _normalize(np.asarray(vec, dtype=float), axis=-1)
    ra = np.rad2deg(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(vec[..., 2], -1.0, 1.0)))
    return ra, dec


def azzen_to_unit(az_deg: np.ndarray | float, zen_deg: np.ndarray | float) -> np.ndarray:
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    zen = np.deg2rad(np.asarray(zen_deg, dtype=float))
    sinz = np.sin(zen)
    x = sinz * np.cos(az)
    y = sinz * np.sin(az)
    z = np.cos(zen)
    return np.stack([x, y, z], axis=-1)


def unit_to_azzen(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vec = _normalize(np.asarray(vec, dtype=float), axis=-1)
    az = np.rad2deg(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    zen = np.rad2deg(np.arccos(np.clip(vec[..., 2], -1.0, 1.0)))
    return az, zen


def orthonormal_spacecraft_axes(
    scz_ra_deg: float,
    scz_dec_deg: float,
    scx_ra_deg: float,
    scx_dec_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a right-handed spacecraft basis from sky pointings of the spacecraft
    Z and X axes.

    The supplied X direction is projected onto the plane perpendicular to Z to
    remove any small non-orthogonality.
    """
    zhat = _normalize(radec_to_unit(scz_ra_deg, scz_dec_deg), axis=-1)
    x_guess = _normalize(radec_to_unit(scx_ra_deg, scx_dec_deg), axis=-1)

    # Project X into the plane perpendicular to Z.
    xhat = x_guess - np.dot(x_guess, zhat) * zhat
    xhat = _normalize(xhat, axis=-1)

    # Right-handed: X x Y = Z  => Y = Z x X
    yhat = np.cross(zhat, xhat)
    yhat = _normalize(yhat, axis=-1)

    # Re-orthogonalize X for numerical cleanliness.
    xhat = np.cross(yhat, zhat)
    xhat = _normalize(xhat, axis=-1)

    return xhat, yhat, zhat


def source_radec_to_spacecraft_azzen(
    src_ra_deg: np.ndarray | float,
    src_dec_deg: np.ndarray | float,
    scz_ra_deg: float,
    scz_dec_deg: float,
    scx_ra_deg: float,
    scx_dec_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert source sky positions to spacecraft-frame azimuth/zenith.

    Returns
    -------
    src_az_deg, src_zen_deg, src_sc_unit
    """
    xhat, yhat, zhat = orthonormal_spacecraft_axes(scz_ra_deg, scz_dec_deg, scx_ra_deg, scx_dec_deg)
    src = radec_to_unit(src_ra_deg, src_dec_deg)

    sx = np.tensordot(src, xhat, axes=([-1], [0]))
    sy = np.tensordot(src, yhat, axes=([-1], [0]))
    sz = np.tensordot(src, zhat, axes=([-1], [0]))
    sc_vec = np.stack([sx, sy, sz], axis=-1)
    sc_vec = _normalize(sc_vec, axis=-1)
    az_deg, zen_deg = unit_to_azzen(sc_vec)
    return az_deg, zen_deg, sc_vec


def incident_angle_deg(detector: str, src_az_deg: np.ndarray, src_zen_deg: np.ndarray) -> np.ndarray:
    meta = DETECTORS[detector.lower()]
    det_vec = azzen_to_unit(float(meta["az_deg"]), float(meta["zen_deg"]))
    src_vec = azzen_to_unit(src_az_deg, src_zen_deg)
    mu = np.sum(src_vec * det_vec, axis=-1)
    return np.rad2deg(np.arccos(np.clip(mu, -1.0, 1.0)))


# -----------------------------------------------------------------------------
# FITS / DRM reading helpers
# -----------------------------------------------------------------------------


def _decompress_drm_rows(drm_data, n_channels: int) -> np.ndarray:
    """
    Decompress the OGIP sparse MATRIX representation.
    """
    n_ebins = len(drm_data)
    out = np.zeros((n_ebins, n_channels), dtype=float)

    for irow in range(n_ebins):
        n_grp = int(drm_data["N_GRP"][irow])
        if n_grp <= 0:
            continue

        f_chan = np.asarray(drm_data["F_CHAN"][irow]).astype(int).ravel()
        n_chan = np.asarray(drm_data["N_CHAN"][irow]).astype(int).ravel()
        matrix = np.asarray(drm_data["MATRIX"][irow], dtype=float).ravel()

        if f_chan.size == 1 and n_grp > 1:
            f_chan = np.repeat(f_chan, n_grp)
        if n_chan.size == 1 and n_grp > 1:
            n_chan = np.repeat(n_chan, n_grp)

        # Database files are commonly 1-based in F_CHAN.
        if f_chan.size > 0 and np.min(f_chan) >= 1:
            f_chan = f_chan - 1

        cursor = 0
        for start, width in zip(f_chan[:n_grp], n_chan[:n_grp]):
            width = int(width)
            start = int(start)
            stop = start + width
            vals = matrix[cursor:cursor + width]
            cursor += width
            if start < 0 or stop > n_channels:
                raise ValueError(
                    f"Invalid MATRIX channel group in row {irow}: start={start}, width={width}, n_channels={n_channels}"
                )
            out[irow, start:stop] = vals

    return out


@lru_cache(maxsize=512)
def read_leaf_effective_area(filepath: str) -> EffectiveAreaCurve:
    path = str(filepath)
    with fits.open(path, memmap=False) as hdul:
        ebounds = hdul["EBOUNDS"].data
        n_channels = len(ebounds)
        drm_hdu = hdul["SPECRESP MATRIX"]
        drm_data = drm_hdu.data

        elo = np.asarray(drm_data["ENERG_LO"], dtype=float)
        ehi = np.asarray(drm_data["ENERG_HI"], dtype=float)
        emid = np.sqrt(elo * ehi)

        drm = _decompress_drm_rows(drm_data, n_channels)
        aeff = drm.sum(axis=1)

        return EffectiveAreaCurve(
            energy_lo_kev=elo,
            energy_hi_kev=ehi,
            energy_mid_kev=emid,
            effective_area_cm2=aeff,
        )


# -----------------------------------------------------------------------------
# Library / nearest-neighbor interpolation on the leaf grid
# -----------------------------------------------------------------------------


def _parse_leaf_filename(path: Path) -> Tuple[str, float, float]:
    m = LEAF_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized leaf filename: {path.name}")
    det = m.group(1).lower()
    zen_deg = int(m.group(2)) / 1000.0
    az_deg = int(m.group(3)) / 1000.0
    if az_deg >= 360.0:
        az_deg = 0.0
    return det, az_deg, zen_deg


class GBMLeafAeffLibrary:
    def __init__(self, db_root: str | Path):
        self.db_root = Path(db_root)
        if not self.db_root.exists():
            raise FileNotFoundError(f"Database root not found: {self.db_root}")
        self.leaf_points: Dict[str, List[LeafPoint]] = self._scan_leaf_database()
        self._leaf_unit_vectors: Dict[str, np.ndarray] = {
            det: np.stack([p.unit_vec for p in pts], axis=0)
            for det, pts in self.leaf_points.items()
        }

    def _scan_leaf_database(self) -> Dict[str, List[LeafPoint]]:
        out: Dict[str, List[LeafPoint]] = {det: [] for det in DETECTORS}
        for det, subdir in DETECTOR_DIRS.items():
            det_dir = self.db_root / subdir
            if not det_dir.exists():
                continue
            for path in sorted(det_dir.glob("glg_leaf_*.rsp")):
                fdet, az_deg, zen_deg = _parse_leaf_filename(path)
                if fdet != det:
                    continue
                out[det].append(
                    LeafPoint(
                        detector=det,
                        az_deg=az_deg,
                        zen_deg=zen_deg,
                        path=path,
                        unit_vec=np.asarray(azzen_to_unit(az_deg, zen_deg), dtype=float),
                    )
                )
            if not out[det]:
                raise RuntimeError(f"No leaf files found for detector {det} in {det_dir}")
        return out

    def _nearest_leaf_indices(
        self,
        detector: str,
        src_sc_unit: np.ndarray,
        k_neighbors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        detector = detector.lower()
        pts = self._leaf_unit_vectors[detector]  # (n_leaf, 3)
        src = np.atleast_2d(np.asarray(src_sc_unit, dtype=float))  # (n_src, 3)

        mu = np.clip(src @ pts.T, -1.0, 1.0)
        ang = np.rad2deg(np.arccos(mu))  # (n_src, n_leaf)

        k = max(1, min(int(k_neighbors), pts.shape[0]))
        idx = np.argpartition(ang, kth=k - 1, axis=1)[:, :k]
        dsel = np.take_along_axis(ang, idx, axis=1)
        order = np.argsort(dsel, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dsel = np.take_along_axis(dsel, order, axis=1)
        return idx, dsel

    @staticmethod
    def _weights_from_distances(dist_deg: np.ndarray, power: float = 2.0) -> np.ndarray:
        d = np.asarray(dist_deg, dtype=float)
        w = np.zeros_like(d)
        exact = d <= 1.0e-8
        row_has_exact = np.any(exact, axis=1)
        if np.any(row_has_exact):
            w[row_has_exact] = exact[row_has_exact].astype(float)
        if np.any(~row_has_exact):
            good = ~row_has_exact
            w[good] = 1.0 / np.maximum(d[good], 1.0e-8) ** power
        wsum = np.sum(w, axis=1, keepdims=True)
        return w / np.maximum(wsum, 1.0e-30)

    def _interpolated_aeff_from_spacecraft_dir(
        self,
        detector: str,
        src_sc_unit: np.ndarray,
        energies_kev: np.ndarray,
        k_neighbors: int = 3,
        distance_power: float = 2.0,
    ) -> np.ndarray:
        detector = detector.lower()
        src_sc_unit = np.atleast_2d(np.asarray(src_sc_unit, dtype=float))
        energies_kev = np.asarray(energies_kev, dtype=float)

        if np.any(energies_kev <= 0):
            raise ValueError("All requested energies must be > 0 keV")

        idx, dist_deg = self._nearest_leaf_indices(detector, src_sc_unit, k_neighbors=k_neighbors)
        weights = self._weights_from_distances(dist_deg, power=distance_power)
        pts = self.leaf_points[detector]

        out = np.zeros((src_sc_unit.shape[0], energies_kev.size), dtype=float)
        cache: Dict[Path, np.ndarray] = {}

        for j in range(idx.shape[1]):
            col_idx = idx[:, j]
            unique_leaf = np.unique(col_idx)
            interp_curves: Dict[int, np.ndarray] = {}
            for leaf_idx in unique_leaf:
                leaf = pts[int(leaf_idx)]
                if leaf.path not in cache:
                    curve = read_leaf_effective_area(str(leaf.path))
                    cache[leaf.path] = np.exp(
                        np.interp(
                            np.log(energies_kev),
                            np.log(curve.energy_mid_kev),
                            np.log(np.maximum(curve.effective_area_cm2, 1.0e-30)),
                            left=np.log(max(curve.effective_area_cm2[0], 1.0e-30)),
                            right=np.log(max(curve.effective_area_cm2[-1], 1.0e-30)),
                        )
                    )
                interp_curves[int(leaf_idx)] = cache[leaf.path]

            for i in range(src_sc_unit.shape[0]):
                out[i] += weights[i, j] * interp_curves[int(col_idx[i])]

        return out

    def compute_detector_from_radec(
        self,
        detector: str,
        scz_ra_deg: float,
        scz_dec_deg: float,
        scx_ra_deg: float,
        scx_dec_deg: float,
        src_ra_deg: Sequence[float] | np.ndarray | float,
        src_dec_deg: Sequence[float] | np.ndarray | float,
        energies_kev: Sequence[float] | np.ndarray,
        k_neighbors: int = 3,
        distance_power: float = 2.0,
    ) -> Dict[str, np.ndarray | float | str]:
        src_az_deg, src_zen_deg, src_sc_unit = source_radec_to_spacecraft_azzen(
            src_ra_deg,
            src_dec_deg,
            scz_ra_deg,
            scz_dec_deg,
            scx_ra_deg,
            scx_dec_deg,
        )
        energies_kev = np.asarray(energies_kev, dtype=float)
        aeff = self._interpolated_aeff_from_spacecraft_dir(
            detector=detector,
            src_sc_unit=src_sc_unit,
            energies_kev=energies_kev,
            k_neighbors=k_neighbors,
            distance_power=distance_power,
        )
        theta = incident_angle_deg(detector, src_az_deg, src_zen_deg)
        return {
            "detector": detector.lower(),
            "detector_type": str(DETECTORS[detector.lower()]["type"]),
            "source_ra_deg": np.asarray(src_ra_deg, dtype=float),
            "source_dec_deg": np.asarray(src_dec_deg, dtype=float),
            "source_sc_az_deg": np.asarray(src_az_deg, dtype=float),
            "source_sc_zen_deg": np.asarray(src_zen_deg, dtype=float),
            "incident_angle_deg": np.asarray(theta, dtype=float),
            "energies_kev": energies_kev,
            "effective_area_cm2": aeff,
        }

    def compute_all_detectors_from_radec(
        self,
        scz_ra_deg: float,
        scz_dec_deg: float,
        scx_ra_deg: float,
        scx_dec_deg: float,
        src_ra_deg: Sequence[float] | np.ndarray | float,
        src_dec_deg: Sequence[float] | np.ndarray | float,
        energies_kev: Sequence[float] | np.ndarray,
        detectors: Optional[Sequence[str]] = None,
        k_neighbors: int = 3,
        distance_power: float = 2.0,
    ) -> Dict[str, object]:
        src_az_deg, src_zen_deg, src_sc_unit = source_radec_to_spacecraft_azzen(
            src_ra_deg,
            src_dec_deg,
            scz_ra_deg,
            scz_dec_deg,
            scx_ra_deg,
            scx_dec_deg,
        )

        energies_kev = np.asarray(energies_kev, dtype=float)
        detectors = [d.lower() for d in (detectors if detectors is not None else DETECTORS.keys())]

        aeff_by_detector: Dict[str, np.ndarray] = {}
        theta_by_detector: Dict[str, np.ndarray] = {}
        for det in detectors:
            aeff_by_detector[det] = self._interpolated_aeff_from_spacecraft_dir(
                detector=det,
                src_sc_unit=src_sc_unit,
                energies_kev=energies_kev,
                k_neighbors=k_neighbors,
                distance_power=distance_power,
            )
            theta_by_detector[det] = np.asarray(incident_angle_deg(det, src_az_deg, src_zen_deg), dtype=float)

        return {
            "detectors": detectors,
            "source_ra_deg": np.asarray(src_ra_deg, dtype=float),
            "source_dec_deg": np.asarray(src_dec_deg, dtype=float),
            "source_sc_az_deg": np.asarray(src_az_deg, dtype=float),
            "source_sc_zen_deg": np.asarray(src_zen_deg, dtype=float),
            "energies_kev": energies_kev,
            "incident_angle_deg": theta_by_detector,
            "aeff_cm2": aeff_by_detector,
        }


# -----------------------------------------------------------------------------
# Output helpers and CLI
# -----------------------------------------------------------------------------


def _parse_array_or_range(text: str) -> np.ndarray:
    text = text.strip()
    if text.startswith("logspace:"):
        _, a, b, n = text.split(":")
        return np.geomspace(float(a), float(b), int(n))
    if text.startswith("linspace:"):
        _, a, b, n = text.split(":")
        return np.linspace(float(a), float(b), int(n))
    if text.startswith("file:"):
        arr = np.loadtxt(text.split(":", 1)[1], ndmin=1)
        return np.asarray(arr, dtype=float)
    return np.array([float(x) for x in text.split(",") if x.strip()], dtype=float)


def _read_radec_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    ras: List[float] = []
    decs: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        if "ra_deg" in cols and "dec_deg" in cols:
            ra_key, dec_key = cols["ra_deg"], cols["dec_deg"]
        elif "ra" in cols and "dec" in cols:
            ra_key, dec_key = cols["ra"], cols["dec"]
        else:
            raise ValueError("CSV must contain ra_deg/dec_deg or ra/dec columns")
        for row in reader:
            ras.append(float(row[ra_key]))
            decs.append(float(row[dec_key]))
    return np.asarray(ras, dtype=float), np.asarray(decs, dtype=float)


def _write_long_csv(result: Mapping[str, object], out_csv: str | Path) -> None:
    detectors = list(result["detectors"])
    src_ra = np.asarray(result["source_ra_deg"], dtype=float)
    src_dec = np.asarray(result["source_dec_deg"], dtype=float)
    src_az = np.asarray(result["source_sc_az_deg"], dtype=float)
    src_zen = np.asarray(result["source_sc_zen_deg"], dtype=float)
    energies = np.asarray(result["energies_kev"], dtype=float)
    theta_by_det = result["incident_angle_deg"]
    aeff_by_det = result["aeff_cm2"]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "detector", "source_index", "source_ra_deg", "source_dec_deg",
            "source_sc_az_deg", "source_sc_zen_deg", "incident_angle_deg",
            "energy_kev", "effective_area_cm2"
        ])
        for det in detectors:
            theta = np.asarray(theta_by_det[det], dtype=float)
            aeff = np.asarray(aeff_by_det[det], dtype=float)
            for i in range(src_ra.size):
                for j in range(energies.size):
                    w.writerow([
                        det, i, float(src_ra[i]), float(src_dec[i]),
                        float(src_az[i]), float(src_zen[i]), float(theta[i]),
                        float(energies[j]), float(aeff[i, j]),
                    ])


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute GBM detector effective areas from GBMDRMdb002 leaf response files."
    )
    p.add_argument("--db-root", required=True, help="Path to data/GBMDRMdb002")
    p.add_argument("--scz-ra", type=float, required=True, help="Spacecraft Z-axis RA (deg)")
    p.add_argument("--scz-dec", type=float, required=True, help="Spacecraft Z-axis Dec (deg)")
    p.add_argument("--scx-ra", type=float, required=True, help="Spacecraft X-axis RA (deg)")
    p.add_argument("--scx-dec", type=float, required=True, help="Spacecraft X-axis Dec (deg)")
    p.add_argument(
        "--src-csv",
        required=True,
        help="CSV with source positions; must contain ra_deg,dec_deg columns",
    )
    p.add_argument(
        "--energies",
        required=True,
        help="Energy grid, e.g. 'logspace:8:1000:80' or '8,10,20,50,100' or 'file:energies.txt'",
    )
    p.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated detector list, default all (n0,n1,...,nb,b0,b1)",
    )
    p.add_argument("--k-neighbors", type=int, default=3, help="Number of nearest leaf files to blend")
    p.add_argument("--distance-power", type=float, default=2.0, help="Inverse-distance weighting power")
    p.add_argument("--out-csv", default=None, help="Optional output CSV in long-table format")
    return p


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    src_ra, src_dec = _read_radec_csv(args.src_csv)
    energies = _parse_array_or_range(args.energies)
    detectors = None if args.detectors is None else [x.strip().lower() for x in args.detectors.split(",") if x.strip()]

    lib = GBMLeafAeffLibrary(args.db_root)
    result = lib.compute_all_detectors_from_radec(
        scz_ra_deg=args.scz_ra,
        scz_dec_deg=args.scz_dec,
        scx_ra_deg=args.scx_ra,
        scx_dec_deg=args.scx_dec,
        src_ra_deg=src_ra,
        src_dec_deg=src_dec,
        energies_kev=energies,
        detectors=detectors,
        k_neighbors=args.k_neighbors,
        distance_power=args.distance_power,
    )

    print(f"Computed {len(result['detectors'])} detector(s) for {len(src_ra)} source position(s) and {len(energies)} energy bin(s).")
    for det in result["detectors"]:
        theta = np.asarray(result["incident_angle_deg"][det], dtype=float)
        aeff = np.asarray(result["aeff_cm2"][det], dtype=float)
        print(
            f"  {det}: incident-angle range = [{theta.min():.3f}, {theta.max():.3f}] deg, "
            f"peak Aeff = {aeff.max():.6g} cm^2"
        )

    if args.out_csv:
        _write_long_csv(result, args.out_csv)
        print(f"Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
