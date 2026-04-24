#!/usr/bin/env python3
"""
Read a Fermi LAT FT2 / pointing-livetime history file (gll_pt_*.fits) for a
GBM trigger folder bn<TRIGGER_ID>, recover the spacecraft axes and local zenith,
and express the GRB position in a local coordinate system with

- Z axis = Earth-up / local zenith (RA_ZENITH, DEC_ZENITH)
- X axis = local east at the zenith direction
- Y axis = Z x X (approximately local north)

The script can also read the GBM trigger catalog file (glg_tcat*.fit*) from the
same trigger folder to retrieve the GRB sky position and trigger time.

Typical trigger folder layout:
    ./bn180720598/
        gll_pt_*.fits
        glg_tcat_all_bn180720598_vXX.fit
        ...

Outputs include:
- spacecraft X/Y/Z in RA/Dec at the selected FT2 row
- zenith/nadir in RA/Dec
- GRB RA/Dec from TCAT (or manual input)
- GRB direction in the local zenith-east frame:
    * Cartesian components (x_east, y_north, z_zenith)
    * theta_deg : angle away from +Z (zenith angle)
    * phi_deg   : azimuth in the X-Y plane measured from +X toward +Y
    * elevation_deg = 90 - theta_deg

Notes
-----
- If RA_ZENITH/DEC_ZENITH are present in the FT2 file, they are used directly.
  Otherwise zenith is derived from the spacecraft position vector.
- The GRB position is searched first in the TCAT primary header using a set of
  common keyword candidates (OBJ_RA, RA_OBJ, RA, TRIGRA, etc.). If that fails,
  the script can optionally accept --grb-ra/--grb-dec.
- The selected FT2 row is the one that covers TRIGTIME from the TCAT when
  available, otherwise the nearest row to --time.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
NORTH_POLE = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass
class AttitudeRecord:
    time_start: float
    time_stop: float
    livetime: Optional[float]
    ra_scx_deg: float
    dec_scx_deg: float
    ra_scy_deg: Optional[float]
    dec_scy_deg: Optional[float]
    ra_scz_deg: float
    dec_scz_deg: float
    ra_zenith_deg: Optional[float]
    dec_zenith_deg: Optional[float]
    ra_nadir_deg: Optional[float]
    dec_nadir_deg: Optional[float]


@dataclass
class TriggerInfo:
    trigger_id: str
    grb_name: Optional[str]
    trigtime: Optional[float]
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    error_radius_deg: Optional[float]
    source_file: Optional[str]
    header_keywords_used: Dict[str, Optional[str]]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("Zero-length vector encountered.")
    return v / n


def _radec_to_vec(ra_deg: np.ndarray | float, dec_deg: np.ndarray | float) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=float) * DEG2RAD
    dec = np.asarray(dec_deg, dtype=float) * DEG2RAD
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


def _vec_to_radec(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vec = _unit(np.asarray(vec, dtype=float))
    x = vec[..., 0]
    y = vec[..., 1]
    z = np.clip(vec[..., 2], -1.0, 1.0)
    ra = (np.arctan2(y, x) * RAD2DEG) % 360.0
    dec = np.arcsin(z) * RAD2DEG
    return ra, dec


def _quat_to_rotmat_xyzw(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(t) for t in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion encountered.")
    x /= n
    y /= n
    z /= n
    w /= n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axes_from_quaternion(q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = _quat_to_rotmat_xyzw(q)
    scx = _unit(r @ np.array([1.0, 0.0, 0.0]))
    scy = _unit(r @ np.array([0.0, 1.0, 0.0]))
    scz = _unit(r @ np.array([0.0, 0.0, 1.0]))
    return scx, scy, scz


def _find_sc_data_hdu(hdul: fits.HDUList):
    for name in ("SC_DATA", "SPACECRAFT", "SCDATA"):
        if name in hdul:
            return hdul[name]
    for hdu in hdul:
        if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            continue
        names = set(hdu.columns.names or [])
        if {"START", "STOP"}.issubset(names) or {"RA_SCZ", "DEC_SCZ"}.issubset(names):
            return hdu
    raise KeyError("Could not find SC_DATA-like extension in FITS file.")


def _colmap(names: Iterable[str]) -> Dict[str, str]:
    return {str(n).upper(): str(n) for n in names}


def _get_col(data, cmap: Dict[str, str], *candidates: str):
    for c in candidates:
        if c.upper() in cmap:
            return data[cmap[c.upper()]]
    return None


def _extract_position_xyz(data, cmap: Dict[str, str]) -> Optional[np.ndarray]:
    vec = _get_col(data, cmap, "SC_POSITION", "SPACECRAFT_POSITION", "POSITION")
    if vec is not None:
        arr = np.asarray(vec, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    x = _get_col(data, cmap, "POS_X", "SC_POS_X", "X")
    y = _get_col(data, cmap, "POS_Y", "SC_POS_Y", "Y")
    z = _get_col(data, cmap, "POS_Z", "SC_POS_Z", "Z")
    if x is not None and y is not None and z is not None:
        return np.stack([x, y, z], axis=-1).astype(float)
    return None


def _extract_quaternions(data, cmap: Dict[str, str]) -> Optional[np.ndarray]:
    q = _get_col(data, cmap, "QUATERNION", "QSJ")
    if q is not None:
        arr = np.asarray(q, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 4:
            return arr
    q1 = _get_col(data, cmap, "QSJ_1", "Q1")
    q2 = _get_col(data, cmap, "QSJ_2", "Q2")
    q3 = _get_col(data, cmap, "QSJ_3", "Q3")
    q4 = _get_col(data, cmap, "QSJ_4", "Q4")
    if q1 is not None and q2 is not None and q3 is not None and q4 is not None:
        return np.stack([q1, q2, q3, q4], axis=-1).astype(float)
    return None


def read_gll_pt_axes(path: str | Path) -> List[AttitudeRecord]:
    with fits.open(path, memmap=True) as hdul:
        hdu = _find_sc_data_hdu(hdul)
        data = hdu.data
        if data is None:
            raise ValueError("Selected SC_DATA extension has no table data.")
        cmap = _colmap(hdu.columns.names or [])

        start_col = _get_col(data, cmap, "START")
        stop_col = _get_col(data, cmap, "STOP")
        if start_col is None or stop_col is None:
            raise KeyError("START/STOP columns are required.")
        start = np.asarray(start_col, dtype=float)
        stop = np.asarray(stop_col, dtype=float)

        livetime_col = _get_col(data, cmap, "LIVETIME")
        livetime = None if livetime_col is None else np.asarray(livetime_col, dtype=float)

        ra_scx = _get_col(data, cmap, "RA_SCX")
        dec_scx = _get_col(data, cmap, "DEC_SCX")
        ra_scz = _get_col(data, cmap, "RA_SCZ")
        dec_scz = _get_col(data, cmap, "DEC_SCZ")
        ra_zen = _get_col(data, cmap, "RA_ZENITH")
        dec_zen = _get_col(data, cmap, "DEC_ZENITH")

        if ra_scx is not None:
            ra_scx = np.asarray(ra_scx, dtype=float)
            dec_scx = np.asarray(dec_scx, dtype=float)
        if ra_scz is not None:
            ra_scz = np.asarray(ra_scz, dtype=float)
            dec_scz = np.asarray(dec_scz, dtype=float)
        if ra_zen is not None:
            ra_zen = np.asarray(ra_zen, dtype=float)
            dec_zen = np.asarray(dec_zen, dtype=float)

        pos_xyz = _extract_position_xyz(data, cmap)
        quats = _extract_quaternions(data, cmap)

        out: List[AttitudeRecord] = []
        for i in range(len(start)):
            if ra_scx is not None and ra_scz is not None:
                scx_vec = _radec_to_vec(float(ra_scx[i]), float(dec_scx[i]))
                scz_vec = _radec_to_vec(float(ra_scz[i]), float(dec_scz[i]))
                scy_vec = _unit(np.cross(scz_vec, scx_vec))
                ra_scy_i, dec_scy_i = _vec_to_radec(scy_vec)
                ra_scx_i, dec_scx_i = float(ra_scx[i]), float(dec_scx[i])
                ra_scz_i, dec_scz_i = float(ra_scz[i]), float(dec_scz[i])
                ra_scy_i, dec_scy_i = float(ra_scy_i), float(dec_scy_i)
            elif quats is not None:
                scx_vec, scy_vec, scz_vec = _axes_from_quaternion(quats[i])
                ra_scx_i, dec_scx_i = [float(x) for x in _vec_to_radec(scx_vec)]
                ra_scy_i, dec_scy_i = [float(x) for x in _vec_to_radec(scy_vec)]
                ra_scz_i, dec_scz_i = [float(x) for x in _vec_to_radec(scz_vec)]
            else:
                raise KeyError(
                    "Need either RA_SCX/DEC_SCX + RA_SCZ/DEC_SCZ columns or quaternion columns."
                )

            if ra_zen is not None:
                ra_zen_i, dec_zen_i = float(ra_zen[i]), float(dec_zen[i])
                zen_vec = _radec_to_vec(ra_zen_i, dec_zen_i)
                ra_nadir_i, dec_nadir_i = [float(x) for x in _vec_to_radec(-zen_vec)]
            elif pos_xyz is not None:
                zen_vec = _unit(np.asarray(pos_xyz[i], dtype=float))
                ra_zen_i, dec_zen_i = [float(x) for x in _vec_to_radec(zen_vec)]
                ra_nadir_i, dec_nadir_i = [float(x) for x in _vec_to_radec(-zen_vec)]
            else:
                ra_zen_i = dec_zen_i = ra_nadir_i = dec_nadir_i = None

            out.append(
                AttitudeRecord(
                    time_start=float(start[i]),
                    time_stop=float(stop[i]),
                    livetime=None if livetime is None else float(livetime[i]),
                    ra_scx_deg=ra_scx_i,
                    dec_scx_deg=dec_scx_i,
                    ra_scy_deg=ra_scy_i,
                    dec_scy_deg=dec_scy_i,
                    ra_scz_deg=ra_scz_i,
                    dec_scz_deg=dec_scz_i,
                    ra_zenith_deg=ra_zen_i,
                    dec_zenith_deg=dec_zen_i,
                    ra_nadir_deg=ra_nadir_i,
                    dec_nadir_deg=dec_nadir_i,
                )
            )
    return out


def select_record_nearest_time(records: Sequence[AttitudeRecord], met: float) -> AttitudeRecord:
    if not records:
        raise ValueError("No records available.")
    starts = np.array([r.time_start for r in records], dtype=float)
    stops = np.array([r.time_stop for r in records], dtype=float)
    idx = np.where((starts <= met) & (met < stops))[0]
    if len(idx):
        return records[int(idx[0])]
    mids = 0.5 * (starts + stops)
    return records[int(np.argmin(np.abs(mids - met)))]


def _open_fits_maybe_gz(path: Path):
    suffixes = ''.join(path.suffixes[-2:]).lower()
    if suffixes.endswith('.fit.gz') or suffixes.endswith('.fits.gz'):
        with gzip.open(path, 'rb') as f:
            data = f.read()
        return fits.open(fits.util.BytesIO(data), memmap=False)
    return fits.open(path, memmap=True)


def _find_file(folder: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_first_header(path: Path) -> fits.Header:
    with _open_fits_maybe_gz(path) as hdul:
        return hdul[0].header.copy()


def _header_get_first(header: fits.Header, candidates: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    for key in candidates:
        if key in header:
            val = header[key]
            try:
                return float(val), key
            except Exception:
                pass
    return None, None


def _header_get_first_str(header: fits.Header, candidates: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    for key in candidates:
        if key in header:
            val = header[key]
            if val is None:
                continue
            return str(val), key
    return None, None


def read_trigger_info_from_tcat(trigger_id: str, tcat_path: Path) -> TriggerInfo:
    header = _read_first_header(tcat_path)

    name, name_key = _header_get_first_str(header, ["OBJECT", "OBJ_NAME", "TRIGNAME", "NAME"])
    trigtime, trigtime_key = _header_get_first(header, ["TRIGTIME", "TRIGTM", "TRIG_MET", "TIME", "TSTART"])
    ra, ra_key = _header_get_first(header, ["OBJ_RA", "RA_OBJ", "RA", "TRIGRA", "LOC_RA", "RA_OBJT"])
    dec, dec_key = _header_get_first(header, ["OBJ_DEC", "DEC_OBJ", "DEC", "TRIGDEC", "LOC_DEC", "DEC_OBJT"])
    err, err_key = _header_get_first(header, ["ERR_RAD", "ERROR", "LOC_ERR", "RA_ERR", "POS_ERR"])

    return TriggerInfo(
        trigger_id=str(trigger_id),
        grb_name=name,
        trigtime=trigtime,
        ra_deg=ra,
        dec_deg=dec,
        error_radius_deg=err,
        source_file=str(tcat_path),
        header_keywords_used={
            "name": name_key,
            "trigtime": trigtime_key,
            "ra": ra_key,
            "dec": dec_key,
            "error": err_key,
        },
    )


def find_trigger_folder(base_dir: Path, trigger_id: str) -> Path:
    tid = str(trigger_id)
    candidates = [base_dir / f"bn{tid}", base_dir / tid]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"Could not find trigger folder bn{tid} or {tid} under {base_dir}")


def find_ft2_and_tcat(trigger_folder: Path, trigger_id: str) -> Tuple[Path, Optional[Path]]:
    ft2 = _find_file(trigger_folder, ["gll_pt*.fits", "gll_pt*.fit", "*pt*.fits", "*pt*.fit"])
    if ft2 is None:
        raise FileNotFoundError(f"Could not find gll_pt_*.fits under {trigger_folder}")
    tcat = _find_file(
        trigger_folder,
        [
            f"glg_tcat_all_bn{trigger_id}_v*.fit*",
            f"glg_tcat_all_{trigger_id}_v*.fit*",
            "glg_tcat*.fit*",
        ],
    )
    return ft2, tcat


def east_north_zenith_basis(ra_zenith_deg: float, dec_zenith_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_hat = _unit(_radec_to_vec(float(ra_zenith_deg), float(dec_zenith_deg)))

    east = np.array([-math.sin(float(ra_zenith_deg) * DEG2RAD), math.cos(float(ra_zenith_deg) * DEG2RAD), 0.0])
    east_norm = np.linalg.norm(east)
    if east_norm < 1e-12:
        east = _unit(np.cross(NORTH_POLE, z_hat)) if abs(z_hat[2]) < 0.999999 else np.array([1.0, 0.0, 0.0])
    else:
        east = east / east_norm

    north = _unit(np.cross(z_hat, east))
    east = _unit(np.cross(north, z_hat))
    return east, north, z_hat


def transform_radec_to_local(ra_deg: float, dec_deg: float, ra_zenith_deg: float, dec_zenith_deg: float) -> Dict[str, float]:
    src = _unit(_radec_to_vec(float(ra_deg), float(dec_deg)))
    x_hat, y_hat, z_hat = east_north_zenith_basis(ra_zenith_deg, dec_zenith_deg)

    x = float(np.dot(src, x_hat))
    y = float(np.dot(src, y_hat))
    z = float(np.dot(src, z_hat))
    z_clip = max(-1.0, min(1.0, z))
    theta_deg = math.degrees(math.acos(z_clip))
    phi_deg = math.degrees(math.atan2(y, x)) % 360.0
    elevation_deg = 90.0 - theta_deg
    return {
        "x_east": x,
        "y_north": y,
        "z_zenith": z,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "elevation_deg": elevation_deg,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trigger_id", help="GBM trigger ID, e.g. 180720598")
    p.add_argument("--base-dir", default=".", help="Directory containing the bn<TRIGGER_ID>/ subfolder")
    p.add_argument("--time", type=float, default=None, help="Override MET time; default uses TRIGTIME from TCAT if available")
    p.add_argument("--ft2", default=None, help="Optional explicit path to gll_pt_*.fits")
    p.add_argument("--tcat", default=None, help="Optional explicit path to glg_tcat*.fit")
    p.add_argument("--grb-ra", type=float, default=None, help="Manual GRB RA in deg if TCAT is absent or incomplete")
    p.add_argument("--grb-dec", type=float, default=None, help="Manual GRB Dec in deg if TCAT is absent or incomplete")
    p.add_argument("--json", action="store_true", help="Print JSON")
    p.add_argument("--indent", type=int, default=2, help="JSON indent level")
    p.add_argument("--list-columns", action="store_true", help="Print FT2 SC_DATA columns, then exit")
    return p.parse_args()


def _list_columns(path: Path) -> None:
    with fits.open(path, memmap=True) as hdul:
        hdu = _find_sc_data_hdu(hdul)
        print(f"Using extension: {hdu.name}")
        for name in hdu.columns.names or []:
            print(name)


def build_payload(trigger_id: str, base_dir: Path, ft2_path: Optional[Path], tcat_path: Optional[Path],
                  time_override: Optional[float], grb_ra_override: Optional[float], grb_dec_override: Optional[float]) -> Dict[str, object]:
    trigger_folder = find_trigger_folder(base_dir, trigger_id)
    auto_ft2, auto_tcat = find_ft2_and_tcat(trigger_folder, trigger_id)
    ft2_file = ft2_path or auto_ft2
    tcat_file = tcat_path or auto_tcat

    triginfo: Optional[TriggerInfo] = None
    if tcat_file is not None and tcat_file.exists():
        triginfo = read_trigger_info_from_tcat(trigger_id, tcat_file)
    else:
        triginfo = TriggerInfo(str(trigger_id), None, None, None, None, None, None, {})

    grb_ra = grb_ra_override if grb_ra_override is not None else triginfo.ra_deg
    grb_dec = grb_dec_override if grb_dec_override is not None else triginfo.dec_deg
    met = time_override if time_override is not None else triginfo.trigtime

    records = read_gll_pt_axes(ft2_file)
    if met is None:
        raise ValueError("No MET time available. Provide --time or use a TCAT with TRIGTIME.")
    rec = select_record_nearest_time(records, met)
    if rec.ra_zenith_deg is None or rec.dec_zenith_deg is None:
        raise ValueError("Could not determine RA_ZENITH/DEC_ZENITH from FT2 row.")

    zenith_vec = _radec_to_vec(rec.ra_zenith_deg, rec.dec_zenith_deg)
    east_hat, north_hat, zen_hat = east_north_zenith_basis(rec.ra_zenith_deg, rec.dec_zenith_deg)
    east_ra, east_dec = [float(x) for x in _vec_to_radec(east_hat)]
    north_ra, north_dec = [float(x) for x in _vec_to_radec(north_hat)]

    grb_local = None
    if grb_ra is not None and grb_dec is not None:
        grb_local = transform_radec_to_local(grb_ra, grb_dec, rec.ra_zenith_deg, rec.dec_zenith_deg)

    payload: Dict[str, object] = {
        "trigger_id": str(trigger_id),
        "trigger_folder": str(trigger_folder),
        "ft2_file": str(ft2_file),
        "tcat_file": None if tcat_file is None else str(tcat_file),
        "selected_met": float(met),
        "attitude": asdict(rec),
        "local_frame": {
            "definition": {
                "z_axis": "Earth-up / local zenith from RA_ZENITH, DEC_ZENITH",
                "x_axis": "local east at the zenith direction",
                "y_axis": "Z x X (approximately local north)",
                "phi_convention": "azimuth measured from +X toward +Y",
                "theta_convention": "angle away from +Z",
            },
            "east_radec_deg": {"ra": east_ra, "dec": east_dec},
            "north_radec_deg": {"ra": north_ra, "dec": north_dec},
            "zenith_radec_deg": {"ra": float(rec.ra_zenith_deg), "dec": float(rec.dec_zenith_deg)},
        },
    }
    if triginfo is not None:
        payload["trigger_info"] = asdict(triginfo)
    if grb_ra is not None and grb_dec is not None:
        payload["grb_radec_deg"] = {"ra": float(grb_ra), "dec": float(grb_dec)}
        payload["grb_local_coordinates"] = grb_local
    else:
        payload["grb_radec_deg"] = None
        payload["grb_local_coordinates"] = None
    return payload


def main() -> None:
    args = _parse_args()
    base_dir = Path('./GRBData/'+args.base_dir).expanduser().resolve()
    ft2_path = None if args.ft2 is None else Path(args.ft2).expanduser().resolve()
    tcat_path = None if args.tcat is None else Path(args.tcat).expanduser().resolve()

    if args.list_columns:
        if ft2_path is None:
            trigger_folder = find_trigger_folder(base_dir, args.trigger_id)
            ft2_path, _ = find_ft2_and_tcat(trigger_folder, args.trigger_id)
        _list_columns(ft2_path)
        return

    payload = build_payload(
        trigger_id=args.trigger_id,
        base_dir=base_dir,
        ft2_path=ft2_path,
        tcat_path=tcat_path,
        time_override=args.time,
        grb_ra_override=args.grb_ra,
        grb_dec_override=args.grb_dec,
    )

    if args.json:
        print(json.dumps(payload, indent=args.indent))
    else:
        print(f"Trigger folder: {payload['trigger_folder']}")
        print(f"FT2 file: {payload['ft2_file']}")
        print(f"TCAT file: {payload['tcat_file']}")
        print(f"Selected MET: {payload['selected_met']}")
        att = payload['attitude']
        print(f"SCZ (RA,Dec): ({att['ra_scz_deg']:.6f}, {att['dec_scz_deg']:.6f}) deg")
        print(f"SCX (RA,Dec): ({att['ra_scx_deg']:.6f}, {att['dec_scx_deg']:.6f}) deg")
        print(f"ZENITH (RA,Dec): ({att['ra_zenith_deg']:.6f}, {att['dec_zenith_deg']:.6f}) deg")
        tr = payload.get('trigger_info', {})
        if payload['grb_radec_deg'] is not None:
            grb = payload['grb_radec_deg']
            loc = payload['grb_local_coordinates']
            print(f"GRB (RA,Dec): ({grb['ra']:.6f}, {grb['dec']:.6f}) deg")
            print(
                "GRB local frame: "
                f"x_east={loc['x_east']:.6f}, y_north={loc['y_north']:.6f}, z_zenith={loc['z_zenith']:.6f}, "
                f"theta={loc['theta_deg']:.6f} deg, phi={loc['phi_deg']:.6f} deg, "
                f"elevation={loc['elevation_deg']:.6f} deg"
            )
        else:
            print("GRB RA/Dec not available from TCAT; use --grb-ra/--grb-dec.")
        if isinstance(tr, dict) and tr.get('header_keywords_used'):
            print(f"TCAT header keys used: {tr['header_keywords_used']}")


if __name__ == "__main__":
    main()
