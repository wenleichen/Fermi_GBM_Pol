#!/usr/bin/env python3
"""
Run the GBM analysis pipeline for one or more (PA, DOP) pairs.

Pipeline steps per pair
-----------------------
1. Run gll_pt_trigger_frame.py to write frame_<trigger>.json
2. Run skymap_from_trigger.py to build the mixed sky map for the given PA/DOP
3. Run gbm_expected_flux_vs_counts.py using the produced histogram

The script creates a pair-specific output folder whose name includes both
PA and DOP, for example:

    ./output/bn180720598/pa147_dop0p100/

This avoids collisions between runs and makes it easy to compare results.

Examples
--------
Single pair:
    python gbm_pipeline_pa_dop.py \
      --trigger-id 180720598 \
      --pa 147 \
      --dop 0.1

Multiple pairs (paired mode):
    python gbm_pipeline_pa_dop.py \
      --trigger-id 180720598 \
      --pa-list 147,120 \
      --dop-list 0.1,0.5 \
      --pairwise

Grid mode over all combinations:
    python gbm_pipeline_pa_dop.py \
      --trigger-id 180720598 \
      --pa-list 0,45,90 \
      --dop-list 0,0.5,1
"""
from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def format_pa_tag(pa_deg: float) -> str:
    s = f"{pa_deg:.3f}".rstrip('0').rstrip('.')
    s = s.replace('-', 'm').replace('.', 'p')
    return f"pa{s}"


def format_dop_tag(dop: float) -> str:
    s = f"{dop:.3f}"
    s = s.replace('-', 'm').replace('.', 'p')
    return f"dop{s}"


def build_pairs(args: argparse.Namespace) -> List[Tuple[float, float]]:
    if args.pa is not None and args.dop is not None:
        return [(float(args.pa), float(args.dop))]

    if args.pa_list is None or args.dop_list is None:
        raise ValueError("Provide either --pa and --dop, or both --pa-list and --dop-list")

    pas = parse_float_list(args.pa_list)
    dops = parse_float_list(args.dop_list)
    if args.pairwise:
        if len(pas) != len(dops):
            raise ValueError("--pairwise requires --pa-list and --dop-list to have the same length")
        return list(zip(pas, dops))
    return list(itertools.product(pas, dops))


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def run_cmd(cmd: Sequence[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    pretty = ' '.join(shlex.quote(x) for x in cmd)
    print(f"\n$ {pretty}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=None if cwd is None else str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trigger-id', required=True)

    p.add_argument('--pa', type=float, default=None, help='Single PA in degrees')
    p.add_argument('--dop', type=float, default=None, help='Single DOP in [0,1]')
    p.add_argument('--pa-list', default=None, help='Comma-separated list of PAs in degrees')
    p.add_argument('--dop-list', default=None, help='Comma-separated list of DOP values')
    p.add_argument('--pairwise', action='store_true', help='Zip pa-list with dop-list instead of taking all combinations')

    p.add_argument('--base-dir', default='.')
    p.add_argument('--sim-root', default='./SimData')
    p.add_argument('--db-root', default='./GBMDRMdb002')
    p.add_argument('--frame-script', default='./gll_pt_trigger_frame.py')
    p.add_argument('--skymap-script', default='./skymap_from_trigger.py')
    p.add_argument('--expected-script', default='./gbm_expected_flux_vs_counts.py')
    p.add_argument('--leaf-tool', default='./gbm_leaf_aeff_tool.py')

    p.add_argument('--output-root', default='./output')
    p.add_argument('--energy-min', type=float, default=50.0)
    p.add_argument('--energy-max', type=float, default=300.0)
    p.add_argument('--n-energy', type=int, default=24)
    p.add_argument('--energy-reduce', choices=['mean', 'sum', 'max'], default='mean')
    p.add_argument('--count-emin', type=float, default=50.0)
    p.add_argument('--count-emax', type=float, default=300.0)
    p.add_argument('--dry-run', action='store_true')

    args = p.parse_args()

    pairs = build_pairs(args)
    if not pairs:
        raise ValueError("No (PA, DOP) pairs to run")

    trigger_id = str(args.trigger_id)
    frame_script = Path(args.frame_script).expanduser().resolve()
    skymap_script = Path(args.skymap_script).expanduser().resolve()
    expected_script = Path(args.expected_script).expanduser().resolve()
    leaf_tool = Path(args.leaf_tool).expanduser().resolve()
    sim_root = Path(args.sim_root).expanduser().resolve()
    db_root = Path(args.db_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    for path, desc in [
        (frame_script, 'frame script'),
        (skymap_script, 'skymap script'),
        (expected_script, 'expected-vs-counts script'),
        (leaf_tool, 'leaf effective-area tool'),
        (sim_root, 'simulation root'),
        (db_root, 'GBMDRMdb002 root'),
    ]:
        ensure_exists(path, desc)

    trigger_root = output_root / f"bn{trigger_id}"
    trigger_root.mkdir(parents=True, exist_ok=True)

    print(f"Trigger: {trigger_id}")
    print(f"Runs: {len(pairs)}")

    for i, (pa, dop) in enumerate(pairs, start=1):
        if not (0.0 <= dop <= 1.0):
            raise ValueError(f"DOP must be in [0,1], got {dop}")

        pair_tag = f"{format_pa_tag(pa)}_{format_dop_tag(dop)}"
        outdir = trigger_root / pair_tag
        outdir.mkdir(parents=True, exist_ok=True)

        frame_json = outdir / f"frame_bn{trigger_id}.json"
        hist_name = f"hist_bn{trigger_id}_Pol_0_0_{format_pa_tag(pa)}_{format_dop_tag(dop)}.npz"
        # Some skymap scripts include the selected folder name in the histogram path and some may choose
        # a different Pol_*_* bin than Pol_0_0. We therefore first run the script, then discover the hist file.

        print(f"\n=== [{i}/{len(pairs)}] trigger={trigger_id} pa={pa} dop={dop} ===")
        print(f"Output directory: {outdir}")

        run_cmd([
            sys.executable, str(frame_script), trigger_id,
            '--base-dir', str(args.base_dir),
            '--json-out', str(frame_json),
        ], dry_run=args.dry_run)

        run_cmd([
            sys.executable, str(skymap_script),
            '--frame-json', str(frame_json),
            '--sim-root', str(sim_root),
            '--outdir', str(outdir),
            '--pa-deg', str(pa),
            '--dop', str(dop),
        ], dry_run=args.dry_run)

        if args.dry_run:
            hist_path = outdir / hist_name
        else:
            hist_candidates = sorted(outdir.glob(f"hist_bn{trigger_id}_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not hist_candidates:
                raise FileNotFoundError(f"No histogram file found in {outdir} after skymap step")
            hist_path = hist_candidates[0]

        run_cmd([
            sys.executable, str(expected_script),
            '--trigger-id', trigger_id,
            '--base-dir', str(args.base_dir),
            '--frame-script', str(frame_script),
            '--leaf-tool', str(leaf_tool),
            '--sky-hist', str(hist_path),
            '--db-root', str(db_root),
            '--energy-min', str(args.energy_min),
            '--energy-max', str(args.energy_max),
            '--n-energy', str(args.n_energy),
            '--energy-reduce', str(args.energy_reduce),
            '--count-emin', str(args.count_emin),
            '--count-emax', str(args.count_emax),
            '--outdir', str(outdir),
            '--tag', pair_tag,
        ], dry_run=args.dry_run)

    print("\nAll requested PA/DOP runs finished.")


if __name__ == '__main__':
    main()
