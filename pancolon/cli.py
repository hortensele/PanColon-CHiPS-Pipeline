"""Command-line interface for PanColon-CHiPS-Pipeline.

    python pancolon_pipeline.py <step> --config config/pipeline.yaml
    python pancolon_pipeline.py all   --config config/pipeline.yaml [--from project --to build_pt]

Steps, in order:
    tile  to_hdf5  project  cluster_filter  assign_hpc  build_pt  infer_survival  attention_map

Use --dry-run to print the exact underlying commands without executing them, and
--no-env-switch if you have already activated the right conda env yourself.
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .steps import STEPS, STEP_MAP, STEP_ORDER


class _Opts:
    def __init__(self, dry_run, no_env_switch):
        self.dry_run = dry_run
        self.no_env_switch = no_env_switch


def _add_common(p):
    p.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved commands without running them.")
    p.add_argument("--no-env-switch", action="store_true",
                   help="Assume the correct conda env is already active.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pancolon_pipeline.py",
        description="WSI -> CHiPS survival score + attention maps (inference pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="<step>")

    for name, _fn, desc in STEPS:
        sp = sub.add_parser(name, help=desc, description=desc)
        _add_common(sp)

    ap = sub.add_parser("all", help="Run every step in order (optionally a sub-range).")
    _add_common(ap)
    ap.add_argument("--from", dest="from_step", choices=STEP_ORDER, default=STEP_ORDER[0],
                    help="First step to run (default: tile).")
    ap.add_argument("--to", dest="to_step", choices=STEP_ORDER, default=STEP_ORDER[-1],
                    help="Last step to run (default: attention_map).")

    ep = sub.add_parser("export",
                        help="Export a self-contained results bundle for the viewer.")
    _add_common(ep)

    sub.add_parser("list", help="List the pipeline steps and exit.")
    return parser


def _run_one(name, cfg, opts):
    print(f"\n{'='*70}\n== STEP: {name}\n{'='*70}")
    return STEP_MAP[name](cfg, opts)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "list":
        print("Pipeline steps (in order):")
        for i, (name, _fn, desc) in enumerate(STEPS, 1):
            print(f"  {i}. {name:16s} {desc}")
        return 0

    cfg = load_config(args.config)
    opts = _Opts(dry_run=args.dry_run, no_env_switch=args.no_env_switch)

    if args.command == "export":
        from .export_bundle import export_bundle
        return export_bundle(cfg, opts)

    if args.command == "all":
        i0 = STEP_ORDER.index(args.from_step)
        i1 = STEP_ORDER.index(args.to_step)
        if i0 > i1:
            raise SystemExit(f"--from ({args.from_step}) is after --to ({args.to_step}).")
        for name in STEP_ORDER[i0:i1 + 1]:
            _run_one(name, cfg, opts)
        print("\nAll requested steps completed.")
        return 0

    return _run_one(args.command, cfg, opts)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
