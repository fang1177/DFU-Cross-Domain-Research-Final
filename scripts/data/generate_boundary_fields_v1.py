"""
Stage 4A Task 3B: generate and persist boundary fields for canonical masks.

Usage examples
--------------
Validate or generate (idempotent, never overwrites conflicts)::

    python scripts/data/generate_boundary_fields_v1.py

Explicitly replace conflicting or already-valid artifacts::

    python scripts/data/generate_boundary_fields_v1.py --regenerate

Debug on a small subset::

    python scripts/data/generate_boundary_fields_v1.py --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/data/generate_boundary_fields_v1.py`` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.boundary_fields import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MASKS_DIR,
    DEFAULT_OUTPUT_DIR,
    MAX_CAP,
    TAU,
    generate_boundary_fields,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate/validate Stage 4A boundary fields (phi_fp, phi_fn)."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--max-cap", type=float, default=MAX_CAP)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Explicitly rewrite artifacts, including conflicts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N eligible records (deterministic order).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.project_root)
    manifest = Path(args.manifest)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    if not manifest.is_absolute():
        manifest = root / manifest
    if not masks_dir.is_absolute():
        masks_dir = root / masks_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    summary = generate_boundary_fields(
        manifest_path=manifest,
        masks_dir=masks_dir,
        output_dir=output_dir,
        tau=args.tau,
        max_cap=args.max_cap,
        regenerate=args.regenerate,
        limit=args.limit,
    )

    print(f"Eligible records: {summary['eligible_records']}")
    print(f"Status counts: {summary['status_counts']}")
    print(f"Empty masks: {summary['empty_mask_records']}")
    print(f"Thin lesions (interior all zero): {summary['interior_all_zero_records']}")
    print(f"Stale manifest records: {summary['stale_manifest_records']}")
    print(f"Conflicts: {len(summary['conflict_ids'])}")
    print(f"Failures: {len(summary['failed_ids'])}")
    print(f"Output: {output_dir}")
    if summary["conflict_ids"]:
        print("Conflicting record ids: " + ", ".join(summary["conflict_ids"]))
    if summary["failed_ids"]:
        print("Failed record ids: " + ", ".join(summary["failed_ids"]))
    return 0 if not (summary["has_failures"] or summary["has_conflicts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())