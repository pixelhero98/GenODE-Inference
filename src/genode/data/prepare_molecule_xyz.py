from __future__ import annotations

import argparse
import json

from genode.data.molecule_xyz import (
    DEFAULT_MOLECULE_DATASET_KEY,
    DEFAULT_MOLECULE_SPLIT_SEED,
    discover_molecule_xyz_strata,
    default_molecule_processed_dir,
    default_molecule_raw_zip,
    prepare_molecule_xyz_all_strata,
    prepare_molecule_xyz_zip,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fixed-order molecule XYZ trajectories for OTFlow training.")
    parser.add_argument("--dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    parser.add_argument("--stratum", default="")
    parser.add_argument("--all_strata", action="store_true", default=False)
    parser.add_argument("--discover_only", action="store_true", default=False)
    parser.add_argument("--zip_path", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--include_pattern", default="*")
    parser.add_argument("--exclude_pattern", default="")
    parser.add_argument("--split_seed", type=int, default=DEFAULT_MOLECULE_SPLIT_SEED)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    zip_path = default_molecule_raw_zip(args.dataset_key, args.stratum) if args.zip_path is None else args.zip_path
    if bool(args.discover_only):
        print(json.dumps(discover_molecule_xyz_strata(zip_path), indent=2))
        return
    if bool(args.all_strata):
        processed_root = default_molecule_processed_dir(args.dataset_key, None) if args.processed_dir is None else args.processed_dir
        metadata = prepare_molecule_xyz_all_strata(
            zip_path,
            processed_root,
            dataset_key=str(args.dataset_key),
            include_pattern=str(args.include_pattern),
            exclude_pattern=str(args.exclude_pattern),
            split_seed=int(args.split_seed),
        )
    else:
        processed_dir = default_molecule_processed_dir(args.dataset_key, args.stratum) if args.processed_dir is None else args.processed_dir
        metadata = prepare_molecule_xyz_zip(
            zip_path,
            processed_dir,
            dataset_key=args.dataset_key,
            stratum=args.stratum,
            split_seed=int(args.split_seed),
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
