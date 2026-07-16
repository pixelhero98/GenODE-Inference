from __future__ import annotations

import argparse
import json

from genode.data.molecule_xyz import (
    DEFAULT_MOLECULE_DATASET_KEY,
    DEFAULT_MOLECULE_SPLIT_SEED,
    MOLECULE_GROUP_DATASET_KEYS,
    build_balanced_molecule_stratum_groups,
    discover_molecule_xyz_strata,
    molecule_processed_path,
    molecule_raw_zip_path,
    prepare_molecule_xyz_group_datasets,
    prepare_molecule_xyz_all_strata,
    prepare_molecule_xyz_zip,
    write_molecule_group_manifests,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fixed-order molecule XYZ trajectories for OTFlow training.")
    parser.add_argument("--scenario_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    parser.add_argument("--stratum", default="")
    parser.add_argument("--all_strata", action="store_true", default=False)
    parser.add_argument("--balanced_groups", action="store_true", default=False)
    parser.add_argument("--prepare_group_data", action="store_true", default=False)
    parser.add_argument("--discover_only", action="store_true", default=False)
    parser.add_argument("--zip_path", default=None)
    parser.add_argument("--zip_paths", default="")
    parser.add_argument("--group_root", default=None)
    parser.add_argument("--group_scenario_keys", default=",".join(MOLECULE_GROUP_DATASET_KEYS))
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--include_pattern", default="*")
    parser.add_argument("--exclude_pattern", default="")
    parser.add_argument("--split_seed", type=int, default=DEFAULT_MOLECULE_SPLIT_SEED)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if bool(args.balanced_groups):
        zip_paths = [part.strip() for part in str(args.zip_paths or "").split(",") if part.strip()]
        if not zip_paths:
            zip_paths = [molecule_raw_zip_path(args.scenario_key, args.stratum) if args.zip_path is None else args.zip_path]
        scenario_keys = tuple(part.strip() for part in str(args.group_scenario_keys).split(",") if part.strip())
        if bool(args.discover_only):
            print(json.dumps(build_balanced_molecule_stratum_groups(zip_paths, dataset_keys=scenario_keys), indent=2))
            return
        if bool(args.prepare_group_data):
            metadata = prepare_molecule_xyz_group_datasets(
                zip_paths,
                args.group_root,
                dataset_keys=scenario_keys,
                split_seed=int(args.split_seed),
            )
        else:
            metadata = write_molecule_group_manifests(zip_paths, args.group_root, dataset_keys=scenario_keys)
        print(json.dumps(metadata, indent=2))
        return
    zip_path = molecule_raw_zip_path(args.scenario_key, args.stratum) if args.zip_path is None else args.zip_path
    if bool(args.discover_only):
        print(json.dumps(discover_molecule_xyz_strata(zip_path), indent=2))
        return
    if bool(args.all_strata):
        processed_root = molecule_processed_path(args.scenario_key, None) if args.processed_dir is None else args.processed_dir
        metadata = prepare_molecule_xyz_all_strata(
            zip_path,
            processed_root,
            dataset_key=str(args.scenario_key),
            include_pattern=str(args.include_pattern),
            exclude_pattern=str(args.exclude_pattern),
            split_seed=int(args.split_seed),
        )
    else:
        processed_dir = molecule_processed_path(args.scenario_key, args.stratum) if args.processed_dir is None else args.processed_dir
        metadata = prepare_molecule_xyz_zip(
            zip_path,
            processed_dir,
            dataset_key=args.scenario_key,
            stratum=args.stratum,
            split_seed=int(args.split_seed),
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
