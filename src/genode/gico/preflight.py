"""Validate common paired measurements without fitting or changing artifacts."""

import argparse
import json

from genode.gico.train_gico import load_config, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_config(load_config(args.config), dry_run=True), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
