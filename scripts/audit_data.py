#!/usr/bin/env python3
"""Generate private file-level and public aggregate dataset inventories."""

import argparse
import json

from exojump.inventory import write_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Root of the recovered experiment data")
    parser.add_argument("--private-output", required=True, help="Ignored directory for detailed paths/IDs")
    parser.add_argument("--public-output", default="data/manifests", help="Anonymous aggregate output")
    args = parser.parse_args()
    summary = write_inventory(args.source, args.private_output, args.public_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
