#!/usr/bin/env python3
"""Evaluate session-level feature baselines with six-participant LOPO validation."""

import argparse
import json

from exojump.baseline import evaluate_lopo_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = evaluate_lopo_baseline(args.aligned_root, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
