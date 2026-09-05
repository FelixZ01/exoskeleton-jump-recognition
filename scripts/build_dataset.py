#!/usr/bin/env python3
"""Build private, model-ready windows from canonical aligned sessions."""

import argparse

from exojump.dataset import build_windows, save_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--max-missing-fraction", type=float, default=0.05)
    args = parser.parse_args()
    arrays = build_windows(
        args.aligned_root,
        window_size=args.window_size,
        stride=args.stride,
        max_missing_fraction=args.max_missing_fraction,
    )
    save_windows(args.output, arrays)
    print(f"Saved {len(arrays['y'])} windows to {args.output}")


if __name__ == "__main__":
    main()
