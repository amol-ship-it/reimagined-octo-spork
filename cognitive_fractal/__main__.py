"""CLI entry point: python3 -m cognitive_fractal data.csv"""

import argparse
import sys

from .csv_discoverer import CSVFunctionDiscoverer
from .pattern_store import DEFAULT_STORE_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Discover the generating function from a CSV of numbers.",
    )
    parser.add_argument("csv_file", help="Path to CSV file (1 or 2 columns)")
    parser.add_argument("--max-degree", type=int, default=3,
                        help="Max polynomial degree inside compositions (default: 3)")
    parser.add_argument("--window-size", type=int, default=50,
                        help="Sliding window size for streaming (default: 50)")
    parser.add_argument("--passes", type=int, default=2,
                        help="Number of streaming passes (default: 2)")
    parser.add_argument("--predict", type=int, default=0,
                        help="Predict this many future values (default: 0)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--db", type=str, default="default",
                        help=f"Pattern store path (default: {DEFAULT_STORE_PATH}). "
                             "Use --no-db to disable persistence.")
    parser.add_argument("--no-db", action="store_true",
                        help="Disable pattern persistence (don't save or load)")

    args = parser.parse_args()

    db = None if args.no_db else args.db

    d = CSVFunctionDiscoverer(
        args.csv_file,
        max_degree=args.max_degree,
        window_size=args.window_size,
        passes=args.passes,
        verbose=not args.quiet,
        db=db,
    )
    result = d.run()

    if args.predict > 0:
        print()
        future = result.predict(args.predict)
        print(f"Next {args.predict} predicted values:")
        for i, v in enumerate(future, start=1):
            print(f"  {i}: {v:.10g}")


if __name__ == "__main__":
    main()
