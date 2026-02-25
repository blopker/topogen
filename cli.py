#!/usr/bin/env python3
"""CLI entrypoint for topogen — generate topo maps from Scaniverse scan exports."""

import argparse
import sys
from pathlib import Path

from topogen import (
    compute_grid,
    compute_levels,
    export_dxf,
    filter_outliers,
    load_ply,
    render_png,
)


def main():
    parser = argparse.ArgumentParser(description="Generate a topo map from a Scaniverse scan export.")
    parser.add_argument("input", help="Path to input file (.ply)")
    parser.add_argument("-o", "--output", default="topo_map.png",
                        help="Output path (.png for image, .dxf for CAD) (default: topo_map.png)")
    parser.add_argument("--interval", type=float, default=1.0, help="Contour interval in feet (default: 1)")
    parser.add_argument("--resolution", type=int, default=500, help="Grid resolution (default: 500)")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI (default: 300)")
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="Gaussian smoothing sigma in feet (default: 1.0, 0 = none)")
    parser.add_argument("--filter", type=float, default=0, dest="filter_strength",
                        help="Outlier filter strength in MADs (default: 0 = off, try 2-5; lower = more aggressive)")
    parser.add_argument("--simplify", type=float, default=0,
                        help="Contour line simplification tolerance in feet (default: 0 = off, try 0.5-2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix == ".ply":
        points = load_ply(input_path)
    else:
        print(f"Error: Unsupported format '{suffix}'. Supported: .ply", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(points)} points from {input_path.name}")

    if args.filter_strength > 0:
        points = filter_outliers(points, strength=args.filter_strength)

    xi_grid, yi_grid, zi_grid, x_range, y_range = compute_grid(
        points, args.resolution, args.smooth,
    )
    levels, index_levels = compute_levels(zi_grid, args.interval)

    out_suffix = Path(args.output).suffix.lower()
    if out_suffix == ".dxf":
        export_dxf(xi_grid, yi_grid, zi_grid, levels, index_levels, args.output,
                   simplify_tolerance=args.simplify)
    else:
        render_png(
            xi_grid, yi_grid, zi_grid, levels, index_levels,
            x_range, y_range, args.output, args.dpi,
            simplify_tolerance=args.simplify,
        )


if __name__ == "__main__":
    main()
