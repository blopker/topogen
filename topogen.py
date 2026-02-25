"""Core library for generating topographic contour maps from 3D scan point clouds."""

from pathlib import Path

import ezdxf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from plyfile import PlyData
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

M_PER_FT = 0.3048


def simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a polyline using the Douglas-Peucker algorithm.

    Args:
        points: (N, 2) array of x, y vertices.
        tolerance: Maximum perpendicular distance (in same units as points)
                   a point can deviate before being kept.

    Returns:
        Simplified (M, 2) array with M <= N.
    """
    if len(points) <= 2:
        return points

    # Find the point with the greatest distance from the start-end line
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-12:
        # Degenerate segment — all points collapse to one
        dists = np.sqrt(np.sum((points - start) ** 2, axis=1))
    else:
        # Perpendicular distance from each point to the start→end line
        t = np.clip(np.dot(points - start, line_vec) / line_len_sq, 0, 1)
        proj = start + t[:, np.newaxis] * line_vec
        dists = np.sqrt(np.sum((points - proj) ** 2, axis=1))

    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]

    if max_dist > tolerance:
        left = simplify_polyline(points[: max_idx + 1], tolerance)
        right = simplify_polyline(points[max_idx:], tolerance)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def filter_outliers(
    points: np.ndarray, strength: float, bin_count: int = 50
) -> np.ndarray:
    """Remove elevation outliers using local median absolute deviation (MAD).

    Divides the x-y plane into bins, computes per-bin median and MAD of z,
    then removes points whose z deviates from their bin median by more than
    `strength` MADs.

    Args:
        points: (N, 3) array of x, y, z.
        strength: Number of MADs a point can deviate before removal.
                  Lower = more aggressive. Typical range: 1.0–5.0.
        bin_count: Number of bins along each x/y axis.

    Returns:
        Filtered (M, 3) array.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_edges = np.linspace(x.min(), x.max(), bin_count + 1)
    y_edges = np.linspace(y.min(), y.max(), bin_count + 1)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, bin_count - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, bin_count - 1)
    bin_id = x_bin * bin_count + y_bin

    keep = np.ones(len(points), dtype=bool)
    for b in np.unique(bin_id):
        mask = bin_id == b
        zb = z[mask]
        if len(zb) < 3:
            continue
        median = np.median(zb)
        mad = np.median(np.abs(zb - median))
        if mad < 1e-9:
            mad = np.std(zb)
        if mad < 1e-9:
            continue
        outlier = np.abs(zb - median) > strength * mad
        idx = np.where(mask)[0]
        keep[idx[outlier]] = False

    n_removed = np.sum(~keep)
    print(
        f"Outlier filter (strength={strength:.1f}): removed {n_removed} of {len(points)} points "
        f"({100 * n_removed / len(points):.1f}%)"
    )
    return points[keep]


def load_ply(path: Path) -> np.ndarray:
    """Load a PLY file and return an (N, 3) array of [x, y, z] vertices."""
    ply = PlyData.read(str(path))
    verts = ply["vertex"]
    x = np.array(verts["x"], dtype=np.float64)
    y = np.array(verts["y"], dtype=np.float64)
    z = np.array(verts["z"], dtype=np.float64)
    return np.column_stack([x, y, z])


def compute_grid(points, grid_resolution, smooth):
    """Interpolate point cloud onto a regular grid and apply smoothing.

    Returns xi_grid, yi_grid, zi_grid (all in feet), x_range, y_range.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_ft = x / M_PER_FT
    y_ft = y / M_PER_FT
    z_ft = z / M_PER_FT

    # Shift origin to (0, 0)
    x_ft -= x_ft.min()
    y_ft -= y_ft.min()

    margin = 0.02
    x_range = x_ft.max() - x_ft.min()
    y_range = y_ft.max() - y_ft.min()
    xi = np.linspace(
        x_ft.min() - margin * x_range, x_ft.max() + margin * x_range, grid_resolution
    )
    yi = np.linspace(
        y_ft.min() - margin * y_range, y_ft.max() + margin * y_range, grid_resolution
    )
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    print(
        f"Interpolating {len(points)} points onto {grid_resolution}x{grid_resolution} grid..."
    )
    zi_grid = griddata((x_ft, y_ft), z_ft, (xi_grid, yi_grid), method="linear")

    if smooth > 0:
        cell_size_ft = x_range / grid_resolution
        sigma_cells = smooth / cell_size_ft
        mask = np.isnan(zi_grid)
        zi_filled = np.where(mask, 0, zi_grid)
        weights = np.where(mask, 0, 1.0)
        zi_smooth = gaussian_filter(zi_filled, sigma=sigma_cells)
        w_smooth = gaussian_filter(weights, sigma=sigma_cells)
        w_smooth[w_smooth == 0] = np.nan
        zi_grid = zi_smooth / w_smooth
        zi_grid[mask] = np.nan
        print(
            f"Applied Gaussian smoothing: sigma = {smooth:.1f} ft ({sigma_cells:.1f} grid cells)"
        )

    return xi_grid, yi_grid, zi_grid, x_range, y_range


def compute_levels(zi_grid, contour_interval_ft):
    """Compute contour levels and index levels from grid data."""
    z_min = np.nanmin(zi_grid)
    z_max = np.nanmax(zi_grid)
    level_min = np.floor(z_min / contour_interval_ft) * contour_interval_ft
    level_max = np.ceil(z_max / contour_interval_ft) * contour_interval_ft
    levels = np.arange(level_min, level_max + contour_interval_ft, contour_interval_ft)

    index_interval = 5.0
    index_levels = levels[
        np.isclose(levels % index_interval, 0, atol=contour_interval_ft * 0.1)
    ]

    print(f"Elevation range: {z_min:.1f} ft – {z_max:.1f} ft")
    print(f"Contour interval: {contour_interval_ft} ft")
    print(f"Generating {len(levels)} contour levels...")

    return levels, index_levels


def _extract_contours(xi_grid, yi_grid, zi_grid, levels, simplify_tolerance):
    """Extract contour segments from the grid using matplotlib.

    Returns a list of (level_value, (N,2) vertices_array) tuples.
    """
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(xi_grid, yi_grid, zi_grid, levels=levels)
    plt.close(fig_tmp)

    contours = []
    total_before = 0
    total_after = 0
    for i, level_val in enumerate(cs.levels):
        for seg in cs.allsegs[i]:
            if len(seg) < 2:
                continue
            total_before += len(seg)
            if simplify_tolerance > 0:
                seg = simplify_polyline(seg, simplify_tolerance)
            total_after += len(seg)
            contours.append((float(level_val), seg))

    if simplify_tolerance > 0:
        print(
            f"Simplified contours: {total_before} → {total_after} vertices "
            f"({100 * (1 - total_after / total_before):.0f}% reduction, tolerance={simplify_tolerance:.1f} ft)"
        )

    return contours


def export_dxf(
    xi_grid, yi_grid, zi_grid, levels, index_levels, output_path, simplify_tolerance=0
):
    """Export contour lines as 3D polylines in a DXF file.

    Regular contours go on the "CONTOUR" layer; index contours (every 5 ft)
    go on the "CONTOUR_INDEX" layer with elevation labels.
    """
    contours = _extract_contours(xi_grid, yi_grid, zi_grid, levels, simplify_tolerance)

    index_set = set(np.round(index_levels, 6))

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Set up layers
    doc.layers.add("CONTOUR", color=7)  # white/black
    doc.layers.add("CONTOUR_INDEX", color=1)  # red
    doc.layers.add("CONTOUR_LABEL", color=3)  # green

    # Set drawing units to feet (Imperial)
    doc.header["$INSUNITS"] = 2  # feet
    doc.header["$LUNITS"] = 2  # decimal
    doc.header["$MEASUREMENT"] = 0  # Imperial

    total_polylines = 0

    for level_val, seg in contours:
        level_rounded = round(level_val, 6)
        is_index = level_rounded in index_set

        layer = "CONTOUR_INDEX" if is_index else "CONTOUR"
        poly = msp.add_lwpolyline(
            [(float(v[0]), float(v[1])) for v in seg],
            dxfattribs={"layer": layer, "elevation": level_val},
        )
        poly.close(False)
        total_polylines += 1

        if is_index and len(seg) > 2:
            mid = seg[len(seg) // 2]
            msp.add_text(
                f"{level_val:.0f}'",
                height=1.0,
                dxfattribs={
                    "layer": "CONTOUR_LABEL",
                    "insert": (float(mid[0]), float(mid[1]), level_val),
                },
            )

    doc.saveas(output_path)
    print(f"Saved DXF with {total_polylines} polylines to {output_path}")


def _add_scale_bar(ax, x_range_ft, y_range_ft):
    """Draw a scale bar in the bottom-left corner."""
    target = x_range_ft * 0.2
    nice_lengths = [1, 2, 5, 10, 15, 20, 25, 50, 100]
    bar_len = min(nice_lengths, key=lambda v: abs(v - target))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.04
    bar_height = (ylim[1] - ylim[0]) * 0.006

    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            bar_len,
            bar_height,
            facecolor="black",
            edgecolor="black",
            linewidth=0.5,
            zorder=10,
        )
    )
    tick_h = bar_height * 3
    for bx in [x0, x0 + bar_len]:
        ax.plot([bx, bx], [y0, y0 + tick_h], color="black", linewidth=0.8, zorder=10)
    ax.text(
        x0 + bar_len / 2,
        y0 + tick_h + (ylim[1] - ylim[0]) * 0.005,
        f"{bar_len} ft",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        zorder=10,
    )


def render_to_figure(
    xi_grid,
    yi_grid,
    zi_grid,
    levels,
    index_levels,
    x_range,
    y_range,
    simplify_tolerance=0,
) -> Figure:
    """Render the topographic map and return a matplotlib Figure (without saving to disk).

    Useful for embedding previews in a GUI.
    """
    z_min = np.nanmin(zi_grid)
    z_max = np.nanmax(zi_grid)

    contours = _extract_contours(xi_grid, yi_grid, zi_grid, levels, simplify_tolerance)
    index_set = set(np.round(index_levels, 6))

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor="white")
    ax.set_facecolor("white")

    # Filled contours (not simplified — these are smooth fills)
    norm = Normalize(vmin=z_min, vmax=z_max)
    cf = ax.contourf(
        xi_grid,
        yi_grid,
        zi_grid,
        levels=levels,
        cmap="YlGn",
        norm=norm,
        alpha=0.5,
    )

    # Draw simplified contour lines manually
    for level_val, seg in contours:
        level_rounded = round(level_val, 6)
        is_index = level_rounded in index_set
        color = "#3d2010" if is_index else "#5a3d2b"
        lw = 0.9 if is_index else 0.3
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=lw)

    # Labels on index contours
    if len(index_levels) > 0:
        for level_val, seg in contours:
            level_rounded = round(level_val, 6)
            if level_rounded in index_set and len(seg) > 4:
                mid = seg[len(seg) // 2]
                ax.annotate(
                    f"{level_val:.0f}'",
                    xy=(mid[0], mid[1]),
                    fontsize=7,
                    color="#3d2010",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7
                    ),
                )

    cbar = fig.colorbar(cf, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Elevation (ft)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    _add_scale_bar(ax, x_range, y_range)

    ax.set_aspect("equal")
    ax.set_xlabel("X (ft)")
    ax.set_ylabel("Y (ft)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.2, alpha=0.3, color="gray")

    fig.tight_layout()
    return fig


def render_png(
    xi_grid,
    yi_grid,
    zi_grid,
    levels,
    index_levels,
    x_range,
    y_range,
    output_path,
    dpi,
    simplify_tolerance=0,
):
    """Render the topographic map as a PNG image."""
    fig = render_to_figure(
        xi_grid,
        yi_grid,
        zi_grid,
        levels,
        index_levels,
        x_range,
        y_range,
        simplify_tolerance=simplify_tolerance,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved PNG to {output_path}")
