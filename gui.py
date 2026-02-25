#!/usr/bin/env python3
"""Desktop GUI for topogen — interactive topographic map generation using DearPyGui."""

import array
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np

from topogen import (
    compute_delaunay,
    compute_grid,
    compute_levels,
    export_dxf,
    filter_outliers,
    load_ply,
    render_png,
    render_to_figure,
)

# --- State ---
_state = {
    "points": None,          # raw (N,3) array from PLY
    "input_path": None,      # Path to loaded PLY
    "precomputed": None,     # cached (tri, x_ft, y_ft, z_ft) from compute_delaunay
    "last_filter": None,     # filter strength used for cached precomputed
    "xi_grid": None,
    "yi_grid": None,
    "zi_grid": None,
    "x_range": None,
    "y_range": None,
    "levels": None,
    "index_levels": None,
}

PREVIEW_W = 1600
PREVIEW_H = 1600
DEBOUNCE_SEC = 0.2

_debounce_lock = threading.Lock()
_debounce_seq = 0  # incremented on every settings change


def _fig_to_rgba(fig, width, height) -> np.ndarray:
    """Render a matplotlib Figure to a flat RGBA float32 numpy array for DearPyGui."""
    fig.set_dpi(200)
    fig.set_size_inches(width / 200, height / 200)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        int(fig.get_figheight() * 200), int(fig.get_figwidth() * 200), 4
    )
    # Resize to exact target if needed
    if buf.shape[0] != height or buf.shape[1] != width:
        from PIL import Image
        img = Image.fromarray(buf)
        img = img.resize((width, height), Image.LANCZOS)
        buf = np.array(img)
    # Normalize to 0-1 floats — stay in numpy (avoid slow array.array copy)
    return np.ascontiguousarray(buf, dtype=np.float32).ravel() * (1.0 / 255.0)


def _on_file_selected(sender, app_data):
    """Callback when a PLY file is selected."""
    selections = app_data.get("selections", {})
    if not selections:
        return
    # Take first selected file
    file_path = list(selections.values())[0]
    path = Path(file_path)
    if path.suffix.lower() != ".ply":
        dpg.set_value("status_text", f"Error: not a .ply file")
        return

    dpg.set_value("status_text", f"Loading {path.name}...")
    try:
        _state["points"] = load_ply(path)
        _state["input_path"] = path
        dpg.set_value("file_label", path.name)
        dpg.set_value("status_text", f"Loaded {len(_state['points']):,} points from {path.name}")
        _schedule_generate()
    except Exception as e:
        dpg.set_value("status_text", f"Error loading file: {e}")


def _on_setting_changed(sender=None, app_data=None, user_data=None):
    """Called when any slider changes — debounces and regenerates."""
    _schedule_generate()


def _schedule_generate():
    """Debounce: wait DEBOUNCE_SEC after the last change before generating."""
    global _debounce_seq
    with _debounce_lock:
        _debounce_seq += 1
        seq = _debounce_seq

    def _worker():
        time.sleep(DEBOUNCE_SEC)
        with _debounce_lock:
            if seq != _debounce_seq:
                return  # a newer change superseded us
        _do_generate()

    threading.Thread(target=_worker, daemon=True).start()


def _do_generate():
    """Generate the topo map with current slider values."""
    if _state["points"] is None:
        return

    dpg.set_value("status_text", "Generating...")

    smooth = dpg.get_value("slider_smooth")
    filter_str = dpg.get_value("slider_filter")
    simplify = dpg.get_value("slider_simplify")
    interval = dpg.get_value("slider_interval")
    resolution = dpg.get_value("slider_resolution")

    try:
        points = _state["points"]
        if filter_str > 0:
            points = filter_outliers(points, strength=filter_str)

        # Reuse cached Delaunay triangulation when filter setting hasn't changed
        precomputed = None
        if _state["last_filter"] == filter_str and _state["precomputed"] is not None:
            precomputed = _state["precomputed"]
        else:
            precomputed = compute_delaunay(points)
            _state["precomputed"] = precomputed
            _state["last_filter"] = filter_str

        xi, yi, zi, xr, yr = compute_grid(points, resolution, smooth,
                                           precomputed=precomputed)
        levels, index_levels = compute_levels(zi, interval)

        _state["xi_grid"] = xi
        _state["yi_grid"] = yi
        _state["zi_grid"] = zi
        _state["x_range"] = xr
        _state["y_range"] = yr
        _state["levels"] = levels
        _state["index_levels"] = index_levels

        # Render preview
        fig = render_to_figure(xi, yi, zi, levels, index_levels, xr, yr,
                               simplify_tolerance=simplify)
        rgba = _fig_to_rgba(fig, PREVIEW_W, PREVIEW_H)
        plt.close(fig)

        dpg.set_value("preview_texture", rgba)
        dpg.set_value("status_text", "Done.")
    except Exception as e:
        dpg.set_value("status_text", f"Error: {e}")
        import traceback
        traceback.print_exc()


def _on_save_png_selected(sender, app_data):
    """Callback for PNG save dialog."""
    file_path = app_data.get("file_path_name", "")
    if not file_path:
        return
    if not file_path.lower().endswith(".png"):
        file_path += ".png"
    if _state["zi_grid"] is None:
        dpg.set_value("status_text", "Generate a map first.")
        return
    dpg.set_value("status_text", f"Saving PNG to {file_path}...")
    try:
        simplify = dpg.get_value("slider_simplify")
        render_png(
            _state["xi_grid"], _state["yi_grid"], _state["zi_grid"],
            _state["levels"], _state["index_levels"],
            _state["x_range"], _state["y_range"],
            file_path, dpi=300, simplify_tolerance=simplify,
        )
        dpg.set_value("status_text", f"Saved PNG to {file_path}")
    except Exception as e:
        dpg.set_value("status_text", f"Error saving PNG: {e}")


def _on_save_dxf_selected(sender, app_data):
    """Callback for DXF save dialog."""
    file_path = app_data.get("file_path_name", "")
    if not file_path:
        return
    if not file_path.lower().endswith(".dxf"):
        file_path += ".dxf"
    if _state["zi_grid"] is None:
        dpg.set_value("status_text", "Generate a map first.")
        return
    dpg.set_value("status_text", f"Saving DXF to {file_path}...")
    try:
        simplify = dpg.get_value("slider_simplify")
        export_dxf(
            _state["xi_grid"], _state["yi_grid"], _state["zi_grid"],
            _state["levels"], _state["index_levels"],
            file_path, simplify_tolerance=simplify,
        )
        dpg.set_value("status_text", f"Saved DXF to {file_path}")
    except Exception as e:
        dpg.set_value("status_text", f"Error saving DXF: {e}")


def main():
    dpg.create_context()

    # -- Blank preview texture --
    blank = array.array("f", [0.9, 0.9, 0.9, 1.0] * (PREVIEW_W * PREVIEW_H))
    with dpg.texture_registry():
        dpg.add_raw_texture(PREVIEW_W, PREVIEW_H, blank,
                            format=dpg.mvFormat_Float_rgba, tag="preview_texture")

    # -- File dialogs --
    with dpg.file_dialog(directory_selector=False, show=False, callback=_on_file_selected,
                         tag="open_dialog", width=700, height=400):
        dpg.add_file_extension(".ply", color=(0, 255, 0, 255))

    with dpg.file_dialog(directory_selector=False, show=False, callback=_on_save_png_selected,
                         tag="save_png_dialog", width=700, height=400, default_filename="topo_map.png"):
        dpg.add_file_extension(".png", color=(0, 255, 0, 255))

    with dpg.file_dialog(directory_selector=False, show=False, callback=_on_save_dxf_selected,
                         tag="save_dxf_dialog", width=700, height=400, default_filename="topo_map.dxf"):
        dpg.add_file_extension(".dxf", color=(0, 255, 0, 255))

    SIDEBAR_W = 260

    # -- Main window --
    with dpg.window(tag="primary"):
        with dpg.group(horizontal=True):
            # --- Sidebar ---
            with dpg.child_window(width=SIDEBAR_W, tag="sidebar"):
                dpg.add_button(label="Open PLY File", width=-1,
                               callback=lambda: dpg.show_item("open_dialog"))
                dpg.add_text("No file loaded", tag="file_label", wrap=SIDEBAR_W - 16)

                dpg.add_separator()

                dpg.add_text("Smooth (ft)")
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Gaussian smoothing sigma. Higher values\nblur terrain more. 0 = off.", wrap=250)
                dpg.add_slider_float(label="##smooth", tag="slider_smooth",
                                     default_value=1.0, min_value=0.0, max_value=5.0,
                                     format="%.1f", width=-1,
                                     callback=_on_setting_changed)

                dpg.add_text("Filter (MADs)")
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Outlier removal strength in median absolute\n"
                                 "deviations. Lower = more aggressive.\n"
                                 "0 = off. Try 2-5.", wrap=250)
                dpg.add_slider_float(label="##filter", tag="slider_filter",
                                     default_value=0.0, min_value=0.0, max_value=10.0,
                                     format="%.1f", width=-1,
                                     callback=_on_setting_changed)

                dpg.add_text("Simplify (ft)")
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Douglas-Peucker contour simplification\n"
                                 "tolerance. Reduces vertex count.\n"
                                 "0 = off. Try 0.3-1.0.", wrap=250)
                dpg.add_slider_float(label="##simplify", tag="slider_simplify",
                                     default_value=0.0, min_value=0.0, max_value=2.0,
                                     format="%.2f", width=-1,
                                     callback=_on_setting_changed)

                dpg.add_text("Interval (ft)")
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Elevation distance between contour lines.\n"
                                 "Smaller = more lines, more detail.", wrap=250)
                dpg.add_slider_float(label="##interval", tag="slider_interval",
                                     default_value=1.0, min_value=0.5, max_value=5.0,
                                     format="%.1f", width=-1,
                                     callback=_on_setting_changed)

                dpg.add_text("Resolution")
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Grid interpolation resolution (cells per axis).\n"
                                 "Higher = finer detail but slower.", wrap=250)
                dpg.add_slider_int(label="##resolution", tag="slider_resolution",
                                   default_value=500, min_value=100, max_value=1000,
                                   width=-1,
                                   callback=_on_setting_changed)

                dpg.add_separator()

                dpg.add_button(label="Save PNG", width=-1,
                               callback=lambda: dpg.show_item("save_png_dialog"))
                dpg.add_button(label="Save DXF", width=-1,
                               callback=lambda: dpg.show_item("save_dxf_dialog"))

                dpg.add_separator()
                dpg.add_text("Ready.", tag="status_text", wrap=SIDEBAR_W - 16)

            # --- Preview area ---
            with dpg.child_window(tag="preview_area"):
                dpg.add_image("preview_texture", tag="preview_image")

    # Resize preview to fill available space
    def _resize_preview():
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        img_size = min(vw - SIDEBAR_W - 24, vh - 16)
        if img_size < 100:
            img_size = 100
        dpg.configure_item("preview_image", width=img_size, height=img_size)

    dpg.set_viewport_resize_callback(lambda: _resize_preview())

    dpg.create_viewport(title="topogen", width=1100, height=860)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    _resize_preview()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
